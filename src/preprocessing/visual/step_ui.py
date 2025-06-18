import streamlit as st
import pandas as pd
import numpy as np
from contextlib import contextmanager

# Import preprocessing steps
from src.preprocessing.steps.missing_values import handle_missing_values, get_missing_summary
from src.preprocessing.steps.outliers import remove_outliers
from src.preprocessing.steps.log_transform import apply_log_transform
from src.preprocessing.steps.normalization import apply_scaling
from src.preprocessing.steps.feature_extraction import apply_feature_extraction
from src.preprocessing.steps.feature_creation import apply_feature_creation
from src.preprocessing.steps.feature_selection import apply_feature_selection
from src.preprocessing.steps.encoding import apply_encoding

# Import pipeline management
from src.preprocessing.pipeline_manager import add_step, undo_last_step, get_applied_steps
from src.data.saver import save_processed_version

# Import visualization helpers
from src.preprocessing.visual.comparison import (
    plot_boxplot_comparison,
    plot_histogram_comparison,
    plot_scatter_comparison,
    plot_heatmap_comparison
)

# Import UI helpers
@contextmanager
def loading_button(message: str):
    """Context manager for showing loading state."""
    progress = st.progress(0)
    status = st.empty()
    try:
        status.text(message)
        yield
    finally:
        progress.empty()
        status.empty()

def tick_animation(message: str):
    """Show success animation with message."""
    st.success(f"✅ {message}")

def _is_step_applied(step_name: str, state: dict) -> bool:
    """Check if a preprocessing step has been applied."""
    return any(s["name"] == step_name for s in state.get("applied_steps", []))

def _apply_step_without_ui(df: pd.DataFrame, step_info: dict) -> pd.DataFrame:
    """Reapply a step given its stored info (after undo)."""
    name = step_info["name"]
    params = step_info["params"]
    if name == "missing_values":
        return handle_missing_values(df, **params)
    elif name == "outliers":
        return remove_outliers(df, **params)
    elif name == "log_transform":
        return apply_log_transform(df, **params)
    elif name == "scaling":
        return apply_scaling(df, **params)
    elif name == "feature_extraction":
        return apply_feature_extraction(df, **params)
    elif name == "feature_creation":
        return apply_feature_creation(df, **params)
    elif name == "feature_selection":
        return apply_feature_selection(df, **params)
    elif name == "encoding":
        return apply_encoding(df, **params)
    else:
        raise ValueError(f"Unknown step name: {name}")

def run_preprocessing_ui(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Main preprocessing UI that guides users through each preprocessing step."""
    # Initialize state if needed
    if "processed_df" not in state:
        state["processed_df"] = df.copy()
    if "applied_steps" not in state:
        state["applied_steps"] = []

    st.markdown("## 🧹 Preprocessing Pipeline")
    st.markdown("Use each section below to apply preprocessing steps.")

    # Handle undo/finish buttons
    col_undo, col_finish = st.columns([1, 1])
    with col_undo:
        if st.button("↩️ Undo Last Step"):
            if state["applied_steps"]:
                undo_info = state["applied_steps"].pop()
                st.info(f"Reverting step: {undo_info['name']}")
                new_df = df.copy()
                for step in state["applied_steps"]:
                    new_df = _apply_step_without_ui(new_df, step)
                state["processed_df"] = new_df
                st.rerun()
            else:
                st.warning("No steps to undo.")
    
    with col_finish:
        if st.button("✅ Finish Preprocessing"):
            if state["applied_steps"]:
                state["preprocessing_done"] = True
                st.success("Preprocessing complete!")
                return state["processed_df"]
            else:
                st.warning("No preprocessing steps applied yet.")

    # Show preprocessing steps
    preprocessing_steps = [
        ("missing_values", _missing_values_ui),
        ("outliers", _outliers_ui),
        ("log_transform", _log_transform_ui),
        ("scaling", _scaling_ui),
        ("feature_extraction", _feature_extraction_ui),
        ("feature_creation", _feature_creation_ui),
        ("feature_selection", _feature_selection_ui),
        ("encoding", _encoding_ui)
    ]

    # Display each preprocessing step in order
    for step_name, step_func in preprocessing_steps:
        if not _is_step_applied(step_name, state):
            step_func(df, state)

    # Return processed_df if preprocessing is done
    if state.get("preprocessing_done"):
        return state["processed_df"]
    
    return None

def _outliers_ui(df: pd.DataFrame, state: dict):
    """UI for outlier detection and removal with a separate visualization step."""
    exp = st.expander("🧯 2. Outlier Detection & Removal", expanded=False)
    with exp:
        st.write("Detect and remove outliers. You can visualize the effect before applying the step.")
        curr_df = state["processed_df"]

        # --- Session state to manage the preview ---
        ui_key = "outlier_preview_state"
        if ui_key not in st.session_state:
            st.session_state[ui_key] = {
                "show_preview": False,
                "preview_df": None,
                "cols_to_plot": [],
                "plot_opts": []
            }

        # --- User Inputs ---
        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        cols = st.multiselect(
            "Select numeric columns to check for outliers",
            options=numeric_cols,
            default=numeric_cols
        )
        method = st.selectbox(
            "Method",
            ["z_score", "iqr", "isolation_forest", "lof", "elliptic_envelope", "dbscan"]
        )
        plot_opts = st.multiselect(
            "Plot types for preview",
            ["Box-plot", "Histogram"],
            default=["Box-plot", "Histogram"]
        )
        params = {"method": method, "columns": cols}

        # --- Buttons for Visualize and Apply ---
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("👁️ Visualize Effect"):
                if not cols:
                    st.warning("Please select at least one column to visualize.")
                else:
                    with st.spinner("Generating preview..."):
                        # Run removal temporarily for visualization
                        preview_df = remove_outliers(curr_df.copy(), **params)
                        # Store preview data in session state
                        st.session_state[ui_key]['preview_df'] = preview_df
                        st.session_state[ui_key]['cols_to_plot'] = cols
                        st.session_state[ui_key]['plot_opts'] = plot_opts
                        st.session_state[ui_key]['show_preview'] = True
                    st.rerun()

        with btn_col2:
            if st.button("✅ Apply Outlier Removal Step"):
                try:
                    with loading_button("Detecting & removing outliers..."):
                        new_df = remove_outliers(curr_df, **params)

                    # Record the step permanently
                    step_info = {"name": "outliers", "params": params}
                    version_path = save_processed_version(new_df, state, step="outliers")
                    step_info["version_path"] = version_path
                    add_step(state, step_info)
                    state["processed_df"] = new_df

                    # Hide the preview plots after applying
                    st.session_state[ui_key]['show_preview'] = False

                    tick_animation("Outliers removed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error removing outliers: {str(e)}")

        # --- Persistent Visualization Section ---
        # This block is outside the buttons and controlled by session state
        if st.session_state[ui_key].get("show_preview"):
            st.markdown("---")
            st.markdown("### 📊 Preview: Before vs. After")
            
            preview_df = st.session_state[ui_key]['preview_df']
            cols_to_plot = st.session_state[ui_key]['cols_to_plot']
            plot_opts_to_show = st.session_state[ui_key]['plot_opts']

            # Loop through the columns selected during the preview
            for col in cols_to_plot:
                st.markdown(f"#### Comparison for Column: `{col}`")

                # Each function is called only ONCE, generating 2 sub-plots
                if "Box-plot" in plot_opts_to_show:
                    plot_boxplot_comparison(curr_df, preview_df, col)
                
                if "Histogram" in plot_opts_to_show:
                    plot_histogram_comparison(curr_df, preview_df, col)

def _scaling_ui(df: pd.DataFrame, state: dict):
    """UI for scaling/normalization preprocessing step."""
    exp = st.expander("✏️ 4. Normalization / Standardization", expanded=False)
    with exp:
        st.write("Scale numeric columns.")
        curr_df = state["processed_df"]
        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        cols = st.multiselect("Select numeric columns to scale", 
                            options=numeric_cols, 
                            default=numeric_cols)

        mode = st.selectbox("Scaling Type", 
                          ["standardization", "normalization", "robust", "maxabs"])
        params = {"method": mode, "columns": cols}

        plot_opts = st.multiselect("Plot types", 
                                 ["Box-plot", "Histogram", "Q-Q Plot"], 
                                 default=["Box-plot", "Histogram"])

        if st.button("Apply Scaling Step"):
            with loading_button("Scaling data..."):
                new_df = apply_scaling(curr_df, columns=cols, method=mode)
            
            # Show before/after
            st.markdown("**Before vs After Scaling**")
            for col in cols:
                before = curr_df[[col]].dropna()
                after = new_df[[col]].dropna()
                cols_disp = st.columns(2)
                with cols_disp[0]:
                    st.write(f"Before: {col}")
                    if "Box-plot" in plot_opts:
                        plot_boxplot_comparison(before, after, col)
                with cols_disp[1]:
                    st.write(f"After: {col}")
                    if "Box-plot" in plot_opts:
                        plot_boxplot_comparison(before, after, col)
                        
            step_info = {"name": "scaling", "params": params}
            version_path = save_processed_version(new_df, state, step="scaling")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Scaling applied")
            st.rerun()


def _missing_values_ui(df: pd.DataFrame, state: dict):
    """UI for handling missing values in the dataset with full dataset view option."""
    exp = st.expander("🔍 1. Handle Missing Values", expanded=True)
    with exp:
        st.write("Detect and handle missing values.")
        curr_df = state["processed_df"]
        
        # Create unique session state keys for this specific UI
        ui_key = f"missing_values_ui_{id(exp)}"
        if ui_key not in st.session_state:
            st.session_state[ui_key] = {
                "show_full_before": False,
                "show_full_after": False,
                "step_applied": False
            }
        ui_state = st.session_state[ui_key]
        
        # Check if we're coming from an undo action
        step_was_applied = any(step.get("name") == "missing_values" for step in state.get("applied_steps", []))
        if step_was_applied and not ui_state["step_applied"]:
            # Reset UI state on undo
            ui_state["show_full_before"] = False
            ui_state["show_full_after"] = False
            ui_state["step_applied"] = False
        
        # Show initial missing values summary
        st.markdown("### Missing Values Summary")
        before_summary = get_missing_summary(curr_df)
        st.dataframe(
            before_summary,
            use_container_width=True,
            hide_index=False,
            column_config={
                "missing_count": st.column_config.NumberColumn("Missing Count"),
                "missing_pct": st.column_config.NumberColumn("Missing %", format="%.2f%%")
            }
        )

        # Before dataset controls
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("👁️ View Full Dataset (Before)"):
                ui_state["show_full_before"] = True
        with col2:
            if ui_state["show_full_before"]:
                if st.button("❌ Close Preview (Before)"):
                    ui_state["show_full_before"] = False
                    
        # Show before dataset if requested
        if ui_state["show_full_before"]:
            st.dataframe(curr_df, height=400)
        
        # Configure missing values handling
        st.markdown("### Configure Missing Values Handling")
        cols = st.multiselect(
            "Select columns to handle missing values",
            options=curr_df.columns.tolist(),
            default=curr_df.columns.tolist()
        )

        action = st.radio(
            "Action",
            ["drop", "impute"],
            index=1 if curr_df[cols].isnull().any().any() else 0
        )
        
        params = {"columns": cols, "action": action}

        if action == "impute":
            strategy = st.selectbox(
                "Imputation Strategy",
                ["mean", "median", "most_frequent", "constant", "ffill", "bfill", "knn"]
            )
            params["strategy"] = strategy
            
            if strategy == "constant":
                fill_value = st.text_input("Fill value (constant):", "")
                if fill_value:
                    try:
                        params["fill_value"] = float(fill_value)
                    except ValueError:
                        params["fill_value"] = fill_value
                else:
                    params["fill_value"] = None

        apply_clicked = st.button("Apply Missing Values Step", use_container_width=True)
        if apply_clicked:
            try:
                with st.spinner("Applying missing values handling..."):
                    new_df = handle_missing_values(curr_df, **params)
                    
                    # Update state and record step
                    step_info = {
                        "name": "missing_values",
                        "params": params
                    }
                    
                    version_path = save_processed_version(new_df, state, step="missing_values")
                    step_info["version_path"] = version_path
                    add_step(state, step_info)
                    state["processed_df"] = new_df
                    
                    # Update UI state
                    ui_state["step_applied"] = True
                    ui_state["show_full_after"] = False  # Reset preview state
                    
                    st.success("✅ Missing values handled successfully!")
                    # Force rerun to refresh UI
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error handling missing values: {str(e)}")
        
        # After dataset controls - show below Apply button
        if ui_state["step_applied"]:
            st.markdown("---")
            st.markdown("### Processed Dataset Preview")
            
            # After dataset controls
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("👁️ View Full Dataset (After)"):
                    ui_state["show_full_after"] = True
            with col2:
                if ui_state["show_full_after"]:
                    if st.button("❌ Close Preview (After)"):
                        ui_state["show_full_after"] = False
                        
            # Show after dataset if requested
            if ui_state["show_full_after"]:
                st.dataframe(state["processed_df"], height=400)
        
        # Undo handling - show before/after comparison if step was applied
        if step_was_applied:
            st.markdown("---")
            st.markdown("### Results Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before:**")
                # Recalculate before summary from original state
                original_df = state["original_df"] if "original_df" in state else df
                original_summary = get_missing_summary(original_df)
                st.dataframe(
                    original_summary,
                    use_container_width=True,
                    hide_index=False
                )
            
            with col2:
                st.markdown("**After:**")
                after_summary = get_missing_summary(state["processed_df"])
                st.dataframe(
                    after_summary,
                    use_container_width=True,
                    hide_index=False
                )
                
def _log_transform_ui(df: pd.DataFrame, state: dict):
    exp = st.expander("🔁 3. Log Transformation", expanded=False)
    with exp:
        st.write("Apply log or similar transformations to reduce skew.")
        curr_df = state["processed_df"]

        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        cols = st.multiselect("Select numeric columns", options=numeric_cols, default=numeric_cols)

        strategy = st.selectbox("Transformation", [
            "box-cox ( +ve and right-skew)", "log1p (zero values)", "log10 (large ranges)", "log2(binary)", "log(x+ k) (-ve values)", "sqrt (mild skewness)"
        ])
        params = {"strategy": strategy, "columns": cols}

        # For 'log(x+ k)', ask for k
        if strategy == "log(x+ k)":
            k = st.number_input("k (offset)", value=1.0)
            params["offset"] = k

        # Plot options
        plot_opts = st.multiselect("Plot types", ["Histogram", "Box-plot", "Q-Q Plot"], default=["Histogram"])

        if st.button("Apply Log Transformation Step"):
            with loading_button("Applying transformation..."):
                new_df = apply_log_transform(curr_df, columns=cols, strategy=strategy, offset=params.get("offset", None))
            # Show before/after
            st.markdown("**Before vs After Log Transformation**")
            for col in cols:
                before = curr_df[[col]].dropna()
                after = new_df[[col]].dropna()
                cols_disp = st.columns(2)
                with cols_disp[0]:
                    st.write(f"Before: {col}")
                    if "Histogram" in plot_opts:
                        plot_histogram_comparison(before, after, col)
                with cols_disp[1]:
                    st.write(f"After: {col}")
                    if "Histogram" in plot_opts:
                        plot_histogram_comparison(before, after, col)
            # Record step
            step_info = {"name": "log_transform", "params": params}
            version_path = save_processed_version(new_df, state, step="log_transform")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Log transformation applied")
            st.experimental_rerun()


def apply_scaling(df: pd.DataFrame, state: dict):
    exp = st.expander("✏️ 4. Normalization / Standardization", expanded=False)
    with exp:
        st.write("Scale numeric columns.")
        curr_df = state["processed_df"]
        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        cols = st.multiselect("Select numeric columns to scale", options=numeric_cols, default=numeric_cols)

        mode = st.selectbox("Scaling Type", ["standardization", "normalization", "robust", "maxabs"])
        params = {"method": mode, "columns": cols}

        plot_opts = st.multiselect("Plot types", ["Box-plot", "Histogram", "Q-Q Plot"], default=["Box-plot", "Histogram"])

        if st.button("Apply Scaling Step"):
            with loading_button("Scaling data..."):
                new_df = apply_scaling(curr_df, columns=cols, method=mode)
            # Show before/after
            st.markdown("**Before vs After Scaling**")
            for col in cols:
                before = curr_df[[col]].dropna()
                after = new_df[[col]].dropna()
                cols_disp = st.columns(2)
                with cols_disp[0]:
                    st.write(f"Before: {col}")
                    if "Box-plot" in plot_opts:
                        plot_boxplot_comparison(before, after, col)
                with cols_disp[1]:
                    st.write(f"After: {col}")
                    if "Box-plot" in plot_opts:
                        plot_boxplot_comparison(before, after, col)
            step_info = {"name": "scaling", "params": params}
            version_path = save_processed_version(new_df, state, step="scaling")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Scaling applied")
            st.experimental_rerun()


def _feature_extraction_ui(df: pd.DataFrame, state: dict):
    exp = st.expander("🍭 5. Feature Extraction (PCA, LDA, etc.)", expanded=False)
    with exp:
        st.write("Dimensionality reduction / feature extraction.")
        curr_df = state["processed_df"]
        numeric_cols = curr_df.select_dtypes(include=["number"]).columns.tolist()
        # For LDA, need labels: prompt user if classification and labels available in state?
        method = st.selectbox("Method", ["PCA", "LDA"], index=0)
        params = {"method": method}

        if method == "PCA":
            n_components = st.number_input("Number of components", min_value=1, max_value=min(len(numeric_cols), 10), value=min(2, len(numeric_cols)))
            params["n_components"] = n_components
        else:  # LDA
            # Assume label column stored in state["label_col"] or ask user
            label_col = st.selectbox("Label column for LDA", options=state.get("label_options", numeric_cols))
            params["n_components"] = st.number_input("Number of components (<= classes-1)", min_value=1, value=1)
            params["label_col"] = label_col

        plot_opts = st.multiselect("Plot types", ["2D Scatter", "3D Scatter", "Heatmap"], default=["2D Scatter"])

        if st.button("Apply Feature Extraction Step"):
            with loading_button("Applying feature extraction..."):
                new_df = apply_feature_extraction(curr_df, **params)
            # Show before/after visuals if feasible
            st.markdown("**Feature Extraction Result Preview**")
            # e.g., for PCA: show scatter of first two components
            if method == "PCA" and "2D Scatter" in plot_opts:
                plot_scatter_comparison(curr_df, new_df, params.get("n_components", 2), title="PCA Components")
            if method == "LDA" and "2D Scatter" in plot_opts:
                plot_scatter_comparison(curr_df, new_df, 2, label_col=params.get("label_col"), title="LDA Projection")
            step_info = {"name": "feature_extraction", "params": params}
            version_path = save_processed_version(new_df, state, step="feature_extraction")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Feature extraction done")
            st.experimental_rerun()


def _feature_creation_ui(df: pd.DataFrame, state: dict):
    exp = st.expander("🛠️ 6. Feature Creation", expanded=False)
    with exp:
        st.write("Create new features (polynomial, binning, date-time, etc.).")
        curr_df = state["processed_df"]
        method = st.selectbox("Method", ["Polynomial Features", "Binning/Discretization", "Date-Time Decomposition", "Aggregation (GroupBy)", "Text Features", "Flag/Boolean Features"])
        params = {"method": method}
        # Depending on method, gather further inputs
        if method == "Polynomial Features":
            cols = st.multiselect("Select numeric columns for polynomial expansion", options=curr_df.select_dtypes(include=["number"]).columns.tolist())
            degree = st.number_input("Degree", min_value=2, max_value=5, value=2)
            params.update({"columns": cols, "degree": degree})
        elif method == "Binning/Discretization":
            col = st.selectbox("Select column to bin", options=curr_df.columns.tolist())
            bins = st.number_input("Number of bins", min_value=2, max_value=50, value=5)
            params.update({"column": col, "bins": bins})
        elif method == "Date-Time Decomposition":
            col = st.selectbox("Select datetime column", options=curr_df.select_dtypes(include=["datetime", "object"]).columns.tolist())
            params.update({"column": col})
        elif method == "Aggregation (GroupBy)":
            group_cols = st.multiselect("Group by columns", options=curr_df.columns.tolist())
            agg_col = st.selectbox("Column to aggregate", options=curr_df.select_dtypes(include=["number"]).columns.tolist())
            agg_func = st.selectbox("Aggregation function", ["sum", "mean", "median", "max", "min", "count"])
            params.update({"group_cols": group_cols, "agg_col": agg_col, "agg_func": agg_func})
        elif method == "Text Features":
            col = st.selectbox("Select text column", options=curr_df.select_dtypes(include=["object", "string"]).columns.tolist())
            # e.g., length, word count
            text_feat = st.selectbox("Feature to extract", ["Length", "Word Count"])
            params.update({"column": col, "text_feat": text_feat})
        elif method == "Flag/Boolean Features":
            col = st.selectbox("Select column to create flag from", options=curr_df.columns.tolist())
            condition = st.text_input("Condition (e.g., df[col] > value):")
            params.update({"column": col, "condition": condition})

        if st.button("Apply Feature Creation Step"):
            with loading_button("Creating features..."):
                new_df = apply_feature_creation(curr_df, **params)
            st.markdown("**Feature Creation Result Preview**")
            st.dataframe(new_df.head())
            step_info = {"name": "feature_creation", "params": params}
            version_path = save_processed_version(new_df, state, step="feature_creation")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Feature creation done")
            st.experimental_rerun()


def _feature_selection_ui(df: pd.DataFrame, state: dict):
    exp = st.expander("📌 7. Feature Selection", expanded=False)
    with exp:
        st.write("Select relevant features.")
        curr_df = state["processed_df"]
        method = st.selectbox("Method", ["Correlation-based", "Chi-Square", "ANOVA", "RFE", "Lasso", "Tree-based Importance", "PCA", "Mutual Information"])
        params = {"method": method}
        if method == "Correlation-based":
            threshold = st.slider("Correlation threshold (absolute)", 0.0, 1.0, 0.5)
            params["threshold"] = threshold
        elif method in {"Chi-Square", "ANOVA"}:
            label_col = st.selectbox("Label column", options=state.get("label_options", curr_df.columns.tolist()))
            params["label_col"] = label_col
            top_k = st.number_input("Select top K features", min_value=1, max_value=curr_df.shape[1]-1, value=5)
            params["top_k"] = top_k
        elif method == "RFE":
            label_col = st.selectbox("Label column", options=state.get("label_options", curr_df.columns.tolist()))
            estimator = st.selectbox("Estimator", ["LogisticRegression", "RandomForestClassifier", "SVR"])  # example
            n_features = st.number_input("Number of features to select", min_value=1, max_value=curr_df.shape[1]-1, value=5)
            params.update({"label_col": label_col, "estimator": estimator, "n_features": n_features})
        elif method == "Lasso":
            alpha = st.number_input("Alpha (regularization)", min_value=0.0, value=1.0)
            params["alpha"] = alpha
        elif method == "Tree-based Importance":
            label_col = st.selectbox("Label column", options=state.get("label_options", curr_df.columns.tolist()))
            params["label_col"] = label_col
        elif method == "PCA":
            n_components = st.number_input("Number of components", min_value=1, max_value=min(curr_df.shape[1], 10), value=2)
            params["n_components"] = n_components
        elif method == "Mutual Information":
            label_col = st.selectbox("Label column", options=state.get("label_options", curr_df.columns.tolist()))
            top_k = st.number_input("Top K features", min_value=1, max_value=curr_df.shape[1]-1, value=5)
            params.update({"label_col": label_col, "top_k": top_k})

        if st.button("Apply Feature Selection Step"):
            with loading_button("Selecting features..."):
                new_df = apply_feature_selection(curr_df, **params)
            st.markdown("**Selected Features Preview**")
            st.dataframe(new_df.head())
            step_info = {"name": "feature_selection", "params": params}
            version_path = save_processed_version(new_df, state, step="feature_selection")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Feature selection done")
            st.experimental_rerun()


def _encoding_ui(df: pd.DataFrame, state: dict):
    exp = st.expander("🧩 8. Encoding Categorical Features", expanded=False)
    with exp:
        st.write("Encode categorical features.")
        curr_df = state["processed_df"]
        cat_cols = curr_df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = st.multiselect("Select categorical columns to encode", options=cat_cols)
        if not cols:
            st.info("No categorical columns selected.")
            return

        method = st.selectbox("Encoding Method", ["One-Hot", "Label", "Target", "Frequency", "Binary", "Hash", "Ordinal"])
        params = {"method": method, "columns": cols}
        if method == "Ordinal":
            order_input = {}
            for col in cols:
                st.markdown(f"Specify order for '{col}' (comma-separated categories in desired order):")
                order = st.text_input(f"Order for {col}", "")
                if order:
                    order_input[col] = [x.strip() for x in order.split(",")]
            params["order"] = order_input

        if st.button("Apply Encoding Step"):
            with loading_button("Applying encoding..."):
                new_df = apply_encoding(curr_df, **params)
            st.markdown("**After Encoding Preview**")
            st.dataframe(new_df.head())
            step_info = {"name": "encoding", "params": params}
            version_path = save_processed_version(new_df, state, step="encoding")
            step_info["version_path"] = version_path
            add_step(state, step_info)
            state["processed_df"] = new_df
            tick_animation("Encoding done")
            st.experimental_rerun()

