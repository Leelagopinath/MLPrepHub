import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from contextlib import contextmanager

# Import preprocessing steps
from src.preprocessing.steps.missing_values import handle_missing_values, get_auto_impute_strategy, get_missing_summary
from src.preprocessing.steps.outliers import remove_outliers
from src.preprocessing.steps.log_transform import apply_log_transform,detect_best_transform
from src.preprocessing.steps.normalization import apply_scaling
from src.preprocessing.steps.feature_extraction import apply_feature_extraction
from src.preprocessing.steps.feature_creation import apply_feature_creation
from src.preprocessing.steps.feature_selection import apply_feature_selection
from src.preprocessing.steps.encoding import apply_encoding
from src.eda.visualization import (plot_correlation_heatmap)
# Import pipeline management
from src.preprocessing.pipeline_manager import add_step
from src.data.saver import save_processed_version
# Import comparison visuals
from src.preprocessing.visual.comparison import show_before_after_plots

@contextmanager
def loading_button(message: str):
    progress = st.progress(0)
    status = st.empty()
    try:
        status.text(message)
        yield
    finally:
        progress.empty()
        status.empty()

def tick_animation(message: str):
    st.success(f"✅ {message}")


def _is_step_applied(step_name: str, state: dict) -> bool:
    return any(s["name"] == step_name for s in state.get("applied_steps", []))


def _apply_step_without_ui(df: pd.DataFrame, step_info: dict) -> pd.DataFrame:
    name = step_info["name"]
    params = step_info.get("params", {})
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
        raise ValueError(f"Unknown step: {name}")
    
def clear_step_ui_session_state():
    """
    Clears all Streamlit session_state keys related to step UIs.
    """
    base_keys = [
        "missing_values_ui", "outliers_ui", "log_transform_ui", "scaling_ui",
        "feature_extraction_ui", "feature_creation_ui", "feature_selection_ui", "encoding_ui"
    ]
    # Also clear any keys with suffixes (e.g., _show_preview, _preview_df, etc.)
    for k in list(st.session_state.keys()):
        for base in base_keys:
            if k.startswith(base):
                del st.session_state[k]


def run_preprocessing_ui(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    # Initialize state
    if "original_df" not in state or state.get("original_df") is None:
        state["original_df"] = df.copy()
        state["processed_df"] = df.copy()
        state["applied_steps"] = []
        # reset target if new dataset
        state.pop("target_col", None)
        state.pop("preprocessing_done", None)
    
    if state.get("preprocessing_done"):
        return state["processed_df"]
    
    st.markdown("## 🧹 Preprocessing Pipeline")

    # Step 1: Target selection
    cols = state["original_df"].columns.tolist()
    previous_target = state.get("target_col")
    if previous_target not in cols or previous_target is None:
        default_index = 0
    else:
        default_index = cols.index(previous_target)

    if "target_col" not in state or state["target_col"] not in cols:
        st.markdown("### 🎯 Select Target Column")
        target = st.selectbox(
            "Choose the target (dependent variable):",
            options=cols,
            index=cols.index(previous_target) if previous_target in cols else 0,
            key="select_target_col"
        )
        if st.button("Confirm Target", key="confirm_target"):
            state["target_col"] = target
            st.success(f"Target column set to `{target}`.")
            st.rerun()
        else:
            st.info("Please confirm target to proceed.")
            return None

    target = state.get("target_col")
    st.markdown(f"**Current target:** `{target}`")
    st.markdown("---")

    # Undo / Finish buttons
    col_undo, col_finish = st.columns([1, 1])
    undo_clicked = False
    finish_clicked = False
    
    with col_undo:
        if st.button("↩️ Undo Last Step", key="undo_step"):
            undo_clicked = True
            
    with col_finish:
        if st.button("✅ Finish Preprocessing", key="finish_preproc"):
            finish_clicked = True
    if undo_clicked:
            if state.get("applied_steps"):
                # Pop last step
                last = state["applied_steps"].pop()
                st.info(f"Undoing: {last['name']}")
                # Rebuild processed_df from original through remaining steps
                new_df = state["original_df"].copy()
                for step in state["applied_steps"]:
                    new_df = _apply_step_without_ui(new_df, step)
                state["processed_df"] = new_df
                clear_step_ui_session_state()
                # Trigger rerun so the next unapplied step UI shows
                try:
                    st.rerun()
                except:
                    st.stop()
                # Clear all step-specific session_state keys so UIs reset
                ui_keys = {
                    "missing_values": "missing_values_ui",
                    "outliers": "outliers_ui",
                    "log_transform": "log_transform_ui",
                    "scaling": "scaling_ui",
                    "feature_extraction": "feature_extraction_ui",
                    "feature_creation": "feature_creation_ui",
                    "feature_selection": "feature_selection_ui",
                    "encoding": "encoding_ui"
                }
                for key in ui_keys.values():
                    if key in st.session_state:
                        del st.session_state[key]
                # Trigger rerun so the next unapplied step UI shows
                try:
                    st.rerun()
                except:
                    st.stop()
            else:
                st.warning("No steps to undo.")
    
    if finish_clicked:
            if state.get("applied_steps"):
                state["preprocessing_done"] = True
                st.success("✅ Preprocessing complete.")
                # Ensure only DataFrame is returned
                return state["processed_df"]
            else:
                st.warning("No preprocessing steps applied yet.")
        # If user has clicked Finish and state["preprocessing_done"] set, return processed_df
    if state.get("preprocessing_done"):
        # Ensure only DataFrame is returned
        return state["processed_df"]

    # Step 2: Preprocessing steps
    steps = [
        ("missing_values", _missing_values_ui),
        ("outliers", _outliers_ui),
        ("log_transform", _log_transform_ui),
        ("scaling", _scaling_ui),
        ("multicollinearity", _multicollinearity_ui),
        ("feature_extraction", _feature_extraction_ui),
        ("feature_creation", _feature_creation_ui),
        ("feature_selection", _feature_selection_ui),
        ("encoding", _encoding_ui)
    ]
    # Show all steps at once (wizard style)
    for name, func in steps:
        func(state["processed_df"], state)
    return None  # until user clicks Finish


# (Removed duplicate _apply_step_without_ui definition to avoid conflicts)

def clear_step_ui_session_state():
    """
    If your individual UI functions use st.session_state keys,
    delete them here so that after undo or re-running a step,
    the UI starts fresh.
    Example:
        keys = ["missing_values_ui", "outliers_ui", ...]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
    """
    # implement based on your UI function keys
    pass

    

# ------------------ Step UIs ------------------

def _missing_values_ui(df: pd.DataFrame, state: dict):
    ui_key = "missing_values_ui"
    exp = st.expander("🔍 1. Handle Missing Values", expanded=not _is_step_applied("missing_values", state))
    with exp:
        original_df = state.get("original_df", df)
        curr_df = state.get("processed_df", original_df)
        target = state.get("target_col")

        # 1️⃣ Missing Summary Before
        st.markdown("### 📋 Missing Values Summary (Before)")
        summary_before = get_missing_summary(original_df)
        st.dataframe(summary_before, use_container_width=True)

        # 2️⃣ Select Imputation Method
        st.markdown("### ⚙️ Imputation Method for Independent Features")
        method = st.selectbox("Choose Imputation Method", ["statistical", "knn"], key=ui_key + "_method")

        # 3️⃣ Select Columns
        missing_cols = [c for c in curr_df.columns if curr_df[c].isnull().any()]
        if target in missing_cols:
            st.info(f"Rows with missing target `{target}` will be dropped.")
            missing_cols.remove(target)

        if missing_cols:
            cols = st.multiselect(
                "Select columns to impute (leave empty to auto-select all)",
                options=missing_cols,
                default=missing_cols,
                key=ui_key + "_cols"
            )
        else:
            st.success("✅ No missing values in independent features.")
            return {}  # Treat as auto-complete step
        
        if st.button("👁️ Preview Dataset", key=ui_key+"_preview"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)

    # 4️⃣ Apply Button
    apply_clicked = st.button("✅ Apply Missing Value Imputation", key=ui_key + "_apply")
    if apply_clicked:
        try:
            with loading_button("Processing missing values..."):
                result = handle_missing_values(
                    curr_df,
                    target_col=target,
                    method=method,
                    columns=cols if cols else None
                )
                if isinstance(result, tuple) and len(result) == 2:
                    new_df, info = result
                else:
                    new_df = result
                    info = {}

            # Save & Record
            version_path = save_processed_version(new_df, state, step="missing_values")
            step_info = {
                "name": "missing_values",
                "params": {"method": method, "columns": cols or None},
                "version_path": version_path
            }
            add_step(state, step_info)
            state["processed_df"] = new_df
            

            # Show Summary After
            st.success("✅ Missing values handled successfully.")
            st.markdown("### ✅ Imputation Details")
            if info.get("imputed_columns"):
                impute_df = pd.DataFrame.from_dict(info["imputed_columns"], orient="index")
                impute_df = impute_df.rename(columns={
                    "strategy": "Method",
                    "missing_before": "Before",
                    "missing_after": "After"
                })
                st.dataframe(impute_df)

            if info.get("summary_after") is not None:
                st.markdown("### 📊 Missing Values Summary (After)")
                st.dataframe(info["summary_after"], use_container_width=True)

            if info.get("dropped_target_rows", 0) > 0:
                st.warning(f"Dropped {info['dropped_target_rows']} rows with missing target `{target}`.")

            # 5️⃣ Preview Final Cleaned Dataset
            if st.button("🔍 Preview Cleaned Dataset"):
                st.markdown("### 🔎 Dataset After Handling Missing Values")
                st.dataframe(new_df.head(50), use_container_width=True)

            # ✅ Most important fix: RETURN PARAMS!
            return {"method": method, "columns": cols or None}

        except Exception as e:
            st.error(f"❌ Error handling missing values: {e}")

    # Already applied
    if _is_step_applied("missing_values", state):
        st.markdown("✅ Missing Values Step Already Applied.")

    return None  # If not applied yet



def _outliers_ui(df: pd.DataFrame, state: dict):
    ui_key = "outliers_ui"
    exp = st.expander("🧯 2. Outlier Detection & Removal", expanded=not _is_step_applied("outliers", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        target = state.get("target_col")
        num_cols = [c for c in curr_df.select_dtypes(include=["number"]).columns if c != target]
        if not num_cols:
            st.info("No numeric columns for outlier detection.")
            return
        if ui_key not in st.session_state:
            st.session_state[ui_key] = {
                "show_preview": False,
                "preview_df": None,
                "cols": num_cols,
                "plot_opts": ["Box-plot", "Histogram", "Violin-plot"]
            }
        ui = st.session_state[ui_key]
        cols = st.multiselect("Select numeric columns", options=num_cols, default=ui.get("cols"), key=ui_key + "_cols")
        ui["cols"] = cols
        method = st.selectbox("Method", ["iqr", "z_score", "isolation_forest", "lof", "elliptic_envelope", "dbscan"], key=ui_key + "_method")
        plot_opts = st.multiselect("Plot types", options=["Box-plot", "Histogram", "Violin-plot"], default=ui.get("plot_opts"), key=ui_key + "_plots")
        ui["plot_opts"] = plot_opts
        params = {"method": method, "columns": cols}
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👁️ Visualize Graphs", key=ui_key + "_viz"):
                if not cols:
                    st.warning("Select columns.")
                else:
                    with loading_button("Generating preview..."):
                        preview_df = remove_outliers(curr_df.copy(), **params)
                        st.session_state[ui_key]["preview_df"] = preview_df
                        st.session_state[ui_key]["show_preview"] = True
                    st.rerun()
        with col2:
            if st.button("👁️ Preview Dataset", key=ui_key+"_preview"):
                st.markdown("### 👁️ Current Preprocessing Dataset")
                st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)


            if st.button("✅ Apply Outlier Removal Step", key=ui_key + "_apply"):
                try:
                    with loading_button("Removing outliers..."):
                        new_df = remove_outliers(curr_df, **params)
                    version = save_processed_version(new_df, state, step="outliers")
                    add_step(state, {"name": "outliers", "params": params, "version_path": version})
                    state["processed_df"] = new_df
                    tick_animation("Outliers removed")
                    # Show before-after comparison like the provided image
                    st.session_state[ui_key]["show_applied_comparison"] = True
                    st.session_state[ui_key]["applied_before"] = curr_df.copy()
                    st.session_state[ui_key]["applied_after"] = new_df.copy()
                    if "log_transform_ui" in st.session_state:
                        del st.session_state["log_transform_ui"]
                    # Also clear any downstream UI states (e.g., scaling_ui, feature_extraction_ui, etc.)
                    for downstream in ["scaling_ui", "feature_extraction_ui", "feature_creation_ui"]:
                        if downstream in st.session_state:
                            del st.session_state[downstream]
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        # Preview before/after (for Visualize Effect)
        if st.session_state[ui_key].get("show_preview"):
            st.markdown("---")
            st.markdown("### Preview Before vs After")
            prev = st.session_state[ui_key]["preview_df"]
            for col in cols:
                st.markdown(f"#### Column: `{col}`")
                for plot_type in plot_opts:
                    st.markdown(f"**{plot_type}**")
                    fig, ax = plt.subplots()
                    before_color = "#ffb3ba"  # light pinkish red
                    after_color = "#4f8cff"   # blue
                    if plot_type == "Box-plot":
                        # Side-by-side boxplots for before and after
                        data = [curr_df[col].dropna(), prev[col].dropna()]
                        box = ax.boxplot(data, patch_artist=True, labels=["Before", "After"])
                        colors = [before_color, after_color]
                        for patch, color in zip(box['boxes'], colors):
                            patch.set_facecolor(color)
                        ax.set_ylabel(col)
                    elif plot_type == "Histogram":
                        ax.hist(curr_df[col].dropna(), bins=30, alpha=0.7, label="Before", color=before_color)
                        ax.hist(prev[col].dropna(), bins=30, alpha=0.7, label="After", color=after_color)
                        ax.legend()
                        ax.set_ylabel("Count")
                        ax.set_xlabel(col)
                    elif plot_type == "Violin-plot":
                        data = [curr_df[col].dropna(), prev[col].dropna()]
                        parts = ax.violinplot(data, showmeans=True)
                        # Set colors for violins
                        for i, pc in enumerate(parts['bodies']):
                            pc.set_facecolor(before_color if i == 0 else after_color)
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.8)
                        # Set mean line color
                        if 'cmeans' in parts:
                            for i, mean in enumerate(parts['cmeans'].get_segments()):
                                parts['cmeans'].set_color('black')
                        ax.set_xticks([1, 2])
                        ax.set_xticklabels(["Before", "After"])
                        ax.set_ylabel(col)
                    st.pyplot(fig)

        # Show before-after comparison after applying outlier removal (like provided image)
        if st.session_state[ui_key].get("show_applied_comparison"):
            before_df = st.session_state[ui_key].get("applied_before")
            after_df = st.session_state[ui_key].get("applied_after")
            if before_df is not None and after_df is not None:
                st.markdown("---")
                st.markdown("### Outlier Removal: Before vs After")
                before_color = "#ffb3ba"  # light pinkish red
                after_color = "#4f8cff"   # blue
                for col in cols:
                    st.markdown(f"#### Column: `{col}`")
                    for plot_type in plot_opts:
                        st.markdown(f"**{plot_type}**")
                        if plot_type == "Histogram":
                            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                            # Before
                            axes[0].hist(before_df[col].dropna(), bins=30, color=before_color, alpha=0.8)
                            axes[0].set_title("Before Outlier Removal")
                            axes[0].set_xlabel("Value")
                            axes[0].set_ylabel("Count")
                            # After
                            axes[1].hist(after_df[col].dropna(), bins=30, color=after_color, alpha=0.8)
                            axes[1].set_title("After Outlier Removal")
                            axes[1].set_xlabel("Value")
                            axes[1].set_ylabel("Count")
                            for ax in axes:
                                ax.grid(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                        elif plot_type == "Box-plot":
                            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                            # Before
                            box1 = axes[0].boxplot(before_df[col].dropna(), patch_artist=True)
                            for patch in box1['boxes']:
                                patch.set_facecolor(before_color)
                            axes[0].set_title("Before Outlier Removal")
                            axes[0].set_ylabel(col)
                            # After
                            box2 = axes[1].boxplot(after_df[col].dropna(), patch_artist=True)
                            for patch in box2['boxes']:
                                patch.set_facecolor(after_color)
                            axes[1].set_title("After Outlier Removal")
                            axes[1].set_ylabel(col)
                            for ax in axes:
                                ax.grid(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                        elif plot_type == "Violin-plot":
                            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                            # Before
                            parts1 = axes[0].violinplot(before_df[col].dropna(), showmeans=True)
                            for pc in parts1['bodies']:
                                pc.set_facecolor(before_color)
                                pc.set_edgecolor('black')
                                pc.set_alpha(0.8)
                            axes[0].set_title("Before Outlier Removal")
                            axes[0].set_ylabel(col)
                            axes[0].set_xticks([1])
                            axes[0].set_xticklabels([""])
                            # After
                            parts2 = axes[1].violinplot(after_df[col].dropna(), showmeans=True)
                            for pc in parts2['bodies']:
                                pc.set_facecolor(after_color)
                                pc.set_edgecolor('black')
                                pc.set_alpha(0.8)
                            axes[1].set_title("After Outlier Removal")
                            axes[1].set_ylabel(col)
                            axes[1].set_xticks([1])
                            axes[1].set_xticklabels([""])
                            for ax in axes:
                                ax.grid(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                # Remove flag so it only shows once after apply
                st.session_state[ui_key]["show_applied_comparison"] = False


def _log_transform_ui(df: pd.DataFrame, state: dict):
    ui_key = "log_transform_ui"
    exp = st.expander("🔁 3. Log Transformation", expanded=not _is_step_applied("log_transform", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        target = state.get("target_col")
        # numeric columns excluding target
        num_cols = [c for c in curr_df.select_dtypes(include=["number"]).columns if c != target]
        if not num_cols:
            st.info("No numeric columns available for log transformation.")
            return None

        # Optionally exclude ID-like columns (e.g. Serial No.) to avoid meaningless transforms
        exclude_cols = []
        for c in num_cols:
            vals = curr_df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 0 and vals.nunique() == len(vals):
                exclude_cols.append(c)
        if exclude_cols:
            st.info(f"Excluding likely ID columns: {exclude_cols}")
        num_cols = [c for c in num_cols if c not in exclude_cols]
        if not num_cols:
            st.info("No suitable numeric columns for log transformation after excluding IDs.")
            return None

        # Initialize session UI state
        if ui_key not in st.session_state:
            st.session_state[ui_key] = {
                "selected_cols": [],
                "method": None,
                "step_applied": False,
                "before_df": None,
                "methods_applied": {},
                "show_preview": False
            }
        ui = st.session_state[ui_key]

        # Column multiselect
        prev_cols = [c for c in ui.get("selected_cols", []) if c in num_cols]
        default_cols = prev_cols or num_cols  # default to all suitable numeric
        cols = st.multiselect(
            "Select numeric columns to transform",
            options=num_cols,
            default=default_cols,
            key=ui_key + "_cols"
        )
        ui["selected_cols"] = cols
        if not cols:
            st.warning("Please select at least one column.")
            return None

        # Suggestion summary for auto-transform
        st.markdown("### Skewness & Suggested Transformation (auto)")
        suggestion_list = []
        for col in cols:
            series = curr_df[col]
            try:
                best_method, orig_skew, new_skew = detect_best_transform(series)
            except Exception as e:
                st.write(f"Error for {col}: {e}")
                best_method, orig_skew, new_skew = None, None, None
            suggestion_list.append({
                "column": col,
                "original_skew": round(orig_skew, 3) if orig_skew is not None else None,
                "new_skew": round(new_skew, 3) if new_skew is not None else None,
                "method": best_method or "(no change)"
            })
        sugg_df = pd.DataFrame(suggestion_list)
        st.table(sugg_df)
        st.markdown("*Columns with reduced |skew| will be transformed using the shown method.*")

        # Method selectbox: manual override or auto
        methods = ["auto", "log", "log10", "log2", "log1p", "boxcox", "yeojohnson"]
        prev_m = ui.get("method")
        default_m = prev_m if prev_m in methods else "auto"
        method = st.selectbox(
            "Select transformation method",
            options=methods,
            index=methods.index(default_m),
            key=ui_key + "_method"
        )
        ui["method"] = method

        # Preview current dataset button with close functionality
        if st.button("👁️ Preview Current Dataset", key=ui_key + "_preview_dataset"):
            ui["show_preview"] = True
        if ui.get("show_preview", False):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(curr_df.head(50), use_container_width=True)
            if st.button("❌ Close Preview", key=ui_key + "_close_preview"):
                ui["show_preview"] = False

        # Apply button
        if st.button("✅ Apply Log Transformation", key=ui_key + "_apply"):
            try:
                # Save a copy of before for preview
                ui["before_df"] = curr_df[cols].copy()
                new_df = curr_df.copy()
                methods_applied = {}
                if method == "auto":
                    # Apply per suggestion
                    for entry in suggestion_list:
                        col = entry["column"]
                        best = entry["method"]
                        if best not in [None, "(no change)", "(insufficient data)"]:
                            # apply best_method to this column
                            new_df = apply_log_transform(new_df, columns=[col], method=best)
                            methods_applied[col] = best
                        else:
                            methods_applied[col] = None
                else:
                    # Apply same method to all selected columns
                    new_df = apply_log_transform(new_df, columns=cols, method=method)
                    for col in cols:
                        methods_applied[col] = method

                # Save processed version and record step
                version = save_processed_version(new_df, state, step="log_transform")
                add_step(state, {
                    "name": "log_transform",
                    "params": {"columns": cols, "method": method, "methods_applied": methods_applied},
                    "version_path": version
                })
                state["processed_df"] = new_df
                ui["step_applied"] = True
                ui["methods_applied"] = methods_applied
                tick_animation("Log transformation applied")
                # Debugging output
                st.write("applied_steps:", state.get("applied_steps"))
                st.write("log_transform_ui state:", ui)
                # Immediately show preview after apply:
                st.success("Log transformation applied. See preview below.")
                after_df = new_df[cols]
                before_df = ui["before_df"]
                for col in cols:
                    b = before_df[col].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                    a = after_df[col].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                    m_used = methods_applied.get(col)
                    if b.size >= 2 and a.size >= 2:
                        st.markdown(f"**Column: {col}** (method: {m_used or 'none'})")
                        show_before_after_plots(pd.DataFrame({col: b}), pd.DataFrame({col: a}), [col])
                    else:
                        st.warning(f"Not enough data to plot for '{col}'")
                st.rerun()
            except Exception as e:
                st.error(f"Error applying log transformation: {e}")
                return None

        # If already applied in this session, show notice + preview button
        if ui.get("step_applied"):
            st.markdown("✅ Log transformation step already applied.")
            if st.button("🔍 Preview Last Log Transformation", key=ui_key + "_preview"):
                before_df = ui.get("before_df")
                after_df = state.get("processed_df", curr_df)[cols]
                for col in cols:
                    b = before_df[col].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                    a = after_df[col].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                    m_used = ui["methods_applied"].get(col)
                    if b.size >= 2 and a.size >= 2:
                        st.markdown(f"**Column: {col}** (method: {m_used or 'none'})")
                        show_before_after_plots(pd.DataFrame({col: b}), pd.DataFrame({col: a}), [col])
                    else:
                        st.warning(f"Not enough data to plot for '{col}'")
    return None


def _scaling_ui(df: pd.DataFrame, state: dict):
    ui_key = "scaling_ui"
    exp = st.expander("✏️ 4. Normalization / Standardization", expanded=not _is_step_applied("scaling", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        target = state.get("target_col")
        num_cols = [c for c in curr_df.select_dtypes(include=["number"]).columns if c != target]
        if not num_cols:
            st.info("No numeric columns for scaling.")
            return None

        # Initialize session state for this UI
        if ui_key not in st.session_state:
            st.session_state[ui_key] = {"cols": num_cols, "method": None, "step_applied": False, "methods_applied": {}}
        ui = st.session_state[ui_key]

        # Column selection
        prev_cols = [c for c in ui.get("cols", []) if c in num_cols]
        default_cols = prev_cols or num_cols
        cols = st.multiselect(
            "Select numeric columns to scale",
            options=num_cols,
            default=default_cols,
            key=ui_key + "_cols"
        )
        ui["cols"] = cols
        if not cols:
            st.warning("No columns selected for scaling.")
            return None

        # Method selection (ensure matches apply_scaling)
        methods = ["standardization", "normalization", "robust", "maxabs"]
        prev_method = ui.get("method")
        default_method = prev_method if prev_method in methods else methods[0]
        mode = st.selectbox(
            "Scaling Type",
            options=methods,
            index=methods.index(default_method),
            key=ui_key + "_method"
        )
        ui["method"] = mode

        # Preview current dataset button
        if st.button("👁️ Preview Current Dataset", key=ui_key + "_preview_dataset"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)

        # Apply scaling
        if st.button("✅ Apply Scaling Step", key=ui_key + "_apply"):
            try:
                with loading_button("Scaling data..."):
                    new_df = apply_scaling(curr_df, columns=cols, method=mode)
                version = save_processed_version(new_df, state, step="scaling")
                # Record methods_applied per column
                methods_applied = {c: mode for c in cols}
                add_step(state, {
                    "name": "scaling",
                    "params": {"columns": cols, "method": mode},
                    "version_path": version
                })
                state["processed_df"] = new_df
                ui["step_applied"] = True
                ui["methods_applied"] = methods_applied
                tick_animation("Scaling applied")
                st.write("applied_steps:", state.get("applied_steps"))
                st.write("scaling_ui state:", ui)
                st.rerun()
            except Exception as e:
                st.error(f"Error applying scaling: {e}")
                return None

    # Post-expander: Preview scaling graphs
    # Use a unique key for this button
    preview_key = ui_key + "_preview_graphs"
    if st.button("🔍 Preview Scaling Graphs", key=preview_key):
        applied = ui.get("step_applied", False) or _is_step_applied("scaling", state)
        if not applied:
            st.warning("Scaling not yet applied.")
        else:
            st.markdown("---")
            st.markdown("### Post-Transformation Preview & Methods")
            after_df = state.get("processed_df", curr_df)
            for col in ui.get("cols", []):
                before_vals = curr_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                after_vals = after_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                method = ui.get("methods_applied", {}).get(col, ui.get("method"))
                st.markdown(f"**Column: {col}** (method: {method})")
                if before_vals.size >= 2 and after_vals.size >= 2:
                    show_before_after_plots(
                        pd.DataFrame({col: before_vals}),
                        pd.DataFrame({col: after_vals}),
                        [col]
                    )
                else:
                    st.warning(f"Not enough data to plot for '{col}'")

def _multicollinearity_ui(df: pd.DataFrame, state: dict):
    ui_key = "multicol_ui"
    exp = st.expander(
        "🔄 Multicollinearity Check",
        expanded=not _is_step_applied("multicollinearity", state)
    )
    with exp:
        # Only numeric columns for collinearity check
        curr_df = state.get("processed_df", df).select_dtypes(include=[np.number]).copy()
        numeric_cols = curr_df.columns.tolist()
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric features for multicollinearity check.")
            return None

        st.markdown("#### Correlation Heatmap (numeric features only)")
# Compute corr
        variances = curr_df.var()
        low_var = variances[variances < 0.01].index.tolist()
        corr_df = curr_df.drop(columns=low_var)

        # 2) compute corr & mask upper triangle
        corr = corr_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # 3) dynamic sizing
        n = corr.shape[0]
        height = max(6, 0.3 * n)
        width  = max(8,  0.3 * n)
        fig, ax = plt.subplots(figsize=(width, height))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink":0.75},
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        st.pyplot(fig)
        
        plt.close()

        st.markdown("#### Variance Inflation Factor (VIF)")
        Xc = sm.add_constant(curr_df)
        vif = pd.Series(
            [variance_inflation_factor(Xc.values, i+1) for i in range(curr_df.shape[1])],
            index=numeric_cols
        )
        st.dataframe(vif.to_frame("VIF"), use_container_width=True)

        st.markdown("### Choose how to reduce collinearity")
        method = st.selectbox(
            "Method",
            ["— none —", "Blockwise PCA", "VIF Drop"],
            index=0,
            key=ui_key + "_method"
        )

        # Parameters
        corr_thresh = st.slider(
            "Correlation threshold (for PCA blocks)",
            0.5, 0.99, 0.8, 0.01,
            key=ui_key + "_corr"
        )
        var_retained = st.slider(
            "Variance to retain (%)",
            50, 100, 95, 5,
            key=ui_key + "_var"
        )
        vif_thresh = st.slider(
            "VIF threshold (for drop)",
            1.0, 20.0, 10.0, 0.5,
            key=ui_key + "_vifthresh"
        )

        if st.button("✅ Apply Multicollinearity Reduction", key=ui_key + "_apply"):
            df_before = state["processed_df"].copy()

            if method == "Blockwise PCA":
                # find correlated blocks
                corr = curr_df.corr().abs()
                visited = set()
                blocks = []
                for col in numeric_cols:
                    if col in visited:
                        continue
                    group = set([col] + list(corr.index[corr[col] > corr_thresh]))
                    if len(group) > 1:
                        blocks.append(list(group))
                    visited |= group

                # drop originals
                to_drop = [c for block in blocks for c in block]
                reduced = state["processed_df"].drop(columns=to_drop)

                # apply PCA per block
                for i, block in enumerate(blocks):
                    sub = state["processed_df"][block]
                    scaled = StandardScaler().fit_transform(sub)
                    pca = PCA(n_components=var_retained/100)
                    comps = pca.fit_transform(scaled)
                    names = [f"MC_PCA_{i+1}_{j+1}" for j in range(comps.shape[1])]
                    reduced = pd.concat([
                        reduced,
                        pd.DataFrame(comps, columns=names, index=state["processed_df"].index)
                    ], axis=1)

                new_df = reduced

            elif method == "VIF Drop":
                Xv = curr_df.copy()
                while True:
                    Xc2 = sm.add_constant(Xv)
                    vifs = pd.Series(
                        [variance_inflation_factor(Xc2.values, i+1) for i in range(Xv.shape[1])],
                        index=Xv.columns
                    )
                    max_v = vifs.max()
                    if max_v <= vif_thresh:
                        break
                    drop = vifs.idxmax()
                    Xv = Xv.drop(columns=[drop])

                new_df = pd.concat([
                    state["processed_df"].drop(columns=numeric_cols),
                    Xv
                ], axis=1)

            else:
                st.warning("Select a method to reduce collinearity.")
                return None

            # record step
            version = save_processed_version(new_df, state, step="multicollinearity")
            add_step(state, {
                "name": "multicollinearity",
                "params": {
                    "method": method,
                    "corr_thresh": corr_thresh,
                    "variance_retained": var_retained,
                    "vif_thresh": vif_thresh
                },
                "version_path": version
            })

            state["processed_df"] = new_df
            tick_animation("Multicollinearity reduced")
            st.success("✅ Multicollinearity step applied.")

            # clear downstream UIs now that columns have changed
            for downstream in [
                "log_transform_ui",
                "scaling_ui",
                "feature_extraction_ui",
                "feature_creation_ui",
                "feature_selection_ui",
                "encoding_ui"
            ]:
                if downstream in st.session_state:
                    del st.session_state[downstream]

            st.rerun()

    return None



def _feature_extraction_ui(df: pd.DataFrame, state: dict):
    ui_key = "feature_extraction_ui"
    exp = st.expander(
        "🍭 5. Feature Extraction",
        expanded=not _is_step_applied("feature_extraction", state)
    )
    with exp:
        curr_df = state.get("processed_df", df).copy()
        target = state.get("target_col")
        num_cols = [c for c in curr_df.select_dtypes(include=["number"]).columns if c != target]
        text_cols = [
            c for c in curr_df.columns
            if pd.api.types.is_string_dtype(curr_df[c]) and c != target
        ]

        # 1) Choose method
        method = st.selectbox(
            "Method",
            [
                "PCA", "t-SNE", "LDA", "Autoencoder",
                "TF-IDF", "Feature Hashing", "Fourier", "DWT", "Word2Vec"
            ],
            key=ui_key + "_method"
        )
        params = {"method": method}

        # 2) If numeric method: pick # components
        if method in ["PCA", "t-SNE", "LDA", "Autoencoder", "Fourier", "DWT", "Word2Vec"]:
            n = st.number_input(
                "Number of components",
                min_value=1,
                max_value=min(len(num_cols), 10),
                value=2,
                key=ui_key + "_n"
            )
            params["n_components"] = n

        # 3) LDA needs a label
        if method == "LDA":
            params["label_col"] = target

        # 4) Text methods: pick a text column safely
        if method in ["TF-IDF", "Feature Hashing", "Word2Vec"]:
            if not text_cols:
                st.error("❌ No text columns available.")
                st.stop()
            widget = ui_key + "_textcol"
            # reset if stale
            # 1. Get current value from session state
            current_val = st.session_state.get(widget)
            
            # 2. Validate or reset to first text column
            if current_val not in text_cols:
                st.session_state[widget] = text_cols[0]
            
            # 3. Create selectbox with validated default
            
            text_col = st.selectbox(
                "Select text column",
                options=text_cols,
                index=text_cols.index(st.session_state.get(widget, text_cols[0])),
                key=widget
            )
            params["text_col"] = text_col

        # 5) Autoencoder placeholder
        if method == "Autoencoder":
            st.info("Autoencoder requires a trained model (not implemented).")
            params["autoencoder_model"] = None

        # 6) Visualization choice
        viz = st.selectbox(
            "Visualization",
            ["Scree Plot", "2D Scatter", "3D Scatter", "Heatmap", "Box Plot", "Q-Q Plot"],
            key=ui_key + "_viz"
        )
        params["viz_type"] = viz

        # 7) Preview dataset button
        if st.button("👁️ Preview Dataset", key=ui_key + "_preview"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(curr_df.head(50), use_container_width=True)

        # 8) Apply feature extraction
        if st.button("✅ Apply Feature Extraction Step", key=ui_key + "_apply"):
            try:
                with loading_button("Extracting features..."):
                    new_df = apply_feature_extraction(curr_df, **params)

                # record step & update state
                version = save_processed_version(new_df, state, step="feature_extraction")
                add_step(state, {
                    "name": "feature_extraction",
                    "params": params,
                    "version_path": version
                })
                state["processed_df"] = new_df

                tick_animation("Feature extraction done")
                st.success("✅ Feature extraction complete.")

                # ─── CLEAR ALL RELEVANT SESSION STATE KEYS ─────────────────
                keys_to_clear = [
                    # own widget keys
                    ui_key + "_method",
                    ui_key + "_n",
                    ui_key + "_textcol",
                    ui_key + "_viz",
                    ui_key + "_preview",
                    ui_key + "_apply",
                    # downstream UIs
                    "log_transform_ui",
                    "scaling_ui",
                    "feature_creation_ui",
                    "feature_selection_ui",
                    "encoding_ui",
                    "multicol_ui",
                    "eda2_ui_cols",     # if you have EDA2 column pickers
                    "eda2_ui_option",
                ]
                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]
                # ────────────────────────────────────────────────────────────

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

        # 9) If already applied, re-show visualization
        if _is_step_applied("feature_extraction", state):
            st.markdown("---")
            st.markdown("### Feature Extraction Visualization")
            try:
                apply_feature_extraction(curr_df, **params)
            except Exception as e:
                st.error(f"Visualization error: {e}")

    return None




def _feature_creation_ui(df: pd.DataFrame, state: dict):
    ui_key = "feature_creation_ui"
    exp = st.expander("🛠️ 6. Feature Creation", expanded=not _is_step_applied("feature_creation", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        method = st.selectbox("Method", ["Polynomial Features","Binning/Discretization","Date-Time Decomposition","Aggregation","Text Features","Flag/Boolean"], key=ui_key+"_method")
        params = {"method": method}
        if method == "Polynomial Features":
            cols = st.multiselect("Select numeric columns", options=curr_df.select_dtypes(include=["number"]).columns.tolist(), key=ui_key+"_cols")
            deg = st.number_input("Degree", min_value=2, max_value=5, value=2, key=ui_key+"_deg")
            params.update({"columns": cols, "degree": deg})
        elif method == "Binning/Discretization":
            col = st.selectbox("Select column", options=curr_df.columns.tolist(), key=ui_key+"_col")
            bins = st.number_input("Bins", min_value=2, max_value=50, value=5, key=ui_key+"_bins")
            params.update({"column": col, "bins": bins})
        elif method == "Date-Time Decomposition":
            col = st.selectbox("Select datetime column", options=curr_df.select_dtypes(include=["datetime","object"]).columns.tolist(), key=ui_key+"_col")
            params.update({"column": col})
        elif method == "Aggregation":
            group_cols = st.multiselect("Group by columns", options=curr_df.columns.tolist(), key=ui_key+"_group")
            agg_col = st.selectbox("Aggregate column", options=curr_df.select_dtypes(include=["number"]).columns.tolist(), key=ui_key+"_aggcol")
            agg_func = st.selectbox("Function", ["sum","mean","median","max","min","count"], key=ui_key+"_aggfunc")
            params.update({"group_cols": group_cols, "agg_col": agg_col, "agg_func": agg_func})
        elif method == "Text Features":
            col = st.selectbox("Select text column", options=curr_df.select_dtypes(include=["object","string"]).columns.tolist(), key=ui_key+"_col")
            feat = st.selectbox("Feature", ["Length","Word Count"], key=ui_key+"_feat")
            params.update({"column": col, "text_feat": feat})
        elif method == "Flag/Boolean":
            col = st.selectbox("Select column", options=curr_df.columns.tolist(), key=ui_key+"_col")
            cond = st.text_input("Condition (e.g., >value)", key=ui_key+"_cond")
            params.update({"column": col, "condition": cond})
        
        if st.button("👁️ Preview Dataset", key=ui_key+"_preview"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)
        
        if st.button("✅ Apply Feature Creation Step", key=ui_key+"_apply"):
            try:
                with loading_button("Creating features..."):
                    new_df = apply_feature_creation(curr_df, **params)
                version = save_processed_version(new_df, state, step="feature_creation")
                add_step(state, {"name":"feature_creation","params":params,"version_path":version})
                state["processed_df"] = new_df
                tick_animation("Feature creation done")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def _feature_selection_ui(df: pd.DataFrame, state: dict):
    ui_key = "feature_selection_ui"
    exp = st.expander("📌 7. Feature Selection", expanded=not _is_step_applied("feature_selection", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        methods = ["Correlation-based","Chi-Square","ANOVA","RFE","Lasso","Tree-based Importance","PCA","Mutual Information"]
        method = st.selectbox("Method", methods, key=ui_key+"_method")
        params = {"method": method}
        if method == "Correlation-based":
            thresh = st.slider("Threshold", 0.0, 1.0, 0.5, key=ui_key+"_thresh")
            params["threshold"] = thresh
        elif method in ["Chi-Square","ANOVA"]:
            label_col = state.get("target_col")
            top_k = st.number_input("Top K", min_value=1, max_value=curr_df.shape[1]-1, value=5, key=ui_key+"_k")
            params.update({"label_col": label_col, "top_k": top_k})
        elif method == "RFE":
            label_col = state.get("target_col")
            estimator = st.selectbox("Estimator", ["LogisticRegression","RandomForestClassifier","SVR"], key=ui_key+"_est")
            n = st.number_input("Num features", min_value=1, max_value=curr_df.shape[1]-1, value=5, key=ui_key+"_n")
            params.update({"label_col": label_col, "estimator": estimator, "n_features": n})
        elif method == "Lasso":
            alpha = st.number_input("Alpha", min_value=0.0, value=1.0, key=ui_key+"_alpha")
            params["alpha"] = alpha
        elif method == "Tree-based Importance":
            label_col = state.get("target_col")
            params["label_col"] = label_col
        elif method == "PCA":
            n = st.number_input("Components", min_value=1, max_value=min(curr_df.shape[1],10), value=2, key=ui_key+"_n")
            params["n_components"] = n
        elif method == "Mutual Information":
            label_col = state.get("target_col")
            top_k = st.number_input("Top K", min_value=1, max_value=curr_df.shape[1]-1, value=5, key=ui_key+"_k")
            params.update({"label_col": label_col, "top_k": top_k})
        
        if st.button("👁️ Preview Dataset", key=ui_key+"_preview"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)
        
        if st.button("✅ Apply Feature Selection Step", key=ui_key+"_apply"):
            try:
                with loading_button("Selecting features..."):
                    new_df = apply_feature_selection(curr_df, **params)
                version = save_processed_version(new_df, state, step="feature_selection")
                add_step(state, {"name":"feature_selection","params":params,"version_path":version})
                state["processed_df"] = new_df
                tick_animation("Feature selection done")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def _encoding_ui(df: pd.DataFrame, state: dict):
    ui_key = "encoding_ui"
    exp = st.expander("🧩 8. Encoding Categorical Features", expanded=not _is_step_applied("encoding", state))
    with exp:
        curr_df = state.get("processed_df", df).copy()
        cat_cols = curr_df.select_dtypes(include=["object","category"]).columns.tolist()
        if not cat_cols:
            st.info("No categorical columns for encoding.")
            return
        cols = st.multiselect("Select categorical columns to encode", options=cat_cols, key=ui_key+"_cols")
        if not cols:
            st.info("No columns selected.")
            return
        method = st.selectbox("Encoding Method", ["One-Hot","Label","Target","Frequency","Binary","Hash","Ordinal"], key=ui_key+"_method")
        params = {"method": method, "columns": cols}
        if method == "Ordinal":
            order_input = {}
            for col in cols:
                order = st.text_input(f"Order for {col}", key=ui_key+f"_order_{col}")
                if order:
                    order_input[col] = [x.strip() for x in order.split(",")]
            params["order"] = order_input
        
        if st.button("👁️ Preview Dataset", key=ui_key+"_preview"):
            st.markdown("### 👁️ Current Preprocessing Dataset")
            st.dataframe(state.get("processed_df", df).head(50), use_container_width=True)
        
        if st.button("✅ Apply Encoding Step", key=ui_key+"_apply"):
            try:
                with loading_button("Applying encoding..."):
                    new_df = apply_encoding(curr_df, **params)
                version = save_processed_version(new_df, state, step="encoding")
                add_step(state, {"name":"encoding","params":params,"version_path":version})
                state["processed_df"] = new_df
                tick_animation("Encoding done")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
