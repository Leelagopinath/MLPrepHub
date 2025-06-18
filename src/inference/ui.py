import streamlit as st
import os
from src.data.loader import load_data
from src.inference.predictor import load_model, predict_with_model
from src.utils.helpers import list_files_with_extension



def client_prediction_ui():
    st.subheader("🔮 Inference - Use Trained Model")

    uploaded_file = st.file_uploader("Upload your new input data for prediction", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        input_df = load_data(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())

        st.markdown("---")

        model_files = list_files_with_extension("project_data/models/", [".pkl", ".joblib", ".onnx"])
        selected_model = st.selectbox("Select a trained model file", model_files)

        if selected_model:
            model_path = os.path.join("project_data/models/", selected_model)
            model, model_format = load_model(model_path)

            if st.button("Run Prediction"):
                with st.spinner("Running model on input data..."):
                    predictions = predict_with_model(model, model_format, input_df)
                st.success("✅ Predictions generated")

                input_df["Prediction"] = predictions
                st.dataframe(input_df)

                st.download_button(
                    label="Download Predictions",
                    data=input_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                                )
                
                # Optionally, add a main entry point to run this UI directly with Streamlit
                if __name__ == "__main__":
                    client_prediction_ui()
