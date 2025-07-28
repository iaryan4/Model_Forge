import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
from src.pipeline.train_pipeline import run_training
import altair as alt
from src.pipeline.predict_pipeline import get_columns,initiate_training
# Config
st.set_page_config(
    page_title="ModelForge",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title('ModelForge AI — Your Smart ML Builder')
st.markdown("""
**ModelForge** automates your ML workflow — upload data, choose a task, and let the magic happen.  
Train, evaluate, and predict — all from one clean, smart interface.
""")

# Upload file
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", 'xls'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        readed_data = pd.read_csv(uploaded_file)
    else:
        readed_data = pd.read_excel(uploaded_file)

    TARGET_COLUMN = '--TARGET COLUMN--'
    columns_list = [col for col in readed_data.columns]
    columns_list.insert(0, TARGET_COLUMN)

    # Target column selection
    target = st.selectbox('Select your `TARGET COLUMN` :', options=columns_list)
    if target == TARGET_COLUMN:
        st.warning('Please select the target column above  ^-^ ')
        st.write("Preview of uploaded data:")
        st.dataframe(readed_data)
    else:
        readed_data.to_csv(os.path.join('artifact', 'raw_data.csv'), index=False)
        with open(os.path.join('artifact', 'target.txt'), 'w') as fp:
            fp.write(target)

        # Task type
        reg_or_class = st.radio('Select which type of dataset is this : ', options=['Regression', 'Classification'], horizontal=True)
        if reg_or_class:
            with open(os.path.join('artifact', 'target.txt'), 'a') as fp:
                fp.write('\n' + reg_or_class)

        st.write("Preview of uploaded data:")
        st.dataframe(readed_data)

        # Train button
        st.write('Click on the `TRAIN` button to train the model ')
        if st.button("🚀 TRAIN MODEL"):
            with st.spinner("🔄 Running Model Training Pipeline... Wait for few seconds 🙂"):
                model_report, best_model_name, best_accuracy, ex_df = run_training()

                # Store in session state
                st.session_state.model_report = model_report
                st.session_state.best_model_name = best_model_name
                st.session_state.best_accuracy = best_accuracy
                st.session_state.ex_df = ex_df

            st.success("✅ Training Completed Successfully! 🪡")
            st.balloons()

        # If model already trained, show results
        if 'model_report' in st.session_state:
            st.markdown("###  Best Performing Model Selected for You:")
            st.markdown(f"- **Model:** `{st.session_state.best_model_name}`  \n- **Accuracy:** `{st.session_state.best_accuracy * 100:.2f}%`")

            st.markdown('##### Model Comparison')
            df_report = pd.DataFrame(list(st.session_state.model_report.items()), columns=['Model', 'Score'])
            df_report['Score'] = df_report['Score'] * 100
            st.bar_chart(df_report.set_index('Model'), color="#49EFE4", x_label='Models', y_label='Score(%)')
        else:
            st.warning("⚠️ Please train the model first before proceeding to the testing section!")

        # ========== Prediction Section ==========
        st.markdown("#### Want to Test the Model?")

        st.write("**How would you like to test the model?** Choose one of the options below:")
        option = st.radio(
            "How would you like to test the model?",
            ("📁 Upload test CSV or Excel File", "📝 Manually enter data")
        )

                # === Download model option right after this ===
        if os.path.exists(os.path.join("artifact", "model.pkl")):
            with open(os.path.join("artifact", "model.pkl"), "rb") as f:
                st.download_button(
                    label="📦 Download Trained Model",
                    data=f,
                    file_name="best_trained_model.pkl",
                    mime="application/octet-stream"
                )
            st.caption("🎯 Or click above to download the trained model for future use.")
        

        user_data = None

        if option == "📝 Manually enter data":
            if 'ex_df' in st.session_state:
                columns, dtypes = get_columns(st.session_state.ex_df)

                with st.form("input_form"):
                    user_input = {}
                    for col in columns:
                        dtype = dtypes[col]
                        if "int" in dtype:
                            user_input[col] = st.number_input(f"{col} (int)", step=1, format="%d")
                        elif "float" in dtype:
                            user_input[col] = st.number_input(f"{col} (float)")
                        else:
                            user_input[col] = st.text_input(f"{col} (text)")

                    submit = st.form_submit_button("Submit")

                if submit:
                    user_data = pd.DataFrame([user_input])
                    st.session_state['user_data'] = user_data
                    user_data.to_csv(os.path.join('artifact', 'user_test.csv'), index=False)
                    st.success("✅ Input saved and ready for prediction!")
            else:
                st.warning("Train the model first to enter input manually.")

        elif option == "📁 Upload test CSV or Excel File":
            uploaded_test = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="UserInput")
            if uploaded_test is not None:
                if uploaded_test.name.endswith(".csv"):
                    user_data = pd.read_csv(uploaded_test)
                else:
                    user_data = pd.read_excel(uploaded_test)

                user_data.to_csv(os.path.join('artifact', 'user_test.csv'), index=False)
                st.session_state['user_data'] = user_data

        # Restore saved input if available (critical for manual flow)
        if user_data is None and 'user_data' in st.session_state:
            user_data = st.session_state['user_data']

        if user_data is not None:
            st.subheader("👀 Preview of Your Input Data")
            st.dataframe(user_data)
            if st.button("🔢 Predict Now"):
                with st.spinner("Running Prediction..."):
                    final_df = initiate_training()
                    st.session_state.final_df = final_df
                    st.success("Prediction Completed!")

        if 'final_df' in st.session_state:
            st.subheader("🔮 Prediction Result")
            st.dataframe(st.session_state.final_df)
            # Download predicted results
            csv = st.session_state.final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Prediction Result as CSV",
                data=csv,
                file_name="prediction_result.csv",
                mime="text/csv"
            )
                # === Download model option right after this ===
            if os.path.exists(os.path.join("artifact", "model.pkl")):
                with open(os.path.join("artifact", "model.pkl"), "rb") as f:
                    st.download_button(
                        label="📦 Download Trained Model",
                        data=f,
                        file_name="best_trained_model.pkl",
                        mime="application/octet-stream",
                        key='LastButton'
                    )

            
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 15px;'>
             <strong>Created By Aryan Sanjot </strong><br>
            Crafted  using <strong>Machine Learning</strong> & <strong>Python 🐍</strong><br>
              <em>Keep exploring, keep innovating — Every model is a step toward the future ✨</em>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Please upload a file to proceed.")
