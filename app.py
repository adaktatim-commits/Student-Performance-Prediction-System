

import streamlit as st
import pandas as pd


from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    get_dataset_info
)

from src.visualization import (
    show_visualizations,
    study_hours_chart,
    attendance_chart,
    correlation_heatmap
)

from src.model_training import (
    train_model,
    predict_student
)


# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide"
)


# Project Title
st.title("STUDENT PERFORMANCE PREDICTION SYSTEM")

st.header("Project Purpose")

st.write("""
This project aims to analyze student academic data
and predict their performance using Machine Learning.
""")

st.header("User Instructions")

st.write("""
1. Upload a valid student dataset in CSV format.
2. Preview and inspect the dataset.
3. Handle missing values and duplicate records.
4. Explore academic performance through charts.
5. Train and evaluate a Machine Learning model.
6. Enter student details to generate predictions.
""")

st.header("Project Workflow")

st.info("""
Dataset Upload → Data Preview → Data Cleaning →
Exploratory Data Analysis → Model Training →
Model Evaluation →  Prediction → Limitations

""")


# STEP 5: Upload Dataset

st.header("Step 5: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload Student Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to start.")
    st.stop()


# Load Dataset
try:
    df = load_dataset(uploaded_file)

    st.success("Dataset uploaded successfully!")

except ValueError as e:
    st.error(str(e))
    st.stop()


# STEP 6: Dataset Exploration

st.header("Step 6: Dataset Exploration")

st.subheader("Original Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Data Types")

data_types = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(data_types)

st.subheader("Statistical Summary")

summary = df.describe(include="all")

st.dataframe(
    summary.astype(str)
) 

st.subheader("Missing Values Before Cleaning")

missing_data = pd.DataFrame({
    "Column Name": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_data)

st.subheader("Duplicate Rows Before Cleaning")
st.write(df.duplicated().sum())


# Data Cleaning
try:
    cleaned_df, cleaning_info = clean_dataset(df)

except ValueError as e:
    st.error(str(e))
    st.stop()


st.subheader("Cleaned Dataset")
st.dataframe(cleaned_df)

st.success("Data cleaning completed!")

st.write("Rows Before Cleaning:", cleaning_info["rows_before"])
st.write("Rows After Cleaning:", cleaning_info["rows_after"])
st.write("Duplicate Rows Removed:", cleaning_info["duplicates_removed"])
st.write("Rows Removed Due to Missing Target:", cleaning_info["target_rows_removed"])

st.subheader("Missing Values After Cleaning")
st.write(cleaned_df.isnull().sum())


# Dataset Information
info = get_dataset_info(cleaned_df)

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Rows", info["shape"][0])

with col2:
    st.metric("Total Columns", info["shape"][1])


# STEP 7: Visualization

st.header("Step 7: Exploratory Data Analysis")

st.subheader("1. Score Distribution")

fig1 = show_visualizations(cleaned_df)
st.pyplot(fig1)
st.write(
    "Observation: This chart shows the distribution "
    "of students' final scores."
)


st.subheader("2. Study Hours vs Final Score")

fig2 = study_hours_chart(cleaned_df)
st.pyplot(fig2)
st.write(
    "Observation: This chart helps to understand "
    "the relationship between study hours and final scores."
)


st.subheader("3. Attendance Distribution")

fig3 = attendance_chart(cleaned_df)
st.pyplot(fig3)
st.write(
    "Observation: This chart shows how student attendance "
    "is distributed."
)


st.subheader("4. Correlation Heatmap")

fig4 = correlation_heatmap(cleaned_df)
st.pyplot(fig4)
st.write(
    "Observation: The heatmap shows the correlation "
    "between academic features and final scores."
)


# STEP 8 and 9: Model Training and Evaluation

st.header("Step 8 & 9: Model Training and Evaluation")

if st.button("Train Linear Regression Model"):

    try:
        model, metrics, y_test, predictions = train_model(
            cleaned_df
        )

        st.session_state["model"] = model

        st.success("Model trained successfully!")

        st.subheader("Model Evaluation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")

        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")

        with col3:
            st.metric("R2 Score", f"{metrics['R2 Score']:.3f}")

        st.write(
            "MAE: Average absolute prediction error."
        )

        st.write(
            "RMSE: Prediction error with greater penalty "
            "for larger errors."
        )

        st.write(
            "R2 Score: Indicates how well the model "
            "explains variation in the target score."
        )

    except Exception as e:
        st.error(f"Model training failed: {e}")


# STEP 10: Prediction Form

st.header("Step 10: Student Performance Prediction")

if "model" not in st.session_state:

    st.info("Please train the model before making predictions.")

else:

    with st.form("prediction_form"):

        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=2.0
        )

        attendance_pct = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        practice_tests = st.number_input(
            "Practice Tests",
            min_value=0,
            value=3
        )

        submitted = st.form_submit_button(
            "Predict Final Score"
        )

    if submitted:

        prediction = predict_student(
            st.session_state["model"],
            study_hours,
            attendance_pct,
            previous_score,
            practice_tests
        )

        st.success(
            f"Predicted Final Score: {prediction:.2f}"
        )

        st.info(
            "Note: This prediction is an estimate "
            "based on the trained Machine Learning model."
        )


# Project Limitations

st.header("Project Limitations")

st.write("""
1. Prediction accuracy depends on dataset quality.
2. Linear Regression assumes a linear relationship.
3. Predicted scores are estimates, not guaranteed results.
""")
