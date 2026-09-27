from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Indian Economy Dashboard",
    page_icon="🇮🇳",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fb;
        }

        .hero {
            padding: 28px;
            border-radius: 18px;
            color: white;
            background: linear-gradient(135deg, #172554, #2563eb);
            margin-bottom: 25px;
        }

        .hero h1 {
            margin-bottom: 5px;
        }

        [data-testid="stMetric"] {
            background-color: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px #00000015;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Load model and dataset
# -----------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache
def load_dataset():
    csv_files = list(BASE_DIR.glob("*.csv"))

    if not csv_files:
        return None, None

    file_path = csv_files[0]
    return pd.read_csv(file_path), file_path.name


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

df, dataset_name = load_dataset()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🇮🇳 Indian Economy Dashboard</h1>
        <p>Explore economic data and generate predictions using Linear Regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Model Prediction",
    ],
)


# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.header("📊 Project Overview")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Dataset Rows", df.shape[0])
        col2.metric("Dataset Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isna().sum().sum()))
        col4.metric("Dataset File", dataset_name)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
    else:
        st.warning("No CSV file was found in the indianeconomy folder.")


# -----------------------------
# EDA
# -----------------------------
elif page == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")

    if df is None:
        st.warning("Please place your CSV dataset in the indianeconomy folder.")
        st.stop()

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    eda_section = st.selectbox(
        "Select EDA section",
        ["Data", "Statistics", "Distribution", "Correlation"],
    )

    if eda_section == "Data":
        st.subheader("Complete Dataset")
        st.dataframe(df)

        st.subheader("Missing Values")
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ["Column", "Missing Values"]
        st.dataframe(missing_values)

    elif eda_section == "Statistics":
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    elif eda_section == "Distribution":
        if numeric_columns:
            selected_column = st.selectbox(
                "Select a numeric column",
                numeric_columns,
            )

            histogram = px.histogram(
                df,
                x=selected_column,
                title="Distribution of " + selected_column,
                color_discrete_sequence=["#2563eb"],
            )

            histogram.update_layout(template="plotly_white")
            st.plotly_chart(histogram)
        else:
            st.info("No numeric columns available.")

    elif eda_section == "Correlation":
        if len(numeric_columns) >= 2:
            correlation = df[numeric_columns].corr()

            heatmap = px.imshow(
                correlation,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Correlation Heatmap",
            )

            st.plotly_chart(heatmap)
        else:
            st.info("At least two numeric columns are required.")


# -----------------------------
# Model prediction
# -----------------------------
elif page == "Model Prediction":
    st.header("🤖 Linear Regression Prediction")

    st.write("Enter a year to predict the economic value.")

    year = st.number_input(
        "Enter Year",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1,
    )

    if st.button("🚀 Generate Prediction"):
        try:
            prediction_result = model.predict([[year]])
            prediction = float(np.asarray(prediction_result).reshape(-1)[0])

            col1, col2 = st.columns(2)
            col1.metric("Selected Year", int(year))
            col2.metric("Predicted Value", f"{prediction:,.2f}")

            st.success("Prediction generated successfully.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.info(
        "The model is loaded from linear_regression_model.pkl and receives "
        "the year as its input feature."
    )