
import streamlit as st
import pandas as pd
import pickle
import io

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Model input features
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

/* Main page */
.stApp {
    background: #101827;
}

/* Centered compact layout */
.block-container {
    max-width: 680px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Heading */
h1 {
    color: white;
    text-align: center;
    font-size: 30px !important;
}

h2, h3 {
    color: white;
}

/* Normal text */
.stMarkdown p {
    color: #CBD5E1;
}

/* White upload and input card effect */
div[data-testid="stFileUploader"],
div[data-testid="stTextArea"] {
    background: #FFFFFF;
    padding: 12px;
    border-radius: 12px;
}

/* Input text */
textarea {
    color: #172033 !important;
}

/* Buttons */
div.stButton > button {
    width: 100%;
    background: #E63946;
    color: white;
    border: none;
    border-radius: 9px;
    height: 45px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #C1121F;
    color: white;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #1E293B;
    padding: 12px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("💳 Credit Card Fraud Detection")



st.divider()

st.subheader("📂 Upload Transaction Data")

st.write(
    "Upload a CSV file" 
)

# ---------------- INPUT OPTIONS ----------------

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"],
    key="csv_uploader"
)

st.markdown("### OR")

transaction_data = st.text_area(
    "Paste Transaction Details (CSV Format)",
    height=150,
    placeholder="Paste your CSV data here..."
)

# ---------------- PREDICTION BUTTON ----------------

if st.button(
    "🔍 Predict Transactions",
    type="primary",
    key="main_predict_button"
):

    try:
        # Read uploaded file or pasted data

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

        elif transaction_data.strip():
            data = pd.read_csv(io.StringIO(transaction_data))

        else:
            st.warning("Please upload a CSV file or paste transaction data.")
            st.stop()

        # Check empty dataset

        if data.empty:
            st.error("The uploaded file contains no transactions.")
            st.stop()

        # Check required columns

        missing_columns = [
            col for col in features if col not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns: {', '.join(missing_columns)}"
            )
            st.stop()

        # Prepare model input

        X = data[features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if X.isnull().any().any():
            st.error(
                "Invalid or missing numerical values found in the CSV."
            )
            st.stop()

        # Check infinite values

        if not X.map(lambda value: abs(value) != float("inf")).all().all():
            st.error("Infinite values are not allowed.")
            st.stop()

        # ---------------- MODEL PREDICTION ----------------

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        # Add predictions to dataset

        results = data.copy()

        results["Prediction"] = [
            "Fraud" if value == 1 else "Genuine"
            for value in predictions
        ]

        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)

        # ---------------- CALCULATE RESULTS ----------------

        total_transactions = len(predictions)

        fraud_count = int((predictions == 1).sum())

        genuine_count = total_transactions - fraud_count

        fraud_percentage = (
            fraud_count / total_transactions
        ) * 100

        # ---------------- DISPLAY DASHBOARD ----------------

        st.divider()

        st.header("📊 Fraud Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            total_transactions
        )

        col2.metric(
            "🚨 Fraud Transactions",
            fraud_count
        )

        col3.metric(
            "✅ Genuine Transactions",
            genuine_count
        )

        st.subheader("Fraud Percentage")

        st.progress(fraud_percentage / 100)

        st.write(f"**{fraud_percentage:.2f}%** transactions predicted as fraud.")

        # ---------------- RESULT TABLE ----------------

        st.subheader("📋 Transaction-wise Predictions")

        st.dataframe(
            results,
            use_container_width=True
        )

        # ---------------- DOWNLOAD REPORT ----------------

        csv_output = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv_output,
            file_name="fraud_prediction_report.csv",
            mime="text/csv",
            key="download_report"
        )

        st.success("Prediction completed successfully!")

    except Exception as error:
        st.error(f"Something went wrong: {error}")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | Machine Learning Project | Logistic Regression"
)
