import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_model, predict_transaction, load_feature_importance
from utils import load_metrics

st.set_page_config(page_title="Fraud Detection AI", layout="wide")

st.title("Fraud Detection AI System")
st.write("Detect suspicious transactions using machine learning.")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ===============================
# Session State
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ===============================
# Single Prediction Section
# ===============================
st.header("Real-Time Fraud Prediction")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    time = st.slider("Transaction Hour", 0, 23, 12)
    location = st.selectbox(
        "Location",
        ["Colombo", "Kandy", "Galle", "Negombo", "NewYork", "London", "Dubai", "Singapore", "Tokyo", "Matara"]
    )

with col2:
    device = st.selectbox("Device", ["Mobile", "Laptop", "Tablet"])
    merchant = st.selectbox(
        "Merchant",
        ["Amazon", "eBay", "Daraz", "AliExpress", "Uber", "Keells", "PickMe"]
    )

if st.button("Predict Fraud"):
    prediction, probability, risk = predict_transaction(
        model, amount, time, location, device, merchant
    )

    st.session_state.last_prediction = {
        "amount": amount,
        "time": time,
        "location": location,
        "device": device,
        "merchant": merchant,
        "probability": probability,
        "risk": risk,
        "prediction": "Fraud" if prediction == 1 else "Normal"
    }

if st.session_state.last_prediction is not None:
    result = st.session_state.last_prediction

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{result['probability']:.2%}")
    c2.metric("Risk Level", result["risk"])
    c3.metric("Prediction", result["prediction"])

    if result["prediction"] == "Fraud":
        st.error("Suspicious Transaction Detected")
    else:
        st.success("Transaction Looks Normal")

# ===============================
# Dashboard Section
# ===============================
st.divider()
st.header("Fraud Analytics Dashboard")

if st.button("Add to Dashboard"):
    if st.session_state.last_prediction is None:
        st.warning("Please click Predict Fraud first.")
    else:
        st.session_state.history.append(st.session_state.last_prediction.copy())
        st.success("Transaction added to dashboard.")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)

    st.subheader("Transaction History")
    st.dataframe(history_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots()
        history_df["risk"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Fraud Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(history_df["probability"], bins=10)
        ax.set_xlabel("Fraud Probability")
        ax.set_ylabel("Count")
        st.pyplot(fig)
else:
    st.info("No transactions added to dashboard yet.")

# ===============================
# Batch Fraud Detection
# ===============================
st.divider()
st.header("Batch Fraud Detection (Upload CSV)")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    # Clean column names
    batch_df.columns = batch_df.columns.str.strip().str.lower()

    st.subheader("Uploaded Data")
    st.dataframe(batch_df, use_container_width=True)

    required_columns = ["amount", "time", "location", "device", "merchant"]
    missing_columns = [col for col in required_columns if col not in batch_df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.write("Your CSV must contain these columns:")
        st.code("amount,time,location,device,merchant")
    else:
        if st.button("Run Fraud Detection"):
            probabilities = []
            risks = []
            predictions = []

            for _, row in batch_df.iterrows():
                pred, prob, risk = predict_transaction(
                    model,
                    float(row["amount"]),
                    int(row["time"]),
                    str(row["location"]),
                    str(row["device"]),
                    str(row["merchant"])
                )

                probabilities.append(prob)
                risks.append(risk)
                predictions.append("Fraud" if pred == 1 else "Normal")

            results_df = batch_df.copy()
            results_df["fraud_probability"] = probabilities
            results_df["risk"] = risks
            results_df["prediction"] = predictions

            st.subheader("Fraud Detection Results")
            st.dataframe(results_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Distribution")
                fig, ax = plt.subplots()
                results_df["prediction"].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            with col2:
                st.subheader("Risk Distribution")
                fig, ax = plt.subplots()
                results_df["risk"].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Risk")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Fraud Results CSV",
                csv,
                "fraud_results.csv",
                "text/csv"
            )

       
# Feature Importance


st.divider()
st.header("Model Feature Importance")

try:
    importance = load_feature_importance()

    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=["feature", "importance"]
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).head(10)

    st.dataframe(importance_df, use_container_width=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Feature importance error: {e}")



# ===============================
# Model Performance
# ===============================

st.divider()
st.header("Model Performance")

try:
    metrics = load_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['f1']:.2f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    ax.imshow(metrics["confusion_matrix"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curve")

    fig, ax = plt.subplots()
    ax.plot(metrics["fpr"], metrics["tpr"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC AUC = {metrics['roc_auc']:.2f}")
    st.pyplot(fig)

except:
    st.warning("Metrics not available. Train model first.")