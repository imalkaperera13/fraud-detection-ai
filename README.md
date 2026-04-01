# 🚨 Fraud Detection AI System

An end-to-end machine learning system that detects **fraudulent transactions** using classification models, real-time prediction, analytics dashboard, and explainable AI.

This project demonstrates a **production-style AI pipeline** including preprocessing, model training, evaluation, feature importance, and batch prediction.

---

# 🧠 Features

* ✅ Real-time fraud prediction
* ✅ Fraud probability scoring
* ✅ Risk classification (Low / Medium / High)
* ✅ Batch fraud detection (CSV upload)
* ✅ Fraud analytics dashboard
* ✅ Feature importance visualization
* ✅ Model performance metrics
* ✅ Confusion matrix
* ✅ ROC curve
* ✅ Downloadable prediction results
* ✅ Explainable AI system

---

# 🏗️ System Architecture

Dataset → Preprocessing → Train Model → Save Model
↓
Prediction Engine
↓
Streamlit Dashboard

---

# 🛠️ Tech Stack

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* Feature Importance
* Model Evaluation Metrics

### Data Processing

* Pandas
* NumPy
* StandardScaler
* OneHotEncoder

### Visualization

* Matplotlib
* Streamlit Charts
* Analytics Dashboard

### Deployment UI

* Streamlit

---

# 📂 Project Structure

```
fraud-detection-ai
│
├── app.py
├── train_model.py
├── preprocess.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── fraud_data.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── feature_importance.pkl
│   └── model_metrics.pkl
```

---

# 🚀 How to Run Locally

### 1. Clone Repository

```
git clone https://github.com/imalkaperera13/fraud-detection-ai.git
cd fraud-detection-ai
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Train Model

```
python train_model.py
```

### 4. Run Application

```
streamlit run app.py
```

---

# 📊 Model Performance

The system evaluates the model using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve

---

# 🔍 Example Prediction

Input:

Transaction Amount: 5000
Time: 02:00 AM
Location: NewYork
Device: Laptop

Output:

Fraud Probability: 0.92
Risk Level: HIGH
Prediction: Fraud

---

# 📈 Dashboard Capabilities

* Fraud distribution chart
* Risk level analytics
* Transaction history
* Probability histogram
* Batch fraud detection results

---

# 🎯 Use Cases

* Banking fraud detection
* Credit card fraud detection
* Payment gateway monitoring
* E-commerce fraud detection
* Telecom fraud detection
* Insurance fraud detection

---

# 🧩 Machine Learning Pipeline

1. Load dataset
2. Data preprocessing
3. Feature encoding
4. Model training
5. Model evaluation
6. Feature importance extraction
7. Save trained model
8. Real-time prediction
9. Dashboard analytics

---

# ⭐ Why This Project Matters

This project demonstrates:

* End-to-end AI system design
* Machine learning pipeline
* Explainable AI
* Model evaluation
* Dashboard development
* Real-time prediction
* Industry-level fraud detection

---

# 🔮 Future Improvements

* XGBoost model integration
* Deep learning fraud detection
* Real-time API deployment
* Database integration
* User authentication
* Cloud deployment (AWS / Streamlit Cloud)

---

# 👨‍💻 Author

Imalka Perera



AI Engineer | DevOps | Machine Learning


GitHub:
https://github.com/imalkaperera13
