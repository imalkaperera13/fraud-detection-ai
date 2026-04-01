# Fraud Detection AI System

An AI-based fraud detection system that predicts suspicious transactions using machine learning.

## Features
- Upload and process fraud transaction data
- Train fraud detection model
- Predict fraud probability
- Show risk level
- Web-based dashboard using Streamlit

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## Project Structure
```bash
fraud-detection-ai/
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
```
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
