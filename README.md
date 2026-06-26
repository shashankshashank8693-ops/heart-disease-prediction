# ❤️ Heart Disease Prediction App

A machine learning web application that predicts the risk of heart disease based on patient health parameters, built using **Logistic Regression** and deployed with **Streamlit**.

🔗 **Live App:** https://your-app-name.streamlit.app

## 📌 Overview

This project uses the UCI Cleveland Heart Disease dataset to train a classification model that predicts whether a patient is at risk of heart disease based on 13 clinical features such as age, cholesterol, blood pressure, and chest pain type.

## 🚀 Features

- Clean, interactive web interface built with Streamlit
- Real-time prediction with probability score
- Data preprocessing pipeline: duplicate removal, outlier handling (IQR method), invalid category filtering
- Feature scaling using StandardScaler
- Logistic Regression model with ~80% test accuracy

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** – data cleaning and preprocessing
- **Scikit-learn** – model training (Logistic Regression)
- **Streamlit** – web app framework

## 📊 Dataset

UCI Cleveland Heart Disease Dataset — 13 clinical features including age, sex, chest pain type, resting blood pressure, cholesterol, ECG results, and more.

## 🧹 Data Preprocessing

- Removed duplicate records
- Removed invalid category values (`thal = 0`, `ca = 4`)
- Removed outliers using the IQR method on continuous columns
- Scaled numerical features using StandardScaler

## 📈 Model Performance

| Metric | Score |
|---|---|
| Train Accuracy | ~87.8% |
| Test Accuracy | ~80.4% |

## 💻 Run Locally

```bash
git clone https://github.com/your-username/heart-disease-app.git
cd heart-disease-app
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📁 Project Structure