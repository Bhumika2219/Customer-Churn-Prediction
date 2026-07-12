
# 📊 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using customer demographic and service information.

The project includes data preprocessing, exploratory data analysis (EDA), machine learning model training, evaluation, and an interactive Streamlit dashboard for real-time predictions.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for telecom companies. This project uses machine learning algorithms to identify customers who are likely to leave the company, helping businesses take proactive retention measures.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Interactive Visualizations using Plotly
- Customer Churn Prediction
- Multiple Machine Learning Models
- Model Performance Comparison
- Business Insights Dashboard
- Interactive Streamlit Web Application

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── customer_churn.csv
│   └── clean_customer_churn.csv
│
├── preprocessing/
│   └── clean_data.py
│
├── notebooks/
│   └── eda.py
│
├── models/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── reports/
│
├── saved_model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── label_encoders.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
│
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Data Analysis

The project includes interactive visualizations for:

- Customer Churn Distribution
- Gender Distribution
- Contract Type Analysis
- Internet Service Analysis
- Payment Method Analysis
- Senior Citizen Analysis
- Monthly Charges Distribution
- Tenure Distribution
- Correlation Heatmap
- Monthly Charges vs Total Charges

---

## 🤖 Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📈 Model Evaluation

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1 Score

The dashboard also compares all trained models visually.

---

## 💼 Business Insights

The analysis suggests:

- Month-to-Month customers are more likely to churn.
- Fiber Optic users show higher churn rates.
- Customers with higher monthly charges are more likely to leave.
- Long-tenure customers are generally more loyal.
- Electronic Check users have comparatively higher churn.

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
```

### Move into the project folder

```bash
cd Customer-Churn-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run dashboard/app.py
```

---

## 📸 Dashboard

The dashboard includes:

- Home Page
- Data Analysis
- Customer Prediction
- Model Performance
- Business Insights
- About Page

---

## 👩‍💻 Author

**Bhumika Patil**

Bachelor of Engineering (Information Technology)

Interested in:

- Data Analytics
- Machine Learning
- Business Intelligence

---

## ⭐ Future Improvements

- Hyperparameter Tuning
- Cross Validation
- XGBoost Implementation
- Model Deployment on Cloud
- Real-time Data Integration
