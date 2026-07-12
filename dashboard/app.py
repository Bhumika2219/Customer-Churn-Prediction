import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Dataset
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_customer_churn.csv")

df = load_data()

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📈 Data Analysis",
        "🤖 Prediction",
        "📊 Model Performance",
        "💼 Business Insights",
        "ℹ About"
    ]
)

# -------------------------------
# HOME PAGE
# -------------------------------

if page == "🏠 Home":

    st.title("📊 Customer Churn Prediction Dashboard")

    st.markdown(
        """
        Welcome to the **Customer Churn Prediction Dashboard**.

        This project demonstrates an end-to-end Machine Learning workflow using

        - Python
        - Pandas
        - Scikit-Learn
        - Streamlit
        - Plotly
        """
    )

    st.divider()

    total_customers = len(df)

    churned = len(df[df["Churn"] == "Yes"])

    retained = len(df[df["Churn"] == "No"])

    churn_rate = round((churned / total_customers) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Customers",
        total_customers
    )

    col2.metric(
        "❌ Churned",
        churned
    )

    col3.metric(
        "✅ Retained",
        retained
    )

    col4.metric(
        "📈 Churn Rate",
        f"{churn_rate}%"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    st.subheader("Dataset Information")

    info1, info2 = st.columns(2)

    info1.write(f"Rows : {df.shape[0]}")
    info1.write(f"Columns : {df.shape[1]}")

    info2.write(f"Missing Values : {df.isnull().sum().sum()}")
    info2.write(f"Duplicate Rows : {df.duplicated().sum()}")

# -------------------------------
# DATA ANALYSIS
# -------------------------------
# -------------------------------
# DATA ANALYSIS
# -------------------------------

elif page == "📈 Data Analysis":

    st.title("📈 Customer Churn Analysis Dashboard")

    st.markdown("Explore customer behaviour using interactive charts.")

    st.divider()

    # ===============================
    # Row 1
    # ===============================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn")

        fig = px.pie(
            df,
            names="Churn",
            hole=0.5,
            color="Churn",
            title="Churn Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Gender Distribution")

        fig = px.histogram(
            df,
            x="gender",
            color="gender",
            title="Gender Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Row 2
    # ===============================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Contract Type")

        fig = px.histogram(
            df,
            x="Contract",
            color="Churn",
            barmode="group",
            title="Contract vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Internet Service")

        fig = px.histogram(
            df,
            x="InternetService",
            color="Churn",
            barmode="group",
            title="Internet Service"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Row 3
    # ===============================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Payment Method")

        fig = px.histogram(
            df,
            x="PaymentMethod",
            color="Churn",
            title="Payment Method Analysis"
        )

        fig.update_layout(xaxis_tickangle=-25)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Senior Citizens")

        fig = px.histogram(
            df,
            x="SeniorCitizen",
            color="Churn",
            title="Senior Citizen Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Row 4
    # ===============================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Monthly Charges")

        fig = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            nbins=30,
            title="Monthly Charges Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Tenure")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            title="Tenure Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Scatter Plot
    # ===============================

    st.subheader("Monthly Charges vs Total Charges")

    fig = px.scatter(
        df,
        x="MonthlyCharges",
        y="TotalCharges",
        color="Churn",
        hover_data=["Contract", "PaymentMethod"],
        title="Monthly Charges vs Total Charges"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Correlation Heatmap
    # ===============================

    st.subheader("Correlation Heatmap")

    numeric_df = df.copy()

    for col in numeric_df.select_dtypes(include="object").columns:
        numeric_df[col] = numeric_df[col].astype("category").cat.codes

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Business Insights
    # ===============================

    st.subheader("Business Insights")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
### Key Findings

• Month-to-Month customers churn the most.

• Fiber Optic users have higher churn.

• High Monthly Charges increase churn.

• Customers with long tenure are more loyal.

• Electronic Check users have higher churn.
""")

    with col2:

        st.info("""
### Business Recommendations

• Offer long-term contract discounts.

• Reward loyal customers.

• Reduce pricing for high-charge customers.

• Improve Fiber Optic customer service.

• Target high-risk customers with retention campaigns.
""")

# -------------------------------
# PREDICTION
# -------------------------------

# -------------------------------
# PREDICTION
# -------------------------------

elif page == "🤖 Prediction":

    import joblib

    st.title("🤖 Customer Churn Prediction")

    st.write("Enter customer details below to predict whether the customer is likely to churn.")

    # =====================================
    # Load Models
    # =====================================

    lr_model = joblib.load("saved_model/logistic_regression.pkl")
    dt_model = joblib.load("saved_model/decision_tree.pkl")
    rf_model = joblib.load("saved_model/random_forest.pkl")

    label_encoders = joblib.load("saved_model/label_encoders.pkl")

    # =====================================
    # Select Model
    # =====================================

    model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    if model_name == "Logistic Regression":
        model = lr_model

    elif model_name == "Decision Tree":
        model = dt_model

    else:
        model = rf_model

    st.divider()

    # =====================================
    # Customer Details
    # =====================================

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Partner", ["No", "Yes"])

    dependents = st.selectbox("Dependents", ["No", "Yes"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0
    )

    st.divider()

    # =====================================
    # Prediction Button
    # =====================================

    if st.button("Predict Churn"):

        input_df = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone_service],
            "MultipleLines":[multiple_lines],
            "InternetService":[internet_service],
            "OnlineSecurity":[online_security],
            "OnlineBackup":[online_backup],
            "DeviceProtection":[device_protection],
            "TechSupport":[tech_support],
            "StreamingTV":[streaming_tv],
            "StreamingMovies":[streaming_movies],
            "Contract":[contract],
            "PaperlessBilling":[paperless],
            "PaymentMethod":[payment],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total]

        })

        # Encode categorical columns

        for col in input_df.columns:

            if col in label_encoders:

                encoder = label_encoders[col]

                input_df[col] = encoder.transform(
                    input_df[col].astype(str)
                )

        prediction = model.predict(input_df)[0]

        st.divider()

        if prediction == 1:

            st.error("⚠ Customer is likely to Churn.")

        else:

            st.success("✅ Customer is likely to Stay.")



# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write("Performance comparison of all trained machine learning models.")

    performance = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            0.80,
            0.73,
            0.78
        ],
        "Precision": [
            0.68,
            0.62,
            0.76
        ],
        "Recall": [
            0.58,
            0.67,
            0.60
        ],
        "F1 Score": [
            0.63,
            0.64,
            0.67
        ]
    })

    st.subheader("Performance Table")

    st.dataframe(performance, use_container_width=True)

    st.divider()

    st.subheader("Accuracy Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Accuracy of Machine Learning Models"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Precision Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Precision",
        color="Model",
        text="Precision"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Recall Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Recall",
        color="Model",
        text="Recall"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("F1 Score Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="F1 Score",
        color="Model",
        text="F1 Score"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.success(
        """
### Best Performing Model

✅ Logistic Regression achieved the highest overall accuracy.

It provides a good balance between precision, recall and interpretability,
making it the preferred model for this project.
"""
    )

# -------------------------------
# BUSINESS INSIGHTS
# -------------------------------

elif page == "💼 Business Insights":

    st.title("💼 Business Insights")

    st.success(
        """
        ✔ Month-to-Month contracts have the highest churn.

        ✔ Longer tenure customers are more loyal.

        ✔ Monthly Charges significantly influence churn.

        ✔ Fiber Optic users have higher churn.

        ✔ Contract type is one of the strongest predictors.
        """
    )

# -------------------------------
# ABOUT
# -------------------------------

else:

    st.title("ℹ About Project")

    st.markdown(
        """
## Customer Churn Prediction Dashboard

### Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-Learn

### Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

### Project Workflow

Data Collection

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Model Training

↓

Model Evaluation

↓

Prediction Dashboard
        """
    )