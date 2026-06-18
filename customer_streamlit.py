import pandas as pd 
import streamlit as st
import joblib


model = joblib.load("customer_churn_model.pkl")
features=joblib.load("features.pkl")
st.title("Customer Churn Prediction System")
#st.markdown("Fill all details before prediction ⬇️")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12, 1)
    internet = st.selectbox("Internet Service", ["Select","DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Select","Month-to-month", "One year", "Two year"])
    gender = st.selectbox("Gender", ["Select","Female", "Male"])

with col2:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.5)
    payment = st.selectbox("Payment Method", ["Select","Credit Card", "Electronic Check", "Mailed Check"])
    security = st.selectbox("Online Security", ["Select","No", "Yes"])
    paperless = st.selectbox("Paperless Billing", ["Select","No", "Yes"])

total_charges=tenure*monthly_charges
st.metric("💰 Total Charges", f"{total_charges:.2f}")



if st.button("Predict"):

    if (
        gender == "Select" or
        internet == "Select" or
        payment == "Select" or
        contract == "Select" or
        security == "Select" or
        paperless == "Select"
    ):
        st.error("❌ Please fill all fields before prediction!")
        st.stop()

    

    gender_male = 1 if gender == "Male" else 0

    fiber = 1 if internet == "Fiber optic" else 0
    no_internet = 1 if internet == "No" else 0

    electronic = 1 if payment == "Electronic Check" else 0

    contract_one = 1 if contract == "One year" else 0
    contract_two = 1 if contract == "Two year" else 0

    security_yes = 1 if security == "Yes" else 0

    paperless_yes = 1 if paperless == "Yes" else 0

    data = pd.DataFrame([{
        "TotalCharges": total_charges,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "InternetService_Fiber optic": fiber,
        "InternetService_No": no_internet,
        "PaymentMethod_Electronic check": electronic,
        "Contract_Two year": contract_two,
        "Contract_One year": contract_one,
        "OnlineSecurity_Yes": security_yes,
        "gender_Male": gender_male,
        "PaperlessBilling_Yes": paperless_yes
    }])

    data=data.reindex(columns=features,fill_value=0)
    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to Churn ({probability:.2%})")

    else:
        st.success(f"✅ Customer is likely to Stay ({1-probability:.2%})")