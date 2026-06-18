# 📊 Customer Churn Prediction App

This project predicts whether a customer is likely to churn (leave the service) based on telecom usage and billing data using Machine Learning.

---

## 🚀 Live App
[Click here to use the app](https://customer-churn-prediction-gkjw.streamlit.app/)

---

## 🧠 Problem Statement
Customer churn is a major issue in the telecom industry.  
This project helps identify customers who are likely to leave so that preventive actions can be taken.

---

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- RandomForest Classifier

---

## 📊 Model Performance

- Model: Random Forest Classifier  
- Final Accuracy: **79.56%**  

---

## 🔬 Data Preprocessing & Improvement

- Missing values in **TotalCharges** were handled using **median imputation**.
- This improved model accuracy from **79.13% to 79.56%**.
- In comparison, removing missing rows using `dropna()` reduced accuracy to **77.47%**.

👉 This shows that proper data imputation significantly improves model performance compared to data deletion.

---
“I improved model performance by applying median imputation instead of row deletion, increasing accuracy from 77.47% to 79.56%.”
---
## 🔑 Features Used

- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Contract Type
- Payment Method
- Online Security
- Gender
- Paperless Billing

---

## 🚀 How It Works

1. User enters customer details in Streamlit UI  
2. Data is preprocessed (encoding + feature alignment)  
3. Trained ML model predicts churn probability  
4. Result is displayed with confidence score  

---

## 📈 Example Output

- Churn Probability: 0.78  
- Prediction: Customer is likely to CHURN  

---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run customer_streamlit.py
```

👨‍💻 Author

Gurmeet Punia
