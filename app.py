#Gender : 1=Female   0=Male     
#Churn  : 1=Yes    0=False
# #Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# order of the X = ['Age', 'Gender', 'Tenure', 'MonthlyCharges']

import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

st.divider()

st.write("Please provide the following details to predict if a customer is likely to churn:")

st.divider()

age = st.number_input("Enter age", min_value=18, max_value=100, value=30)


Tenure  = st.number_input("Enter tenure (in months)", min_value=0, max_value=130, value=10)

monthly_charge = st.number_input("Enter monthly charges", min_value=30, max_value=150)

gender = st.selectbox("Select gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict Churn")

st.divider()

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, Tenure, monthly_charge]
    X1 = np.array(X) 
    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]
    
    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.balloons()

    st.write(f"The customer is likely to: **{predicted}**")

else:
    st.write("Please enter all details and click on Predict Churn button.")