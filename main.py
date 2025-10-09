import streamlit as st
import pickle
import numpy as np

# Load the trained ML model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Header
st.title("🩺 Diabetes Prediction App")
st.markdown("""
Enter your health parameters below to predict the risk of Diabetes.
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 79)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.number_input("Age", 1, 120, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = np.array(list(data.values())).reshape(1, -1)
    return features

input_data = user_input_features()

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of being diabetic
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes! Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes! Probability: {probability*100:.2f}%")





