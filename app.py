import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

@st.cache_resource
def train_model():
    # Pima Indians Diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(url, names=columns)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Replace medically impossible zero values with median values
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_columns:
        median = X.loc[X[col] != 0, col].median()
        X[col] = X[col].replace(0, median)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, accuracy

st.title("🩺 Diabetes Prediction using Machine Learning")
st.write("Enter the patient's health information to get a machine-learning prediction.")

try:
    model, scaler, accuracy = train_model()

    st.sidebar.header("Model Information")
    st.sidebar.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    st.sidebar.info(
        "Model: Logistic Regression\n\n"
        "Dataset: Pima Indians Diabetes Dataset"
    )

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]])

        input_df = pd.DataFrame(input_data, columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])

        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        # Use reasonable fallback values for zero inputs
        fallback = {
            "Glucose": 120, "BloodPressure": 70,
            "SkinThickness": 20, "Insulin": 80, "BMI": 25
        }
        for col in zero_columns:
            if input_df.loc[0, col] == 0:
                input_df.loc[0, col] = fallback[col]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Prediction: Higher likelihood of diabetes ({probability*100:.1f}%)")
        else:
            st.success(f"✅ Prediction: Lower likelihood of diabetes ({(1-probability)*100:.1f}%)")

        st.caption(
            "This project is for educational purposes only and is not a medical diagnosis."
        )

except Exception as e:
    st.error("The dataset could not be loaded. Please check your internet connection and try again.")
    st.code(str(e))
