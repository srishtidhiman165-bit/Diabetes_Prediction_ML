# 🩺 Diabetes Prediction using Machine Learning

A beginner-friendly Machine Learning project that predicts the likelihood of diabetes using the **Pima Indians Diabetes Dataset** and **Logistic Regression**.

## 📌 Project Overview

This project demonstrates a complete basic ML workflow:

1. Load a healthcare dataset
2. Clean and preprocess the data
3. Split data into training and testing sets
4. Scale numerical features
5. Train a Logistic Regression model
6. Evaluate model accuracy
7. Create an interactive Streamlit web application
8. Make predictions from user input

## 🧠 Machine Learning Algorithm

**Logistic Regression** is used because the target variable has two classes:

- `0` → No diabetes
- `1` → Diabetes

## 📊 Features

The model uses these 8 inputs:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## 📁 Project Structure

```text
Diabetes-Prediction-ML/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ How to Run

### Step 1: Clone or download the project

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd Diabetes-Prediction-ML
```

### Step 2: Install libraries

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit app

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Train/Test Model

To see the model training and evaluation in the terminal:

```bash
python train_model.py
```

## 📈 Expected Output

The program displays:

- Dataset shape
- Sample records
- Model accuracy
- Classification report

The exact accuracy can vary slightly depending on the environment and dataset version.

## 🌐 Dataset

This project uses the **Pima Indians Diabetes Dataset**, commonly used for educational machine-learning classification tasks.

Dataset source:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- GitHub

## ⚠️ Disclaimer

This project is created for **educational purposes only**. It is not a medical diagnostic tool and should not be used to make medical decisions.

## 👩‍💻 Author

**Srishti Dhiman**

GitHub: Add your GitHub profile link here.
