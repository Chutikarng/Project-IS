import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# 🎯 Configure Streamlit Web App
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")

# ✅ Load the pre-trained model
@st.cache_data
def load_model():
    with open("rf_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# 📂 Define CSV file path
csv_path = "Dataset/train.csv"  

# ✅ Check if the file exists before loading
if not os.path.exists(csv_path):
    st.error(f"🚨 File Not Found: `{csv_path}`. Please check that the file is in the correct directory.")
else:
    # 📌 Load the CSV file
    df = pd.read_csv(csv_path)
    st.write("📌 **Titanic CSV Data Preview:**")
    st.write(df.head())

    # 📌 Drop unnecessary text columns before prediction
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # 📌 Convert categorical features to numeric
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})  # Encode gender
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Encode embarkation port

    # 📌 Ensure all columns are numeric before filling missing values
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
    df.fillna(df.median(), inplace=True)  # Fill missing values with median

    # 📌 Make Predictions
    if st.button("🔍 Predict"):
        predictions = model.predict(df)
        df['Prediction'] = ["✅ Survived" if p == 1 else "❌ Not Survived" for p in predictions]
        st.write("📌 **Predicted Results:**")
        st.write(df)

        # 📊 Show Bar Chart for Predictions
        st.write("📊 **Prediction Distribution**")
        pred_counts = df["Prediction"].value_counts()
        st.bar_chart(pred_counts)
