import streamlit as st

st.set_page_config(page_title="AI Model Development")

st.title("⚙️ AI Model Development")

st.header("📌 Machine Learning Model Development (Random Forest)")
st.write("""
**Steps:**
1. **Prepare Data** → Handle missing values & encode categorical data.
2. **Split Data** → `train_test_split()`
3. **Train Model** → `RandomForestClassifier`
4. **Evaluate Model** → `accuracy_score()`
""")

st.code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("dataset/train.csv")
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
""", language="python")

st.header("📌 Neural Network Model Development (CNN)")
st.write("""
**Steps:**
1. **Load Image Data** → `ImageDataGenerator()`
2. **Build CNN Model**
3. **Compile Model**
4. **Train Model**
5. **Evaluate Performance**
""")

st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "images/train", target_size=(150,150), batch_size=32, class_mode="categorical", subset="training")

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
""", language="python")
