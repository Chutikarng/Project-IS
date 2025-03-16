import streamlit as st

st.set_page_config(page_title="Data Preparation & Algorithm Theory")

st.title("📊 Data Preparation & Algorithms")

## 🔹 1. Data Preparation
st.header("📌 Data Preparation")
st.write("""
Data preparation is a crucial step in AI model development to achieve high accuracy.
The selected datasets for this project are:

- **Machine Learning Model Dataset** → Kaggle Titanic dataset:  
  [🔗 Titanic - Machine Learning from Disaster](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)
  
- **CNN Model Dataset** → Kaggle Butterfly Image Classification dataset:  
  [🔗 Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

### ✨ **Steps in Data Preparation**
1. **Handling Missing Values**  
   - Titanic dataset: Missing values in **Age**, **Cabin**, and **Embarked** columns are filled with median or most frequent values.
   - Butterfly dataset: Artificially modified by adding **NaN values** using ChatGPT to simulate real-world data issues.
  
2. **Encoding Categorical Data**  
   - Titanic dataset: Converted categorical features (**Sex**, **Embarked**) to numerical using **Label Encoding / One-Hot Encoding**.
   
3. **Feature Scaling**  
   - Standardization or Normalization applied to numerical features.

4. **Image Preprocessing**  
   - Butterfly dataset: Images resized to (150, 150) pixels.
   - Data Augmentation applied (Rotation, Flip, Brightness Adjustment).
""")

## 🔹 2. Algorithm Theory
st.header("📌 Algorithm Theory")

### 🎯 **Machine Learning Model: Titanic Classification**
st.subheader("🚢 Titanic Survival Prediction Model")
st.write("""
The Titanic dataset is used to predict survival using multiple machine learning algorithms.
Three models were compared:

1. **Random Forest Classifier** 🌳  
   - Uses multiple Decision Trees.
   - Reduces overfitting with Bagging.
   - Achieved **highest accuracy**, so it was selected as the final model.

2. **Support Vector Machine (SVM)**  
   - Uses hyperplanes for classification.
   - Performs well on complex patterns but is computationally expensive.

3. **XGBoost Classifier** 🚀  
   - A powerful boosting algorithm.
   - Slightly lower accuracy compared to Random Forest in this case.

📌 **Final Model Selected: Random Forest Classifier (Highest Accuracy).**
""")

st.code("""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
""", language="python")



### 🎯 **Neural Network Model: Butterfly Classification**
st.subheader("🦋 Butterfly Image Classification Model")
st.write("""
The Butterfly dataset was chosen for image classification using **Convolutional Neural Networks (CNN)**.
To introduce data imperfections, **missing values were manually added** using ChatGPT.

Steps in model development:
1. **Load Images & Handle Missing Data**  
   - Some image filenames were randomly replaced with NaN values.
   - Missing data handled by removing or replacing them.

2. **Build CNN Model**  
   - Uses **Conv2D + MaxPooling** layers to extract image features.

3. **Compile & Train the Model**  
   - Optimizer: **Adam**
   - Loss function: **Categorical Crossentropy**

4. **Evaluate Performance**  
   - Accuracy and loss are monitored during training.
""")

st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
""", language="python")

st.write("📌 This CNN model is trained to classify different species of butterflies.")

