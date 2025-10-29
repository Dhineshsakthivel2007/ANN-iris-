import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and preprocessing files
model = load_model("my_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

classes = np.load("classes.npy")

# Streamlit UI
st.title("🌸 Iris Species Predictor (Pretrained ANN)")
st.write("Model loaded from `iris_ann_model.h5`")

# Input fields
st.subheader("Enter flower features:")
sl = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
pl = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
pw = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button("🔮 Predict Species"):
    sample = np.array([[sl, sw, pl, pw]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    pred_class = np.argmax(prediction, axis=1)[0]
    pred_label = classes[pred_class]
    st.success(f"🌼 Predicted Species: **{pred_label}**")
