import streamlit as st
import pickle
import pandas as pd

df_fake = pd.read_csv("../Fake.csv")
df_true = pd.read_csv("../True.csv")

# Load the model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Title
st.title("📰 Fake News Detector")

# Input box
input_text = st.text_area("Enter the news text below:")

# Predict button
if st.button("Check if it's Fake or Real"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success("✅ This looks like **REAL** news.")
        else:
            st.error("🚨 This looks like **FAKE** news.")
