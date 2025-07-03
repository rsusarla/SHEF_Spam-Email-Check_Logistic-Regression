import streamlit as st
import numpy as np
import joblib


model=joblib.load('/content/logistic_regression_model.joblib')

vectorizer = joblib.load('/content/vectorizer.joblib')

st.title("Message Check - Spam or Ham (Not a Spam)")
st.subheader("Enter a message to identify Spam or Ham")

user_message = st.text_input("Enter SMS message:")
if st.button("Check Message"):
  transformed_input = vectorizer.transform([user_message])  # This returns a sparse matrix (2D)
  message_type=model.predict(transformed_input)[0]
  # model.predict(transformed_input) returns a NumPy array containing predictions.
  # [0] grabs the first element of the array, which represents the prediction of the single input message.

  if message_type == 0:
    result = "Ham Message"
  else:
    result = "Spam Message"
  st.success(f"The entered message is a: {result}")
