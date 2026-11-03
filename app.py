import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "gulpara/sms-spam-detection"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message below and click Predict.")

message = st.text_area("SMS Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter an SMS message.")
    else:
        inputs = tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
            
            if prediction == 1:
                st.error(f"🚨 Spam ({confidence:.2f}% confidence)")
            else:
                st.success(f"✅ Ham ({confidence:.2f}% confidence)")
