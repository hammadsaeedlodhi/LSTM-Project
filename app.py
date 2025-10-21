# app.py
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("model_word_lstm.keras")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# -----------------------------
# Predict next word
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    if not token_list:
        return None
    token_list = token_list[-(max_sequence_len - 1):] if len(token_list) >= (max_sequence_len - 1) else token_list
    padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    preds = model.predict(padded, verbose=0)
    predicted_index = int(np.argmax(preds, axis=1)[0])
    return tokenizer.index_word.get(predicted_index, None)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hamlet LSTM Predictor", page_icon="🎭", layout="centered")

st.title("🎭 Shakespeare Next Word Predictor (Hamlet-LSTM)")
st.markdown("Type a short phrase and the model will guess the next word in Hamlet's style!")

# User input
input_text = st.text_input("Enter your text:", "fran you come most carefully vpon your")

if st.button("Predict Next Word"):
    with st.spinner("Thinking like Shakespeare... 🧠"):
        # derive max sequence length from model input
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"**Predicted next word:** {next_word}")
        else:
            st.warning("Could not predict next word. Try a different phrase!")

st.markdown("---")
st.caption("Built with 🧡 using TensorFlow & Streamlit")
