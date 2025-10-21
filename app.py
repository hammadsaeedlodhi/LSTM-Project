import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1) Load model and tokenizer
# -----------------------------
MODEL_PATH = "model_word_lstm.keras"
TOKENIZER_PATH = "tokenizer.pickle"

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.stop()

# -----------------------------
# 2) Helper: Predict next word
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text.lower()])[0]
        token_list = token_list[-(max_sequence_len - 1):] if len(token_list) >= (max_sequence_len - 1) else token_list
        padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        preds = model.predict(padded, verbose=0)
        predicted_index = np.argmax(preds, axis=1)[0]

        # Get the predicted word
        if hasattr(tokenizer, "index_word"):
            return tokenizer.index_word.get(predicted_index, None)
        else:
            inv_map = {v: k for k, v in tokenizer.word_index.items()}
            return inv_map.get(predicted_index, None)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -----------------------------
# 3) Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Next Word Predictor (LSTM)", page_icon="🔮", layout="centered")

st.title("🔮 Next Word Prediction using LSTM")
st.write("This app uses a trained LSTM model on *Shakespeare’s Hamlet* to predict the next word from a given input sentence.")

# Input box
input_text = st.text_input("Enter a text prompt:", placeholder="e.g. fran you come most carefully vpon your")

if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"✨ Predicted next word: **{next_word}**")
        else:
            st.error("No prediction could be made. Try a different input!")

# -----------------------------
# 4) Footer
# -----------------------------
st.markdown("---")
st.caption("Developed with ❤️ using TensorFlow + Streamlit")
