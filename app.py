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

st.success("✅ Model and tokenizer loaded successfully!")

# -----------------------------
# 2) Helper: Temperature sampling
# -----------------------------
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# -----------------------------
# 3) Helper: Predict next word
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len, temperature=0.8):
    try:
        token_list = tokenizer.texts_to_sequences([text.lower()])[0]
        if not token_list:
            return None  # No valid tokens in input
        token_list = token_list[-(max_sequence_len - 1):]
        padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        preds = model.predict(padded, verbose=0)
        predicted_index = sample_with_temperature(preds[0], temperature)
        return tokenizer.index_word.get(predicted_index, None)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -----------------------------
# 4) Streamlit UI
# -----------------------------
st.set_page_config(page_title="Next Word Predictor (LSTM)", page_icon="🔮", layout="centered")

st.title("🔮 Next Word Prediction using LSTM")
st.write("This app uses a trained LSTM model on *Shakespeare’s Hamlet* (and other plays) to predict the next word from a given input sentence.")

# Input box
input_text = st.text_input("Enter a text prompt:", placeholder="e.g. to be or not to")

temperature = st.slider("Adjust creativity (Temperature)", 0.3, 1.5, 0.8, 0.1)

if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len, temperature)
        if next_word:
            st.success(f"✨ Predicted next word: **{next_word}**")
        else:
            st.error("No prediction could be made. Try a different or longer input!")

# -----------------------------
# 5) Footer
# -----------------------------
st.markdown("---")
st.caption("Developed with ❤️ using TensorFlow + Streamlit")
