import os
import pickle
import nltk
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 0) Reproducibility
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 1) Load or fetch Hamlet text
# -----------------------------
nltk.download('gutenberg', quiet=True)

if os.path.exists('hamlet.txt'):
    with open('hamlet.txt', 'r', encoding='utf-8') as f:
        text = f.read().lower()
else:
    from nltk.corpus import gutenberg
    text = gutenberg.raw('shakespeare-hamlet.txt').lower()
    with open('hamlet.txt', 'w', encoding='utf-8') as f:
        f.write(text)

text = text.replace('\r', '')
text = "\n".join(line.strip() for line in text.splitlines())

# -----------------------------
# 2) Tokenization and sequence building
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print(f"[INFO] Vocabulary size (total_words): {total_words}")

input_sequences = []
for line in text.split('\n'):
    line = line.strip()
    if not line:
        continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        seq = token_list[: i + 1]
        if len(seq) > 1:
            input_sequences.append(seq)

# Fallback if lines were too short
if not input_sequences:
    print("[WARN] No input sequences from lines; using full text.")
    token_list_full = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list_full)):
        input_sequences.append(token_list_full[: i + 1])

if not input_sequences:
    raise RuntimeError("No sequences could be created. Check text input.")

max_sequence_len = max(len(seq) for seq in input_sequences)
print(f"[INFO] Max sequence length: {max_sequence_len}")

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

# -----------------------------
# 3) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# 4) Callback
# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -----------------------------
# 5) Build model
# -----------------------------
embedding_dim = 100
lstm_units_1 = 150
lstm_units_2 = 100

model = Sequential([
    Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_len - 1),
    LSTM(lstm_units_1, return_sequences=True),
    Dropout(0.2),
    LSTM(lstm_units_2),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔧 Build before summary (so parameters show correctly)
model.build(input_shape=(None, max_sequence_len - 1))
model.summary()

# -----------------------------
# 6) Train model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------
# 7) Predict helper
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    token_list = token_list[-(max_sequence_len - 1):] if len(token_list) >= (max_sequence_len - 1) else token_list
    padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    preds = model.predict(padded, verbose=0)
    predicted_index = int(np.argmax(preds, axis=1)[0])
    if predicted_index == 0:
        return None
    return tokenizer.index_word.get(predicted_index, None)

# -----------------------------
# 8) Example prediction
# -----------------------------
input_text = "fran you come most carefully vpon your"
print(f"\nInput text: {input_text}")
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
print(f"Predicted next word: {next_word}")

# -----------------------------
# 9) Save model and tokenizer
# -----------------------------
model_filename = "model_word_lstm.keras"
model.save(model_filename)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n✅ Saved model to {model_filename} and tokenizer to tokenizer.pickle")
