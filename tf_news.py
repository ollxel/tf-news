import re
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

fake = fake[["text", "label"]]
true = true[["text", "label"]]

df = pd.concat([fake, true], ignore_index=True)
df.drop_duplicates(inplace=True)
df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
maxlen = 150

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

model = Sequential([
    Embedding(vocab_size, 100, input_length=maxlen),
    Dropout(0.5),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_pad, y_train_cat,
          epochs=5,
          batch_size=64,
          validation_data=(X_test_pad, y_test_cat))

loss, acc = model.evaluate(X_test_pad, y_test_cat)
print("Test Loss:", loss, "Test Accuracy:", acc)

def predict_text(text, model, tokenizer, maxlen):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(pad)
    idx = np.argmax(pred, axis=1)[0]
    return "Real" if idx == 1 else "Fake"

# Example usage
while True:
    user_input = input("Enter news headline (or 'exit'): ")
    if user_input.lower() in ("exit", "quit"):
        break
    print("Prediction:", predict_text(user_input, model, tokenizer, maxlen))