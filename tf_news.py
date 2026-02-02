# Consolidated runnable script: preprocessing, training, evaluation, prediction
import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data (if not already available)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    text = re.sub(r'\s+', ' ', str(text), flags=re.I)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if w not in stop_words and len(w) > 3]
    indices = np.unique(words, return_index=True)[1]
    return np.array(words)[np.sort(indices)].tolist()

def build_and_train(epochs=15):
    # Load and prepare data
    Fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    Fake['label'] = 0
    true['label'] = 1
    Fake.drop(columns=['title','date','subject'], inplace=True)
    true.drop(columns=['title','date','subject'], inplace=True)
    News = pd.concat([Fake, true], ignore_index=True)
    News.drop_duplicates(inplace=True)
    x = News.drop('label', axis=1)
    y = News['label']
    texts = list(x['text'])
    cleaned = [process_text(t) for t in texts]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(cleaned, y, test_size=0.2, random_state=42)

    # Tokenize and pad
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    v = len(tokenizer.word_index)
    maxlen = 150
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_train_one = tf.keras.utils.to_categorical(y_train_enc)
    y_test_one = tf.keras.utils.to_categorical(y_test_enc)

    # Build model
    inputt = Input(shape=(maxlen,))
    x = Embedding(v+1, 100)(inputt)
    x = Dropout(0.5)(x)
    x = LSTM(150, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation='softmax')(x)
    model = Model(inputt, out)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train (may take time)
    history = model.fit(X_train_pad, y_train_one, epochs=epochs, validation_data=(X_test_pad, y_test_one))

    # Evaluate
    loss, acc = model.evaluate(X_test_pad, y_test_one)
    print('Test Loss:', loss, 'Test Accuracy:', acc)
    return model, tokenizer, maxlen, le

def predict_text(text, model, tokenizer, maxlen):
    toks = process_text(text)
    seq = tokenizer.texts_to_sequences([toks])
    pad = pad_sequences(seq, maxlen=maxlen)
    preds = model.predict(pad)
    idx = np.argmax(preds, axis=1)[0]
    return 'Real' if idx == 1 else 'Fake'

# Usage note: running build_and_train() will train the model in one cell and return (model, tokenizer, maxlen, label_encoder).
# Example: model, tokenizer, maxlen, le = build_and_train(epochs=3)
# Then: print(predict_text('Some news text here', model, tokenizer, maxlen))
# Запуск: обучение модели + интерактивное предсказание
print('🚀 Starting build_and_train() — это займёт время...')
model, tokenizer, maxlen, le = build_and_train(epochs=3)
print('✅ Обучение завершено! Введите текст новости для классификации.')
print('   Введите "quit" или "exit" для выхода.\n')

while True:
    try:
        user_input = input('📝 Введите текст новости: ')
    except (EOFError, KeyboardInterrupt):
        break
    
    if not user_input.strip():
        print('   ⚠️  Пусто, повторите.')
        continue
    
    if user_input.strip().lower() in ('quit', 'exit', 'выход'):
        print('👋 До свидания!')
        break
    
    result = predict_text(user_input, model, tokenizer, maxlen)
    print(f'   🎯 Предсказание: {result}\n')