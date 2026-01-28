import pandas as pd
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

true['tf'] = 0
fake['tf'] = 1

print(true.columns)
print(true)

del true['text']
del fake['text']
del true['date']
del fake['date']
del true['subject']
del fake['subject']

combined = pd.concat([true, fake], ignore_index=True)
print(combined['tf'].mean())

x = combined['title']
y = combined['tf']


vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 3),  # Учитываем не только слова, но и словосочетания
    min_df=5,  # Игнорируем слишком редкие слова
    max_df=0.7  # Игнорируем слишком частые слова
)
X_vectorized = vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

scores = cross_val_score(model, X_vectorized, y, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean CV accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')


def analyze_prediction(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    proba = model.predict_proba(text_vectorized)[0]

    # Получаем важные слова для предсказания
    feature_names = vectorizer.get_feature_names_out()
    coef = model.feature_importances_  # Важность признаков

    # Находим топ-5 слов, повлиявших на решение
    if hasattr(model, 'feature_importances_'):
        indices = np.argsort(coef)[::-1][:10]
        top_features = [feature_names[i] for i in indices if coef[i] > 0]

    result = "FAKE" if prediction == 1 else "REAL"
    confidence = proba[1] if prediction == 1 else proba[0]

    print(f"\nТекст: '{text}'")
    print(f"Предсказание: {result} (уверенность: {confidence:.2%})")
    if confidence < 0.7:
        print("⚠️  Модель не уверена в ответе!")
    print(f"Вероятности: REAL={proba[0]:.2%}, FAKE={proba[1]:.2%}")

    return prediction, confidence

while True:
    title = (input("Enter title: "))
    title_vectorized = vectorizer.transform([title])
    prediction = model.predict(title_vectorized)

    if prediction[0] == 0:
        print("Prediction: REAL news")
    else:
        print("Prediction: FAKE news")
    z = input("Enter again? y/n")
    if z == 'y' or 'Y':
        print("Okay")
    else:
        break