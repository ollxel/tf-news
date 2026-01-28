import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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


vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vectorized = vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

while True:
    title = (input("Enter title: "))
    title_vectorized = vectorizer.transform([title])
    prediction = model.predict(title_vectorized)

    if prediction[0] == 0:
        print("Prediction: REAL news")
    else:
        print("Prediction: FAKE news")
    z = (input("Enter again? y/n"))
    if z == 'y' or 'Y':
        print("Okay")
    else:
        break