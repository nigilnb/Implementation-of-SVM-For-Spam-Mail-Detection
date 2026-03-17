# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset, remove empty columns, and convert labels (ham = 0, spam = 1).
2.Split the data into training and testing sets.
3.Transform text data into numerical features using TF-IDF vectorization.
4.Train an SVM model on the training data and evaluate its accuracy on test data.

## Program:
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NIGIL.S
RegisterNumber:212225240100
*/
# SVM for Spam Mail Detection (with your dataset format)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset and drop empty columns
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]   # keep only useful columns
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
## Output:
<img width="347" height="43" alt="image" src="https://github.com/user-attachments/assets/32a69cfc-4bd8-4f3d-9b44-3723c5606386" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
