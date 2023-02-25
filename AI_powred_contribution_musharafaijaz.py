import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('wikipedia_articles.csv')

# Preprocess the data
# Extract features from the data

data['word_count'] = data['article_text'].apply(lambda x: len(x.split()))
data['ref_count'] = data['article_text'].apply(lambda x: x.count('<ref>'))
data['img_count'] = data['article_text'].apply(lambda x: x.count('<img>'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['quality'], axis=1), data['quality'], test_size=0.2, random_state=42)

# Train a support vector machine (SVM) classifier on the training data
svm_clf = SVC(kernel='linear', C=1)
svm_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, f1_score

# Evaluate the performance of the model on the testing data
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', acc)
print('F1-score:', f1)
