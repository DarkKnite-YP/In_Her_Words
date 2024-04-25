# import pandas as pd
# import numpy as np
# from collections import Counter
# import math
#
# # Read data from Excel file
# excel_file = 'Book1.xlsx'  # Change this to your Excel file path
# df = pd.read_excel(excel_file)
#
#
# # Tokenization and preprocessing
# def preprocess_text(text):
#     # Simple tokenization - split by whitespace and convert to lowercase
#     return text.lower().split()
#
#
# # Split data into features (text) and labels (hostile or non-hostile)
# X = df['Text'].apply(preprocess_text)
# y = df['Hostile']
#
# # Build vocabulary
# vocabulary = Counter()
# for text in X:
#     vocabulary.update(text)
#
# # Map words to indices
# word_to_index = {word: index for index, (word, _) in enumerate(vocabulary.items())}
#
#
# # Convert text data into feature vectors (bag-of-words representation)
# def text_to_feature_vector(text):
#     vector = np.zeros(len(vocabulary))
#     for word in text:
#         if word in word_to_index:
#             vector[word_to_index[word]] += 1
#     return vector
#
#
# X_vectorized = np.array([text_to_feature_vector(text) for text in X])
#
# # Split data into training and testing sets (80% training, 20% testing)
# split_index = int(0.8 * len(X))
# X_train, X_test = X_vectorized[:split_index], X_vectorized[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]
#
#
# # Logistic regression classifier
# class LogisticRegression:
#     def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
#         self.learning_rate = learning_rate
#         self.max_iter = max_iter
#         self.tol = tol
#
#     def fit(self, X, y):
#         self.coef_ = np.zeros(X.shape[1])
#         for _ in range(self.max_iter):
#             prev_coef = np.copy(self.coef_)
#             for i in range(X.shape[0]):
#                 prediction = self.predict_proba(X[i])
#                 error = y[i] - prediction
#                 self.coef_ += self.learning_rate * error * X[i]
#             if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
#                 break
#
#     def predict_proba(self, X):
#         z = np.dot(X, self.coef_)
#         return 1 / (1 + np.exp(-z))
#
#     def predict(self, X):
#         return (self.predict_proba(X) >= 0.5).astype(int)
#
#
# # Train logistic regression classifier
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = classifier.predict(X_test)
#
# # Evaluate the classifier
# accuracy = np.mean(y_pred == y_test)
# print("Accuracy:", accuracy)
