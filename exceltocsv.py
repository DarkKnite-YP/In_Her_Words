# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report
# from sklearn.externals import joblib
#
# # Read data from Excel file
# excel_file = 'data.xlsx'  # Change this to your Excel file path
# df = pd.read_excel(excel_file)
#
# # Split data into features (text) and labels (hostile or non-hostile)
# X = df['Text']
# y = df['Hostile']
#
# # Feature extraction using TF-IDF
# vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
# X_vectorized = vectorizer.fit_transform(X)
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
#
# # Train a linear SVM classifier
# classifier = LinearSVC()
# classifier.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = classifier.predict(X_test)
#
# # Evaluate the classifier
# print(classification_report(y_test, y_pred))
#
# # Save the trained classifier and vectorizer for future use
# joblib.dump(classifier, 'classifier.pkl')
# joblib.dump(vectorizer, 'vectorizer.pkl')
#
# print("Classifier trained and saved successfully.")
