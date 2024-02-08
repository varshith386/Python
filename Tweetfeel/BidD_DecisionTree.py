#Working

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

df = pd.read_csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv") 
texts = df['text'].tolist()
labels = df['label'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert texts to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
tree_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_tree = tree_classifier.predict(X_test_tfidf)

# Evaluate the performance of the Decision Tree model
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))
