#Working

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE

df = pd.read_csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv") 
texts = df['text'].tolist()
labels = df['label'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert texts to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


# Initialize the MLP (Multi-Layer Perceptron) classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the MLP classifier
mlp_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_mlp = mlp_classifier.predict(X_test_tfidf)

# Evaluate the performance of the MLP model
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp))