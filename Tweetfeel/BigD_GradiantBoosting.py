import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, auc, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

df = pd.read_csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv")
texts = df['text'].tolist()
labels = df['label'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert texts to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Initialize the XGBoost classifier
xgb_classifier = XGBClassifier()

# Train the XGBoost classifier
xgb_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_xgb = xgb_classifier.predict(X_test_tfidf)

# Evaluate the performance of the XGBoost model
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Get predicted probabilities for each class
y_scores = xgb_classifier.predict_proba(X_test_tfidf)

# Compute ROC-AUC score for each class
roc_auc_scores = []
for i in range(n_classes):
    roc_auc = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
    roc_auc_scores.append(roc_auc)

# Plot ROC-AUC score for each class
plt.figure(figsize=(8, 6))
plt.plot(range(n_classes), roc_auc_scores, marker='o', color='skyblue', linestyle='-', label='ROC-AUC')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('ROC-AUC and Accuracy for Each Class (XGBoost Classifier)')
plt.xticks(range(n_classes), labels=[f'Class {i}' for i in range(n_classes)])
plt.grid(True)
plt.legend(loc='upper left')

class_accuracies = []
class_f1_scores = []

for i in range(n_classes):
    class_accuracy = accuracy_score([1 if label == i else 0 for label in y_test], [1 if pred == i else 0 for pred in y_pred_xgb])
    class_f1 = f1_score([1 if label == i else 0 for label in y_test], [1 if pred == i else 0 for pred in y_pred_xgb])

    class_accuracies.append(class_accuracy)
    class_f1_scores.append(class_f1)
    
# Plot accuracy and F1 score for each class
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Class')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(range(n_classes), class_accuracies, color=color, marker='o', label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('F1 Score', color=color)  
ax2.plot(range(n_classes), class_f1_scores, color=color, marker='s', linestyle='--', label='F1 Score')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.xticks(range(n_classes), labels=[f'Class {i}' for i in range(n_classes)])
plt.show()