import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, auc, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np
from imblearn.over_sampling import SMOTE


df = pd.read_csv("file:///C:/Users/hp/OneDrive/Desktop/Pyhton/test.csv") 
texts = df['text'].tolist()
labels = df['label'].tolist()


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


svm_classifier = SVC(kernel='linear', C=1.0, probability=True)


svm_classifier.fit(X_train_resampled, y_train_resampled)


y_pred = svm_classifier.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]


y_scores = svm_classifier.predict_proba(X_test_tfidf)


roc_auc_scores = []
for i in range(n_classes):
    roc_auc = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
    roc_auc_scores.append(roc_auc)


plt.figure(figsize=(8, 6))
plt.plot(range(n_classes), roc_auc_scores, marker='o', color='skyblue', linestyle='-', label='ROC-AUC')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('ROC-AUC and Accuracy for Each Class (SVM)')
plt.xticks(range(n_classes), labels=[f'Class {i}' for i in range(n_classes)])
plt.grid(True)
plt.legend(loc='upper left')

class_accuracies = []
class_f1_scores = []

for i in range(n_classes):
    class_accuracy = accuracy_score([1 if label == i else 0 for label in y_test], [1 if pred == i else 0 for pred in y_pred])
    class_accuracies.append(class_accuracy)

    class_f1 = f1_score([1 if label == i else 0 for label in y_test], [1 if pred == i else 0 for pred in y_pred])
    class_f1_scores.append(class_f1)


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