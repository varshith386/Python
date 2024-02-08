import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
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


d2 = pd.DataFrame(X_train_resampled.toarray(), columns=vectorizer.get_feature_names_out())
d2['label'] = y_train_resampled


forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
forest_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_forest = forest_classifier.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred_forest))
print("\nClassification Report:\n", classification_report(y_test, y_pred_forest))

label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
d2['emotion'] = d2['label'].map(label_mapping)

count = d2['emotion'].value_counts()

count.plot(kind='bar')
plt.xlabel('Emotion')
plt.ylabel('Number of Instances')
plt.xticks(rotation=0)
plt.show()





texts = d2['text'].tolist()
labels = d2['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_distribution = pd.Series(y_train).value_counts()
train_distribution.plot(kind='bar')


plt.xlabel('Emotion')
plt.ylabel('Number of Instances')
plt.xticks(rotation=0)
plt.title('Distribution of Emotions in Training Data After Train-Test Split')
plt.show()