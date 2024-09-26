import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import numpy as np

data = pd.read_csv(r'Drug prescription Dataset.csv')

label_encoders = {}
for column in ['disease', 'gender', 'severity', 'drug']:  
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

print(data.head())

X = data[['disease', 'age', 'gender', 'severity']]
y = data['drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_drug(disease, age, gender, severity):
    inputs = [
        label_encoders['disease'].transform([disease])[0],
        age,
        label_encoders['gender'].transform([gender])[0],
        severity  
    ]

    prediction = model.predict([inputs])
    return label_encoders['drug'].inverse_transform(prediction)[0] 
a = input("Press 'Y' to find the drug required: ")
if a == "Y":
    x = input("Enter the disease: ")
    y = int(input("Enter your age: "))
    z = input("Enter your gender: ")
    p = int(input("Enter the severity (0: Low, 1: Normal, 2: High): "))
    predicted_drug = predict_drug(x, y, z, p)
    print(f"Predicted Drug: {predicted_drug}")



data = pd.read_csv(r'Drug prescription Dataset.csv')

label_encoders = {}
for column in ['disease', 'gender', 'severity', 'drug']: 
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['disease', 'age', 'gender', 'severity']]
y = data['drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

training_accuracy = []

testing_accuracy = []

for epoch in range(1, 501):  
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    training_accuracy.append(train_acc)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    testing_accuracy.append(test_acc)

plt.plot(range(1, 501), training_accuracy, label='Training Accuracy', color='blue')
plt.plot(range(1, 501), testing_accuracy, label='Testing Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.legend()
plt.show()
