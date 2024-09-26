import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'Drug prescription Dataset.csv')

label_encoders = {}
for column in ['disease', 'gender', 'severity']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['disease', 'age', 'gender', 'severity']]
y = data['drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
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
    return prediction[0]

if input("Press 'Y' to find the drug required: ") == "Y":
    x = input("Enter the disease: ")
    y = int(input("Enter your age: "))
    z = input("Enter your gender: ")
    p = int(input("Enter the severity: "))
    predicted_drug = predict_drug(x, y, z, p)
    print(f"Predicted Drug: {predicted_drug}")
