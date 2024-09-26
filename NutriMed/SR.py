import speech_recognition as sr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def transcribe_audio(audio_file):

    recognizer = sr.Recognizer()


    with sr.AudioFile(audio_file) as source:

        audio_data = recognizer.record(source)

        try:

            transcribed_text = recognizer.recognize_google(audio_data)
            print("Transcribed text:", transcribed_text)
            return transcribed_text.split() 
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return []
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return []

def convert_word_to_number(word):
    word_to_number = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }
    return word_to_number.get(word, word)

audio_file = r"C:\Users\hp\Downloads\WhatsApp Ptt 2024-05-19 at 16.30.54.wav"
word_list = transcribe_audio(audio_file)

if word_list:
    print("Parsed Sentence:", word_list)


indices = [5, 11, 16, 21]
extracted_words = [word_list[i] for i in indices]
extracted_words = [convert_word_to_number(word) for word in extracted_words]
word5, word11, word16, word21 = extracted_words
data = pd.read_csv(r'Drug prescription Dataset.csv')


label_encoders = {}
for column in ['disease', 'gender', 'severity']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


print(data.head())

X = data[['disease', 'age', 'gender', 'severity']]  
y = data['drug']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
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


a= input("Press 'Y' to find the drug requried ")
if(a=="Y"):
    x=word5
    y=word11
    z=word16
    p=word21
    predicted_drug = predict_drug(x,y,z,p)
    print(f"Predicted Drug: {predicted_drug}")

