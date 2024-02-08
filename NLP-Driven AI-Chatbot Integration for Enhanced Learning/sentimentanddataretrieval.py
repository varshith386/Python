from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import tensorflow as tf
import csv
import mysql.connector
import random

tweet = 'im going to kill varshith'

tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = TFAutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='tf')  # Use 'tf' to get TensorFlow tensors
output = model(encoded_tweet)

scores = output[0][0].numpy()
scores = softmax(scores)

# Find the maximum score and its index
max_score, max_index = max((score, index) for index, score in enumerate(scores))

print(f"The maximum score is {max_score} at index {max_index}.")

# MySQL database configuration
db_config = {
    'user': 'root',
    'password': 'Gokss^^',
    'host': 'localhost',
    'database': 'dbms'
}

# CSV file path
csv_file_path = r'C:\Users\admin\Downloads\archive\StudentsPerformance.csv'

try:
    # Connect to MySQL database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Open and read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip header row
        header = next(csv_reader)

        # Create table (if not exists)
        table_name = 'students'
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in header])})"
        cursor.execute(create_table_query)

        # Insert data into MySQL table
        insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['%s' for _ in header])})"
        for row in csv_reader:
            cursor.execute(insert_query, row)

    # Commit changes
    conn.commit()
    print("Data imported successfully!")

    # Select random scores
    select_random_query = "SELECT mathscore, readingscore, writingcore FROM students ORDER BY RAND() LIMIT 1"
    cursor.execute(select_random_query)
    random_scores = cursor.fetchone()

    # Display the randomly selected scores
    print("Randomly selected scores:")
    print(f"Math Score: {random_scores[0]}")
    print(f"Reading Score: {random_scores[1]}")
    print(f"Writing Score: {random_scores[2]}")

    advice_dict = {
        '0': [
            "Take a deep breath and count to ten.",
            "Reflect on positive aspects of the situation.",
            "Consider talking to a friend or a therapist.",
            "Engage in physical activity to release tension.",
        ],
        '1': [
            "Practice mindfulness and stay in the present moment.",
            "Find activities that bring you a sense of calm.",
            "Maintain a balanced perspective on situations.",
            "Consider journaling to express your thoughts.",
        ],
        '2': [
            "Celebrate small victories and achievements.",
            "Surround yourself with positive influences.",
            "Engage in activities that bring you joy.",
            "Express gratitude for the positive aspects of your life.",
        ]
    }

    # Analyze sentiment and scores
    if max_index == 0:
        if sum(int(score) for score in random_scores) < 100:
            print("Angry and Low Score", random.choice(advice_dict['0']))
        else:
            print("Angry and High Score", random.choice(advice_dict['0']))
    elif max_index == 1:
        if sum(int(score) for score in random_scores) < 100:
            print("Neutral and Low Score", random.choice(advice_dict['1']))
        else:
            print("Neutral and High Score", random.choice(advice_dict['1']))
    elif max_index == 2:
        if sum(int(score) for score in random_scores) < 100:
            print("Happy and Low Score", random.choice(advice_dict['2']))
        else:
            print("Happy and High Score", random.choice(advice_dict['2']))

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Close connections
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed.")
