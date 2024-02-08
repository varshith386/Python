import socket
import threading
import websocket
import json
import openai
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import tensorflow as tf
import csv
import mysql.connector
import random
import os

import time


# Your OpenAI API key
api_key = "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"  # Replace with your actual API key

# Create a socket object for user input
s_user_input = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and the port on which to listen for user input
user_input_host = 'localhost'
user_input_port = 7777

# Bind the socket for user input to the host and port
s_user_input.bind((user_input_host, user_input_port))

# Listen for incoming user input connections
s_user_input.listen(5)

print(f"Listening for user input on port {user_input_port}...")

# Create a WebSocket server for communication with HTML
from websocket_server import WebsocketServer

def send_message_to_html(message):
    server.send_message_to_all(message)

def new_client(client, server):
    print(f"New WebSocket client connected and was given id {client['id']}")

def client_left(client, server):
    print(f"WebSocket Client({client['id']}) disconnected")

def message_received(client, server, message):
    print(f"Message Received on WebSocket: {message}")

# Start the WebSocket server in a separate thread
def start_websocket_server():
    global server
    server = WebsocketServer(port=9002, host='localhost')
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()

threading.Thread(target=start_websocket_server).start()

# Function to check if the user's recent interactions contain keywords related to the day
def is_day_related(input_messages):
    keywords = ["day", "today", "life"]
    input_text = "".join([message["content"] for message in input_messages]).lower()
    return any(keyword in input_text for keyword in keywords)

# Function to generate a chatbot reply using OpenAI's GPT-3.5 Turbo
def generate_chatbot_reply(api_key, user_input, conversation_history):
    openai.api_key = api_key
    response = ""
    
    # Analyze the user responses for relevant questions
    if is_day_related(conversation_history[-3:]):
        response = "How was your day?"
    else:
        # Strip "stu|" from the user input
        user_input_stripped = user_input.replace("stu|", "")
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_input_stripped}],
        ).choices[0].message['content']
    
    return response

# Function to perform sentiment analysis and scoring calculation
def analyze_sentiment_and_scores(tweet):
    # Load RoBERTa model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = TFAutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    encoded_tweet = tokenizer(tweet, return_tensors='tf')
    output = model(encoded_tweet)

    scores = output[0][0].numpy()
    scores = softmax(scores)

    max_score, max_index = max((score, index) for index, score in enumerate(scores))

    # Calculate the sentiment index based on max_index
    sentiment_index = max_index

    return sentiment_index, max_score, max_index

# Maintain a history of user responses
conversation_history = []
interaction_count = 0  # Counter for interactions

key = "g" # Inside the main loop
while key == "g":
    # Accept a connection for user input
    client_input, addr_input = s_user_input.accept()
    print(f"Got a connection from {addr_input}")

    # Receive the user input
    message_input = client_input.recv(1024)
    received_message_input = message_input.decode()

    # Strip "stu|" from the user input
    received_message_stripped = received_message_input.replace("stu|", "")
    print(f"Received user input: {received_message_stripped}")

    # Add the user's input to the conversation history
    conversation_history.append({"role": "user", "content": received_message_stripped})

    # Generate a chatbot reply using GPT-3.5 Turbo and check for a relevant question
    chatbot_reply = generate_chatbot_reply(api_key, received_message_input, conversation_history)
    print(f"Chatbot reply: {chatbot_reply}")

    # Send the chatbot reply to HTML via WebSocket
    send_message_to_html(json.dumps(chatbot_reply))

    interaction_count += 1

    if interaction_count == 3:  
        # Ask about the user's day
        send_message_to_html(json.dumps({"chatbot_reply": "How was your day?"}))

    if interaction_count == 4:   
        # Perform sentiment analysis and scoring calculation using user_day_response as "tweet"
        sentiment_index, max_score, max_index = analyze_sentiment_and_scores(received_message_input)

        # Prepare data to send to HTML
        data_to_send = {
            "user_input": received_message_input,
            "sentiment": str(sentiment_index),
            "max_score": str(max_score),
            "max_index": str(max_index),
        }
        
        key ="f"

    # Close the connection for user input
    client_input.close()

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
    select_random_query = "SELECT mathscore, readingscore, writingscore FROM students ORDER BY RAND() LIMIT 1"
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
            q="Angry and Low Score"+random.choice(advice_dict['0'])
            print(q)
            send_message_to_html(json.dumps(q))
        else:
            q="Angry and High Score"+random.choice(advice_dict['0'])
            print(q)
            send_message_to_html(json.dumps(q))
    elif max_index == 1:
        if sum(int(score) for score in random_scores) < 100:
            q="Neutral and Low Score", random.choice(advice_dict['1'])
            print(q)
            send_message_to_html(json.dumps(q))
        else:
            q="Neutral and High Score"+random.choice(advice_dict['1'])
            print(q)
            send_message_to_html(json.dumps(q))
    elif max_index == 2:
        if sum(int(score) for score in random_scores) < 100:
            q="Happy and Low Score"+random.choice(advice_dict['2'])
            print(q)
            send_message_to_html(json.dumps(q))
        else:
           q="Happy and High Score"+random.choice(advice_dict['2'])
           print(q)
           send_message_to_html(json.dumps(q))
           
           data_to_send = {
                "scores": [int(random_scores[0]), int(random_scores[1]), int(random_scores[2])],  # Replace with actual scores
                "advice": q,
                }
except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Close connections
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed.")
        
os.system("start C:\\Users\\admin\\Desktop\\dbms\\website\\students\\asd.html")

def start_websocket_server():
    global server
    server = WebsocketServer(port=9004, host='localhost')
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()

threading.Thread(target=start_websocket_server).start()

while True:
    send_message_to_html(json.dumps(data_to_send))
    time.sleep(1) 


