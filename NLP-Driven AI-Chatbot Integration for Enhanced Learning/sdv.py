import socket
import threading
import websocket
import json
import openai
import mysql.connector

# Your OpenAI API key
api_key = "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"  # Replace with your actual API key

# Database credentials
db_host = "localhost"
db_user = "root"
db_password = "Gokss^^"  # Replace with your actual password
db_database = "dbms"  # Replace with your actual database name

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and the port on which to listen
host = 'localhost'
port = 8888

# Bind the socket to the host and port
s.bind((host, port))

# Listen for incoming connections
s.listen(5)

print(f"Listening on port {port}...")

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
    server = WebsocketServer(port=9001, host='localhost')
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()

threading.Thread(target=start_websocket_server).start()

def generate_mysql_query(api_key, user_input):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant capable of generating MySQL queries."},
                {"role": "user", "content": f"Generate a MySQL query for: '{user_input}'without explanatory line just the sql query"}
            ]
        )
        full_response = response.choices[0].message['content']
        return full_response.strip()
    
    except Exception as e:
        print("Error:", e)
        return None

def execute_query(query, host, user, password, database):
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cursor = connection.cursor()
        cursor.execute(query)
        if query.lower().startswith("select"):
            results = cursor.fetchall()
            return results
        else:
            connection.commit()
            return "Query executed successfully."
    except mysql.connector.Error as e:
        return f"Error: {e}"
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

while True:
    # Accept a connection
    client, addr = s.accept()
    print(f"Got a connection from {addr}")

    # Receive the message
    message = client.recv(1024)
    received_message = message.decode()
    print(f"Received message: {received_message}")

    # Separate user input
    if received_message.startswith("Teachers|"):
        user_input = received_message.replace("Teachers|", "")

        # Generate MySQL query using ChatGPT
        mysql_query = generate_mysql_query(api_key, user_input)
        print(f"MySQL Query: {mysql_query}")

        if mysql_query:
            # Execute the query
            query_result = execute_query(mysql_query, db_host, db_user, db_password, db_database)
            print(f"Query Result: {query_result}")

            # Send the result to HTML via WebSocket
            send_message_to_html(json.dumps({"user_input": user_input, "query_result": query_result}))
        else:
            print("Unable to generate a valid MySQL query.")

    # Close the connection
    client.close()
