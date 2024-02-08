import socket
import threading
from websocket_server import WebsocketServer
import openai
import mysql.connector
import json
import websocket

    # Your OpenAI API key
api_key = "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"  # Replace with your actual API key

# Database credentials
db_host = "localhost"
db_user = "root"
db_password = "Gokss^^"  # Replace with your actual password
db_database = "dbms"  # Replace with your actual database name

# Global WebSocket server
global websocket_server

def generate_mysql_query(user_input, api_key=api_key):
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(    
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant capable of generating MySQL queries."},
                {"role": "user", "content": f"Generate a MySQL query for: '{user_input}' without explanatory line just the sql query"},
            ]
        )
        full_response = response.choices[0].message['content']
        return full_response.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        print("Error:", e)
        return None


# Function to execute MySQL query
def execute_query(query):
    try:
        connection = mysql.connector.connect(host=db_host, user=db_user, password=db_password, database=db_database)
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

def send_message_over_websocket(message, address="ws://127.0.0.1:9001"):
    try:
        ws = websocket.create_connection(address)
        ws.send(message)
        ws.close()
    except Exception as e:
        print(f"WebSocket Error: {e}")

def new_client(client, server):
    print(f"New WebSocket client connected and was given id {client['id']}")

def client_left(client, server):
    print(f"WebSocket Client({client['id']}) disconnected")

def message_received(client, server, message):
    print(f"Message Received on WebSocket: {message}")
    # Sending 'Hi' to all connected clients
    server.send_message_to_all("Hi")

# Start the WebSocket server in a separate thread
def start_websocket_server():
    global websocket_server
    websocket_server = WebsocketServer(port=9001, host='127.0.0.1')
    websocket_server.set_fn_new_client(new_client)
    websocket_server.set_fn_client_left(client_left)
    websocket_server.set_fn_message_received(message_received)
    websocket_server.run_forever()

threading.Thread(target=start_websocket_server).start()

# Main server function
def start_server(port=8888):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', port))
    server.listen(5)
    print(f"Server listening on port {port}")

    try:
        while True:
            client_socket, addr = server.accept()
            print(f"Connection from {addr}")

            data = client_socket.recv(1024)
            message = data.decode('utf-8')
            print(f"Received data: {message}")

            mysql_query = generate_mysql_query(message)
            print(f"MySQL Query: {mysql_query}")
            query_result = execute_query(mysql_query)
            print(f"Query Result: {query_result}")
           
            send_message_over_websocket(message)

            client_socket.close()

    except KeyboardInterrupt:
        print("\nServer shutting down.")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()