# import socket

# def listen_socket(port):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(('localhost', port))
#         s.listen()
#         print(f"Listening on port {port}...")

#         while True:  # Run forever
#             print("Waiting for a connection...")
#             conn, addr = s.accept()
#             with conn:
#                 print(f"Connected by {addr}")
#                 data = conn.recv(1024).decode()
#                 print(f"Received data: {data}")
#                 # Add any additional processing of 'data' here

# if __name__ == "__main__":
#     listen_socket(9999)

import socket
import mysql.connector
import os

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",  # Update as needed
        user="root",  # Replace with your MySQL username
        password="Gokss^^",  # Replace with your MySQL password
        database="dbms"
    )

def insert_into_db(name, email, password):
    db = connect_to_db()
    cursor = db.cursor()
    query = "INSERT INTO pwd (name, email, password) VALUES (%s, %s, %s)"
    cursor.execute(query, (name, email, password))
    db.commit()
    cursor.close()
    db.close()

def check_credentials(email, password):
    db = connect_to_db()
    cursor = db.cursor()
    query = "SELECT password FROM pwd WHERE email = %s"
    cursor.execute(query, (email,))
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result and result[0] == password

def open_html_file():
    # Update the path to your HTML file
    os.system("start C:\\Users\\admin\\Desktop\\dbms\\website\\teacher\\Untitled-2.html")
def listen_socket(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))
        s.listen()
        print(f"Listening on port {port}...")

        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024).decode()
                data_parts = data.split("|")

                if len(data_parts) == 4 and data_parts[0] == "Register":
                    _, name, email, password = data_parts
                    insert_into_db(name, email, password)
                    print("Registration data inserted into database.")

                elif len(data_parts) == 3 and data_parts[0] == "Login":
                    _, email, password = data_parts
                    if check_credentials(email, password):
                        print("Login successful.")
                        open_html_file()
                    else:
                        print("Invalid login credentials.")

if __name__ == "__main__":
    listen_socket(9999)
