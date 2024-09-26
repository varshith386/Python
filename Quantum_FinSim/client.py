# sender.py

import socket

def start_sender(host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))  # Bind to the sender port
        s.listen()
        print(f'Sender listening on {host}:{port}...')
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                crypto_name = conn.recv(1024).decode()
                if not crypto_name:
                    break
                predicted_value = send_to_receiver(crypto_name)
                conn.sendall(predicted_value.encode())

def send_to_receiver(crypto_name):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65433))  # Connect to the receiver
        s.sendall(crypto_name.encode())
        predicted_value = s.recv(1024).decode()
        return predicted_value

if __name__ == "__main__":
    start_sender()
