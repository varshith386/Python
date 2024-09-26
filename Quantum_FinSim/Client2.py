# receiver.py

import socket
from try6 import predict_price

def receiver():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 65433))  # Port for receiver
        s.listen()
        print('Receiver listening on port 65433...')
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                crypto_name = data.decode()
                predicted_value = predict_price(crypto_name)
                conn.sendall(str(predicted_value).encode())

if __name__ == "__main__":
    receiver()
