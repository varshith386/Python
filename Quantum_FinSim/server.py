import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class MyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/check_price':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            crypto_name = json.loads(post_data.decode())['crypto_name']
            send_to_sender(crypto_name)
            predicted_value = receive_from_sender()
            response = {'predicted_value': predicted_value}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

def send_to_sender(crypto_name):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65432))  # Port for sender
        s.sendall(crypto_name.encode())

def receive_from_sender():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65432))  # Port for sender
        data = s.recv(1024)
    return data.decode()

def run(server_class=HTTPServer, handler_class=MyHandler, port=1000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
