from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket

def send_data_through_socket(data, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))  # Connect to the specified port
        s.sendall(data.encode())

class RequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Check the content type and decode JSON only if correct
        content_type = self.headers.get('Content-Type')
        if content_type == 'application/json':
            try:
                data = json.loads(post_data)
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b"Bad JSON payload")
                return
        else:
            self.send_response(400)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"Incorrect Content-Type")
            return

        if self.path == '/login':
            email = data.get('email', 'Not found')
            password = data.get('password', 'Not found')
            send_data_through_socket(f"Login|{email}|{password}", 9999)

        elif self.path == '/register':
            name = data.get('name', 'Not found')
            email = data.get('email', 'Not found')
            password = data.get('password', 'Not found')
            send_data_through_socket(f"Register|{name}|{email}|{password}", 9999)
        
        elif self.path == '/teachers':
            message = data.get('message', 'Not found')
            send_data_through_socket(f"Teachers|{message}", 8888)  # Using a different port for teachers
        
        elif self.path == '/stu':  # Handle the path used by your HTML form
            message = data.get('message', 'Not found')
            send_data_through_socket(f"stu|{message}", 7777)  # Use the same port or another as per your requirement
            print(f"Received message: {message}") 
        
        else:
            print(f"Unknown POST request to {self.path}")

        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
