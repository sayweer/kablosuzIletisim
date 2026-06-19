import socket

HOST = "0.0.0.0"
PORT = 53200

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("HTTP Server çalışıyor, bağlantı bekleniyor...")

conn, addr = server_socket.accept()
print(f"Bağlanan cihaz: {addr}")

request = conn.recv(1024)
print("Gelen istek:\n", request.decode(errors="ignore"))

response = """\
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8

Merhaba telefon 👋
Server seni gördü.
"""
conn.sendall(response.encode())

conn.close()
server_socket.close()