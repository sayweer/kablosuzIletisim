import socket
import time  # Saat bilgisi almak için

HOST = "0.0.0.0"
PORT = 53200

request_count = 0  # Server boyunca kaç istek geldiğini tutan sayaç

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("Server çalışıyor. Sürekli bağlantı bekliyor...")

while True:
    conn, addr = server_socket.accept()
    request_count += 1  # Her bağlantıda sayacı artır

    current_time = time.strftime("%H:%M:%S")  # Anlık saat

    print(f"{request_count}. istek alındı - {addr}")

    request = conn.recv(1024)

    response = f"""\
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8

Server ayakta 💪

Gelen istek sayısı: {request_count}
Sunucu saati: {current_time}
"""
    conn.sendall(response.encode())
    conn.close()
