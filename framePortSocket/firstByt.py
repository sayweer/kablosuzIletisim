import socket

HOST = "0.0.0.0"
PORT = 53200     # 49152 - 65535 portları arası bizim at kosturma noktamiz.

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #buradaki parametrelerden AF_INET IPv4 kullanacagimizi soyluyor. sock_stream ise TCP kullanacağım dememnin pythone dili :))
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Server çalışıyor, bağlantı bekleniyor...")

conn, addr = server_socket.accept()
print(f"Bağlanan cihaz: {addr}")

data = conn.recv(1024)
print("Gelen mesaj:", data.decode())

conn.close()
server_socket.close()