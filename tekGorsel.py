import socket
import cv2
import time

HOST = "0.0.0.0"
PORT = 53200

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("Kamera server çalışıyor, bağlantı bekleniyor...")

while True:
    conn, addr = server_socket.accept()
    print(f"Bağlanan cihaz: {addr}")

    conn.recv(1024)  

    cap = cv2.VideoCapture(0)
    time.sleep(0.2)  # Kamera uyandı uyansın diye
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Kamera görüntüsü alınamadı")
        conn.close()
        continue

    _, jpeg = cv2.imencode(".jpg", frame)
    image_bytes = jpeg.tobytes()

    response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(image_bytes)}\r\n"
        "\r\n"
    ).encode() + image_bytes

    conn.sendall(response)
    conn.close()
