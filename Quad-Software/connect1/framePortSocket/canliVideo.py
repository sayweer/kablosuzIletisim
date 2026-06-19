import socket
import cv2
import time

HOST = "0.0.0.0"
PORT = 53200

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("MJPEG kamera server çalışıyor...")

# Kamerayı BİR KEZ aç
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    conn, addr = server_socket.accept()
    print(f"Bağlanan cihaz: {addr}")

    conn.recv(1024)

    # MJPEG başlıkları
    headers = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "\r\n"
    )
    conn.sendall(headers.encode())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # JPEG'e çevir
            _, jpeg = cv2.imencode(".jpg", frame)
            image_bytes = jpeg.tobytes()

            # Her kare için parça (boundary)
            part = (
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(image_bytes)}\r\n"
                "\r\n"
            ).encode() + image_bytes + b"\r\n"

            conn.sendall(part)
            time.sleep(0.05)  # 20 FPS falan 
    except Exception as e:
        print("Bağlantı koptu")
    finally:
        conn.close()
