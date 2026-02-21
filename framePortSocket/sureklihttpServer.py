import socket  # İşletim sisteminin ağ (network) özelliklerini kullanabilmek için socket modülü

HOST = "0.0.0.0"  # Bu makinedeki tüm ağ arayüzlerinden (Wi-Fi, Ethernet vs.) bağlantı kabul et
PORT = 53200      # Server'ın dinleyeceği port (49152–65535 arası güvenli alan)

# IPv4 (AF_INET) ve TCP (SOCK_STREAM) kullanan bir socket oluştur
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bu socket'i belirtilen IP ve port'a bağla (bu port artık bu programa ait)
server_socket.bind((HOST, PORT))

# Socket'i dinleme moduna al
# 5 = aynı anda kapı önünde bekleyebilecek maksimum bağlantı sayısı
server_socket.listen(5)

print("Server çalışıyor. Sürekli bağlantı bekliyor...")

# Server kapanmasın diye sonsuz döngü
while True:
    # Bir client bağlanana kadar burada bekler
    # Bağlantı gelince:
    # conn -> client ile birebir haberleşme socket'i
    # addr -> client'ın IP ve port bilgisi
    conn, addr = server_socket.accept()
    print(f"Bağlanan cihaz: {addr}")

    # Client'tan gelen veriyi al (en fazla 1024 byte)
    request = conn.recv(1024)
    print("İstek alındı")

    # HTTP formatında bir cevap hazırla
    response = """\
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8

Server ayakta 💪
Sayfa her yenilendiğinde ben buradayım.
"""

    # Cevabı byte'a çevirip client'a gönder
    conn.sendall(response.encode())

    # Bu client ile işimiz bitti, bağlantıyı kapat
    conn.close()
