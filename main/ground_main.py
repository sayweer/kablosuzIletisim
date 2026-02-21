#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###255'ten 0'a dönerken yaşanabilecek saliselik bir atlama ihtimali için bulanık eşleştirme algoritması hazırlayacağız sonraki prototiplerde!!!!!)


#os = işletim sistemiyle konuşmak için(dosya yolları falan filan)
#cv2 = openCV ,görüntü okuma ,çizim yapma ,ekrana basma
#time = FPS hesaplamak için
#json = ağdan gelen metni pythonun anlayacağı dile çevirmek için.
#socket = UDP ve Mavlink gibi ağ bağlantı yollarını açmak için.
#threading = işleri asenkron yapabilmek için 
import os, cv2, time, json, socket, threading
import numpy as np #matris ve matematik kütüphanesi. np adıyla kısaltıyoruz. numpy nin mean fonksiyonunu 8x8 piksel meselesinde kullandık
from pymavlink import mavutil #dronun beyniyle (pixhawk/ardupilot) standart havacılık dili yani MAVlinkle konuşmak için.
from collections import deque # Bufferlama için gerekli /çift yönlü kuyruk. baştan ve sondan veri atılabilir.

# --- ağ AYARLARI ---
SRT_LISTEN_PORT = 9000 #SRT protokolünün yer istasyonundaki kapı numarası. 9000 standartmış.
SRT_LATENCY_MS = 120  # SRT nin paket kaybına karşı ne kadar süre bekleyeceğidir. Wi-Fi ortamında biraz artırmak titremeyi azaltır

MAVLINK_CONN_STR = "udp:0.0.0.0:14550" # demek "Bana gelen tüm ağ kartlarını dinle" demektir. 14550 standart UDP MAVLink portudur.
MAVLINK_BAUD = 57600 #Bağlantı hızı... UDP üzerinden çok önemli olmasa da standart Ardupilot hızı 57600 olarak girilir.
MAVLINK_SOURCE_SYS = 255 #yer istasyonunun mavlinkdeki kimlik numarası
MAVLINK_SOURCE_COMP = 190

META_LISTEN_IP = "0.0.0.0" #UDP üzerinden gelen JSON (Kutu) verilerini tüm IP adreslerinden dinlemek için 0.0.0.0
META_LISTEN_PORT = 5005  #görüntü işleme verisinin ineceği kapı. Videodan bağımsız olması için 5005 seçildi.
META_BUFFER_LEN = 120    # Geçmiş metadataları tutacak kuyruk uzunluğu

SHOW_WINDOW = True #Görüntüyü ekrana basıp basmayacağımızı belirten anahtar.
WINDOW_NAME = "Ground Station - NVIDIA Accelerated"  #Açılacak OpenCV penceresinin sol üstünde yazacak başlık.

# --- GLOBAL DEĞIŞKENLER ---
meta_lock = threading.Lock() #Thread kilidi. UDP dinleyicisi listeye veri yazarken, ana kod oradan okumaya kalkışıp çökmesin diye konulan polis.

# Yapı: [(ts_recv, payload), (ts_recv, payload), ...]
meta_queue = deque(maxlen=META_BUFFER_LEN) # Gelen ham metadataları tutan kuyruk. deque yapısı sayesinde maxlen=50'yi geçince 51. veri gelirse en eski olan 1. veriyi kendiliğinden siler.

def opencv_has_gstreamer(): #Bilgisayarına kurduğun OpenCV kütüphanesinin GStreamer (donanım hızlandırma) desteğiyle derlenip derlenmediğini kontrol eden fonksiyon.
    try:
        bi = cv2.getBuildInformation() #OpenCV'nin nasıl kurulduğunun devasa bir metin raporunu verir.
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi) #Bu raporun içinde "GStreamer: YES" kelimesi geçiyorsa True (Var) döner, yoksa False döner.
    except Exception:
        return False #kodun herhangi bir yerinde hata çıkarsa default olarak GStreamer yokmuş gibi davranacak.

def build_receiver_pipeline_nvidia(): #SRT üzerinden gelen videoyu ekran kartı (GPU) ile çözecek GStreamer boru hattını string (metin) olarak oluşturur. ekran kartı olmayan bilgisayarda denersek büyük sıkıntı
    uri = "srt://:{}?mode=listener&latency={}&transtype=live".format(SRT_LISTEN_PORT, SRT_LATENCY_MS) # uri: SRT dinleyici adresi. mode=listener (PC bağlantıyı bekler). latency=120ms. transtype=live (canlı yayın profili, gecikmeyi düşük tutar).
    appsink = "appsink drop=true max-buffers=1 sync=false" #appsink = OpenCV'nin videoyu GStreamer'dan teslim aldığı son nokta. drop=true (bilgisayar kasarsa eski kareleri çöpe at), max-buffers=1 (sadece en son gelen 1 kareyi RAM'de tut), sync=false (Videonun zaman damgasını bekleme, gelir gelmez ekrana bas).
    
    return (# pipeline birleştirme kısmı
        "srtsrc uri=\"{uri}\" ! queue ! "  #SRT URI'sini dinle. queue: İşlemci yorulursa veriyi silmek yerine GStreamer'ın iç tamponunda beklet.
        "h264parse ! nvh264dec ! "      #h264parse: Gelen düz byte akışını anlamlı H264 bloklarına (NAL units) ayırır. # nvh264dec: NVIDIA ekran kartının özel şifre çözücüsü. İşlemciyi (CPU) kullanmadan donanımsal olarak H264'ü çözer.
        "videoconvert ! video/x-raw,format=BGR ! " # videoconvert: Ekran kartından çıkan ham formatı, OpenCV'nin sevdiği Mavi-Yeşil-Kırmızı (BGR) formatına çevirir.
        "{appsink}" # En son appsink değişkenini ekler.
    ).format(uri=uri, appsink=appsink)
#################################################################################################
# [YENİ] Barkodu okuyup sayıyı bulan fonksiyon  ------ kutuyu yerde çizmek için 2 yeni 1 de değiştirilmiş fonksiyon-------
# [DEĞİŞTİRİLDİ] H264 bozulmalarına karşı merkez ortalaması alan yeni okuyucu eskisinde tek piksele bakıyorduk baya bi sorunluydu.
def read_barcode(frame): #Görüntünün sol üst köşesindeki barkod tarzı yapımızı okuyup 0-255 arası tam sayıyı (Sıra Numarasını) bulur.
    binary_str = "" #01 leri yan yana koyup okuyacağımız değişken
    for i in range(8):
        # 20x20'lik kutunun sadece tam merkezindeki 8x8'lik güvenli alanı alıyoruz
        # Y ekseni: 6 ile 14 arası, X ekseni: (i*20)+6 ile (i*20)+14 arası
        core_region = frame[6:14, (i*20)+6 : (i*20)+14]
        
        # Bu 64 pikselin parlaklık ortalamasını alıyoruz (np.mean hayat kurtarır)
        mean_brightness = np.mean(core_region) #seçilen 64 adet (8x8) pikselin tüm Mavi, Yeşil, Kırmızı değerlerini toplar ve ortalamasını tek bir sayı olarak verir.
        
        # Ortalama 127'den büyükse Beyaz, değilse Siyahtır
        if mean_brightness > 127:
            binary_str += "1"  
        else:
            binary_str += "0"  
            
    # '01010010' gibi bir metni onluk tabanda tam sayıya (Örn: 82) çevir
    return int(binary_str, 2)

# [YENİ] Barkodu ekrandan gizleyen (siyah bant çeken) fonksiyon
def mask_barcode(frame):
    # Sol üst köşedeki (0,0) noktasından (160,20) noktasına kadar siyah kutu çiz
    cv2.rectangle(frame, (0, 0), (160, 20), (0, 0, 0), -1) #resim(frame), sol üst köşe(0,0), sağ alt köşe(X:160, Y:20), Renk BGR(0,0,0 Siyah), Kalınlık(-1 yani içini tamamen doldur).
    return frame

# [YENİ ve DEĞİŞTİRİLDİ] Artık zamana göre değil, SIRA NUMARASINA göre arıyoruz
def get_synced_metadata_by_seq(target_seq): #Okuduğumuz barkod numarasını (Örn: 82) alır, UDP kuyruğundaki 120 veriyi tarayıp 82 olanı bulur.
    best_payload = None #boş atadık
    with meta_lock:
        if not meta_queue: #Eğer kuyruk tamamen boşsa (henüz UDP gelmediyse) hiç uğraşmadan direkt boş (None) döneriz.
            return None
        
        # Kuyruktaki tüm verileri dön, barkod numarası eşleşeni bul!
        for ts, payload in eta_queue:
            if payload.get("seq") == target_seq:
                best_payload = payload
                break  # Bulduğumuz an döngüden çık
                
    return best_payload
######################################################################################################

def draw_dets(frame, dets): #Gelen JSON içindeki "detections" (hedefler) listesini alıp ekrana yeşil kutuları çizen fonksiyon.
    for d in dets:
        x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
        x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
        cls = int(d.get("cls", 0)); conf = float(d.get("conf", 0.0))
     
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2) #Hedefin etrafına (x1,y1)'den (x2,y2)'ye kadar, yeşil renkte (0,255,0), 2 piksel kalınlığında bir kutu çizer.
        label = "ID:{} %{:.0f}".format(cls, conf*100) #Ekranın üzerine yazılacak "ID:0 %85" gibi formatlı metin. conf*100 ile ondalık skoru yüzdeye çeviriyoruz.
        
        # Etiket arka planı (okunabilirlik için)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] #yazacağımız yazının piksel olarak genişliğini ve yüksekliğini hesaplar ki arkasına tam o boyutta bir yeşil bant koyalım (t_size içine atar). FONT_HERSHEY_SIMPLEX tipini kullanır.
        cv2.rectangle(frame, (x1, y1-20), (x1+t_size[0], y1), (0,255,0), -1) #Yazının altına, okunaklı olsun diye (x1, y1-20)'den başlayan, yazı genişliği kadar uzanan yeşil renkli, -1 kalınlığında (dolu) bir arkaplan çizer.
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1) #yeşil dolgunun üzerine, siyah renkte (0,0,0), 1 kalınlığında yazımızı yazar.
    return frame

class MetaReceiver(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self, daemon=True)
        self.ip = ip; self.port = port
        self.stop_event = threading.Event()
        self.sock = None

    def stop(self):
        self.stop_event.set()
        if self.sock: self.sock.close()

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # [DÜZELTME]: PC tarafında da paket kaybını önlemek için buffer 1MB yapıldı.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        print("[META] Dinleniyor {}:{}".format(self.ip, self.port))
        
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
                # Senin kontrol ettiğin kısım: Try-Except burada aktif.
                payload = json.loads(data.decode("utf-8"))
                
                # SADECE ALIP KAYDETMEK YERİNE KUYRUĞA EKLİYORUZ
                recv_time = time.time()
                with meta_lock:
                    meta_queue.append((recv_time, payload))
                    
            except socket.timeout:
                pass
            except Exception as e:
                # JSON hatası olursa sessizce geç
                pass

class MavlinkReceiver(threading.Thread):
    def __init__(self, conn_str, baud):
        threading.Thread.__init__(self, daemon=True)
        self.conn_str = conn_str; self.baud = baud
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.data = {"mode":"N/A","armed":False,"lat":0,"lon":0,"alt_m":0,"battery_v":0}
        self.m_conn = None

    def stop(self):
        self.stop_event.set()

    def request_data_stream(self):
        """3 Saniyede bir (yaklaşık 0.33 Hz) veri isteği gönderir"""
        if self.m_conn is None: return
        
        # MAVLink'te interval mikrosaniye cinsindendir. 3 sn = 3,000,000 us
        interval_us = 3000000 
        
        # GLOBAL_POSITION_INT iste
        self.m_conn.mav.command_long_send(
            self.m_conn.target_system, self.m_conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, interval_us, 0, 0, 0, 0, 0)
        
        # HEARTBEAT zaten otomatiktir ama SYS_STATUS (pil için) isteyebiliriz
        self.m_conn.mav.command_long_send(
            self.m_conn.target_system, self.m_conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, interval_us, 0, 0, 0, 0, 0)

    def run(self):
        last_req_time = 0
        
        while not self.stop_event.is_set():
            try:
                print("[MAV] Baglaniyor:", self.conn_str)
                self.m_conn = mavutil.mavlink_connection(self.conn_str, baud=self.baud,
                                               source_system=MAVLINK_SOURCE_SYS,
                                               source_component=MAVLINK_SOURCE_COMP,
                                               autoreconnect=True)
                self.m_conn.wait_heartbeat(timeout=10)
                print("[MAV] Baglandi.")
            except Exception:
                time.sleep(1.0)
                continue

            while not self.stop_event.is_set():
                # Periyodik veri isteği (3 saniyede bir)
                now = time.time()
                if now - last_req_time > 3.0:
                    self.request_data_stream()
                    last_req_time = now

                try:
                    msg = self.m_conn.recv_match(blocking=True, timeout=1.0)
                    if msg is None: continue
                    t = msg.get_type()
                    with self.lock:
                        if t == "HEARTBEAT":
                            self.data["mode"] = mavutil.mode_string_v10(msg)
                            self.data["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        elif t == "GLOBAL_POSITION_INT":
                            self.data["lat"] = msg.lat / 1e7
                            self.data["lon"] = msg.lon / 1e7
                            self.data["alt_m"] = msg.relative_alt / 1000.0
                        elif t == "SYS_STATUS":
                            if msg.voltage_battery != 65535:
                                self.data["battery_v"] = msg.voltage_battery / 1000.0
                except Exception:
                    break

def draw_overlay(frame, tel, fps, meta_payload, sync_diff):
    # Sol Üst: Sistem Bilgisi
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, "NVIDIA GPU: ON", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # Sağ Üst: Telemetri
    h, w = frame.shape[:2]
    cv2.putText(frame, "MOD: {}".format(tel["mode"]), (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    cv2.putText(frame, "ARMED: {}".format(tel["armed"]), (w-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if not tel["armed"] else (0,255,0), 2)
    cv2.putText(frame, "BAT: {:.1f}V".format(tel["battery_v"]), (w-200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, "ALT: {:.1f}m".format(tel["alt_m"]), (w-200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Alt: Yapay Zeka Durumu
    det_count = 0
    infer_ms = 0
    if meta_payload:
        det_count = len(meta_payload.get("detections", []))
        infer_ms = float(meta_payload.get("infer_ms", 0))

    cv2.putText(frame, "AI: {} Obj | Infer: {:.1f}ms | SeQ Numarası: {}".format(   #bu kısımdaki sync diff seQ numarasını olarak değiştirdim.
        det_count, infer_ms, sync_diff), (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok!")
        return

    pipe = build_receiver_pipeline_nvidia()
    print("[VID] NVIDIA Pipeline Hazir:\n", pipe)

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    meta = MetaReceiver(META_LISTEN_IP, META_LISTEN_PORT)
    mav.start()
    meta.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps = 0.0
    n = 0
    t0 = time.time()
    
    cap = None # Başlangıçta boş

    try:
        while True:
            # 1. BAĞLANTI KONTROLÜ VE YENİDEN BAĞLANMA
            if cap is None or not cap.isOpened():
                print("[VID] Jetson bekleniyor / Baglanti kuruluyor...")
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                
                if not cap.isOpened():
                    # Bağlanamadıysa 2 saniye bekle ve tekrar dene (ÇÖKME YOK)
                    time.sleep(2.0)
                    continue
                else:
                    print("[VID] Baglanti Kuruldu!")

            # 2. FRAME OKUMA
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Stream koptu veya frame hatasi! Yeniden baslatiliyor...")
                cap.release()
                cap = None # Cap'i sıfırla ki üstteki if bloğu tekrar bağlanmaya çalışsın
                time.sleep(1.0)
                continue
            
            #########################################################################################
            # [YENİ SİHİR BURADA BAŞLIYOR]
            # 1. Görüntünün köşesindeki barkodu oku (Örn: seq_num = 82)
            seq_num = read_barcode(frame)

            # 2. Barkodun üzerini siyah bantla kapat ki gözümüzü yormasın
            frame = mask_barcode(frame)

            # 3. Geçmiş kutusuna (Queue) gidip 82 numaralı metadatayı iste!
            payload = get_synced_metadata_by_seq(seq_num)
            #########################################################################################

            # FPS Hesabı
            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = n / (now - t0)
                n = 0
                t0 = now

            # Telemetriyi al (Thread safe)
            with mav.lock:
                tel = dict(mav.data)

            if payload:
                dets = payload.get("detections", [])
                if dets:
                    frame = draw_dets(frame, dets)

            draw_overlay(frame, tel, fps, payload, seq_num)     # sync_diff yerine seq_num gönderdik
            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        mav.stop()
        meta.stop()
        print("[SYS] Kapatiliyor.")

if __name__ == "__main__":
    main()