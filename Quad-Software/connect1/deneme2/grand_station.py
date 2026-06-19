#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, json, socket, threading
import numpy as np
from pymavlink import mavutil
from collections import deque # Bufferlama için gerekli

# --- AYARLAR ---
SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 120         # Wi-Fi ortamında biraz artırmak titremeyi azaltır
VIDEO_LATENCY_SEC = 0.15     # Tahmini video decode gecikmesi (Senkronizasyon ayarı)

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

META_LISTEN_IP = "0.0.0.0"
META_LISTEN_PORT = 5005
META_BUFFER_LEN = 50         # Geçmiş metadataları tutacak kuyruk uzunluğu

SHOW_WINDOW = True
WINDOW_NAME = "Ground Station - NVIDIA Accelerated"

# --- GLOBAL DEĞIŞKENLER ---
meta_lock = threading.Lock()
# Gelen ham metadayı bir kuyrukta tutuyoruz.
# Yapı: [(ts_recv, payload), (ts_recv, payload), ...]
meta_queue = deque(maxlen=META_BUFFER_LEN) 

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_receiver_pipeline_nvidia():
    """
    NVIDIA GPU kullanarak decode yapan optimize pipeline.
    nvh264dec -> nvvidconv (GPU'da renk dönüşümü) -> CPU'ya transfer
    """
    uri = "srt://:{}?mode=listener&latency={}&transtype=live".format(SRT_LISTEN_PORT, SRT_LATENCY_MS)
    appsink = "appsink drop=true max-buffers=1 sync=false"
    
    # NOT: Bu pipeline Linux üzerinde NVIDIA sürücüleri yüklüyse çalışır.
    # Windows kullanıyorsan 'd3d11h264dec' kullanılması gerekebilir.
    return (
        "srtsrc uri=\"{uri}\" ! queue ! "
        "h264parse ! nvh264dec ! "             # GPU Decode
        "nvvidconv ! video/x-raw,format=BGRx ! " # GPU üzerinde BGRx formatına çevir
        "videoconvert ! video/x-raw,format=BGR ! " # OpenCV için BGR (CPU işlemi çok azaldı)
        "{appsink}"
    ).format(uri=uri, appsink=appsink)

def get_synced_metadata(current_time):
    """
    Video zamanına en uygun metadatayı kuyruktan çeker.
    """
    target_time = current_time - VIDEO_LATENCY_SEC
    best_payload = None
    min_diff = 100.0 # Başlangıçta büyük bir fark
    
    with meta_lock:
        if not meta_queue:
            return None, 0.0
        
        # Kuyruktaki tüm paketlere bak, zamanı en yakın olanı bul
        for ts, payload in meta_queue:
            diff = abs(ts - target_time)
            if diff < min_diff:
                min_diff = diff
                best_payload = payload
            
        # Eğer en iyi eşleşme bile çok eskiyse (örn 1 saniye) çizme
        if min_diff > 0.5:
            return None, min_diff
            
    return best_payload, min_diff

def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
        x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
        cls = int(d.get("cls", 0)); conf = float(d.get("conf", 0.0))
        
        # Görsel İyileştirme: Daha kalın ve okunur kutular
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = "ID:{} %{:.0f}".format(cls, conf*100)
        
        # Etiket arka planı (okunabilirlik için)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1-20), (x1+t_size[0], y1), (0,255,0), -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
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

    cv2.putText(frame, "AI: {} Obj | Infer: {:.1f}ms | Sync Diff: {:.3f}s".format(
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

            # Senkronize Metadata Çekimi
            payload, sync_diff = get_synced_metadata(now)

            if payload:
                dets = payload.get("detections", [])
                if dets:
                    frame = draw_dets(frame, dets)

            draw_overlay(frame, tel, fps, payload, sync_diff)
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