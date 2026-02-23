#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, json, socket, threading
import numpy as np
from pymavlink import mavutil
from collections import deque

# --- ağ AYARLARI ---
SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 300  # 120 -> 300 (Wi-Fi için daha gerçekçi)

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

META_LISTEN_IP = "0.0.0.0"
META_LISTEN_PORT = 5005
META_BUFFER_LEN = 240

SHOW_WINDOW = True
WINDOW_NAME = "Ground Station - Stable TS Demux"

# --- GLOBAL ---
meta_lock = threading.Lock()
meta_queue = deque(maxlen=META_BUFFER_LEN)

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_receiver_pipeline(decoder_element):
    """
    MPEG-TS aldığımız için decodebin yerine NET boru:
    srtsrc -> tsdemux -> h264parse -> decoder -> videoconvert -> appsink
    """
    uri = "srt://:{}?mode=listener&transtype=live&latency={}".format(SRT_LISTEN_PORT, SRT_LATENCY_MS)
    appsink = "appsink drop=true max-buffers=1 sync=false"

    return (
        "srtsrc uri=\"{uri}\" ! queue max-size-buffers=8 leaky=downstream ! "
        "tsdemux ! queue max-size-buffers=8 leaky=downstream ! "
        "h264parse ! {dec} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "{appsink}"
    ).format(uri=uri, dec=decoder_element, appsink=appsink)

# Barkod okuyucu
def read_barcode(frame):
    binary_str = ""
    for i in range(8):
        core_region = frame[6:14, (i*20)+6:(i*20)+14]
        mean_brightness = np.mean(core_region)
        binary_str += "1" if mean_brightness > 127 else "0"
    return int(binary_str, 2)

def mask_barcode(frame):
    cv2.rectangle(frame, (0, 0), (160, 20), (0, 0, 0), -1)
    return frame

def get_synced_metadata_by_seq(target_seq):
    with meta_lock:
        if not meta_queue:
            return None
        for ts, payload in reversed(meta_queue):
            if payload.get("seq") == target_seq:
                return payload
    return None

def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
        x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
        cls = int(d.get("cls", 0)); conf = float(d.get("conf", 0.0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = "ID:{} %{:.0f}".format(cls, conf*100)
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
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        print("[META] Dinleniyor {}:{}".format(self.ip, self.port))

        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
                payload = json.loads(data.decode("utf-8"))
                recv_time = time.time()
                with meta_lock:
                    meta_queue.append((recv_time, payload))
            except socket.timeout:
                pass
            except Exception:
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
        if self.m_conn is None:
            return
        interval_us = 3000000
        self.m_conn.mav.command_long_send(
            self.m_conn.target_system, self.m_conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, interval_us, 0, 0, 0, 0, 0)

        self.m_conn.mav.command_long_send(
            self.m_conn.target_system, self.m_conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, interval_us, 0, 0, 0, 0, 0)

    def run(self):
        last_req_time = 0
        while not self.stop_event.is_set():
            try:
                print("[MAV] Baglaniyor:", self.conn_str)
                self.m_conn = mavutil.mavlink_connection(
                    self.conn_str, baud=self.baud,
                    source_system=MAVLINK_SOURCE_SYS,
                    source_component=MAVLINK_SOURCE_COMP,
                    autoreconnect=True
                )
                self.m_conn.wait_heartbeat(timeout=10)
                print("[MAV] Baglandi.")
            except Exception:
                time.sleep(1.0)
                continue

            while not self.stop_event.is_set():
                now = time.time()
                if now - last_req_time > 3.0:
                    self.request_data_stream()
                    last_req_time = now

                try:
                    msg = self.m_conn.recv_match(blocking=True, timeout=1.0)
                    if msg is None:
                        continue
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

def draw_overlay(frame, tel, fps, meta_payload, seq_num):
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    h, w = frame.shape[:2]
    cv2.putText(frame, "MOD: {}".format(tel["mode"]), (w-240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    cv2.putText(frame, "ARMED: {}".format(tel["armed"]), (w-240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,255,0) if tel["armed"] else (0,0,255), 2)
    cv2.putText(frame, "BAT: {:.1f}V".format(tel["battery_v"]), (w-240, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, "ALT: {:.1f}m".format(tel["alt_m"]), (w-240, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    det_count = 0
    infer_ms = 0.0
    if meta_payload:
        det_count = len(meta_payload.get("detections", []))
        infer_ms = float(meta_payload.get("infer_ms", 0.0))

    cv2.putText(frame, "AI: {} Obj | Infer: {:.1f}ms | Seq: {}".format(
        det_count, infer_ms, seq_num), (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok!")
        return

    # Decoder denemeleri:
    # 1) NVIDIA decode (varsa)
    # 2) CPU decode (garanti)
    pipeline_candidates = [
        ("nvh264dec", build_receiver_pipeline("nvh264dec")),
        ("avdec_h264", build_receiver_pipeline("avdec_h264")),
    ]

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    meta = MetaReceiver(META_LISTEN_IP, META_LISTEN_PORT)
    mav.start()
    meta.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps = 0.0
    n = 0
    t0 = time.time()

    cap = None
    active_pipe = None

    try:
        while True:
            # 1) Bağlan / yeniden bağlan
            if cap is None or not cap.isOpened():
                print("[VID] Jetson bekleniyor / Baglanti kuruluyor...")

                opened = False
                for dec_name, pipe in pipeline_candidates:
                    tmp = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    if tmp.isOpened():
                        cap = tmp
                        active_pipe = dec_name
                        print("[VID] Baglanti Kuruldu! Decoder:", dec_name)
                        opened = True
                        break
                    else:
                        tmp.release()

                if not opened:
                    time.sleep(2.0)
                    continue

            # 2) Frame oku
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Stream koptu veya frame hatasi! Yeniden baslatiliyor...")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(1.0)
                continue

            # 3) Barkod oku, maskle, metadata eşle
            seq_num = read_barcode(frame)
            frame = mask_barcode(frame)
            payload = get_synced_metadata_by_seq(seq_num)

            # FPS
            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = n / (now - t0)
                n = 0
                t0 = now

            # Telemetri
            with mav.lock:
                tel = dict(mav.data)

            if payload:
                dets = payload.get("detections", [])
                if dets:
                    frame = draw_dets(frame, dets)

            draw_overlay(frame, tel, fps, payload, seq_num)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        mav.stop()
        meta.stop()
        print("[SYS] Kapatiliyor.")

if __name__ == "__main__":
    main()