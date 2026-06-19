#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, json, socket, threading
import numpy as np
from pymavlink import mavutil

SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 120

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

META_LISTEN_IP = "0.0.0.0"
META_LISTEN_PORT = 5005
META_STALE_SEC = 0.8

SHOW_WINDOW = True
WINDOW_NAME = "Ground Station"

meta_lock = threading.Lock()
meta_state = {"ts_recv": 0.0, "payload": {"ts":0.0, "infer_ms":0.0, "detections":[], "seq":0}}

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_receiver_pipeline():
    uri = "srt://:{}?mode=listener&latency={}&transtype=live".format(SRT_LISTEN_PORT, SRT_LATENCY_MS)
    appsink = "appsink drop=true max-buffers=1 sync=false"
    # Sender MPEG-TS gönderiyor -> tsdemux şart
    return (
        "srtsrc uri=\"{uri}\" ! queue ! "
        "tsdemux ! queue ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! {appsink}"
    ).format(uri=uri, appsink=appsink)

class MetaReceiver(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self, daemon=True)
        self.ip = ip; self.port = port
        self.stop_event = threading.Event()
        self.sock = None

    def stop(self):
        self.stop_event.set()
        try:
            if self.sock: self.sock.close()
        except Exception:
            pass

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        print("[META] Dinleniyor {}:{}".format(self.ip, self.port))
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
                payload = json.loads(data.decode("utf-8"))
                with meta_lock:
                    meta_state["ts_recv"] = time.time()
                    meta_state["payload"] = payload
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
        self.data = {"mode":"N/A","armed":False,"lat":None,"lon":None,"alt_m":None,"groundspeed":None,"battery_v":None}

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                print("[MAV] Baglaniyor:", self.conn_str)
                m = mavutil.mavlink_connection(self.conn_str, baud=self.baud,
                                               source_system=MAVLINK_SOURCE_SYS,
                                               source_component=MAVLINK_SOURCE_COMP,
                                               autoreconnect=True)
                m.wait_heartbeat(timeout=10)
                print("[MAV] Baglandi.")
            except Exception:
                time.sleep(1.0)
                continue

            while not self.stop_event.is_set():
                try:
                    msg = m.recv_match(blocking=True, timeout=1.0)
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
                            self.data["groundspeed"] = ((msg.vx**2 + msg.vy**2)**0.5) / 100.0
                        elif t == "SYS_STATUS":
                            if msg.voltage_battery != 65535:
                                self.data["battery_v"] = msg.voltage_battery / 1000.0
                except Exception:
                    break

def draw_overlay(frame, tel, fps):
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Mode:{} Armed:{}".format(tel["mode"], tel["armed"]), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer yok -> bu script bu halde calismaz.")
        return

    if SHOW_WINDOW and os.name != "nt" and "DISPLAY" not in os.environ:
        print("[UI] DISPLAY yok -> headless.")
        return

    pipe = build_receiver_pipeline()
    print("[VID] Pipeline:\n", pipe)

    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERR] Video acilmadi. Jetson stream geliyor mu?")
        return

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    meta = MetaReceiver(META_LISTEN_IP, META_LISTEN_PORT)
    mav.start(); meta.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps = 0.0
    n = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[VID] Frame yok (stream koptu?)")
                time.sleep(0.2)
                continue

            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = n / (now - t0)
                n = 0
                t0 = now

            with mav.lock:
                tel = dict(mav.data)

            draw_overlay(frame, tel, fps)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mav.stop()
        meta.stop()
        print("[SYS] PC bitti.")

if __name__ == "__main__":
    main()
