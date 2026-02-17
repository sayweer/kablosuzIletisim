#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import json
import socket
import threading
import numpy as np
from pymavlink import mavutil

SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 120

# ÖNEMLİ: sender MPEG-TS gönderiyor -> TS MODE
VIDEO_MODE = "ts"   # <-- auto yerine bunu kullan
VIDEO_CONNECT_TIMEOUT_SEC = 30.0
VIDEO_RECONNECT_SLEEP_SEC = 1.0

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

META_LISTEN_IP = "0.0.0.0"
META_LISTEN_PORT = 5005
META_STALE_SEC = 0.8

SHOW_WINDOW = True
WINDOW_NAME = "Ground Station"
PRINT_INTERVAL_SEC = 0.5

meta_lock = threading.Lock()
meta_state = {"ts_recv": 0.0, "payload": {"ts": 0.0, "infer_ms": 0.0, "detections": [], "seq": 0}}

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_video_candidates(port, latency_ms):
    uri = "srt://:{}?mode=listener&latency={}&transtype=live".format(port, latency_ms)
    appsink = "appsink max-buffers=1 drop=true sync=false"

    # TS -> tsdemux -> h264parse -> decode
    gst_ts = (
        "srtsrc uri=\"{uri}\" ! queue ! "
        "tsdemux ! queue ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
    ).format(uri=uri, appsink=appsink)

    # decodebin alternatif (bazı sistemlerde daha iyi)
    gst_ts_decodebin = (
        "srtsrc uri=\"{uri}\" ! queue ! "
        "tsdemux ! queue ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
    ).format(uri=uri, appsink=appsink)

    return [
        ("gst_ts_h264parse", gst_ts),
        ("gst_ts_decodebin", gst_ts_decodebin),
    ]

def wait_first_frame(cap, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
        time.sleep(0.02)
    return None

def connect_video_with_timeout(port, latency_ms, total_timeout_sec):
    cands = build_video_candidates(port, latency_ms)
    deadline = time.time() + total_timeout_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        for name, pipe in cands:
            if time.time() >= deadline:
                break
            print("[VID] Deneme #{} | {}".format(attempt, name))
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            first = wait_first_frame(cap, timeout_sec=3.0)
            if first is not None:
                print("[VID] Baglandi! pipeline={}".format(name))
                return cap, name, first

            cap.release()
            time.sleep(0.2)

        time.sleep(0.3)

    return None, None, None

def fmt_num(v, digits=2):
    if v is None:
        return "N/A"
    try:
        return "{:.{}f}".format(float(v), digits)
    except Exception:
        return str(v)

class MetaReceiver(threading.Thread):
    def __init__(self, ip="0.0.0.0", port=5005):
        threading.Thread.__init__(self, daemon=True)
        self.ip = ip
        self.port = port
        self.stop_event = threading.Event()
        self.sock = None

    def stop(self):
        self.stop_event.set()
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        print("[META] Dinleniyor: {}:{}".format(self.ip, self.port))
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
    def __init__(self, conn_str, baud=57600):
        threading.Thread.__init__(self, daemon=True)
        self.conn_str = conn_str
        self.baud = baud
        self._stop_event = threading.Event()
        self.master = None
        self.lock = threading.Lock()
        self.data = {
            "mode": "N/A", "armed": False,
            "lat": None, "lon": None, "alt_m": None, "groundspeed": None,
            "roll_deg": None, "pitch_deg": None, "yaw_deg": None, "battery_v": None
        }

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                print("[MAV] Baglaniyor:", self.conn_str)
                self.master = mavutil.mavlink_connection(
                    self.conn_str,
                    baud=self.baud,
                    source_system=MAVLINK_SOURCE_SYS,
                    source_component=MAVLINK_SOURCE_COMP,
                    autoreconnect=True
                )
                print("[MAV] Heartbeat bekleniyor...")
                self.master.wait_heartbeat(timeout=10)
                print("[MAV] Baglanti tamam.")
            except Exception:
                time.sleep(1.5)
                continue

            while not self._stop_event.is_set():
                try:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    if msg is None:
                        continue
                    mtype = msg.get_type()
                    with self.lock:
                        if mtype == "HEARTBEAT":
                            self.data["mode"] = mavutil.mode_string_v10(msg)
                            self.data["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        elif mtype == "GLOBAL_POSITION_INT":
                            self.data["lat"] = msg.lat / 1e7
                            self.data["lon"] = msg.lon / 1e7
                            self.data["alt_m"] = msg.relative_alt / 1000.0
                            self.data["groundspeed"] = ((msg.vx ** 2 + msg.vy ** 2) ** 0.5) / 100.0
                        elif mtype == "ATTITUDE":
                            self.data["roll_deg"] = np.degrees(msg.roll)
                            self.data["pitch_deg"] = np.degrees(msg.pitch)
                            self.data["yaw_deg"] = np.degrees(msg.yaw)
                        elif mtype == "SYS_STATUS":
                            if msg.voltage_battery != 65535:
                                self.data["battery_v"] = msg.voltage_battery / 1000.0
                except Exception:
                    break

def draw_meta_boxes(frame):
    with meta_lock:
        ts_recv = meta_state["ts_recv"]
        payload = dict(meta_state["payload"])

    stale = (time.time() - ts_recv) > META_STALE_SEC
    dets = payload.get("detections", [])
    infer_ms = float(payload.get("infer_ms", 0.0))
    seq = payload.get("seq", 0)

    if not stale:
        for d in dets:
            x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
            x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
            cls = d.get("cls", -1); conf = d.get("conf", 0.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, "cls:{} {:.2f}".format(cls, conf),
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, "YOLO(meta) det:{} infer:{:.1f}ms seq:{}".format(len(dets), infer_ms, seq),
                    (20, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    else:
        cv2.putText(frame, "YOLO(meta): STALE",
                    (20, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_overlay(frame, tel, fps, pipe_name):
    cv2.putText(frame, "SRT LIVE ({})".format(pipe_name), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y = 90
    lines = [
        "Mode: {} Armed: {}".format(tel["mode"], tel["armed"]),
        "Alt: {}m  Spd: {}m/s".format(fmt_num(tel["alt_m"]), fmt_num(tel["groundspeed"])),
        "Bat: {}V".format(fmt_num(tel["battery_v"])),
        "Lat: {} Lon: {}".format(fmt_num(tel["lat"], 6), fmt_num(tel["lon"], 6)),
    ]
    for line in lines:
        cv2.putText(frame, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y += 24

def draw_wait_screen(msg, sub=""):
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(img, msg, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(img, sub, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img

def main():
    global SHOW_WINDOW
    cv2.setNumThreads(1)

    if SHOW_WINDOW and os.name != "nt" and "DISPLAY" not in os.environ:
        print("[UI] DISPLAY yok, SHOW_WINDOW=False yapiliyor.")
        SHOW_WINDOW = False

    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer:", opencv_has_gstreamer())
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok -> bu script bu halde calismaz.")
        return

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    meta_rx = MetaReceiver(META_LISTEN_IP, META_LISTEN_PORT)
    mav.start()
    meta_rx.start()

    cap = None
    pipe_name = "none"
    first_frame = None

    fps_counter = 0
    fps_t0 = time.time()
    fps = 0.0
    last_console = 0.0

    try:
        while True:
            if cap is None:
                if SHOW_WINDOW:
                    wait_img = draw_wait_screen(
                        "BAGLANTI BEKLENIYOR...",
                        "SRT Port:{}  Mode:{}  (q cikis)".format(SRT_LISTEN_PORT, VIDEO_MODE)
                    )
                    cv2.imshow(WINDOW_NAME, wait_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap, pipe_name, first_frame = connect_video_with_timeout(
                    SRT_LISTEN_PORT, SRT_LATENCY_MS, 6.0
                )
                if cap is None:
                    time.sleep(0.2)
                    continue
                frame = first_frame
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[VID] Sinyal koptu, reconnect...")
                    cap.release()
                    cap = None
                    time.sleep(VIDEO_RECONNECT_SLEEP_SEC)
                    continue

            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0 = now

            with mav.lock:
                tel = dict(mav.data)

            draw_overlay(frame, tel, fps, pipe_name)
            draw_meta_boxes(frame)

            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if now - last_console >= 1.0:
                    last_console = now
                    print("[LIVE] fps={:.1f} mode={} armed={} det_meta={}".format(
                        fps, tel["mode"], tel["armed"],
                        len(meta_state["payload"].get("detections", []))
                    ))
    finally:
        if cap:
            cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        mav.stop()
        meta_rx.stop()
        print("[SYS] Program sonlandi.")

if __name__ == "__main__":
    main()
