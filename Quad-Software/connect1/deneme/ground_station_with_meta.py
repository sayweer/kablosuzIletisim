#!/usr/bin/env python3
import os
import cv2
import time
import json
import socket
import threading
import subprocess
import numpy as np
from pymavlink import mavutil
# KONFIG
SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 100
VIDEO_MODE = "raw_h264"

VIDEO_CONNECT_TIMEOUT_SEC = 30.0
VIDEO_RECONNECT_SLEEP_SEC = 1.0

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190
PRINT_INTERVAL_SEC = 0.5

# Raspberry metadata UDP listener (PC tarafinda)
META_LISTEN_IP = "0.0.0.0"
META_LISTEN_PORT = 5005

# Metadata eskimesi (sn)
META_STALE_SEC = 0.8

# META PAYLASIMI
meta_lock = threading.Lock()
meta_state = {"ts_recv": 0.0, "payload": {"ts": 0.0, "infer_ms": 0.0, "detections": []}}

def opencv_has_gstreamer() -> bool:
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_srt_pipelines(port: int, latency_ms: int, mode: str = "raw_h264"):
    uri = f'srt://:{port}?mode=listener&latency={latency_ms}'
    appsink = "appsink max-buffers=1 drop=true sync=false"

    pipelines = {
        "raw_h264": (
            f'srtsrc uri="{uri}" ! queue ! '
            f'h264parse ! avdec_h264 ! videoconvert ! {appsink}'
        ),
        "ts": (
            f'srtsrc uri="{uri}" ! queue ! tsdemux ! queue ! '
            f'h264parse ! avdec_h264 ! videoconvert ! {appsink}'
        ),
    }
    if mode in pipelines:
        return [(mode, pipelines[mode])]
    return [("raw_h264", pipelines["raw_h264"])]

def wait_first_frame(cap, timeout_sec: float):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
        time.sleep(0.01)
    return None

def connect_video_with_timeout(port: int, latency_ms: int, mode: str, total_timeout_sec: float):
    pipelines = build_srt_pipelines(port, latency_ms, mode)
    deadline = time.time() + total_timeout_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        for name, pipe in pipelines:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            print(f"[VID] Deneme #{attempt} | pipeline={name}")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                continue

            first_frame = wait_first_frame(cap, timeout_sec=min(5.0, remaining))
            if first_frame is not None:
                print(f"[VID] Baglandi! Aktif pipeline: {name}")
                return cap, name, first_frame

            cap.release()
        time.sleep(0.5)

    return None, None, None

def fmt_num(v, digits=2):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)

class MetaReceiver(threading.Thread):
    def __init__(self, ip="0.0.0.0", port=5005):
        super().__init__(daemon=True)
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
        print(f"[META] Dinleniyor: {self.ip}:{self.port}")

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
        super().__init__(daemon=True)
        self.conn_str = conn_str
        self.baud = baud
        self._stop_event = threading.Event()
        self.master = None
        self.last_heartbeat = 0.0
        self.last_print = 0.0
        self.data = {
            "mode": "N/A", "armed": False, "lat": None, "lon": None,
            "alt_m": None, "groundspeed": None, "roll_deg": None,
            "pitch_deg": None, "yaw_deg": None, "battery_v": None,
        }
        self.lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def _connect(self):
        while not self._stop_event.is_set():
            try:
                print(f"[MAV] Baglaniyor: {self.conn_str}")
                self.master = mavutil.mavlink_connection(
                    self.conn_str, baud=self.baud, source_system=MAVLINK_SOURCE_SYS,
                    source_component=MAVLINK_SOURCE_COMP, autoreconnect=True
                )
                print("[MAV] Heartbeat bekleniyor...")
                self.master.wait_heartbeat(timeout=5)
                self.last_heartbeat = time.time()
                print("[MAV] Baglanti tamam.")
                return
            except Exception:
                time.sleep(2)

    def _request_streams(self):
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
            )
        except Exception:
            pass

    def run(self):
        while not self._stop_event.is_set():
            self._connect()
            if not self.master:
                continue
            self._request_streams()

            while not self._stop_event.is_set():
                try:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    now = time.time()
                    if msg:
                        mtype = msg.get_type()
                        with self.lock:
                            if mtype == "HEARTBEAT":
                                self.last_heartbeat = now
                                self.data["mode"] = mavutil.mode_string_v10(msg)
                                self.data["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                            elif mtype == "GLOBAL_POSITION_INT":
                                self.data["lat"] = msg.lat / 1e7
                                self.data["lon"] = msg.lon / 1e7
                                self.data["alt_m"] = msg.relative_alt / 1000.0
                                self.data["groundspeed"] = ((msg.vx**2 + msg.vy**2)**0.5) / 100.0
                            elif mtype == "ATTITUDE":
                                self.data["roll_deg"] = np.degrees(msg.roll)
                                self.data["pitch_deg"] = np.degrees(msg.pitch)
                                self.data["yaw_deg"] = np.degrees(msg.yaw)
                            elif mtype == "SYS_STATUS":
                                if msg.voltage_battery != 65535:
                                    self.data["battery_v"] = msg.voltage_battery / 1000.0

                    if now - self.last_print >= PRINT_INTERVAL_SEC:
                        self.last_print = now

                except Exception:
                    break

def draw_meta_boxes(frame):
    with meta_lock:
        ts_recv = meta_state["ts_recv"]
        payload = dict(meta_state["payload"])

    stale = (time.time() - ts_recv) > META_STALE_SEC
    dets = payload.get("detections", [])
    infer_ms = float(payload.get("infer_ms", 0.0))

    if not stale:
        for d in dets:
            x1 = int(d.get("x1", 0))
            y1 = int(d.get("y1", 0))
            x2 = int(d.get("x2", 0))
            y2 = int(d.get("y2", 0))
            cls = d.get("cls", -1)
            conf = d.get("conf", 0.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            txt = f"cls:{cls} {conf:.2f}"
            cv2.putText(frame, txt, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.putText(frame, f"YOLO(meta) Det:{len(dets)} Infer:{infer_ms:.1f}ms",
                    (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    else:
        cv2.putText(frame, "YOLO(meta): STALE", (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_overlay(frame, tel, fps, pipe_name):
    cv2.putText(frame, f"SRT LIVE ({pipe_name})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y = 90
    lines = [
        f"Mode: {tel['mode']} Armed: {tel['armed']}",
        f"Alt: {fmt_num(tel['alt_m'])}m  Spd: {fmt_num(tel['groundspeed'])}m/s",
        f"Bat: {fmt_num(tel['battery_v'])}V",
        f"Lat: {fmt_num(tel['lat'],6)} Lon: {fmt_num(tel['lon'],6)}"
    ]
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += 25

    cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

def draw_wait_screen(text, subtext=""):
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, subtext, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok!")
        return

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    meta_rx = MetaReceiver(META_LISTEN_IP, META_LISTEN_PORT)
    mav.start()
    meta_rx.start()

    cap = None
    fps_counter, fps_t0, fps = 0, time.time(), 0.0
    pipe_name = VIDEO_MODE

    try:
        while True:
            if cap is None:
                wait_img = draw_wait_screen("BAGLANTI BEKLENIYOR...", f"Port: {SRT_LISTEN_PORT} Mode: {VIDEO_MODE}")
                cv2.imshow("Ground Station", wait_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cap, pipe_name, first_frame = connect_video_with_timeout(
                    SRT_LISTEN_PORT, SRT_LATENCY_MS, VIDEO_MODE, VIDEO_CONNECT_TIMEOUT_SEC
                )
                if cap:
                    print("[SYS] Goruntu geldi!")
                    frame = first_frame
                else:
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    print("[SYS] Sinyal koptu, tekrar baglaniliyor...")
                    cap.release()
                    cap = None
                    time.sleep(VIDEO_RECONNECT_SLEEP_SEC)
                    continue

            fps_counter += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_counter / (time.time() - fps_t0)
                fps_counter = 0
                fps_t0 = time.time()

            with mav.lock:
                tel = dict(mav.data)

            draw_overlay(frame, tel, fps, pipe_name)
            draw_meta_boxes(frame)

            cv2.imshow("Ground Station", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        mav.stop()
        meta_rx.stop()
        print("[SYS] Program sonlandi.")

if __name__ == "__main__":
    main()
