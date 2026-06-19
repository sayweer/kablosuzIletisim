#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import threading
import subprocess
import numpy as np
from pymavlink import mavutil

# =========================
# KONFİGÜRASYON
# =========================
SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 120

# "auto", "ts", "raw_h264", "rtp_h264", "decodebin"
VIDEO_MODE = "auto"

# İlk bağlantı bekleme süresi (senin istediğin gibi 30 sn)
VIDEO_CONNECT_TIMEOUT_SEC = 30.0

# Akış başladıktan sonra frame gelmezse bu kadar bekleyip reconnect dener
VIDEO_STALL_RECONNECT_SEC = 3.0

# Yeniden bağlanma denemeleri arasında bekleme
VIDEO_RECONNECT_SLEEP_SEC = 1.0

# MAVLink bağlantısı:
# Ağdan geliyorsa:
MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
# USB serial ise:
# MAVLINK_CONN_STR = "/dev/ttyACM0"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

PRINT_INTERVAL_SEC = 0.5  # telemetri terminal yazdırma aralığı

# GStreamer debug açmak istersen:
# os.environ["GST_DEBUG"] = "2"


# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def opencv_has_gstreamer() -> bool:
    """OpenCV'nin GStreamer backend ile derlenip derlenmediğini kontrol eder."""
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False


def check_gst_plugin(plugin_name: str) -> bool:
    """Sistemde gst plugin var mı kontrol eder."""
    try:
        res = subprocess.run(
            ["gst-inspect-1.0", plugin_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return res.returncode == 0
    except FileNotFoundError:
        return False


def build_srt_pipelines(port: int, latency_ms: int, mode: str = "auto"):
    """
    Drone tarafındaki stream formatı farklı olabilir.
    Bu yüzden birden fazla pipeline fallback veriyoruz.
    """
    uri = f'srt://:{port}?mode=listener&latency={latency_ms}'
    appsink = "appsink max-buffers=1 drop=true sync=false"

    pipelines = {
        # SRT içinde MPEG-TS + H264
        "ts": (
            f'srtsrc uri="{uri}" ! queue ! tsdemux ! queue ! '
            f'h264parse ! avdec_h264 ! videoconvert ! {appsink}'
        ),
        # SRT içinde raw H264 (annex-b)
        "raw_h264": (
            f'srtsrc uri="{uri}" ! queue ! '
            f'h264parse ! avdec_h264 ! videoconvert ! {appsink}'
        ),
        # SRT içinde RTP/H264
        "rtp_h264": (
            f'srtsrc uri="{uri}" ! queue ! '
            f'application/x-rtp,media=video,encoding-name=H264,payload=96 ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! {appsink}'
        ),
        # Genel fallback (bazı akışlarda decodebin kurtarır)
        "decodebin": (
            f'srtsrc uri="{uri}" ! queue ! decodebin ! videoconvert ! {appsink}'
        ),
    }

    if mode == "auto":
        order = ["ts", "raw_h264", "rtp_h264", "decodebin"]
        return [(name, pipelines[name]) for name in order]
    if mode in pipelines:
        return [(mode, pipelines[mode])]

    # Geçersiz mode girilirse auto'ya düş
    order = ["ts", "raw_h264", "rtp_h264", "decodebin"]
    return [(name, pipelines[name]) for name in order]


def wait_first_frame(cap, timeout_sec: float):
    """Açılan capture'dan ilk frame'i timeout ile bekler."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.03)
    return None


def connect_video_with_timeout(port: int, latency_ms: int, mode: str, total_timeout_sec: float):
    """
    30 sn boyunca (veya verilen timeout kadar) pipeline'ları deneyerek bağlantı açar.
    Başarılı olursa (cap, pipeline_name, first_frame) döner.
    """
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
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            # Pipeline açıldı ama frame henüz gelmemiş olabilir.
            first_frame = wait_first_frame(cap, timeout_sec=min(4.0, remaining))
            if first_frame is not None:
                print(f"[VID] Bağlandı. Aktif pipeline: {name}")
                return cap, name, first_frame

            # Açıldı gibi görünüp frame gelmediyse kapatıp diğerini dene
            cap.release()

        time.sleep(0.2)

    return None, None, None


def fmt_num(v, digits=2):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


# =========================
# MAVLINK THREAD
# =========================
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
            "mode": "N/A",
            "armed": False,
            "lat": None,
            "lon": None,
            "alt_m": None,
            "groundspeed": None,
            "roll_deg": None,
            "pitch_deg": None,
            "yaw_deg": None,
            "battery_v": None,
        }
        self.lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def _connect(self):
        while not self._stop_event.is_set():
            try:
                print(f"[MAV] Bağlanıyor: {self.conn_str}")

                is_serial = self.conn_str.startswith("/dev/") or self.conn_str.upper().startswith("COM")
                if is_serial:
                    self.master = mavutil.mavlink_connection(
                        self.conn_str,
                        baud=self.baud,
                        source_system=MAVLINK_SOURCE_SYS,
                        source_component=MAVLINK_SOURCE_COMP,
                        autoreconnect=True,
                        dialect="common",
                    )
                else:
                    self.master = mavutil.mavlink_connection(
                        self.conn_str,
                        source_system=MAVLINK_SOURCE_SYS,
                        source_component=MAVLINK_SOURCE_COMP,
                        autoreconnect=True,
                        dialect="common",
                    )

                try:
                    self.master.mav.set_proto_version(2)
                except Exception:
                    pass

                print("[MAV] Heartbeat bekleniyor...")
                self.master.wait_heartbeat(timeout=10)
                self.last_heartbeat = time.time()
                print("[MAV] Heartbeat alındı. Bağlantı tamam.")
                return
            except Exception as e:
                print(f"[MAV] Bağlanamadı: {e}. 2 sn sonra tekrar denenecek.")
                time.sleep(2)

    def _request_streams(self):
        # Eski/uyumsuz FC'lerde bazı komutlar reddedilebilir; sorun değil.
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # Hz
                1,
            )
        except Exception:
            pass

        try:
            def set_interval(msg_id, hz):
                interval_us = int(1e6 / hz)
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,
                    interval_us,
                    0, 0, 0, 0, 0
                )

            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2)
        except Exception:
            pass

    def _parse_msg(self, msg):
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            return

        with self.lock:
            if mtype == "HEARTBEAT":
                self.last_heartbeat = time.time()
                try:
                    self.data["mode"] = mavutil.mode_string_v10(msg)
                except Exception:
                    self.data["mode"] = "N/A"
                self.data["armed"] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )

            elif mtype == "GLOBAL_POSITION_INT":
                self.data["lat"] = msg.lat / 1e7
                self.data["lon"] = msg.lon / 1e7
                self.data["alt_m"] = msg.relative_alt / 1000.0
                # vx/vy cm/s
                self.data["groundspeed"] = ((msg.vx ** 2 + msg.vy ** 2) ** 0.5) / 100.0

            elif mtype == "ATTITUDE":
                # rad -> deg
                self.data["roll_deg"] = np.degrees(msg.roll)
                self.data["pitch_deg"] = np.degrees(msg.pitch)
                self.data["yaw_deg"] = np.degrees(msg.yaw)

            elif mtype == "SYS_STATUS":
                if msg.voltage_battery != 65535:
                    self.data["battery_v"] = msg.voltage_battery / 1000.0

    def _print_status(self):
        with self.lock:
            d = dict(self.data)

        print(
            f"[TEL] mode={d['mode']} armed={d['armed']} "
            f"lat={fmt_num(d['lat'], 7)} lon={fmt_num(d['lon'], 7)} alt={fmt_num(d['alt_m'])}m "
            f"spd={fmt_num(d['groundspeed'])}m/s "
            f"roll={fmt_num(d['roll_deg'])} pitch={fmt_num(d['pitch_deg'])} yaw={fmt_num(d['yaw_deg'])} "
            f"bat={fmt_num(d['battery_v'])}V"
        )

    def run(self):
        while not self._stop_event.is_set():
            self._connect()
            if self.master is None:
                time.sleep(1)
                continue

            self._request_streams()

            while not self._stop_event.is_set():
                try:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    now = time.time()

                    if msg is not None:
                        self._parse_msg(msg)

                    # 5 sn heartbeat yoksa reconnect
                    if now - self.last_heartbeat > 5:
                        raise RuntimeError("Heartbeat timeout (5s)")

                    if now - self.last_print >= PRINT_INTERVAL_SEC:
                        self.last_print = now
                        self._print_status()

                except Exception as e:
                    print(f"[MAV] Bağlantı koptu / hata: {e}. Yeniden bağlanılıyor...")
                    try:
                        if self.master:
                            self.master.close()
                    except Exception:
                        pass
                    self.master = None
                    time.sleep(1)
                    break


# =========================
# OVERLAY
# =========================
def draw_overlay(frame, tel, fps, video_pipeline_name):
    h, w = frame.shape[:2]

    cv2.putText(frame, "SRT VIDEO + MAVLINK2 TELEMETRY", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}  PIPE: {video_pipeline_name}", (20, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    cv2.putText(frame, f"MODE: {tel['mode']}  ARMED: {tel['armed']}", (20, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    cv2.putText(frame,
                f"ALT: {fmt_num(tel['alt_m'])} m   SPD: {fmt_num(tel['groundspeed'])} m/s",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)

    cv2.putText(frame,
                f"LAT/LON: {fmt_num(tel['lat'], 7)} / {fmt_num(tel['lon'], 7)}",
                (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    cv2.putText(frame,
                f"ATT(deg): R={fmt_num(tel['roll_deg'])} P={fmt_num(tel['pitch_deg'])} Y={fmt_num(tel['yaw_deg'])}",
                (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    cv2.putText(frame, f"BAT: {fmt_num(tel['battery_v'])} V", (20, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

    cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def draw_wait_screen(text1, text2="", width=960, height=540):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text1, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if text2:
        cv2.putText(img, text2, (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(img, "Press 'q' to quit", (30, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return img


# =========================
# MAIN
# =========================
def main():
    print("[SYS] Başlatılıyor...")

    # 1) OpenCV-GStreamer kontrol
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destekli görünmüyor.")
        print("      cv2.getBuildInformation() içinde 'GStreamer: YES' olmalı.")
        print("      Kurulum notlarını aşağıda verdim.")
        return

    # 2) GStreamer plugin kontrol (uyarı amaçlı)
    required_plugins = ["srtsrc", "tsdemux", "h264parse", "avdec_h264", "appsink"]
    missing = [p for p in required_plugins if not check_gst_plugin(p)]
    if missing:
        print(f"[WARN] Eksik olabilecek plugin(ler): {missing}")
        print("       Yine de çalıştırmayı deneyeceğim, ama pipeline açılmazsa bunları kur.")

    # 3) MAVLink thread
    mav_thread = MavlinkReceiver(MAVLINK_CONN_STR, baud=MAVLINK_BAUD)
    mav_thread.start()

    cap = None
    first_frame = None
    active_pipeline = "N/A"
    last_good_frame_ts = 0.0

    # FPS sayaç
    fps_counter = 0
    fps_t0 = time.time()
    fps = 0.0

    print("[SYS] Video bağlantısı bekleniyor...")

    try:
        while True:
            # Video bağlı değilse bağlan
            if cap is None:
                wait_img = draw_wait_screen(
                    f"VIDEO WAITING... (timeout {int(VIDEO_CONNECT_TIMEOUT_SEC)}s)",
                    f"SRT listen port: {SRT_LISTEN_PORT}, mode: {VIDEO_MODE}"
                )
                cv2.imshow("Ground Station", wait_img)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

                cap, pipe_name, ff = connect_video_with_timeout(
                    port=SRT_LISTEN_PORT,
                    latency_ms=SRT_LATENCY_MS,
                    mode=VIDEO_MODE,
                    total_timeout_sec=VIDEO_CONNECT_TIMEOUT_SEC
                )

                if cap is None:
                    # 30 sn doldu, kısa bekleyip tekrar dene
                    msg = draw_wait_screen("NO VIDEO IN 30s. Re-trying...",
                                           "Drone tarafı srtsink mode=caller ve doğru IP/port kontrol et.")
                    cv2.imshow("Ground Station", msg)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    time.sleep(VIDEO_RECONNECT_SLEEP_SEC)
                    continue

                active_pipeline = pipe_name
                first_frame = ff
                last_good_frame_ts = time.time()
                print(f"[VID] Akış başladı. Pipeline={active_pipeline}")

            # Frame çek
            if first_frame is not None:
                frame = first_frame
                ok = True
                first_frame = None
            else:
                ok, frame = cap.read()

            now = time.time()

            if not ok or frame is None:
                # Kısa kesinti olabilir, hemen kopmuş sayma
                lost_for = now - last_good_frame_ts
                text = f"VIDEO SIGNAL LOST ({lost_for:.1f}s) - waiting..."
                blank = draw_wait_screen(text, f"Pipeline: {active_pipeline}")
                cv2.imshow("Ground Station", blank)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # Uzun sürdüyse reconnect
                if lost_for >= VIDEO_STALL_RECONNECT_SEC:
                    print(f"[VID] Frame yok ({lost_for:.1f}s). Reconnect başlatılıyor...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    first_frame = None
                    time.sleep(VIDEO_RECONNECT_SLEEP_SEC)
                else:
                    time.sleep(0.01)

                continue

            # Başarılı frame
            last_good_frame_ts = now

            # FPS güncelle
            fps_counter += 1
            elapsed = now - fps_t0
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_t0 = now

            # Telemetri snapshot
            with mav_thread.lock:
                tel = dict(mav_thread.data)

            draw_overlay(frame, tel, fps, active_pipeline)

            cv2.imshow("Ground Station", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        mav_thread.stop()
        print("[SYS] Kapandı.")


if __name__ == "__main__":
    main()
