import os
import cv2
import time
import threading
import subprocess
import numpy as np
from pymavlink import mavutil

# YOLO11 (Ultralytics)
# KENDI BILGISAYARINA GORE: Bu kutuphane kurulu degilse:
# pip install -U ultralytics
from ultralytics import YOLO

# =========================
# KONFIGURASYON
# =========================
SRT_LISTEN_PORT = 9000
SRT_LATENCY_MS = 100
VIDEO_MODE = "raw_h264"

VIDEO_CONNECT_TIMEOUT_SEC = 30.0
VIDEO_STALL_RECONNECT_SEC = 3.0
VIDEO_RECONNECT_SLEEP_SEC = 1.0

MAVLINK_CONN_STR = "udp:0.0.0.0:14550"
MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190
PRINT_INTERVAL_SEC = 0.5

# =========================
# YOLO AYARLARI
# =========================
YOLO_ENABLED = True

# >>> KENDI BILGISAYARINA GORE DEGISTIR <<<
# 1) seyit.pt dosyasinin tam yolunu yaz
# 2) Dosya script ile ayni klasorde ise sadece "seyit.pt" yeterli
YOLO_MODEL_PATH = "seyit.pt"

# >>> KENDI BILGISAYARINA GORE DEGISTIR <<<
# "auto" -> CUDA varsa GPU(0), yoksa CPU
# "cpu"  -> zorla CPU
# "0"    -> zorla ilk GPU
YOLO_DEVICE = "auto"

YOLO_IMGSZ = 640
YOLO_CONF = 0.35
YOLO_IOU = 0.45

# Performans ayari:
# 1 = her frame infer
# 2 = 2 frame'de 1 infer
# 3 = 3 frame'de 1 infer ...
INFER_EVERY_N_FRAMES = 1

# Sinif filtresi (opsiyonel):
# None -> tum siniflar
# ornek: [0, 2] -> sadece class 0 ve 2
YOLO_CLASSES = None


# =========================
# YARDIMCI FONKSIYONLAR
# =========================
def opencv_has_gstreamer() -> bool:
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False


def check_gst_plugin(plugin_name: str) -> bool:
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


# =========================
# YOLO WRAPPER
# =========================
class YoloRunner:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        classes=None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model dosyasi bulunamadi: {model_path}")

        self.model = YOLO(model_path)
        self.device = self._resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes

        print(f"[YOLO] Model yuklendi: {model_path}")
        print(f"[YOLO] Device: {self.device}")

    def _resolve_device(self, device: str):
        if device != "auto":
            return device
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def infer_and_annotate(self, frame):
        t0 = time.time()

        results = self.model.predict(
            source=frame,          # OpenCV ndarray (BGR)
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,  # None ise tum siniflar
            verbose=False,
        )
        r = results[0]
        annotated = r.plot()       # BGR numpy frame

        det_count = 0
        if r.boxes is not None:
            det_count = len(r.boxes)

        infer_ms = (time.time() - t0) * 1000.0
        return annotated, det_count, infer_ms


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
                print(f"[MAV] Baglaniyor: {self.conn_str}")
                self.master = mavutil.mavlink_connection(
                    self.conn_str,
                    baud=self.baud,
                    source_system=MAVLINK_SOURCE_SYS,
                    source_component=MAVLINK_SOURCE_COMP,
                    autoreconnect=True,
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
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
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
                                self.data["armed"] = bool(
                                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                                )

                            elif mtype == "GLOBAL_POSITION_INT":
                                self.data["lat"] = msg.lat / 1e7
                                self.data["lon"] = msg.lon / 1e7
                                self.data["alt_m"] = msg.relative_alt / 1000.0

                                # Duzeltme: msg.vy2 diye alan yok, vx/vy var
                                self.data["groundspeed"] = float(np.hypot(msg.vx, msg.vy) / 100.0)

                            elif mtype == "ATTITUDE":
                                self.data["roll_deg"] = np.degrees(msg.roll)
                                self.data["pitch_deg"] = np.degrees(msg.pitch)
                                self.data["yaw_deg"] = np.degrees(msg.yaw)

                            elif mtype == "SYS_STATUS":
                                if msg.voltage_battery != 65535:
                                    self.data["battery_v"] = msg.voltage_battery / 1000.0

                    if now - self.last_print >= PRINT_INTERVAL_SEC:
                        self.last_print = now
                        # with self.lock:
                        #     d = dict(self.data)
                        # print(f"[TEL] {d['mode']} Bat:{d['battery_v']}V Alt:{d['alt_m']}m")

                except Exception:
                    break


# =========================
# GORSELLESTIRME
# =========================
def draw_overlay(frame, tel, fps, pipe_name, yolo_enabled=False, det_count=0, infer_ms=0.0):
    cv2.putText(frame, f"SRT LIVE ({pipe_name})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y = 90
    lines = [
        f"Mode: {tel['mode']} Armed: {tel['armed']}",
        f"Alt: {fmt_num(tel['alt_m'])}m  Spd: {fmt_num(tel['groundspeed'])}m/s",
        f"Bat: {fmt_num(tel['battery_v'])}V",
        f"Lat: {fmt_num(tel['lat'],6)} Lon: {fmt_num(tel['lon'],6)}"
    ]

    if yolo_enabled:
        lines.append(f"YOLO: ON  Det: {det_count}  Infer: {infer_ms:.1f} ms")

    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += 25

    cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def draw_wait_screen(text, subtext=""):
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, subtext, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


# =========================
# MAIN
# =========================
def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok!")
        return

    # YOLO init
    yolo = None
    if YOLO_ENABLED:
        try:
            yolo = YoloRunner(
                model_path=YOLO_MODEL_PATH,
                device=YOLO_DEVICE,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                classes=YOLO_CLASSES,
            )
        except Exception as e:
            print(f"[YOLO][ERR] Model yuklenemedi: {e}")
            print("[YOLO] YOLO kapatiliyor, yalnizca ham video devam edecek.")
            yolo = None

    mav = MavlinkReceiver(MAVLINK_CONN_STR, MAVLINK_BAUD)
    mav.start()

    cap = None
    pipe_name = VIDEO_MODE
    fps_counter, fps_t0, fps = 0, time.time(), 0.0
    frame_idx = 0

    last_annotated = None
    last_det_count = 0
    last_infer_ms = 0.0

    try:
        while True:
            if cap is None:
                wait_img = draw_wait_screen(
                    "BAGLANTI BEKLENIYOR...",
                    f"Port: {SRT_LISTEN_PORT} Mode: {VIDEO_MODE}",
                )
                cv2.imshow("Ground Station", wait_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                cap, pipe_name, first_frame = connect_video_with_timeout(
                    SRT_LISTEN_PORT,
                    SRT_LATENCY_MS,
                    VIDEO_MODE,
                    VIDEO_CONNECT_TIMEOUT_SEC,
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

            frame_idx += 1
            display_frame = frame

            # ---------------- YOLO ENTEGRASYONU ----------------
            if yolo is not None:
                # N frame'de bir infer
                if frame_idx % INFER_EVERY_N_FRAMES == 0:
                    try:
                        display_frame, last_det_count, last_infer_ms = yolo.infer_and_annotate(frame)
                        last_annotated = display_frame
                    except Exception as e:
                        print(f"[YOLO][ERR] Inference hatasi: {e}")
                        display_frame = frame
                else:
                    # Infer yapilmayan frame'lerde son annotate'i goster (daha akici goruntu)
                    if last_annotated is not None:
                        display_frame = last_annotated
            # ---------------------------------------------------

            fps_counter += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_counter / (time.time() - fps_t0)
                fps_counter = 0
                fps_t0 = time.time()

            with mav.lock:
                tel = dict(mav.data)

            draw_overlay(
                display_frame,
                tel,
                fps,
                pipe_name,
                yolo_enabled=(yolo is not None),
                det_count=last_det_count,
                infer_ms=last_infer_ms,
            )

            cv2.imshow("Ground Station", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        mav.stop()
        print("[SYS] Program sonlandi.")


if __name__ == "__main__":
    main()
