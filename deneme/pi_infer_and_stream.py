#!/usr/bin/env python3
import os
import cv2
import time
import json
import socket
import threading
import queue
import subprocess
from ultralytics import YOLO

# RASPBERRY KONFIG
CAM_DEVICE = "/dev/video0"
WIDTH = 640
HEIGHT = 480
FPS = 30

# - KENDI AGINA GORE DEGISTIR -
PC_IP = "192.168.1.50"      # Ground station PC IP
SRT_PORT = 9000
META_PORT = 5005            # Metadata UDP port (PC tarafinda ayni olmali)

# KENDI DOSYA YOLUNA GORE DEGISTIR
MODEL_PATH = "seyit.pt"

# YOLO ayarlari
IMGSZ = 640
CONF = 0.35
IOU = 0.45
INFER_EVERY_N = 2           # 1=her frame, 2=2 frame'de 1 infer

# Stream olarak annotate gondermek ister misin?
STREAM_ANNOTATED = True     # True -> kutulu goruntu gonderir, False -> ham goruntu gonderir

# PAYLASILAN DURUM
running = True
latest_frame = None
latest_annotated = None
latest_meta = {"ts": 0.0, "detections": [], "infer_ms": 0.0}

frame_lock = threading.Lock()

# THREADS
def capture_loop():
    global latest_frame, running

    cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERR] Kamera acilamadi: {CAM_DEVICE}")
        running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print("[CAP] Kamera acildi.")
    while running:
        ok, frame = cap.read()
        if not ok:
            continue
        with frame_lock:
            latest_frame = frame

    cap.release()
    print("[CAP] Kamera kapandi.")


def infer_loop():
    global latest_annotated, latest_meta, running

    if not os.path.isfile(MODEL_PATH):
        print(f"[ERR] Model bulunamadi: {MODEL_PATH}")
        running = False
        return

    model = YOLO(MODEL_PATH)
    print(f"[YOLO] Model yuklendi: {MODEL_PATH}")

    idx = 0
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        idx += 1
        if idx % INFER_EVERY_N != 0:
            # infer yapilmayan framelerde annotate'i koru
            continue

        t0 = time.time()
        try:
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device="cpu",   # Raspberry'de genelde cpu
                verbose=False
            )
            r = results[0]
            ann = r.plot()

            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls = int(b.cls[0])
                    confv = float(b.conf[0])

                    x1, y1, x2, y2 = xyxy
                    detections.append({
                        "cls": cls,
                        "conf": confv,
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "cx": float((x1 + x2) / 2.0),
                        "cy": float((y1 + y2) / 2.0),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1)
                    })

            infer_ms = (time.time() - t0) * 1000.0

            with frame_lock:
                latest_annotated = ann
                latest_meta = {
                    "ts": time.time(),
                    "infer_ms": infer_ms,
                    "detections": detections
                }

        except Exception as e:
            print(f"[YOLO][ERR] {e}")
            time.sleep(0.01)


def metadata_udp_sender():
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[META] UDP -> {PC_IP}:{META_PORT}")

    while running:
        with frame_lock:
            meta = dict(latest_meta)
        try:
            payload = json.dumps(meta).encode("utf-8")
            sock.sendto(payload, (PC_IP, META_PORT))
        except Exception:
            pass
        time.sleep(0.03)  # ~33Hz

    sock.close()


def srt_stream_loop():
    """
    OpenCV frame -> ffmpeg stdin -> SRT H264 stream
    """
    global running

    ffmpeg_cmd = [
    "ffmpeg",
    "-re",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-g", "30",                         # <<< EKLE >>> keyframe araligi
    "-keyint_min", "30",                # <<< EKLE >>>
    "-x264-params", "repeat-headers=1", # <<< EKLE >>> SPS/PPS tekrar
    "-b:v", "2000k",
    "-maxrate", "2000k",                # <<< EKLE >>> bitrate stabil
    "-bufsize", "4000k",                # <<< EKLE >>>
    "-pix_fmt", "yuv420p",
    "-f", "mpegts",
    f"srt://{PC_IP}:{SRT_PORT}?mode=caller&latency=100"
]


    print("[SRT] ffmpeg process baslatiliyor...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        while running:
            with frame_lock:
                if STREAM_ANNOTATED and latest_annotated is not None:
                    frame = latest_annotated.copy()
                else:
                    frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            # Boyut garanti
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            try:
                proc.stdin.write(frame.tobytes())
            except Exception:
                print("[SRT] ffmpeg pipe koptu.")
                break

    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        proc.wait()
        print("[SRT] ffmpeg kapandi.")


def main():
    global running

    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=infer_loop, daemon=True),
        threading.Thread(target=metadata_udp_sender, daemon=True),
        threading.Thread(target=srt_stream_loop, daemon=True),
    ]

    for t in threads:
        t.start()

    print("[SYS] Raspberry pipeline calisiyor. Cikmak icin Ctrl+C.")
    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        time.sleep(1.0)
        print("[SYS] Durdu.")


if __name__ == "__main__":
    main()
