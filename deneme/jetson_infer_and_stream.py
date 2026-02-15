#!/usr/bin/env python3
import os
import cv2
import time
import json
import socket
import threading
import subprocess
from ultralytics import YOLO


# JETSON KONFIG


# ---- KENDI AGINA GORE DEGISTIR ----
PC_IP = "192.168.1.50"              # <<< DEGISTIR >>> Ground Station PC IP
SRT_PORT = 9000                     # <<< DEGISTIR >>> PC tarafiyla ayni olmali
META_PORT = 5005                    # <<< DEGISTIR >>> PC tarafiyla ayni olmali

# ---- MODEL ----
MODEL_PATH = "/home/jetson/models/seyit.pt"   # <<< DEGISTIR >>> model tam yolu

# Kamera ayarlari
WIDTH = 640                        # <<< DEGISTIR >>>
HEIGHT = 480                       # <<< DEGISTIR >>>
FPS = 30                           # <<< DEGISTIR >>>

# YOLO ayarlari
YOLO_DEVICE = 0                    # <<< DEGISTIR >>> 0=GPU(CUDA), "cpu"=CPU
IMGSZ = 640                        # <<< DEGISTIR >>> yavassa 512/416
CONF = 0.35                        # <<< DEGISTIR >>>
IOU = 0.45                         # <<< DEGISTIR >>>
INFER_EVERY_N = 1                  # <<< DEGISTIR >>> 2 yaparsan daha hafif

# Stream ayarlari
STREAM_ANNOTATED = True            # <<< DEGISTIR >>> True: kutulu video
BITRATE_KBPS = 2500                # <<< DEGISTIR >>> 1500-4000 arasi deneyin
SRT_LATENCY_MS = 100               # <<< DEGISTIR >>> 80/100/120 test et

# USB kamera (/dev/video0) için
CAM_DEVICE = "/dev/video0"         # <<< DEGISTIR >>> /dev/video1 olabilir

# CSI kamera kullanacaksan True yap (IMX219 vb.)
USE_CSI_CAMERA = False             # <<< DEGISTIR >>>

running = True
latest_frame = None
latest_annotated = None
latest_meta = {"ts": 0.0, "detections": [], "infer_ms": 0.0}
frame_lock = threading.Lock()


def build_jetson_capture():
    """
    Jetson'da kamera capture:
    - CSI ise nvarguscamerasrc
    - USB ise v4l2src
    """
    if USE_CSI_CAMERA:
        gst = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, framerate={FPS}/1, format=NV12 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=1 max-buffers=1 sync=false"
        )
    else:
        gst = (
            f"v4l2src device={CAM_DEVICE} ! "
            f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1 sync=false"
        )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    return cap


def capture_loop():
    global latest_frame, running
    cap = build_jetson_capture()

    if not cap.isOpened():
        print("[ERR] Kamera acilamadi (Jetson GStreamer capture).")
        running = False
        return

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
    print(f"[YOLO] Model yuklendi: {MODEL_PATH} | device={YOLO_DEVICE}")

    idx = 0
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        idx += 1
        if idx % INFER_EVERY_N != 0:
            continue

        t0 = time.time()
        try:
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=YOLO_DEVICE,
                verbose=False
            )
            r = results[0]
            ann = r.plot()

            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    detections.append({
                        "cls": int(b.cls[0]),
                        "conf": float(b.conf[0]),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "cx": float((x1 + x2) / 2.0),
                        "cy": float((y1 + y2) / 2.0),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
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
            payload = json.dumps(dict(latest_meta)).encode("utf-8")
        try:
            sock.sendto(payload, (PC_IP, META_PORT))
        except Exception:
            pass
        time.sleep(0.03)
    sock.close()


def srt_stream_loop():
    """
    Jetson donanim encoder (nvv4l2h264enc) ile TS + SRT
    OpenCV BGR frame -> GStreamer appsrc pipeline
    """
    global running

    pipeline = (
        f'appsrc is-live=true block=true format=time '
        f'caps=video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! '
        f'videoconvert ! video/x-raw,format=I420 ! '
        f'nvv4l2h264enc bitrate={BITRATE_KBPS*1000} insert-sps-pps=true idrinterval=30 iframeinterval=30 ! '
        f'h264parse config-interval=1 ! mpegtsmux ! '
        f'srtsink uri="srt://{PC_IP}:{SRT_PORT}?mode=caller&latency={SRT_LATENCY_MS}" '
        f'wait-for-connection=false sync=false'
    )

    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if not out.isOpened():
        print("[ERR] SRT VideoWriter acilamadi. GStreamer pluginlerini kontrol et.")
        running = False
        return

    print("[SRT] Jetson GStreamer stream basladi.")
    while running:
        with frame_lock:
            if STREAM_ANNOTATED and latest_annotated is not None:
                frame = latest_annotated.copy()
            else:
                frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        out.write(frame)

    out.release()
    print("[SRT] Stream durdu.")


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

    print("[SYS] Jetson pipeline calisiyor. Cikmak icin Ctrl+C.")
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
