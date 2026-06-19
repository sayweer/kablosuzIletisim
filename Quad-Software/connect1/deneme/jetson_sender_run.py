#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, cv2, time, json, socket, threading
import numpy as np

# ========= KONFIG (JETSON) =========
PC_IP = "192.168.1.50"
SRT_PORT = 9000
META_PORT = 5005

WIDTH  = 640
HEIGHT = 480
FPS    = 30
BITRATE_KBPS = 2500
SRT_LATENCY_MS = 120

USE_CSI_CAMERA = False
CAM_DEVICE = "/dev/video0"

# ========= GLOBAL =========
running = True
frame_lock = threading.Lock()
latest_tx = None
latest_meta = {"ts":0.0, "infer_ms":0.0, "detections":[], "seq":0}

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_camera_candidates():
    cands = []
    if USE_CSI_CAMERA:
        pipe = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("csi_nvargus", pipe))
    else:
        pipe1 = (
            "v4l2src device={dev} ! "
            "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("usb_mjpeg", pipe1))

        pipe2 = (
            "v4l2src device={dev} ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("usb_raw", pipe2))
    return cands

def open_camera():
    for name, pipe in build_camera_candidates():
        print("[CAM] Deneniyor:", name)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None and fr.size > 0:
                print("[CAM] Acildi:", name)
                return cap, name
        cap.release()

    print("[CAM] GStreamer acilmadi, V4L2 fallback...")
    cap = cv2.VideoCapture(CAM_DEVICE)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, fr = cap.read()
        if ok and fr is not None and fr.size > 0:
            print("[CAM] V4L2 fallback acildi.")
            return cap, "opencv_v4l2"
    cap.release()
    return None, None

def build_writer_pipeline():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(PC_IP, SRT_PORT, SRT_LATENCY_MS)
    # FIX: nvvidconv + NVMM + NV12 + mux sonrası queue + wait-for-connection=true
    pipe = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! queue ! "
        "srtsink uri=\"{uri}\" wait-for-connection=true sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS*1000, uri=uri)
    return pipe

def open_writer():
    pipe = build_writer_pipeline()
    print("[SRT] Writer pipeline:\n", pipe)
    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi (NVENC/NVMM)")
        return wr
    wr.release()
    return None

def capture_loop():
    global running, latest_tx, latest_meta
    cap = None
    seq = 0
    try:
        while running:
            if cap is None:
                cap, _ = open_camera()
                if cap is None:
                    print("[CAM] Kamera yok, 2sn sonra...")
                    time.sleep(2.0)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[CAM] Frame yok, reset...")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            frame = np.ascontiguousarray(frame)  # KRİTİK

            seq += 1
            with frame_lock:
                latest_tx = frame
                latest_meta = {"ts": time.time(), "infer_ms": 0.0, "detections": [], "seq": int(seq)}

    except Exception as e:
        print("[CAM][FATAL] crash:", e)
        running = False
    finally:
        try:
            if cap: cap.release()
        except Exception:
            pass

def stream_loop():
    global running
    wr = None
    sent = 0
    t0 = time.time()

    while running:
        if wr is None:
            wr = open_writer()
            if wr is None:
                print("[SRT] Writer acilamadi, 2sn sonra...")
                time.sleep(2.0)
                continue

        with frame_lock:
            frame = None if latest_tx is None else latest_tx.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = np.ascontiguousarray(frame)

        try:
            wr.write(frame)
            sent += 1
        except Exception as e:
            print("[SRT] write hata, reset:", e)
            try: wr.release()
            except Exception: pass
            wr = None
            time.sleep(0.5)
            continue

        if time.time() - t0 > 5.0:
            print("[SRT] sent={} frame".format(sent))
            t0 = time.time()

    try:
        if wr: wr.release()
    except Exception:
        pass

def meta_loop():
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[META] UDP -> {}:{}".format(PC_IP, META_PORT))
    while running:
        with frame_lock:
            payload = dict(latest_meta)
        try:
            sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))
        except Exception:
            pass
        time.sleep(0.03)
    sock.close()

def main():
    global running
    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer:", opencv_has_gstreamer())
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer yok. Jetson'da apt opencv kullanman gerek.")
        return

    ths = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=stream_loop, daemon=True),
        threading.Thread(target=meta_loop, daemon=True),
    ]
    for t in ths: t.start()

    print("[SYS] Calisiyor. Cikmak icin Ctrl+C")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        running = False
        time.sleep(1.0)
        print("[SYS] Bitti.")

if __name__ == "__main__":
    main()
