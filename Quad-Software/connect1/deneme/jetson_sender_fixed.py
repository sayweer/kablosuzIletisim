#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano Sender (Python 3.6 uyumlu)
- Kamera al
- (Opsiyonel) TRT tespit
- SRT ile MPEG-TS(H264) gönder
- UDP ile metadata gönder

ÖNEMLİ FIX:
nvv4l2h264enc öncesi nvvidconv + video/x-raw(memory:NVMM),format=NV12
"""

from __future__ import print_function

import os
import cv2
import time
import json
import socket
import threading
import numpy as np

# =========================
# KONFIG (JETSON)
# =========================
PC_IP = "192.168.1.50"
SRT_PORT = 9000
META_PORT = 5005

WIDTH = 640
HEIGHT = 480
FPS = 30
BITRATE_KBPS = 2500
SRT_LATENCY_MS = 120

STREAM_ANNOTATED = True

USE_CSI_CAMERA = False
CAM_DEVICE = "/dev/video0"

ENABLE_INFERENCE = True
ENGINE_PATH = "quad_yolov11n.engine"
IMGSZ = 640
CONF_THRES = 0.35
IOU_THRES = 0.45
INFER_EVERY_N = 1
CLASS_NAMES = []

# =========================
running = True
frame_lock = threading.Lock()
latest_raw = None
latest_tx = None
latest_meta = {"ts": 0.0, "infer_ms": 0.0, "detections": [], "seq": 0}


def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False


# =========================
# Kamera
# =========================
def build_camera_candidates():
    cands = []
    if USE_CSI_CAMERA:
        cands.append((
            "csi_nvargus",
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=WIDTH, h=HEIGHT, fps=FPS))
    else:
        cands.append((
            "usb_mjpeg",
            "v4l2src device={dev} ! "
            "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS))

        cands.append((
            "usb_raw",
            "v4l2src device={dev} ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS))

    return cands


def open_camera():
    for name, pipe in build_camera_candidates():
        print("[CAM] Deneniyor:", name)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print("[CAM] Acildi:", name)
                return cap, name
        cap.release()

    print("[CAM] GStreamer kamera acilmadi, V4L2 fallback...")
    cap = cv2.VideoCapture(CAM_DEVICE)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            print("[CAM] V4L2 fallback acildi.")
            return cap, "opencv_v4l2"
    cap.release()
    return None, None


# =========================
# Writer (FIXED)
# =========================
def build_srt_writer_pipeline_nvv4l2():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(PC_IP, SRT_PORT, SRT_LATENCY_MS)

    # appsrc -> videoconvert (CPU) -> nvvidconv (NVMM) -> NV12 -> nvv4l2h264enc
    pipe = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS * 1000, uri=uri)

    return pipe


def open_writer():
    pipe = build_srt_writer_pipeline_nvv4l2()
    print("[SRT] Writer pipeline (FIXED):\n", pipe)
    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi: nvv4l2 (NVMM)")
        return wr, "ts_nvv4l2_nvmm"
    wr.release()

    # CPU fallback: x264enc (iş görür ama Jetson'ı yorabilir)
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(PC_IP, SRT_PORT, SRT_LATENCY_MS)
    pipe2 = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate={br_kbps} key-int-max=30 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br_kbps=BITRATE_KBPS, uri=uri)

    print("[SRT] NVENC acilmadi, CPU fallback deneniyor:\n", pipe2)
    wr = cv2.VideoWriter(pipe2, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi: ts_x264 (CPU)")
        return wr, "ts_x264"
    wr.release()
    return None, None


# =========================
# Threadler
# =========================
def capture_loop():
    global latest_raw, running
    cap = None

    while running:
        if cap is None:
            cap, _ = open_camera()
            if cap is None:
                print("[CAM] Kamera acilamadi, 2 sn sonra tekrar...")
                time.sleep(2.0)
                continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[CAM] Frame alinamadi, kamera yeniden aciliyor...")
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with frame_lock:
            latest_raw = frame

    if cap:
        cap.release()


def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d["x1"]); y1 = int(d["y1"])
        x2 = int(d["x2"]); y2 = int(d["y2"])
        cls = int(d["cls"]); conf = float(d["conf"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        txt = "cls:{} {:.2f}".format(cls, conf)
        cv2.putText(frame, txt, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


def infer_loop():
    global latest_tx, latest_meta, running
    seq = 0

    # İstersen TRT kısmını sonra tekrar açarsın, önce stream stabil olsun:
    infer_enabled = False  # <-- STABILITE için ilk etapta kapat

    while running:
        with frame_lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        seq += 1
        dets = []
        infer_ms = 0.0

        tx = frame
        if STREAM_ANNOTATED and dets:
            tx = draw_dets(tx, dets)

        with frame_lock:
            latest_tx = tx
            latest_meta = {
                "ts": time.time(),
                "infer_ms": float(infer_ms),
                "detections": dets,
                "seq": int(seq)
            }

        if not infer_enabled:
            time.sleep(0.001)


def stream_loop():
    global running
    wr = None
    wr_name = None

    sent = 0
    t_last = time.time()

    while running:
        if wr is None:
            wr, wr_name = open_writer()
            if wr is None:
                print("[SRT] Writer acilamadi, 2 sn sonra tekrar...")
                time.sleep(2.0)
                continue

        with frame_lock:
            frame = None if latest_tx is None else latest_tx.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        try:
            wr.write(frame)
            sent += 1
        except Exception as e:
            print("[SRT] write hata, writer reset:", e)
            try:
                wr.release()
            except Exception:
                pass
            wr = None
            time.sleep(0.5)
            continue

        now = time.time()
        if now - t_last >= 5.0:
            print("[SRT] {} frame gonderildi | writer={}".format(sent, wr_name))
            t_last = now

    if wr:
        wr.release()


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
        print("[ERR] OpenCV GStreamer destegi yok. Jetson'da gstreamer'li opencv lazim.")
        return

    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=infer_loop, daemon=True),
        threading.Thread(target=stream_loop, daemon=True),
        threading.Thread(target=meta_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    print("[SYS] Calisiyor. Cikmak icin Ctrl+C")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        running = False
        time.sleep(1.0)
        print("[SYS] Program sonlandi.")


if __name__ == "__main__":
    main()
