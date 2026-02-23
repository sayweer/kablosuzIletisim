#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import json
import socket
import threading
import numpy as np

cv2.setNumThreads(1)

# =========================
# AYARLAR
# =========================
PC_IP = "192.168.1.100"
SRT_PORT = 9000
META_PORT = 5005

USB_CAM_DEV = "/dev/video0"
FRAME_W = 640
FRAME_H = 480
FPS = 30

ENGINE_PATH = "model.engine"
IMGSZ = 640
CONF_THRES = 0.40
IOU_THRES = 0.45

BITRATE = 2000000
INFER_EVERY_N = 3

# SRT dayanıklılık için (Wi-Fi jitter/paket kaybında şart)
SRT_LATENCY_MS = 300

# =========================
# GLOBALLER VE KİLİTLER
# =========================
latest_frame_seq = 0
running = True

stream_lock = threading.Lock()
infer_lock = threading.Lock()

latest_stream_frame = None
latest_infer_frame = None
latest_frame_ts = 0.0

# =========================
# TENSORRT YÜKLEME
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    HAS_TRT = True
    print("[SYS] TensorRT modulleri hazir.")
except ImportError as e:
    print("[ERR] TensorRT hatasi:", e)

def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(float(new_shape) / float(h), float(new_shape) / float(w))
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = float(new_shape - new_unpad[0]) / 2.0
    dh = float(new_shape - new_unpad[1]) / 2.0
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, dw, dh

def clip_box(x1, y1, x2, y2, w, h):
    return (max(0.0, min(float(w - 1), x1)),
            max(0.0, min(float(h - 1), y1)),
            max(0.0, min(float(w - 1), x2)),
            max(0.0, min(float(h - 1), y2)))

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = np.where(union > 0, inter / union, 0.0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def draw_barcode(frame, seq_num):
    binary_str = format(seq_num, '08b')
    for i, bit in enumerate(binary_str):
        color = (255, 255, 255) if bit == '1' else (0, 0, 0)
        frame[0:20, i*20:(i+1)*20] = color
    return frame

class YOLO_TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            safe_shape = tuple(s if s > 0 else 1 for s in shape)
            size = trt.volume(safe_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'dev': dev_mem})
            else:
                self.outputs.append({'host': host_mem, 'dev': dev_mem})

    def infer(self, img):
        orig_h, orig_w = img.shape[:2]
        padded_img, ratio, dw, dh = letterbox_bgr(img, IMGSZ)

        input_image = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        np.copyto(self.inputs[0]['host'], input_image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['dev'], self.inputs[0]['host'], self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['dev'], self.stream)
        self.stream.synchronize()

        output = self.outputs[0]['host']
        num_channels = len(output) // 8400
        output = output.reshape((num_channels, 8400)).T

        boxes, scores, class_ids = [], [], []
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence > CONF_THRES:
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])

                x1 = (cx - w / 2.0 - dw) / ratio
                y1 = (cy - h / 2.0 - dh) / ratio
                x2 = (cx + w / 2.0 - dw) / ratio
                y2 = (cy + h / 2.0 - dh) / ratio

                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)

                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
                class_ids.append(class_id)

        indices = nms(boxes, scores, IOU_THRES)
        detections = []
        for i in indices:
            detections.append({
                "x1": float(boxes[i][0]), "y1": float(boxes[i][1]),
                "x2": float(boxes[i][2]), "y2": float(boxes[i][3]),
                "conf": float(scores[i]), "cls": int(class_ids[i])
            })
        return detections

# =========================
# THREAD 1: KAMERA
# =========================
def capture_thread():
    global latest_stream_frame, latest_infer_frame, latest_frame_ts, latest_frame_seq, running

    current_seq = 0
    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        ts = time.time()

        clean_frame = frame.copy()
        barcode_frame = draw_barcode(frame, current_seq)

        with stream_lock:
            latest_stream_frame = barcode_frame

        with infer_lock:
            latest_infer_frame = clean_frame
            latest_frame_ts = ts
            latest_frame_seq = current_seq

        current_seq = (current_seq + 1) % 256

# =========================
# THREAD 2: SRT VİDEO YAYINI
# =========================
def stream_thread():
    global latest_stream_frame, running

    # HATA KAYNAĞI OLAN ŞEY: zayıf timestamping + düşük SRT toleransı + parse config yok
    # ÇÖZÜM: appsrc is-live/do-timestamp/block + h264parse config-interval=1 + SRT latency
    gst_out = (
        "appsrc is-live=true do-timestamp=true block=true format=time ! "
        "video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} maxperf-enable=1 preset-level=1 control-rate=1 "
        "insert-sps-pps=true idrinterval=15 ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! "
        "queue max-size-buffers=4 leaky=downstream ! "
        "srtsink uri=srt://{ip}:{port}?mode=caller&latency={lat}&transtype=live "
        "sync=false wait-for-connection=true"
    ).format(w=FRAME_W, h=FRAME_H, fps=FPS, br=BITRATE, ip=PC_IP, port=SRT_PORT, lat=SRT_LATENCY_MS)

    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS), (FRAME_W, FRAME_H), True)
    if not out.isOpened():
        print("[ERR] GStreamer VideoWriter acilamadi! Pipeline:\n{}".format(gst_out))
        # Stream thread'i öldürme, ama boşta kal
        while running:
            time.sleep(1.0)
        return

    while running:
        frame_to_send = None
        with stream_lock:
            if latest_stream_frame is not None:
                frame_to_send = latest_stream_frame.copy()

        if frame_to_send is not None:
            out.write(frame_to_send)
        else:
            time.sleep(0.01)

# =========================
# THREAD 3: AI VE METADATA
# =========================
def inference_thread():
    global latest_infer_frame, latest_frame_ts, latest_frame_seq, running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

    SEND_MIN_INTERVAL_SEC = 0.01
    last_send_t = 0.0

    last_known_detections = []
    last_infer_ms = 0.0

    last_infer_seq = None

    if not HAS_TRT:
        print("[AI] TensorRT/PyCUDA yok. Bos metadata ile devam (trt_ok=False).")
        while running:
            with infer_lock:
                frame_ts = latest_frame_ts
                frame_seq = latest_frame_seq

            payload = {
                "ts": frame_ts,
                "seq": frame_seq,
                "infer_ms": 0.0,
                "detections": [],
                "det_count": 0,
                "trt_ok": False
            }

            now = time.time()
            if now - last_send_t >= SEND_MIN_INTERVAL_SEC:
                try:
                    sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))
                except Exception:
                    pass
                last_send_t = now

            time.sleep(0.02)
        return

    ctx = None
    yolo = None
    trt_ok = False

    try:
        try:
            cuda.init()
            ctx = cuda.Device(0).make_context()
        except Exception as e:
            print("[ERR] CUDA context kurulamadi:", e)
            while running:
                with infer_lock:
                    frame_ts = latest_frame_ts
                    frame_seq = latest_frame_seq

                payload = {
                    "ts": frame_ts,
                    "seq": frame_seq,
                    "infer_ms": 0.0,
                    "detections": [],
                    "det_count": 0,
                    "trt_ok": False
                }
                now = time.time()
                if now - last_send_t >= SEND_MIN_INTERVAL_SEC:
                    try:
                        sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))
                    except Exception:
                        pass
                    last_send_t = now
                time.sleep(0.02)
            return

        try:
            yolo = YOLO_TRT(ENGINE_PATH)
            trt_ok = True
            print("[AI] YOLO TensorRT aktif.")
        except Exception as e:
            print("[ERR] YOLO engine acilamadi (bos dets ile devam):", e)
            yolo = None
            trt_ok = False

        while running:
            frame = None
            frame_ts = 0.0
            frame_seq = 0

            with infer_lock:
                if latest_infer_frame is not None:
                    frame = latest_infer_frame
                    frame_ts = latest_frame_ts
                    frame_seq = latest_frame_seq

            if frame is None:
                time.sleep(0.01)
                continue

            do_infer = False
            if yolo is not None and (last_infer_seq != frame_seq):
                last_infer_seq = frame_seq
                if (frame_seq % INFER_EVERY_N) == 0:
                    do_infer = True

            if do_infer:
                t0 = time.time()
                try:
                    last_known_detections = yolo.infer(frame)
                    last_infer_ms = (time.time() - t0) * 1000.0
                except Exception:
                    pass

            payload = {
                "ts": frame_ts,
                "seq": frame_seq,
                "infer_ms": float(last_infer_ms),
                "detections": last_known_detections,
                "det_count": len(last_known_detections),
                "trt_ok": True if (trt_ok and yolo is not None) else False
            }

            now = time.time()
            if now - last_send_t >= SEND_MIN_INTERVAL_SEC:
                try:
                    sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))
                except Exception:
                    pass
                last_send_t = now

            time.sleep(0.005)

    finally:
        try:
            if ctx is not None:
                ctx.pop()
                try:
                    ctx.detach()
                except Exception:
                    pass
        except Exception:
            pass

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread),
        threading.Thread(target=stream_thread),
        threading.Thread(target=inference_thread),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        for t in threads:
            t.join()