#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import json
import socket
import threading
import numpy as np

# Python 3.6 uyumlu TRT + CUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# =========================
# JETSON KONFIG
# =========================

# ---- AG AYARLARI ----
PC_IP = "192.168.1.50"              # <<< DEGISTIR >>> Ground Station PC IP
SRT_PORT = 9000                     # <<< DEGISTIR >>> PC tarafiyla ayni olmali
META_PORT = 5005                    # <<< DEGISTIR >>> PC tarafiyla ayni olmali

# ---- MODEL ----
ENGINE_PATH = "/home/jetson/models/seyit.engine"   # <<< DEGISTIR >>> engine tam yol

# Kamera ayarlari
WIDTH = 640                         # <<< DEGISTIR >>>
HEIGHT = 480                        # <<< DEGISTIR >>>
FPS = 30                            # <<< DEGISTIR >>>

# YOLO ayarlari
IMGSZ = 640                         # <<< DEGISTIR >>> engine export boyutu ile AYNI olmali
CONF_THRES = 0.35                   # <<< DEGISTIR >>>
IOU_THRES = 0.45                    # <<< DEGISTIR >>>
INFER_EVERY_N = 1                   # <<< DEGISTIR >>> 2/3 yaparsan daha hafif

# Stream ayarlari
STREAM_ANNOTATED = True             # <<< DEGISTIR >>> True: kutulu video
BITRATE_KBPS = 2500                 # <<< DEGISTIR >>> 1500-4000 arasi deneyin
SRT_LATENCY_MS = 100                # <<< DEGISTIR >>> 80/100/120 test et

# Kamera secimi
CAM_DEVICE = "/dev/video0"          # <<< DEGISTIR >>> /dev/video1 olabilir
USE_CSI_CAMERA = False              # <<< DEGISTIR >>> CSI kamera ise True

# Sınıf isimleri (opsiyonel)
CLASS_NAMES = []                    # <<< DEGISTIR >>> ["balloon","target"] gibi

running = True
latest_frame = None
latest_annotated = None
latest_meta = {"ts": 0.0, "detections": [], "infer_ms": 0.0}
frame_lock = threading.Lock()


# =========================
# TRT YARDIMCILAR
# =========================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def nms(boxes, scores, iou_thres):
    # boxes: [N,4] xyxy
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        iou = inter / union

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def letterbox_bgr(image, new_shape=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return out, r, (dw, dh)


class TRTModel(object):
    """
    Python 3.6 uyumlu TensorRT runner.
    Not: Output decode kısmı engine yapısına göre değişebilir.
    """
    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise RuntimeError("Engine bulunamadi: {}".format(engine_path))

        self.engine_path = engine_path
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Engine deserialize basarisiz")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context olusturulamadi")

        # TRT sürüm farklarına tolerans
        self.num_bindings = self.engine.num_bindings
        self.bindings = [None] * self.num_bindings
        self.host_inputs = []
        self.cuda_inputs = []
        self.ht_outputs = []
        self.cuda_outputs = []
        self.input_binding_idx = None
        self.output_binding_indices = []

        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Dinamik shape ise runtime'da set edeceğiz
            if is_input:
                self.input_binding_idx = i
                if -1 in shape:
                    # NCHW varsayıp dinamik boyutu set et
                    self.context.set_binding_shape(i, (1, 3, IMGSZ, IMGSZ))
                    shape = self.context.get_binding_shape(i)

                size = int(trt.volume(shape))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(cuda_mem)

                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_binding_indices.append(i)
                if -1 in shape:
                    shape = self.context.get_binding_shape(i)

                size = int(trt.volume(shape))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(cuda_mem)

                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        self.stream = cuda.Stream()
        print("[TRT] Engine yuklendi:", engine_path)
        for i in range(self.num_bindings):
            print("[TRT] binding", i, self.engine.get_binding_name(i), self.engine.get_binding_shape(i),
                  "input" if self.engine.binding_is_input(i) else "output")

    def preprocess(self, bgr):
        img, ratio, (dw, dh) = letterbox_bgr(bgr, new_shape=IMGSZ)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, 0)        # NCHW
        return x, ratio, dw, dh

    def infer_raw(self, input_tensor):
        np.copyto(self.host_inputs[0], input_tensor.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for i in range(len(self.cuda_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
        self.stream.synchronize()

        outs = []
        for out_idx, bind_idx in enumerate(self.output_binding_indices):
            shape = self.context.get_binding_shape(bind_idx)
            arr = np.array(self.host_outputs[out_idx]).reshape(tuple(shape))
            outs.append(arr)
        return outs

    def postprocess(self, outs, orig_w, orig_h, ratio, dw, dh):
        """
        !!! KRITIK !!!
        Bu fonksiyon engine output formatina gore ayarlanmalidir.

        Varsayim-1 (yaygin): output[0] -> [1, N, 6] (x1,y1,x2,y2,conf,cls)
        Varsayim-2: output[0] -> [1,84,N] vs olabilir.

        Simdilik [1,N,6] formatini baz alan decode.
        """
        dets = []

        if len(outs) == 0:
            return dets

        out = outs[0]
        # Güvenli şekillendirme
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] >= 6:
            pred = out[0]  # [N,>=6]
        elif out.ndim == 2 and out.shape[1] >= 6:
            pred = out
        else:
            # Buraya düşüyorsa engine formatin farkli, log basip geç
            print("[TRT][WARN] Beklenmeyen output shape:", out.shape)
            return dets

        boxes = []
        scores = []
        clses = []

        for row in pred:
            conf = float(row[4])
            if conf < CONF_THRES:
                continue
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            cls = int(row[5])

            # letterbox geri dönüşümü
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1 = max(0.0, min(float(orig_w - 1), x1))
            y1 = max(0.0, min(float(orig_h - 1), y1))
            x2 = max(0.0, min(float(orig_w - 1), x2))
            y2 = max(0.0, min(float(orig_h - 1), y2))

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            clses.append(cls)

        keep = nms(boxes, scores, IOU_THRES)
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            c = clses[i]
            s = scores[i]
            dets.append({
                "cls": int(c),
                "conf": float(s),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            })

        return dets

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        x, ratio, dw, dh = self.preprocess(bgr)
        outs = self.infer_raw(x)
        dets = self.postprocess(outs, w, h, ratio, dw, dh)
        return dets


# =========================
# PIPELINE THREADS
# =========================
def draw_detections(frame, dets):
    for d in dets:
        x1 = int(d["x1"]); y1 = int(d["y1"])
        x2 = int(d["x2"]); y2 = int(d["y2"])
        cls = int(d["cls"]); conf = float(d["conf"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if CLASS_NAMES and 0 <= cls < len(CLASS_NAMES):
            label = "{} {:.2f}".format(CLASS_NAMES[cls], conf)
        else:
            label = "cls:{} {:.2f}".format(cls, conf)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


def build_jetson_capture():
    if USE_CSI_CAMERA:
        gst = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1, format=NV12 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ) % (WIDTH, HEIGHT, FPS)
    else:
        gst = (
            "v4l2src device=%s ! "
            "image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ) % (CAM_DEVICE, WIDTH, HEIGHT, FPS)
    return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)


def capture_loop():
    global latest_frame, running
    cap = build_jetson_capture()
    if not cap.isOpened():
        print("[ERR] Kamera acilamadi.")
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

    try:
        model = TRTModel(ENGINE_PATH)
    except Exception as e:
        print("[ERR] TRT model acilamadi:", e)
        running = False
        return

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
            dets = model.predict(frame)
            infer_ms = (time.time() - t0) * 1000.0

            ann = frame.copy()
            ann = draw_detections(ann, dets)

            with frame_lock:
                latest_annotated = ann
                latest_meta = {
                    "ts": time.time(),
                    "infer_ms": infer_ms,
                    "detections": dets
                }
        except Exception as e:
            print("[TRT][ERR] infer:", e)
            time.sleep(0.01)


def metadata_udp_sender():
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[META] UDP -> {}:{}".format(PC_IP, META_PORT))
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
    global running

    pipeline = (
        'appsrc is-live=true block=true format=time '
        'caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ! '
        'videoconvert ! video/x-raw,format=I420 ! '
        'nvv4l2h264enc bitrate=%d insert-sps-pps=true idrinterval=30 iframeinterval=30 ! '
        'h264parse config-interval=1 ! mpegtsmux ! '
        'srtsink uri="srt://%s:%d?mode=caller&latency=%d" '
        'wait-for-connection=false sync=false'
    ) % (WIDTH, HEIGHT, FPS, BITRATE_KBPS * 1000, PC_IP, SRT_PORT, SRT_LATENCY_MS)

    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if not out.isOpened():
        print("[ERR] SRT VideoWriter acilamadi.")
        running = False
        return

    print("[SRT] Stream basladi.")
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

    print("[SYS] Python:", os.popen("python3 --version").read().strip())
    print("[SYS] Engine:", ENGINE_PATH)

    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=infer_loop, daemon=True),
        threading.Thread(target=metadata_udp_sender, daemon=True),
        threading.Thread(target=srt_stream_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    print("[SYS] Pipeline calisiyor. Cikmak icin Ctrl+C.")
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
