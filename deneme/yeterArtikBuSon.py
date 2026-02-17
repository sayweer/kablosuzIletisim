#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano Sender (Python 3.6)
- Kamera al (GStreamer / fallback V4L2)
- TensorRT engine ile tespit
- SRT (MPEG-TS/H264) video gönder
- UDP ile metadata gönder

Stabilite notları:
1) build_camera_candidates tuple.format hatası yok (format sadece string'e uygulanıyor)
2) TRT output decode:
   - (1,7,8400) / (C,N) YOLO-like -> conf = obj * max(cls_scores)
   - (N,6+) xyxy+conf+cls
   - (N,C) ve (C,N) iki yönü de dener
3) Inference çökse bile stream devam eder
4) STREAM_ANNOTATED=True ise kutuları Jetson üstünde çizer
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

# Senin isteğin: kutuları Jetson çizsin
STREAM_ANNOTATED = True

USE_CSI_CAMERA = False
CAM_DEVICE = "/dev/video0"

ENABLE_INFERENCE = True
ENGINE_PATH = "quad_yolov11n.engine"
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
INFER_EVERY_N = 1          # 2 veya 3 yaparsan Nano rahatlar
DET_STALE_SEC = 0.35       # eski detections bu süreden sonra çizilmez

# =========================
# GLOBALS
# =========================
running = True
state_lock = threading.Lock()

latest_raw = None
latest_raw_seq = 0

latest_dets = []
latest_det_ts = 0.0
latest_infer_ms = 0.0

latest_meta = {
    "ts": 0.0,
    "seq": 0,
    "infer_ms": 0.0,
    "detections": [],
    "trt_ok": False
}

# =========================
# TRT IMPORT
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    HAS_TRT = True
except Exception as e:
    print("[TRT] TensorRT/PyCUDA import yok:", e)
    HAS_TRT = False


def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False


# =========================
# CAMERA
# =========================
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, fr = cap.read()
        if ok and fr is not None and fr.size > 0:
            print("[CAM] V4L2 fallback acildi.")
            return cap, "opencv_v4l2"
    cap.release()
    return None, None


# =========================
# SRT WRITER (NVENC)
# =========================
def build_srt_writer_pipeline():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(
        PC_IP, SRT_PORT, SRT_LATENCY_MS
    )

    # wait-for-connection=false -> PC yokken bile sender kilitlenmez
    pipe = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! queue ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS * 1000, uri=uri)

    return pipe


def open_writer():
    pipe = build_srt_writer_pipeline()
    print("[SRT] Writer pipeline:\n", pipe)

    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi (NVENC/NVMM)")
        return wr

    try:
        wr.release()
    except Exception:
        pass
    return None


# =========================
# HELPERS
# =========================
def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = float(new_shape[1] - new_unpad[0]) / 2.0
    dh = float(new_shape[0] - new_unpad[1]) / 2.0

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return out, r, dw, dh


def nms_xyxy(boxes, scores, iou_thres):
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
        i = int(order[0])
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


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return x1, y1, x2, y2


def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d["x1"])
        y1 = int(d["y1"])
        x2 = int(d["x2"])
        y2 = int(d["y2"])
        cls = int(d["cls"])
        conf = float(d["conf"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        txt = "cls:{} {:.2f}".format(cls, conf)
        cv2.putText(
            frame,
            txt,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1
        )
    return frame


# =========================
# TRT DETECTOR
# =========================
class TRTDetector(object):
    def __init__(self, engine_path):
        if not HAS_TRT:
            raise RuntimeError("TensorRT/PyCUDA yok")
        if not os.path.isfile(engine_path):
            raise RuntimeError("Engine yok: {}".format(engine_path))

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Engine deserialize fail")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Context create fail")

        self.input_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
                break
        if self.input_idx is None:
            raise RuntimeError("Input binding yok")

        in_shape = tuple(self.context.get_binding_shape(self.input_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.input_idx, (1, 3, IMGSZ, IMGSZ))

        self.bindings = [None] * self.engine.num_bindings
        self.host_in = None
        self.dev_in = None

        self.host_out = []
        self.dev_out = []
        self.out_bind_idxs = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            if -1 in shape:
                shape = tuple([1 if d < 0 else int(d) for d in shape])

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[i] = int(dev_mem)

            if self.engine.binding_is_input(i):
                self.host_in = host_mem
                self.dev_in = dev_mem
            else:
                self.out_bind_idxs.append(i)
                self.host_out.append(host_mem)
                self.dev_out.append(dev_mem)

        self.stream = cuda.Stream()
        self._shape_logged = False

        print("[TRT] Engine yuklendi:", engine_path)
        for i in range(self.engine.num_bindings):
            print("[TRT] binding", i, self.engine.get_binding_name(i), self.context.get_binding_shape(i))

    def preprocess(self, bgr):
        img, ratio, dw, dh = letterbox_bgr(bgr, IMGSZ)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x, ratio, dw, dh

    def infer_raw(self, x):
        np.copyto(self.host_in, x.ravel())
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for i in range(len(self.dev_out)):
            cuda.memcpy_dtoh_async(self.host_out[i], self.dev_out[i], self.stream)
        self.stream.synchronize()

        outs = []
        for out_i, bind_i in enumerate(self.out_bind_idxs):
            shape = tuple(self.context.get_binding_shape(bind_i))
            if -1 in shape:
                shape = tuple([1 if d < 0 else int(d) for d in shape])
            arr = np.array(self.host_out[out_i]).reshape(shape)
            outs.append(arr)
        return outs

    def _decode_cn(self, out_cn, orig_w, orig_h, ratio, dw, dh):
        # out_cn: (C, N)
        C = int(out_cn.shape[0])
        N = int(out_cn.shape[1])

        if C < 6:
            return []

        xywh = out_cn[0:4, :].astype(np.float32)

        # normalize mi pixel mi? kaba ama pratik:
        # eğer max küçükse (<=2) normalize kabul edip IMGSZ ile çarp.
        max_xywh = float(np.max(np.abs(xywh))) if xywh.size > 0 else 0.0
        if max_xywh <= 2.0:
            xywh = xywh * float(IMGSZ)

        obj = out_cn[4, :].astype(np.float32)

        if C > 5:
            cls_scores = out_cn[5:, :].astype(np.float32)
            cls_ids = np.argmax(cls_scores, axis=0).astype(np.int32)
            cls_best = np.max(cls_scores, axis=0).astype(np.float32)
            conf = obj * cls_best
        else:
            cls_ids = np.zeros((N,), dtype=np.int32)
            conf = obj

        idxs = np.where(conf >= CONF_THRES)[0]
        if idxs.size == 0:
            return []

        if idxs.size > 300:
            top = np.argsort(conf[idxs])[::-1][:300]
            idxs = idxs[top]

        boxes = []
        scores = []
        clses = []

        for j in idxs:
            cx = float(xywh[0, j])
            cy = float(xywh[1, j])
            w = float(xywh[2, j])
            h = float(xywh[3, j])

            x1 = cx - (w / 2.0)
            y1 = cy - (h / 2.0)
            x2 = cx + (w / 2.0)
            y2 = cy + (h / 2.0)

            # letterbox geri çözüm
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf[j]))
            clses.append(int(cls_ids[j]))

        keep = nms_xyxy(boxes, scores, IOU_THRES)

        dets = []
        for i in keep:
            b = boxes[i]
            dets.append({
                "cls": int(clses[i]),
                "conf": float(scores[i]),
                "x1": float(b[0]),
                "y1": float(b[1]),
                "x2": float(b[2]),
                "y2": float(b[3])
            })
        return dets

    def _decode_n6(self, out_n6, orig_w, orig_h, ratio, dw, dh):
        # out_n6: (N, 6+) -> x1,y1,x2,y2,conf,cls
        arr = out_n6.astype(np.float32)
        if arr.shape[1] < 6:
            return []

        # normalize mi?
        max_xy = float(np.max(np.abs(arr[:, :4]))) if arr.shape[0] > 0 else 0.0
        if max_xy <= 2.0:
            arr[:, :4] *= float(IMGSZ)

        boxes = []
        scores = []
        clses = []

        for i in range(arr.shape[0]):
            conf = float(arr[i, 4])
            if conf < CONF_THRES:
                continue

            x1 = float(arr[i, 0]); y1 = float(arr[i, 1])
            x2 = float(arr[i, 2]); y2 = float(arr[i, 3])
            cls = int(arr[i, 5]) if arr.shape[1] > 5 else 0

            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            clses.append(cls)

        keep = nms_xyxy(boxes, scores, IOU_THRES)

        dets = []
        for k in keep:
            b = boxes[k]
            dets.append({
                "cls": int(clses[k]),
                "conf": float(scores[k]),
                "x1": float(b[0]),
                "y1": float(b[1]),
                "x2": float(b[2]),
                "y2": float(b[3])
            })
        return dets

    def decode_auto(self, out, orig_w, orig_h, ratio, dw, dh):
        # out: squeezed tek output
        # olasılar:
        # (C,N) -> YOLO-like
        # (N,C) -> YOLO-like veya N,6
        # (N,6+) -> xyxy conf cls
        if out.ndim != 2:
            return []

        r, c = int(out.shape[0]), int(out.shape[1])

        # 1) (C,N) aday: C küçük, N büyük
        if r <= 128 and c > r:
            # C,N
            if r >= 6:
                dets = self._decode_cn(out, orig_w, orig_h, ratio, dw, dh)
                if len(dets) > 0:
                    return dets

        # 2) (N,6+) aday
        if c >= 6 and r > 0:
            dets = self._decode_n6(out, orig_w, orig_h, ratio, dw, dh)
            if len(dets) > 0:
                return dets

        # 3) transpose deneyelim
        out_t = out.T
        rt, ct = int(out_t.shape[0]), int(out_t.shape[1])

        if rt <= 128 and ct > rt and rt >= 6:
            dets = self._decode_cn(out_t, orig_w, orig_h, ratio, dw, dh)
            if len(dets) > 0:
                return dets

        if ct >= 6 and rt > 0:
            dets = self._decode_n6(out_t, orig_w, orig_h, ratio, dw, dh)
            if len(dets) > 0:
                return dets

        return []

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        x, ratio, dw, dh = self.preprocess(bgr)
        outs = self.infer_raw(x)
        if not outs:
            return []

        out = np.squeeze(np.array(outs[0]))

        if not self._shape_logged:
            print("[TRT] output shape:", out.shape)
            self._shape_logged = True

        return self.decode_auto(out, w, h, ratio, dw, dh)


# =========================
# THREADS
# =========================
def capture_loop():
    global running, latest_raw, latest_raw_seq, latest_meta
    cap = None

    while running:
        if cap is None:
            cap, _ = open_camera()
            if cap is None:
                print("[CAM] Kamera acilmadi, 2sn sonra...")
                time.sleep(2.0)
                continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[CAM] Frame yok, kamera reset...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(0.4)
            continue

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        frame = np.ascontiguousarray(frame)

        with state_lock:
            latest_raw = frame
            latest_raw_seq += 1
            if latest_meta["seq"] == 0:
                latest_meta = {
                    "ts": time.time(),
                    "seq": int(latest_raw_seq),
                    "infer_ms": 0.0,
                    "detections": [],
                    "trt_ok": False
                }

    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass


def infer_loop():
    global running, latest_dets, latest_det_ts, latest_infer_ms, latest_meta

    infer_enabled = bool(ENABLE_INFERENCE and HAS_TRT)
    trt_ok = False
    detector = None
    ctx = None

    frame_counter = 0

    if infer_enabled:
        try:
            cuda.init()
            ctx = cuda.Device(0).make_context()
            detector = TRTDetector(ENGINE_PATH)
            trt_ok = True
            print("[TRT] Hazir.")
        except Exception as e:
            print("[TRT] Baslatilamadi, inference kapaniyor:", e)
            infer_enabled = False
            trt_ok = False
            detector = None
            try:
                if ctx is not None:
                    ctx.pop()
                    ctx.detach()
            except Exception:
                pass
            ctx = None

    while running:
        with state_lock:
            frame = None if latest_raw is None else latest_raw.copy()
            seq = int(latest_raw_seq)

        if frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1
        run_infer = infer_enabled and detector is not None and (frame_counter % INFER_EVERY_N == 0)

        dets = []
        infer_ms = 0.0

        if run_infer:
            t0 = time.time()
            try:
                dets = detector.predict(frame)
            except Exception as e:
                # Inference patlasa da sistem yaşamaya devam
                print("[TRT] infer hata:", e)
                dets = []
            infer_ms = (time.time() - t0) * 1000.0

            with state_lock:
                latest_dets = dets
                latest_det_ts = time.time()
                latest_infer_ms = float(infer_ms)

                latest_meta = {
                    "ts": time.time(),
                    "seq": int(seq),
                    "infer_ms": float(infer_ms),
                    "detections": dets,
                    "trt_ok": bool(trt_ok)
                }
        else:
            # Inference çalışmadığı frame'lerde de meta akışı sürsün
            with state_lock:
                latest_meta = {
                    "ts": time.time(),
                    "seq": int(seq),
                    "infer_ms": float(latest_infer_ms),
                    "detections": list(latest_dets),
                    "trt_ok": bool(trt_ok)
                }

        if not infer_enabled:
            time.sleep(0.01)
        else:
            time.sleep(0.001)

    try:
        if ctx is not None:
            ctx.pop()
            ctx.detach()
    except Exception:
        pass


def stream_loop():
    global running
    wr = None
    sent = 0
    t_log = time.time()

    while running:
        if wr is None:
            wr = open_writer()
            if wr is None:
                print("[SRT] Writer acilamadi, 2sn sonra...")
                time.sleep(2.0)
                continue

        with state_lock:
            frame = None if latest_raw is None else latest_raw.copy()
            dets = list(latest_dets)
            det_ts = float(latest_det_ts)
            infer_ms = float(latest_infer_ms)

        if frame is None:
            time.sleep(0.01)
            continue

        if STREAM_ANNOTATED:
            if (time.time() - det_ts) <= DET_STALE_SEC and len(dets) > 0:
                frame = draw_dets(frame, dets)

        frame = np.ascontiguousarray(frame)

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
            time.sleep(0.3)
            continue

        now = time.time()
        if now - t_log >= 5.0:
            print("[SRT] sent={} det={} infer_ms={:.1f}".format(sent, len(dets), infer_ms))
            t_log = now

    try:
        if wr is not None:
            wr.release()
    except Exception:
        pass


def meta_loop():
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[META] UDP -> {}:{}".format(PC_IP, META_PORT))

    while running:
        with state_lock:
            payload = dict(latest_meta)
        try:
            sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))
        except Exception:
            pass
        time.sleep(0.03)

    try:
        sock.close()
    except Exception:
        pass


def main():
    global running
    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer:", opencv_has_gstreamer())
    print("[SYS] HAS_TRT:", HAS_TRT, "ENABLE_INFERENCE:", ENABLE_INFERENCE)
    print("[SYS] STREAM_ANNOTATED:", STREAM_ANNOTATED)

    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer yok. Bu haliyle calismaz.")
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
        print("[SYS] Bitti.")


if __name__ == "__main__":
    main()
