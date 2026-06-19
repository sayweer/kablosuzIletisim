#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano Sender — Senkronize Bounding Box Versiyonu

Temel fark:
- Inference hangi frame'i işlediyse kutular o frame'e çizilir.
- SRT ile sadece annotated frame gönderilir.
- Kutu kayması (frame vs meta timing) minimuma iner.

Python 3.6 uyumlu.
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

# NanoStation link jitter için genelde 150-200 daha stabil olabiliyor
SRT_LATENCY_MS = 150

USE_CSI_CAMERA = False
CAM_DEVICE = "/dev/video0"

ENABLE_INFERENCE = True
ENGINE_PATH = "quad_yolov11n_jetson.engine"
IMGSZ = 640
CONF_THRES = 0.40
IOU_THRES = 0.45

MAX_INFER_FPS = 15
MIN_INFER_INTERVAL = 1.0 / MAX_INFER_FPS

# YOLO decode mod (engine formatına göre)
DECODE_MODE = "noobj"   # "noobj" veya "obj" veya "auto"
LOG_EVERY_SEC = 3.0

# =========================
# SHARED STATE
# =========================
running = True
state_lock = threading.Lock()
new_frame_event = threading.Event()

latest_raw = None
latest_raw_seq = 0

latest_annotated = None
latest_annotated_seq = 0

latest_meta = {
    "ts": 0.0, "seq": 0, "infer_ms": 0.0,
    "detections": [], "det_count": 0, "trt_ok": False
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


def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return "GStreamer: YES" in bi or "GStreamer:                   YES" in bi
    except:
        return False


# =========================
# CAMERA
# =========================
def build_camera_candidates():
    cands = []
    if USE_CSI_CAMERA:
        p = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("csi_nvargus", p))
    else:
        p1 = (
            "v4l2src device={dev} ! "
            "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("usb_mjpeg", p1))

        p2 = (
            "v4l2src device={dev} ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)
        cands.append(("usb_raw", p2))
    return cands


def open_camera():
    for name, pipe in build_camera_candidates():
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None and fr.size > 0:
                print("[CAM] Acildi:", name)
                return cap, name
        try:
            cap.release()
        except:
            pass

    cap = cv2.VideoCapture(CAM_DEVICE)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, fr = cap.read()
        if ok and fr is not None and fr.size > 0:
            print("[CAM] Acildi: opencv_v4l2")
            return cap, "opencv_v4l2"
    try:
        cap.release()
    except:
        pass
    return None, None


# =========================
# SRT WRITER
# =========================
def build_srt_writer_pipeline():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(
        PC_IP, SRT_PORT, SRT_LATENCY_MS)

    return (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 max-size-time=0 max-size-bytes=0 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
        "queue leaky=downstream max-size-buffers=2 max-size-time=0 max-size-bytes=0 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS * 1000, uri=uri)


def open_writer():
    pipe = build_srt_writer_pipeline()
    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi")
        return wr
    try:
        wr.release()
    except:
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
    top = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)); right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color), r, dw, dh


def nms_xyxy(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def clip_box(x1, y1, x2, y2, w, h):
    return (max(0.0, min(float(w-1), x1)), max(0.0, min(float(h-1), y1)),
            max(0.0, min(float(w-1), x2)), max(0.0, min(float(h-1), y2)))


def draw_dets(frame, dets, color=(0, 200, 255)):
    for d in dets:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = "c{} {:.2f}".format(d["cls"], d["conf"])
        cv2.putText(frame, txt, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


# =========================
# TRT DETECTOR
# =========================
class TRTDetector(object):
    def __init__(self, engine_path):
        if not HAS_TRT:
            raise RuntimeError("TensorRT/PyCUDA yok")
        if not os.path.isfile(engine_path):
            raise RuntimeError("Engine yok: " + engine_path)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
                break

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

    def preprocess(self, bgr):
        img, ratio, dw, dh = letterbox_bgr(bgr, IMGSZ)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return np.expand_dims(x, 0), ratio, dw, dh

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
            outs.append(np.array(self.host_out[out_i]).reshape(shape))
        return outs

    def _decode_cn(self, out_cn, orig_w, orig_h, ratio, dw, dh, use_obj):
        C, N = int(out_cn.shape[0]), int(out_cn.shape[1])
        if C < 6 or N < 1:
            return []

        xywh = out_cn[0:4, :].astype(np.float32)
        if xywh.size > 0 and float(np.max(np.abs(xywh))) <= 2.0:
            xywh *= float(IMGSZ)

        score_mat = out_cn[4:, :].astype(np.float32)

        if use_obj and score_mat.shape[0] >= 2:
            obj = score_mat[0, :]
            cls_scores = score_mat[1:, :]
            cls_ids = np.argmax(cls_scores, axis=0).astype(np.int32)
            conf = (obj * np.max(cls_scores, axis=0)).astype(np.float32)
        else:
            cls_ids = np.argmax(score_mat, axis=0).astype(np.int32)
            conf = np.max(score_mat, axis=0).astype(np.float32)

        idxs = np.where(conf >= CONF_THRES)[0]
        if idxs.size == 0:
            return []

        if idxs.size > 300:
            idxs = idxs[np.argsort(conf[idxs])[::-1][:300]]

        boxes, scores, clses = [], [], []
        for j in idxs:
            cx, cy = float(xywh[0, j]), float(xywh[1, j])
            w, h = float(xywh[2, j]), float(xywh[3, j])

            x1 = (cx - w/2.0 - dw) / ratio
            y1 = (cy - h/2.0 - dh) / ratio
            x2 = (cx + w/2.0 - dw) / ratio
            y2 = (cy + h/2.0 - dh) / ratio

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf[j]))
            clses.append(int(cls_ids[j]))

        keep = nms_xyxy(boxes, scores, IOU_THRES)
        return [{
            "cls": int(clses[i]), "conf": float(scores[i]),
            "x1": float(boxes[i][0]), "y1": float(boxes[i][1]),
            "x2": float(boxes[i][2]), "y2": float(boxes[i][3])
        } for i in keep]

    def decode_auto(self, out, orig_w, orig_h, ratio, dw, dh):
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        out = np.squeeze(out)
        if not self._shape_logged:
            print("[TRT] output shape:", out.shape)
            self._shape_logged = True

        if out.ndim != 2:
            return []

        cands = []

        def add_cand(name, dets):
            if dets:
                cands.append((name, dets, len(dets), sum(d["conf"] for d in dets)))

        r, c = int(out.shape[0]), int(out.shape[1])
        if r >= 6 and c > r:
            if DECODE_MODE in ("auto", "noobj"):
                add_cand("cn_noobj", self._decode_cn(out, orig_w, orig_h, ratio, dw, dh, False))
            if DECODE_MODE in ("auto", "obj"):
                add_cand("cn_obj", self._decode_cn(out, orig_w, orig_h, ratio, dw, dh, True))

        out_t = out.T
        rt, ct = int(out_t.shape[0]), int(out_t.shape[1])
        if rt >= 6 and ct > rt:
            if DECODE_MODE in ("auto", "noobj"):
                add_cand("t_cn_noobj", self._decode_cn(out_t, orig_w, orig_h, ratio, dw, dh, False))
            if DECODE_MODE in ("auto", "obj"):
                add_cand("t_cn_obj", self._decode_cn(out_t, orig_w, orig_h, ratio, dw, dh, True))

        if not cands:
            return []

        cands.sort(key=lambda x: (x[2], x[3]), reverse=True)
        return cands[0][1]

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        x, ratio, dw, dh = self.preprocess(bgr)
        outs = self.infer_raw(x)
        if not outs:
            return []
        return self.decode_auto(outs[0], w, h, ratio, dw, dh)


# =========================
# THREAD'LER
# =========================
def capture_loop():
    global running, latest_raw, latest_raw_seq
    cap = None
    target_interval = 1.0 / FPS

    while running:
        if cap is None:
            cap, _ = open_camera()
            if cap is None:
                time.sleep(2.0)
                continue

        t0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.release()
            except:
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

        new_frame_event.set()

        elapsed = time.time() - t0
        if target_interval - elapsed > 0.001:
            time.sleep(target_interval - elapsed)

    try:
        if cap:
            cap.release()
    except:
        pass


def infer_loop():
    global running, latest_annotated, latest_annotated_seq, latest_meta

    infer_enabled = bool(ENABLE_INFERENCE and HAS_TRT)
    detector = None
    ctx = None
    last_processed_seq = -1
    last_infer_time = 0.0

    if infer_enabled:
        try:
            cuda.init()
            ctx = cuda.Device(0).make_context()
            detector = TRTDetector(ENGINE_PATH)
            print("[TRT] Hazir.")
        except Exception as e:
            print("[TRT] Baslatilamadi:", e)
            infer_enabled = False

    while running:
        new_frame_event.wait(timeout=0.1)
        new_frame_event.clear()
        if not running:
            break

        with state_lock:
            if latest_raw is None:
                continue
            seq = int(latest_raw_seq)
            if seq == last_processed_seq:
                continue
            frame = latest_raw.copy()

        # rate limit
        now = time.time()
        wait = MIN_INFER_INTERVAL - (now - last_infer_time)
        if wait > 0.001:
            time.sleep(wait)

        dets = []
        infer_ms = 0.0

        if infer_enabled and detector is not None:
            t0 = time.time()
            try:
                dets = detector.predict(frame)
            except Exception as e:
                print("[TRT] hata:", e)
                dets = []
            infer_ms = (time.time() - t0) * 1000.0

        last_processed_seq = seq
        last_infer_time = time.time()

        if dets:
            frame = draw_dets(frame, dets)

        frame = np.ascontiguousarray(frame)

        with state_lock:
            latest_annotated = frame
            latest_annotated_seq = seq
            latest_meta = {
                "ts": time.time(), "seq": seq,
                "infer_ms": infer_ms,
                "detections": dets,
                "det_count": len(dets),
                "trt_ok": infer_enabled
            }

    try:
        if ctx:
            ctx.pop()
            ctx.detach()
    except:
        pass


def stream_loop():
    global running
    wr = None
    sent = 0
    t_log = time.time()
    target_interval = 1.0 / FPS
    last_logged_seq = -1

    while running:
        if wr is None:
            wr = open_writer()
            if wr is None:
                time.sleep(2.0)
                continue

        t0 = time.time()

        with state_lock:
            frame = latest_annotated
            seq = latest_annotated_seq

        if frame is None:
            time.sleep(0.03)
            continue

        frame_to_send = np.ascontiguousarray(frame)

        try:
            wr.write(frame_to_send)
            sent += 1
        except Exception as e:
            print("[SRT] hata:", e)
            try:
                wr.release()
            except:
                pass
            wr = None
            time.sleep(0.3)
            continue

        now = time.time()
        if now - t_log >= LOG_EVERY_SEC:
            tag = "yeni" if seq != last_logged_seq else "tekrar"
            print("[SRT] sent={} seq={} ({})".format(sent, seq, tag))
            t_log = now
            last_logged_seq = seq

        elapsed = time.time() - t0
        if target_interval - elapsed > 0.001:
            time.sleep(target_interval - elapsed)

    try:
        if wr:
            wr.release()
    except:
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
        except:
            pass
        time.sleep(0.1)

    try:
        sock.close()
    except:
        pass


def main():
    global running
    print("=" * 50)
    print("[SYS] SENKRON BBOX SENDER")
    print("[SYS] MAX_INFER_FPS:", MAX_INFER_FPS)
    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer:", opencv_has_gstreamer())
    print("[SYS] HAS_TRT:", HAS_TRT)
    print("=" * 50)

    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer yok!")
        return

    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=infer_loop, daemon=True),
        threading.Thread(target=stream_loop, daemon=True),
        threading.Thread(target=meta_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    print("[SYS] Calisiyor. Ctrl+C ile cik.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        running = False
        new_frame_event.set()
        time.sleep(1.0)
        print("[SYS] Bitti.")


if __name__ == "__main__":
    main()