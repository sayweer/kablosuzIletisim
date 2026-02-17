#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano Sender
- Kamera al
- (Opsiyonel) TensorRT ile tespit yap
- SRT ile videoyu PC'ye gönder
- UDP ile metadata gönder
- (Opsiyonel) MAVLink'i FC'den alıp PC'ye forward et

Python 3.6 uyumlu
"""

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

# ---- AG ----
PC_IP = "192.168.1.50"              # <<< DEGISTIR >>> Yer istasyonu PC IP
SRT_PORT = 9000                     # <<< DEGISTIR >>> PC script ile ayni
META_PORT = 5005                    # <<< DEGISTIR >>> PC script ile ayni
PC_MAVLINK_UDP_PORT = 14550         # <<< DEGISTIR >>> PC script ile ayni

# ---- VIDEO ----
WIDTH = 640                         # <<< DEGISTIR >>>
HEIGHT = 480                        # <<< DEGISTIR >>>
FPS = 30                            # <<< DEGISTIR >>>
BITRATE_KBPS = 2500                 # <<< DEGISTIR >>> 1500-4000 dene
SRT_LATENCY_MS = 120                # <<< DEGISTIR >>> 80/100/120 test et
STREAM_ANNOTATED = True             # True ise kutulu goruntu gider

# ---- KAMERA ----
USE_CSI_CAMERA = False              # <<< DEGISTIR >>> CSI ise True
CAM_DEVICE = "/dev/video0"          # <<< DEGISTIR >>> USB kamera device

# ---- INFERENCE ----
ENABLE_INFERENCE = True             # <<< DEGISTIR >>> test icin False yapabilirsin
ENGINE_PATH = "quad_yolov11n.engine"  # <<< DEGISTIR >>> engine dosya yolu
IMGSZ = 640                         # <<< DEGISTIR >>> engine export size ile ayni
CONF_THRES = 0.35                   # <<< DEGISTIR >>>
IOU_THRES = 0.45                    # <<< DEGISTIR >>>
INFER_EVERY_N = 1                   # <<< DEGISTIR >>> 2/3 yaparsan hafifler
CLASS_NAMES = []                    # <<< DEGISTIR >>> ["balloon","target"] gibi

# ---- MAVLINK FORWARD (opsiyonel) ----
ENABLE_MAVLINK_FORWARD = False      # <<< DEGISTIR >>> FC Jetson'a bagliysa True
FC_MAVLINK_IN = "/dev/ttyACM0"      # <<< DEGISTIR >>> "/dev/ttyACM0" veya "udp:0.0.0.0:14551"
FC_MAVLINK_BAUD = 57600             # <<< DEGISTIR >>>

# =========================
# GLOBAL STATE
# =========================
running = True
frame_lock = threading.Lock()
latest_raw = None
latest_tx = None
latest_meta = {
    "ts": 0.0,
    "infer_ms": 0.0,
    "detections": [],
    "seq": 0
}

# =========================
# YARDIMCI
# =========================

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False


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


def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return out, r, (dw, dh)


def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d["x1"]); y1 = int(d["y1"])
        x2 = int(d["x2"]); y2 = int(d["y2"])
        cls = int(d["cls"]); conf = float(d["conf"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if CLASS_NAMES and 0 <= cls < len(CLASS_NAMES):
            txt = "{} {:.2f}".format(CLASS_NAMES[cls], conf)
        else:
            txt = "cls:{} {:.2f}".format(cls, conf)

        cv2.putText(frame, txt, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


# =========================
# TRT DETECTOR (Opsiyonel)
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    HAS_TRT = True
except Exception as e:
    print("[TRT] TensorRT/PyCUDA import yok -> inference kapanacak:", e)


class TRTDetector(object):
    def __init__(self, engine_path):
        if not HAS_TRT:
            raise RuntimeError("TensorRT/PyCUDA yok")

        if not os.path.isfile(engine_path):
            raise RuntimeError("Engine dosyasi yok: {}".format(engine_path))

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Engine deserialize fail")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution context fail")

        self.num_bindings = self.engine.num_bindings
        self.bindings = [None] * self.num_bindings
        self.host_in = []
        self.dev_in = []
        self.host_out = []
        self.dev_out = []
        self.out_bind_idxs = []

        # Input shape set (dinamik destek)
        self.input_idx = None
        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
                break
        if self.input_idx is None:
            raise RuntimeError("Input binding bulunamadi")

        in_shape = tuple(self.context.get_binding_shape(self.input_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.input_idx, (1, 3, IMGSZ, IMGSZ))
            in_shape = tuple(self.context.get_binding_shape(self.input_idx))

        # Binding alloc
        for i in range(self.num_bindings):
            is_input = self.engine.binding_is_input(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))

            # Hala dinamikse güvenli fallback
            if -1 in shape:
                shape = tuple([1 if d < 0 else int(d) for d in shape])

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[i] = int(dev_mem)
            if is_input:
                self.host_in.append(host_mem)
                self.dev_in.append(dev_mem)
            else:
                self.out_bind_idxs.append(i)
                self.host_out.append(host_mem)
                self.dev_out.append(dev_mem)

        self.stream = cuda.Stream()
        print("[TRT] Engine yüklendi:", engine_path)
        for i in range(self.num_bindings):
            print("[TRT] binding {} {} shape={} {}".format(
                i,
                self.engine.get_binding_name(i),
                self.context.get_binding_shape(i),
                "INPUT" if self.engine.binding_is_input(i) else "OUTPUT"
            ))

    def preprocess(self, bgr):
        img, ratio, (dw, dh) = letterbox_bgr(bgr, new_shape=IMGSZ)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, 0)        # NCHW
        return x, ratio, dw, dh

    def infer_raw(self, x):
        np.copyto(self.host_in[0], x.ravel())

        cuda.memcpy_htod_async(self.dev_in[0], self.host_in[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for i in range(len(self.dev_out)):
            cuda.memcpy_dtoh_async(self.host_out[i], self.dev_out[i], self.stream)

        self.stream.synchronize()

        outs = []
        for out_i, bind_i in enumerate(self.out_bind_idxs):
            shape = tuple(self.context.get_binding_shape(bind_i))
            if -1 in shape:
                # Güvenli fallback
                shape = tuple([1 if d < 0 else int(d) for d in shape])
            arr = np.array(self.host_out[out_i]).reshape(shape)
            outs.append(arr)
        return outs

    def decode(self, outs, orig_w, orig_h, ratio, dw, dh):
        # İki format destek:
        # 1) [1, N, 6] -> x1,y1,x2,y2,conf,cls
        # 2) [1, 84, N] veya [84, N] -> x,y,w,h + class scores
        dets = []
        if len(outs) == 0:
            return dets

        out = np.array(outs[0])
        out = np.squeeze(out)

        boxes = []
        scores = []
        clses = []

        # Format-1: [N,6+]
        if out.ndim == 2 and out.shape[1] >= 6:
            pred = out
            for row in pred:
                conf = float(row[4])
                if conf < CONF_THRES:
                    continue
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                cls = int(row[5])

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

        # Format-2: [84,N] benzeri
        elif out.ndim == 2 and out.shape[0] >= 6 and out.shape[1] > 6:
            # [C,N] varsayalim
            # ilk 4: cx,cy,w,h
            # 4..: class score
            xywh = out[0:4, :]            # [4,N]
            cls_scores = out[4:, :]       # [num_cls,N]

            # top-k ile hafiflet
            confs = np.max(cls_scores, axis=0)
            cls_ids = np.argmax(cls_scores, axis=0)

            # düşük conf ele
            idxs = np.where(confs >= CONF_THRES)[0]
            if idxs.size > 300:
                # en yüksek 300
                top = np.argsort(confs[idxs])[::-1][:300]
                idxs = idxs[top]

            for j in idxs:
                conf = float(confs[j])
                cls = int(cls_ids[j])

                cx = float(xywh[0, j]); cy = float(xywh[1, j])
                w = float(xywh[2, j]); h = float(xywh[3, j])

                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

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

        else:
            print("[TRT][WARN] Desteklenmeyen output shape:", out.shape)
            return dets

        keep = nms_xyxy(boxes, scores, IOU_THRES)
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
        dets = self.decode(outs, w, h, ratio, dw, dh)
        return dets


# =========================
# CAMERA
# =========================

def build_camera_candidates():
    cands = []
    if USE_CSI_CAMERA:
        # CSI Kamera (Jetson)
        cands.append((
            "csi_nvargus",
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=WIDTH, h=HEIGHT, fps=FPS))
    else:
        # USB MJPEG
        cands.append((
            "usb_mjpeg",
            "v4l2src device={dev} ! "
            "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS))

        # USB RAW (MJPEG degilse bu tutar)
        cands.append((
            "usb_raw",
            "v4l2src device={dev} ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS))

    return cands


def open_camera():
    cands = build_camera_candidates()
    for name, pipe in cands:
        print("[CAM] Deneniyor:", name)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print("[CAM] Acildi:", name)
                return cap, name
        cap.release()

    # En son OpenCV V4L2 fallback
    print("[CAM] GStreamer kamera acilmadi, V4L2 fallback deneniyor...")
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
# STREAM WRITER
# =========================

def build_srt_writer_candidates():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(PC_IP, SRT_PORT, SRT_LATENCY_MS)

    base = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
    ).format(w=WIDTH, h=HEIGHT, fps=FPS)

    cands = []

    # 1) Jetson HW encoder (en iyi)
    cands.append((
        "ts_nvv4l2",
        base +
        "nvv4l2h264enc bitrate={br} insert-sps-pps=1 idrinterval=30 iframeinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
        .format(br=BITRATE_KBPS * 1000, uri=uri)
    ))

    # 2) omxh264enc (eski jetpack fallback)
    cands.append((
        "ts_omx",
        base +
        "omxh264enc bitrate={br} control-rate=2 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
        .format(br=BITRATE_KBPS * 1000, uri=uri)
    ))

    # 3) x264enc (CPU fallback)
    cands.append((
        "ts_x264",
        base +
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate={br_kbps} key-int-max=30 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
        .format(br_kbps=BITRATE_KBPS, uri=uri)
    ))

    # 4) RAW H264 over SRT (TS demux sorunu olursa)
    cands.append((
        "raw_x264",
        base +
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate={br_kbps} key-int-max=30 ! "
        "h264parse config-interval=1 ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
        .format(br_kbps=BITRATE_KBPS, uri=uri)
    ))

    return cands


def open_writer():
    for name, pipe in build_srt_writer_candidates():
        print("[SRT] Writer deneniyor:", name)
        print("[SRT] Pipeline:", pipe)
        wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
        if wr.isOpened():
            print("[SRT] Writer acildi:", name)
            return wr, name
        wr.release()
    return None, None


# =========================
# THREADS
# =========================

def capture_loop():
    global latest_raw, running

    cap = None
    cam_name = None

    while running:
        if cap is None:
            cap, cam_name = open_camera()
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
    print("[CAM] Thread kapandi.")


def infer_loop():
    global latest_tx, latest_meta, running

    detector = None
    infer_enabled = ENABLE_INFERENCE and HAS_TRT
    ctx = None
    seq = 0

    if infer_enabled:
        try:
            cuda.init()
            dev = cuda.Device(0)
            ctx = dev.make_context()

            print("[TRT] Model yukleniyor...")
            detector = TRTDetector(ENGINE_PATH)
            print("[TRT] Model hazir.")
        except Exception as e:
            print("[TRT] Acilamadi, inference kapanacak:", e)
            infer_enabled = False
            detector = None
            if ctx is not None:
                try:
                    ctx.pop()
                except Exception:
                    pass
                ctx = None

    while running:
        with frame_lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        seq += 1
        dets = []
        infer_ms = 0.0

        if infer_enabled and detector is not None and (seq % INFER_EVERY_N == 0):
            t0 = time.time()
            try:
                dets = detector.predict(frame)
            except Exception as e:
                print("[TRT] infer hata:", e)
                dets = []
            infer_ms = (time.time() - t0) * 1000.0

        tx = frame.copy()
        if STREAM_ANNOTATED and len(dets) > 0:
            tx = draw_dets(tx, dets)

        with frame_lock:
            latest_tx = tx
            latest_meta = {
                "ts": time.time(),
                "infer_ms": float(infer_ms),
                "detections": dets,
                "seq": int(seq)
            }

        # inference kapaliysa da stream aksin
        if not infer_enabled:
            time.sleep(0.001)

    if ctx is not None:
        try:
            ctx.pop()
        except Exception:
            pass
    print("[TRT] Thread kapandi.")


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
            print("[SRT] write hata, writer yeniden aciliyor:", e)
            try:
                wr.release()
            except Exception:
                pass
            wr = None
            time.sleep(0.5)
            continue

        # log
        now = time.time()
        if now - t_last >= 5.0:
            print("[SRT] {} frame gonderildi | writer={}".format(sent, wr_name))
            t_last = now

    if wr:
        wr.release()
    print("[SRT] Thread kapandi.")


def meta_loop():
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[META] UDP -> {}:{}".format(PC_IP, META_PORT))

    while running:
        with frame_lock:
            payload = dict(latest_meta)

        try:
            msg = json.dumps(payload).encode("utf-8")
            sock.sendto(msg, (PC_IP, META_PORT))
        except Exception:
            pass

        time.sleep(0.03)  # ~33Hz

    sock.close()
    print("[META] Thread kapandi.")


def mavlink_forward_loop():
    global running
    if not ENABLE_MAVLINK_FORWARD:
        print("[MAVFWD] Disabled.")
        return

    try:
        from pymavlink import mavutil
    except Exception as e:
        print("[MAVFWD] pymavlink import yok, kapandi:", e)
        return

    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while running:
        master = None
        try:
            print("[MAVFWD] FC baglaniyor:", FC_MAVLINK_IN)
            master = mavutil.mavlink_connection(
                FC_MAVLINK_IN,
                baud=FC_MAVLINK_BAUD,
                autoreconnect=True
            )
            print("[MAVFWD] Heartbeat bekleniyor...")
            master.wait_heartbeat(timeout=10)
            print("[MAVFWD] FC baglandi. PC'ye forward basladi.")

            while running:
                msg = master.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                try:
                    raw = msg.get_msgbuf()
                    if raw:
                        out_sock.sendto(raw, (PC_IP, PC_MAVLINK_UDP_PORT))
                except Exception:
                    pass

        except Exception as e:
            print("[MAVFWD] Hata, yeniden denenecek:", e)
            time.sleep(2.0)
        finally:
            try:
                if master is not None:
                    master.close()
            except Exception:
                pass

    out_sock.close()
    print("[MAVFWD] Thread kapandi.")


def main():
    global running

    print("[SYS] Python:", os.popen("python3 --version").read().strip())
    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer destek:", opencv_has_gstreamer())
    print("[SYS] PC_IP:", PC_IP, "SRT:", SRT_PORT, "META:", META_PORT)

    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok. Jetson'da apt opencv kullan.")
        return

    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=infer_loop, daemon=True),
        threading.Thread(target=stream_loop, daemon=True),
        threading.Thread(target=meta_loop, daemon=True),
        threading.Thread(target=mavlink_forward_loop, daemon=True),
    ]

    for t in threads:
        t.start()

    print("[SYS] Calisiyor. Cikmak icin Ctrl+C")

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        time.sleep(1.0)
        print("[SYS] Program sonlandi.")


if __name__ == "__main__":
    main()
