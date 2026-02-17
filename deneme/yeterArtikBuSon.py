#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano Sender (Python 3.6 uyumlu)
- Kamera al (GStreamer)
- TensorRT ile tespit (engine)
- SRT ile MPEG-TS(H264) video gönder
- UDP ile metadata (detection kutuları) gönder

NOT:
- STREAM_ANNOTATED=True ise Jetson kutulu görüntüyü gönderir.
- STREAM_ANNOTATED=False ise ham görüntü gönderir, kutuları PC çizer.
"""

from __future__ import print_function
import os, cv2, time, json, socket, threading
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

STREAM_ANNOTATED = False   # <- önerim: False (PC çizsin), istersen True yap
USE_CSI_CAMERA = False
CAM_DEVICE = "/dev/video0"

ENABLE_INFERENCE = True
ENGINE_PATH = "quad_yolov11n.engine"
IMGSZ = 640
CONF_THRES = 0.35
IOU_THRES = 0.45
INFER_EVERY_N = 1  # 2/3 yaparsan Jetson rahatlar

# =========================
running = True
frame_lock = threading.Lock()
latest_raw = None
latest_tx = None
latest_meta = {"ts": 0.0, "infer_ms": 0.0, "detections": [], "seq": 0}

# =========================
# TRT import
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    HAS_TRT = True
except Exception as e:
    print("[TRT] TensorRT/PyCUDA yok:", e)
    HAS_TRT = False

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
# SRT Writer (NVENC FIX)
# =========================
def build_srt_writer_pipeline_nvv4l2():
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(PC_IP, SRT_PORT, SRT_LATENCY_MS)
    pipe = (
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! queue ! "
        "srtsink uri=\"{uri}\" wait-for-connection=true sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS * 1000, uri=uri)
    return pipe

def open_writer():
    pipe = build_srt_writer_pipeline_nvv4l2()
    print("[SRT] Writer pipeline:\n", pipe)
    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if wr.isOpened():
        print("[SRT] Writer acildi: nvv4l2 (NVMM)")
        return wr, "ts_nvv4l2_nvmm"
    wr.release()
    return None, None

# =========================
# Utils: letterbox + NMS
# =========================
def letterbox_bgr(img, new_shape=640, color=(114,114,114)):
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
    top = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)); right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh

def nms_xyxy(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def draw_dets(frame, dets):
    for d in dets:
        x1 = int(d["x1"]); y1 = int(d["y1"])
        x2 = int(d["x2"]); y2 = int(d["y2"])
        cls = int(d["cls"]); conf = float(d["conf"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.putText(frame, "cls:{} {:.2f}".format(cls, conf),
                    (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
    return frame

# =========================
# TRT Detector (YOLOv11 engine için basit decode)
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
            raise RuntimeError("Context fail")

        # input binding index
        self.input_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
                break
        if self.input_idx is None:
            raise RuntimeError("Input binding yok")

        # dynamic ise set
        in_shape = tuple(self.context.get_binding_shape(self.input_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.input_idx, (1,3,IMGSZ,IMGSZ))

        # alloc
        self.bindings = [None]*self.engine.num_bindings
        self.host_in = []
        self.dev_in  = []
        self.host_out = []
        self.dev_out  = []
        self.out_bind_idxs = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            if -1 in shape:
                shape = tuple([1 if d<0 else int(d) for d in shape])
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[i] = int(dev_mem)
            if self.engine.binding_is_input(i):
                self.host_in.append(host_mem); self.dev_in.append(dev_mem)
            else:
                self.out_bind_idxs.append(i)
                self.host_out.append(host_mem); self.dev_out.append(dev_mem)

        self.stream = cuda.Stream()
        print("[TRT] Engine yüklendi:", engine_path)

    def preprocess(self, bgr):
        img, ratio, dw, dh = letterbox_bgr(bgr, IMGSZ)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))
        x = np.expand_dims(x, 0)
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
                shape = tuple([1 if d<0 else int(d) for d in shape])
            arr = np.array(self.host_out[out_i]).reshape(shape)
            outs.append(arr)
        return outs

    def decode_yolo(self, outs, orig_w, orig_h, ratio, dw, dh):
        # Bu decode, sizin engine output'unuza bağlı.
        # Çoğu YOLO TensorRT export: (1, 84, 8400) gibi gelir.
        if not outs:
            return []
        out = np.squeeze(np.array(outs[0]))

        dets = []
        boxes = []
        scores = []
        clses = []

        if out.ndim == 2 and out.shape[0] >= 6 and out.shape[1] > 6:
            # [C,N] varsayımı, first4=xywh, rest=class scores
            xywh = out[0:4, :]
            cls_scores = out[4:, :]
            confs = np.max(cls_scores, axis=0)
            cls_ids = np.argmax(cls_scores, axis=0)

            idxs = np.where(confs >= CONF_THRES)[0]
            if idxs.size > 300:
                top = np.argsort(confs[idxs])[::-1][:300]
                idxs = idxs[top]

            for j in idxs:
                conf = float(confs[j]); cls = int(cls_ids[j])
                cx = float(xywh[0,j]); cy = float(xywh[1,j])
                w  = float(xywh[2,j]); h  = float(xywh[3,j])
                x1 = cx - w/2.0; y1 = cy - h/2.0
                x2 = cx + w/2.0; y2 = cy + h/2.0

                x1 = (x1 - dw)/ratio; y1 = (y1 - dh)/ratio
                x2 = (x2 - dw)/ratio; y2 = (y2 - dh)/ratio

                x1 = max(0.0, min(float(orig_w-1), x1))
                y1 = max(0.0, min(float(orig_h-1), y1))
                x2 = max(0.0, min(float(orig_w-1), x2))
                y2 = max(0.0, min(float(orig_h-1), y2))

                boxes.append([x1,y1,x2,y2]); scores.append(conf); clses.append(cls)

        else:
            print("[TRT][WARN] Output shape desteklenmiyor:", out.shape)
            return []

        keep = nms_xyxy(boxes, scores, IOU_THRES)
        for i in keep:
            x1,y1,x2,y2 = boxes[i]
            dets.append({
                "cls": int(clses[i]),
                "conf": float(scores[i]),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2)
            })
        return dets

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        x, ratio, dw, dh = self.preprocess(bgr)
        outs = self.infer_raw(x)
        return self.decode_yolo(outs, w, h, ratio, dw, dh)

# =========================
# Threads
# =========================
def capture_loop():
    global latest_raw, running
    cap = None
    while running:
        if cap is None:
            cap, _ = open_camera()
            if cap is None:
                print("[CAM] Kamera acilamadi, 2 sn sonra...")
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

        with frame_lock:
            latest_raw = np.ascontiguousarray(frame)
    try:
        if cap: cap.release()
    except Exception:
        pass

def infer_loop():
    global latest_tx, latest_meta, running
    seq = 0

    infer_enabled = bool(ENABLE_INFERENCE and HAS_TRT)
    detector = None
    ctx = None

    if infer_enabled:
        try:
            cuda.init()
            ctx = cuda.Device(0).make_context()
            detector = TRTDetector(ENGINE_PATH)
            print("[TRT] Hazır.")
        except Exception as e:
            print("[TRT] Açılamadı, inference kapandı:", e)
            infer_enabled = False
            detector = None
            try:
                if ctx: ctx.pop()
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

        tx = frame
        if STREAM_ANNOTATED and dets:
            tx = draw_dets(tx.copy(), dets)

        with frame_lock:
            latest_tx = np.ascontiguousarray(tx)
            latest_meta = {
                "ts": time.time(),
                "infer_ms": float(infer_ms),
                "detections": dets,
                "seq": int(seq)
            }

        if not infer_enabled:
            time.sleep(0.001)

    try:
        if ctx: ctx.pop()
    except Exception:
        pass

def stream_loop():
    global running
    wr = None
    wr_name = "none"
    sent = 0
    t_last = time.time()

    while running:
        if wr is None:
            wr, wr_name = open_writer()
            if wr is None:
                print("[SRT] Writer acilamadi, 2 sn sonra...")
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
            try:
                wr.release()
            except Exception:
                pass
            wr = None
            time.sleep(0.5)
            continue

        now = time.time()
        if now - t_last >= 5.0:
            with frame_lock:
                det_cnt = len(latest_meta.get("detections", []))
            print("[SRT] sent={} writer={} det={}".format(sent, wr_name, det_cnt))
            t_last = now

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
    try:
        sock.close()
    except Exception:
        pass

def main():
    global running
    print("[SYS] OpenCV:", cv2.__version__)
    print("[SYS] GStreamer:", opencv_has_gstreamer())
    print("[SYS] TRT:", HAS_TRT, "ENABLE_INFERENCE:", ENABLE_INFERENCE)
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer yok.")
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
