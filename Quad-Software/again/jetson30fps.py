#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JETSON MOD-A (Max FPS Video)
- Kamera 30 FPS akar, video 30 FPS gönderilir.
- Inference daha seyrek (INFER_EVERY_N), kutu "son det" ile en güncel frame'e çizilir.
- Bu yüzden bazen kutu biraz geriden gelir ama video akıcıdır.

OTONOM:
- Otonom kontrol, bu kutu çizimine değil, inference çıktısına bakmalı.
- Bu kodda inference çıktısı (last_dets + timestamp) zaten ayrı state'te tutuluyor.
"""

import cv2
import time
import threading
import signal
import sys
import numpy as np

cv2.setNumThreads(1)

# -------------------------
# Ağ / Video
# -------------------------
PC_IP           = "192.168.1.50"
SRT_PORT        = 9000

USB_CAM_DEV     = "/dev/video0"
FRAME_W         = 640
FRAME_H         = 480
FPS             = 30

BITRATE         = 2000000
SRT_LATENCY_MS  = 300      # 150 düşük gecikme; saha için 300-600 daha güvenli
SRT_PAYLOADSIZE = 1316

# -------------------------
# Inference
# -------------------------
ENGINE_PATH     = "model.engine"
IMGSZ           = 640
CONF_THRES      = 0.40
IOU_THRES       = 0.45

INFER_EVERY_N   = 2        # 30 FPS -> ~15 FPS infer (ısı/kalite dengesi)
DET_MAX_AGE_S   = 0.12     # çok eski det çizilmesin (ghost/trailing azalsın)

# seq farkı kontrolü: det çok geride kaldıysa çizme
MAX_SEQ_LAG     = INFER_EVERY_N + 1

# -------------------------
# State
# -------------------------
running = True

frame_lock = threading.Lock()
latest_frame = None
latest_seq   = 0

det_lock = threading.Lock()
last_dets     = []
last_det_time = 0.0
last_det_seq  = -1
last_infer_ms = 0.0

# -------------------------
# TensorRT / CUDA
# -------------------------
HAS_TRT = False
_cuda = None
_trt  = None

try:
    import tensorrt as _trt
    import pycuda.driver as _cuda
    _cuda.init()
    HAS_TRT = True
    print("[SYS] TensorRT/PyCUDA hazir.")
except Exception as e:
    print("[WARN] TensorRT yok:", e)

def _shutdown(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# -------------------------
# Utils
# -------------------------
def letterbox_bgr(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(float(new_shape)/h, float(new_shape)/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh-0.1)); bottom = int(round(dh+0.1))
    left = int(round(dw-0.1)); right = int(round(dw+0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh

def clip_box(x1,y1,x2,y2,w,h):
    return (max(0.0, min(float(w-1), x1)),
            max(0.0, min(float(h-1), y1)),
            max(0.0, min(float(w-1), x2)),
            max(0.0, min(float(h-1), y2)))

def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
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
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = "ID:{} {:.0f}%".format(cls, conf*100.0)
        cv2.putText(frame, txt, (x1, max(20,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

# -------------------------
# TRT Wrapper (basit)
# -------------------------
class YOLO_TRT(object):
    def __init__(self, engine_path):
        logger = _trt.Logger(_trt.Logger.WARNING)
        runtime = _trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = _cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            safe_shape = tuple(max(int(s), 1) for s in shape)
            size = int(_trt.volume(safe_shape))
            dtype = _trt.nptype(self.engine.get_binding_dtype(binding))

            host = _cuda.pagelocked_empty(size, dtype)
            dev  = _cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host, "dev": dev})
            else:
                self.outputs.append({"host": host, "dev": dev})

    def infer(self, bgr):
        oh, ow = bgr.shape[:2]
        padded, ratio, dw, dh = letterbox_bgr(bgr, IMGSZ)

        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        x = np.expand_dims(x.transpose(2,0,1), 0)

        np.copyto(self.inputs[0]["host"], x.ravel())
        _cuda.memcpy_htod_async(self.inputs[0]["dev"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            _cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
        self.stream.synchronize()

        out = self.outputs[0]["host"]
        num_channels = int(len(out) // 8400)
        out = out.reshape((num_channels, 8400)).T

        boxes, scores, clses = [], [], []
        for row in out:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < CONF_THRES:
                continue

            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = (cx - bw/2.0 - dw) / ratio
            y1 = (cy - bh/2.0 - dh) / ratio
            x2 = (cx + bw/2.0 - dw) / ratio
            y2 = (cy + bh/2.0 - dh) / ratio
            x1,y1,x2,y2 = clip_box(x1,y1,x2,y2, ow, oh)

            boxes.append([x1,y1,x2,y2])
            scores.append(conf)
            clses.append(cls_id)

        keep = nms(boxes, scores, IOU_THRES)
        dets = []
        for i in keep:
            dets.append({"x1": boxes[i][0], "y1": boxes[i][1],
                         "x2": boxes[i][2], "y2": boxes[i][3],
                         "conf": scores[i], "cls": clses[i]})
        return dets

# -------------------------
# Threads
# -------------------------
def capture_thread():
    global latest_frame, latest_seq, running

    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[FATAL] Kamera acilamadi:", USB_CAM_DEV)
        running = False
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # 1 -> kamera iç buffer küçük, gecikme az

    seq = 0
    try:
        while running:
            ok, fr = cap.read()
            if not ok or fr is None:
                time.sleep(0.01)
                continue

            if fr.shape[1] != FRAME_W or fr.shape[0] != FRAME_H:
                fr = cv2.resize(fr, (FRAME_W, FRAME_H))

            fr = np.ascontiguousarray(fr)  # appsrc için güvenli

            with frame_lock:
                latest_frame = fr
                latest_seq = seq

            seq = (seq + 1) % 1000000
    finally:
        cap.release()

def inference_thread():
    global last_dets, last_det_time, last_det_seq, last_infer_ms, running

    if not HAS_TRT:
        while running:
            time.sleep(0.5)
        return

    ctx = None
    yolo = None
    try:
        ctx = _cuda.Device(0).make_context()
        yolo = YOLO_TRT(ENGINE_PATH)
        print("[AI] Engine yuklendi.")
    except Exception as e:
        print("[ERR] TRT init fail:", e)
        running = False
        return

    last_seq_seen = -1
    try:
        while running:
            fr = None
            seq = -1
            with frame_lock:
                if latest_frame is not None:
                    seq = latest_seq
                    if seq != last_seq_seen:
                        fr = latest_frame.copy()

            if fr is None:
                time.sleep(0.003)
                continue

            last_seq_seen = seq

            if (seq % INFER_EVERY_N) != 0:
                continue

            t0 = time.perf_counter()
            try:
                ctx.push()
                dets = yolo.infer(fr)
                ctx.pop()
                infer_ms = (time.perf_counter() - t0) * 1000.0

                with det_lock:
                    last_dets = dets
                    last_det_time = time.time()
                    last_det_seq = seq
                    last_infer_ms = float(infer_ms)
            except Exception as ex:
                print("[AI] infer hata:", ex)
                try:
                    ctx.pop()
                except Exception:
                    pass
    finally:
        try:
            ctx.pop()
            ctx.detach()
        except Exception:
            pass

def stream_thread():
    global running

    # MUX sonrası queue:
    # - finite: max-size-time ile sınırlı backlog
    # - non-leaky: TS paketlerini parçalama (macroblock üretmesin)
    TS_QUEUE_MAX_TIME_NS = 500000000  # 0.5s. İstersen 1_000_000_000 (1s) yap.

    uri = "srt://{ip}:{port}?mode=caller&latency={lat}&transtype=live&payloadsize={ps}".format(
        ip=PC_IP, port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE
    )

    gst_out = (
        "appsrc is-live=true do-timestamp=true block=true format=time ! "
        "video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "

        # RAW tarafında leaky=downstream doğru:
        # yetişemezsen eski frame düşer -> gecikme birikmez
        "queue max-size-buffers=2 leaky=downstream ! "

        "videoconvert ! video/x-raw,format=I420 ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "

        "nvv4l2h264enc bitrate={br} control-rate=1 preset-level=3 "
        "insert-sps-pps=true idrinterval=30 iframeinterval=30 ! "

        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! "

        # Bitstream tarafında leaky YOK. finite time ile limit var.
        "queue max-size-buffers=0 max-size-bytes=0 max-size-time={tmax} ! "

        "srtsink uri=\"{uri}\" sync=false wait-for-connection=true"
    ).format(w=FRAME_W, h=FRAME_H, fps=FPS, br=BITRATE, uri=uri, tmax=TS_QUEUE_MAX_TIME_NS)

    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS), (FRAME_W, FRAME_H), True)
    if not out.isOpened():
        print("[FATAL] VideoWriter acilmadi!\n", gst_out)
        running = False
        return

    frame_interval = 1.0 / float(FPS)

    while running:
        loop_t0 = time.perf_counter()

        with frame_lock:
            fr = None if latest_frame is None else latest_frame.copy()
            seq = int(latest_seq)

        if fr is None:
            time.sleep(0.005)
            continue

        with det_lock:
            dets = list(last_dets)
            det_age = time.time() - last_det_time if last_det_time > 0 else 999.0
            det_seq = int(last_det_seq)
            infer_ms = float(last_infer_ms)

        # seq farkı
        seq_ok = False
        if det_seq >= 0:
            diff = seq - det_seq
            if diff < 0:
                diff += 1000000
            seq_ok = (diff <= MAX_SEQ_LAG)

        # MOD-A: video akıcı, kutu sadece "taze" ise çiz
        if dets and det_age <= DET_MAX_AGE_S and seq_ok:
            fr = draw_dets(fr, dets)

        cv2.putText(fr, "Inf:{:.1f}ms Age:{:.0f}ms".format(infer_ms, det_age*1000.0),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        out.write(np.ascontiguousarray(fr))

        # Rate limiter: CPU boşuna yanmasın.
        # Bu sleep "lag biriktirmez" çünkü raw leaky queue + do-timestamp var.
        dt = time.perf_counter() - loop_t0
        s = frame_interval - dt
        if s > 0:
            time.sleep(s)

    try:
        out.release()
    except Exception:
        pass

if __name__ == "__main__":
    ths = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=inference_thread, daemon=True),
        threading.Thread(target=stream_thread, daemon=True),
    ]
    for t in ths:
        t.start()

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        running = False

    running = False
    time.sleep(1.0)
    sys.exit(0)