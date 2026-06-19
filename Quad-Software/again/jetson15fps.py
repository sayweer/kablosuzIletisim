#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JETSON MOD-B (Locked Boxes)
- Sadece inference yapılan kareler yayınlanır.
- Kutular tam oturur (inference hangi frame'deyse o frame gönderilir).
- Video FPS: FPS_OUT ~ FPS / INFER_EVERY_N (örn 30/2=15)
"""

import cv2
import time
import threading
import signal
import sys
import numpy as np

cv2.setNumThreads(1)

PC_IP           = "192.168.1.50"
SRT_PORT        = 9000

USB_CAM_DEV     = "/dev/video0"
FRAME_W         = 640
FRAME_H         = 480
FPS             = 30

BITRATE         = 2000000
SRT_LATENCY_MS  = 300
SRT_PAYLOADSIZE = 1316

ENGINE_PATH     = "model.engine"
IMGSZ           = 640
CONF_THRES      = 0.40
IOU_THRES       = 0.45

INFER_EVERY_N   = 2
FPS_OUT         = max(1, int(FPS // INFER_EVERY_N))  # 30/2=15. (20 istiyorsan INFER_EVERY_N=1 ya da FPS=20)

running = True

frame_lock = threading.Lock()
latest_frame = None
latest_seq   = 0

# inference sonucu "gönderilecek frame"
send_lock = threading.Lock()
send_cond = threading.Condition(send_lock)
send_frame = None
send_seq   = -1
send_infer_ms = 0.0

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
        i = int(order[0]); keep.append(i)
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

def capture_thread():
    global latest_frame, latest_seq, running

    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        running = False
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    seq = 0
    try:
        while running:
            ok, fr = cap.read()
            if not ok or fr is None:
                time.sleep(0.01)
                continue

            if fr.shape[1] != FRAME_W or fr.shape[0] != FRAME_H:
                fr = cv2.resize(fr, (FRAME_W, FRAME_H))

            fr = np.ascontiguousarray(fr)

            with frame_lock:
                latest_frame = fr
                latest_seq = seq

            seq = (seq + 1) % 1000000
    finally:
        cap.release()

def inference_thread():
    global running, send_frame, send_seq, send_infer_ms

    if not HAS_TRT:
        while running:
            time.sleep(0.5)
        return

    ctx = None
    yolo = None
    try:
        ctx = _cuda.Device(0).make_context()
        yolo = YOLO_TRT(ENGINE_PATH)
    except Exception:
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

                # TAM OTURMA: kutuyu inference yapılan frame'e çiziyoruz (aynı frame!)
                if dets:
                    fr = draw_dets(fr, dets)
                cv2.putText(fr, "Infer:{:.1f}ms Seq:{}".format(infer_ms, seq),
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

                fr = np.ascontiguousarray(fr)

                with send_cond:
                    send_frame = fr
                    send_seq = seq
                    send_infer_ms = float(infer_ms)
                    send_cond.notify_all()

            except Exception:
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
    global running, send_frame, send_seq

    TS_QUEUE_MAX_TIME_NS = 500000000  # 0.5s

    uri = "srt://{ip}:{port}?mode=caller&latency={lat}&transtype=live&payloadsize={ps}".format(
        ip=PC_IP, port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE
    )

    # framerate=FPS_OUT/1: bu modda zaten o civarda frame gönderiyoruz.
    gst_out = (
        "appsrc is-live=true do-timestamp=true block=true format=time ! "
        "video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} control-rate=1 preset-level=3 "
        "insert-sps-pps=true idrinterval=30 iframeinterval=30 ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! "
        "queue max-size-buffers=0 max-size-bytes=0 max-size-time={tmax} ! "
        "srtsink uri=\"{uri}\" sync=false wait-for-connection=true"
    ).format(w=FRAME_W, h=FRAME_H, fps=FPS_OUT, br=BITRATE, uri=uri, tmax=TS_QUEUE_MAX_TIME_NS)

    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS_OUT), (FRAME_W, FRAME_H), True)
    if not out.isOpened():
        running = False
        return

    frame_interval = 1.0 / float(FPS_OUT)
    last_sent = -1

    while running:
        loop_t0 = time.perf_counter()

        with send_cond:
            # yeni frame bekle (timeout, Ctrl+C yakalayabilelim)
            send_cond.wait(timeout=0.2)
            fr = None if send_frame is None else send_frame.copy()
            seq = int(send_seq)

        if fr is None or seq == last_sent:
            time.sleep(0.005)
            continue

        out.write(np.ascontiguousarray(fr))
        last_sent = seq

        # Rate limit
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