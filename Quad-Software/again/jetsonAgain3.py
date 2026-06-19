#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import threading
import signal
import sys
import numpy as np

cv2.setNumThreads(1)

# ==========================================
# AYARLAR
# ==========================================
PC_IP           = "192.168.1.50"
SRT_PORT        = 9000

USB_CAM_DEV     = "/dev/video0"
FRAME_W         = 640
FRAME_H         = 480
FPS             = 30

ENGINE_PATH     = "model.engine"
IMGSZ           = 640
CONF_THRES      = 0.40
IOU_THRES       = 0.45

# Yayın Ayarları
BITRATE         = 2000000    # Hızlı hareketlerde bozulmayı önlemek için optimum seviye
SRT_LATENCY_MS  = 150        # Gerçek zamanlılık için düşürüldü
SRT_PAYLOADSIZE = 1316

# Her N karede bir yapay zeka çıkarımı (30 FPS'de N=2 ise 15 FPS inference yapar)
INFER_EVERY_N   = 2

# ==========================================
# GLOBAL DURUM VE BUFFER
# ==========================================
running = True

frame_lock   = threading.Lock()
frame_buffer = {}  # {seq: frame_copy} - Kutuları doğru kareye senkronize etmek için
latest_seq   = 0

det_lock      = threading.Lock()
last_dets     = []
last_infer_ms = 0.0
last_det_seq  = -1

# ==========================================
# TensorRT ve Kapanış Sinyalleri
# ==========================================
HAS_TRT = False
cuda    = None
trt     = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
    print("[SYS] TensorRT/PyCUDA hazir.")
except Exception as e:
    print("[WARN] TensorRT yok, kutular olmayacak:", e)

def _shutdown(sig, frame):
    global running
    print("\n[SYS] Durdurma sinyali alindi (Ctrl+C), guvenli sekilde kapatiliyor...")
    running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(float(new_shape) / h, float(new_shape) / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, dw, dh

def clip_box(x1, y1, x2, y2, w, h):
    return (max(0.0, min(float(w - 1), x1)), max(0.0, min(float(h - 1), y1)),
            max(0.0, min(float(w - 1), x2)), max(0.0, min(float(h - 1), y2)))

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    boxes  = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou   = np.where(union > 0, inter / union, 0.0)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

def draw_dets(frame, dets):
    for d in dets:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        cls, conf = int(d["cls"]), float(d["conf"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "ID:{} {:.0f}%".format(cls, conf * 100.0)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

# ==========================================
# YOLO TRT SINIFI
# ==========================================
class YOLO_TRT:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            safe_shape = tuple(max(s, 1) for s in shape)
            size = trt.volume(safe_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "dev": dev_mem})
            else:
                self.outputs.append({"host": host_mem, "dev": dev_mem})

    def infer(self, img: np.ndarray) -> list:
        orig_h, orig_w = img.shape[:2]
        padded, ratio, dw, dh = letterbox_bgr(img, IMGSZ)

        inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)

        np.copyto(self.inputs[0]["host"], inp.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["dev"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
        self.stream.synchronize()

        output = self.outputs[0]["host"]
        num_channels = len(output) // 8400
        output = output.reshape((num_channels, 8400)).T

        boxes, scores, class_ids = [], [], []
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < CONF_THRES: continue
            
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = (cx - w/2.0 - dw) / ratio
            y1 = (cy - h/2.0 - dh) / ratio
            x2 = (cx + w/2.0 - dw) / ratio
            y2 = (cy + h/2.0 - dh) / ratio
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)

        keep = nms(boxes, scores, IOU_THRES)
        return [{"x1": boxes[i][0], "y1": boxes[i][1], "x2": boxes[i][2], "y2": boxes[i][3],
                 "conf": scores[i], "cls": class_ids[i]} for i in keep]

# ==========================================
# THREAD: KAMERA
# ==========================================
def capture_thread():
    global latest_seq, running
    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[FATAL] Kamera acilamadi:", USB_CAM_DEV)
        running = False
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    seq = 0
    try:
        while running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            
            with frame_lock:
                frame_buffer[seq] = frame.copy()
                if len(frame_buffer) > 30: # Hafiza sızıntısını önler
                    oldest_seq = min(frame_buffer.keys())
                    del frame_buffer[oldest_seq]
                latest_seq = seq
            seq = (seq + 1) % 1_000_000
    finally:
        cap.release()
        print("[CAM] Kamera kapatildi.")

# ==========================================
# THREAD: INFERENCE (YAPAY ZEKA)
# ==========================================
def inference_thread():
    global last_dets, last_infer_ms, last_det_seq, running
    if not HAS_TRT:
        while running: time.sleep(0.5)
        return

    ctx, yolo = None, None
    try:
        ctx = cuda.Device(0).make_context()
        yolo = YOLO_TRT(ENGINE_PATH)
        print("[AI] Engine yuklendi.")
    except Exception as e:
        print("[ERR] TRT/Engine acilamadi:", e)
        if ctx: 
            try: ctx.pop() 
            except: pass
        while running: time.sleep(0.5)
        return

    last_seq_seen = -1
    try:
        while running:
            frame, seq = None, -1
            with frame_lock:
                if latest_seq in frame_buffer:
                    frame = frame_buffer[latest_seq]
                    seq = latest_seq

            if frame is None or seq == last_seq_seen:
                time.sleep(0.005)
                continue
            last_seq_seen = seq

            if (seq % INFER_EVERY_N) != 0: continue

            t0 = time.perf_counter()
            try:
                ctx.push()
                dets = yolo.infer(frame)
                ctx.pop()
                infer_ms = (time.perf_counter() - t0) * 1000.0
                
                with det_lock:
                    last_dets = dets
                    last_infer_ms = infer_ms
                    last_det_seq = seq # Kutuların hangi kareye ait olduğunu damgalıyoruz
            except Exception as ex:
                print("[AI] Infer hata:", ex)
                try: ctx.pop()
                except: pass
    finally:
        try:
            ctx.pop()
            ctx.detach()
        except: pass
        print("[AI] Inference thread kapandi.")

# ==========================================
# THREAD: STREAM (GSTREAMER / SRT)
# ==========================================
def stream_thread():
    global running

    # JETSON ISINMA ÇÖZÜMÜ: format=BGRx ve nvvidconv kullanılarak CPU yükü GPU'ya alındı.
    gst_out = (
        "appsrc is-live=true do-timestamp=true block=true format=time ! "
        "video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} control-rate=1 preset-level=1 "
        "insert-sps-pps=true idrinterval=15 iframeinterval=15 ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux alignment=7 ! "
        "queue max-size-buffers=4 leaky=downstream ! "
        "srtsink uri=\"srt://{ip}:{port}?mode=caller&latency={lat}&transtype=live&payloadsize={ps}\" "
        "sync=false wait-for-connection=true"
    ).format(w=FRAME_W, h=FRAME_H, fps=FPS, br=BITRATE,
             ip=PC_IP, port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE)

    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS), (FRAME_W, FRAME_H), True)
    if not out.isOpened():
        print("[FATAL] VideoWriter acilmadi!\nPipeline:\n", gst_out)
        running = False
        return

    n, t0, fps_now = 0, time.perf_counter(), 0.0
    last_sent_seq = -1

    while running:
        frame_to_send, seq_to_send, current_dets, infer_time = None, -1, [], 0.0

        with det_lock:
            seq_to_send = last_det_seq
            current_dets = list(last_dets)
            infer_time = last_infer_ms

        # KUTU KAYMA ÇÖZÜMÜ: Sadece inference tamamlanmış kareyi bekleyip gönderiyoruz.
        if seq_to_send == last_sent_seq or seq_to_send == -1:
            time.sleep(0.01)
            continue

        with frame_lock:
            if seq_to_send in frame_buffer:
                frame_to_send = frame_buffer[seq_to_send].copy()
            
        if frame_to_send is not None:
            if current_dets:
                frame_to_send = draw_dets(frame_to_send, current_dets)

            n += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                fps_now = n / (now - t0)
                n, t0 = 0, now

            cv2.putText(frame_to_send, "Stream FPS:{:.1f}  Infer:{:.1f}ms  Seq:{}".format(
                fps_now, infer_time, seq_to_send),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            out.write(frame_to_send)
            last_sent_seq = seq_to_send

    out.release()
    print("[STREAM] VideoWriter kapatildi.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread,   daemon=True, name="CamThread"),
        threading.Thread(target=inference_thread, daemon=True, name="InfThread"),
        threading.Thread(target=stream_thread,    daemon=True, name="SrtThread"),
    ]
    for t in threads: t.start()

    try:
        while running: time.sleep(0.5)
    except KeyboardInterrupt:
        running = False

    running = False
    time.sleep(1.0)
    print("[SYS] Cikis yapildi.")
    sys.exit(0)