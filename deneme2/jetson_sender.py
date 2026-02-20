#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import json
import socket
import threading
import numpy as np

# İşlemci (CPU) ısınmasını engellemek için OpenCV thread sayısını kısıtla
cv2.setNumThreads(1)

# =========================
# AYARLAR
# =========================
PC_IP = "192.168.1.100"      # Yer istasyonu IP adresi
SRT_PORT = 9000
META_PORT = 5005

# Kamera Ayarları (USB Kamera)
USB_CAM_DEV = "/dev/video0"
FRAME_W = 640
FRAME_H = 480
FPS = 30

# Model Ayarları (YOLOv8 veya YOLOv11)
ENGINE_PATH = "model.engine" # Jetson'da derlenen engine dosyasi (v8 veya v11 fark etmez)
IMGSZ = 640                  # Modelin giriş boyutu
CONF_THRES = 0.40
IOU_THRES = 0.45

# Isınma ve Performans Kontrolleri
BITRATE = 1500000            # 1.5 Mbps
INFER_EVERY_N = 3            # AI sadece 3 karede bir çalışsın

# =========================
# GLOBALLER
# =========================
running = True
video_lock = threading.Lock()
meta_lock = threading.Lock()

latest_frame = None
latest_frame_ts = 0.0

latest_detections = []
latest_infer_ms = 0.0

# =========================
# TENSORRT YÜKLEME
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
    print("[SYS] TensorRT ve PyCUDA hazir.")
except ImportError as e:
    print("[ERR] TensorRT/PyCUDA hatasi. Sadece video aktarilacak:", e)

# =========================
# MATEMATİKSEL YARDIMCI FONKSİYONLAR
# =========================
def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):
    """ Görüntüyü sündürmeden (aspect ratio koruyarak) istenen boyuta getirir """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Ölçekleme oranını bul (en dar kenara göre)
    r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
    new_unpad = (int(round(w * r)), int(round(h * r)))

    # Boşluk (padding) miktarlarını hesapla
    dw = float(new_shape[1] - new_unpad[0]) / 2.0
    dh = float(new_shape[0] - new_unpad[1]) / 2.0

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Kenarlara gri renk ekle
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh

def clip_box(x1, y1, x2, y2, w, h):
    """ Kutuları orijinal görüntü sınırları içinde tutar """
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return x1, y1, x2, y2

def nms(boxes, scores, iou_threshold):
    """ Non-Maximum Suppression (Üst üste binen kutuları eler) """
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
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

# =========================
# YOLOv8 / YOLOv11 ORTAK TENSORRT SINIFI
# =========================
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
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
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
        
        # 1. Preprocess (Letterbox ile)
        # Sündürme yok, modelin beklediği orana güvenli şekilde getirilir
        padded_img, ratio, dw, dh = letterbox_bgr(img, IMGSZ)
        
        input_image = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1)) # HWC -> CHW
        input_image = np.expand_dims(input_image, axis=0)
        
        # GPU'ya Kopyala ve Çalıştır
        np.copyto(self.inputs[0]['host'], input_image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['dev'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['dev'], self.stream)
        self.stream.synchronize()
        
        # 2. Postprocess (YOLOv8 ve YOLOv11 uyumlu)
        output = self.outputs[0]['host']
        
        # Orijinal matris boyutunu dinamik bul (örneğin 84 x 8400)
        num_channels = len(output) // 8400 
        output = output.reshape((num_channels, 8400)).T # Matrisi devir (8400, C)
        
        boxes, scores, class_ids = [], [], []
        
        for row in output:
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > CONF_THRES:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Letterbox padding ve oranlarını geriye doğru hesapla
                x1 = (cx - w / 2.0 - dw) / ratio
                y1 = (cy - h / 2.0 - dh) / ratio
                x2 = (cx + w / 2.0 - dw) / ratio
                y2 = (cy + h / 2.0 - dh) / ratio
                
                # Görüntü dışına taşan kutuları kırp
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
                
        # NMS (Üst üste binen kutuları temizle)
        indices = nms(boxes, scores, IOU_THRES)
        
        detections = []
        for i in indices:
            detections.append({
                "x1": boxes[i][0], "y1": boxes[i][1], 
                "x2": boxes[i][2], "y2": boxes[i][3],
                "conf": scores[i], "cls": class_ids[i]
            })
            
        return detections

# =========================
# THREAD 1: USB KAMERA
# =========================
def capture_thread():
    global latest_frame, latest_frame_ts, running
    
    print("[CAM] USB Kamera aciliyor:", USB_CAM_DEV)
    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("[ERR] Kamera bulunamadi!")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        with video_lock:
            latest_frame = frame
            latest_frame_ts = time.time()
            
    cap.release()

# =========================
# THREAD 2: SRT VİDEO YAYINI
# =========================
def stream_thread():
    global latest_frame, running
    
    gst_out = (
        "appsrc ! video/x-raw, format=BGR ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "videoconvert ! video/x-raw, format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        "nvv4l2h264enc bitrate={} maxperf-enable=0 preset-level=1 control-rate=1 ! "  #maxperf-enable=0 kısmını 1 yaparsak performansı arttırmış oluruz.  control-rate=1 sabit bit hızı anlamına geliyor bu iyidir ama burada 1.5mbps kullandığımız için kalite düşüyor bunun yerine 2 kullanırsak yani değişken bit hızı kullanırsak kalite yükselebilir.
        "video/x-h264, stream-format=byte-stream ! "
        "h264parse ! mpegtsmux alignment=7 ! "  #alignment=1 daha uyumlu diye bir şey duydum ama emin değilim.
        "srtsink uri=srt://{}:{} "
        "sync=false wait-for-connection=false".format(BITRATE, PC_IP, SRT_PORT)
    )
    
    print("[STREAM] SRT Yayini basladi.")
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS), (FRAME_W, FRAME_H), True)

    while running:
        frame_to_send = None
        with video_lock:
            if latest_frame is not None:
                frame_to_send = latest_frame.copy()
        
        if frame_to_send is not None:
            out.write(frame_to_send)
        else:
            time.sleep(0.01)

    out.release()

# =========================
# THREAD 3: AI VE METADATA (SENKRONİZE)
# =========================
def inference_thread():
    global latest_frame, latest_frame_ts, running
    
    yolo = None
    if HAS_TRT and os.path.exists(ENGINE_PATH):
        try:
            yolo = YOLO_TRT(ENGINE_PATH)
            print("[AI] YOLO Engine (v8/v11) aktif.")
        except Exception as e:
            print("[ERR] AI Baslatilamadi:", e)
            
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_count = 0
    
    while running:
        frame = None
        frame_ts = 0.0
        
        with video_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                frame_ts = latest_frame_ts
                
        if frame is None:
            time.sleep(0.01)
            continue
            
        detections = []
        infer_ms = 0.0

        if yolo and (frame_count % INFER_EVERY_N == 0):
            t0 = time.time()
            try:
                detections = yolo.infer(frame)
            except Exception as e:
                print("[ERR] Çıkarım hatası:", e)
            infer_ms = (time.time() - t0) * 1000

        frame_count += 1
        
        payload = {
            "ts": frame_ts,
            "infer_ms": infer_ms,
            "detections": detections,
            "det_count": len(detections),
            "trt_ok": True if yolo else False
        }
        
        try:
            sock.sendto(json.dumps(payload).encode('utf-8'), (PC_IP, META_PORT))
        except:
            pass
            
        time.sleep(0.01)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread),
        threading.Thread(target=stream_thread),
        threading.Thread(target=inference_thread)
    ]
    
    for t in threads:
        t.start()
        
    print("[SYS] Sistem calisiyor. Cikmak icin Ctrl+C")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SYS] Kapatiliyor...")
        running = False
        for t in threads:
            t.join()