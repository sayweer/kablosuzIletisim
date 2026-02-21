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
SRT_PORT = 9000  #doğrudan işlenmemiş ham görüntünün sıkıştırılıp srt ile gönderildiği port numarası... (srt otoyolu)
META_PORT = 5005  #yapay zekanın işeleme sonucu elde ettiği verileri json ile ilettiği port numarası...   (UDP patikası)

USB_CAM_DEV = "/dev/video0"
FRAME_W = 640
FRAME_H = 480
FPS = 30

ENGINE_PATH = "model.engine"
IMGSZ = 640
CONF_THRES = 0.40
IOU_THRES = 0.45

BITRATE = 1500000
INFER_EVERY_N = 3

# =========================
# GLOBALLER VE KİLİTLER
# =========================
latest_frame_seq = 0  # [YENİ] Son karenin barkod sıra numarası

running = True

# [DÜZELTME]: Thread Kilitlerini (Locks) Ayırdık!
# Eskiden tek bir kilit vardı ve yapay zeka çalışırken video yayını beklemek zorunda kalıyordu.
# Şimdi video yayını için ayrı, yapay zeka için ayrı kilit ve değişken kullanıyoruz.
stream_lock = threading.Lock()
infer_lock = threading.Lock()

latest_stream_frame = None  #video yayıncısı kabini
latest_infer_frame = None   #yapay zekanın kabini
latest_frame_ts = 0.0

# =========================
# TENSORRT YÜKLEME (PyCUDA DÜZELTİLDİ)
# =========================
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # [DÜZELTME]: 'import pycuda.autoinit' BURADAN SİLİNDİ! 
    # Çünkü bu komut CUDA'yı yanlış yerde (ana thread'de) başlatıp çökmeye neden oluyordu.
    HAS_TRT = True
    print("[SYS] TensorRT modulleri hazir.")
except ImportError as e:
    print("[ERR] TensorRT hatasi:", e)

# (letterbox_bgr, clip_box ve nms fonksiyonları aynı, kalabalık yapmasın diye atlıyorum ama kodunda kalacak)
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
    return max(0.0, min(float(w - 1), x1)), max(0.0, min(float(h - 1), y1)), max(0.0, min(float(w - 1), x2)), max(0.0, min(float(h - 1), y2))

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
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
        # [DÜZELTME]: Claude'un uyardığı "Sıfıra Bölünme (Division by Zero)" hatası giderildi.
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = np.where(union > 0, inter / union, 0.0) 
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# [YENİ] Sıra numarasını (0-255) alıp 8 bitlik siyah-beyaz barkod çizen fonksiyon
def draw_barcode(frame, seq_num):
    # Sayıyı 8 haneli binary (ikilik) metne çevir. Örn: 5 -> '00000101'
    binary_str = format(seq_num, '08b')
    
    # 8 bitin her birini tek tek dön
    for i, bit in enumerate(binary_str):
        # 1 ise Bembeyaz (255,255,255), 0 ise Zifiri Siyah (0,0,0) yap
        color = (255, 255, 255) if bit == '1' else (0, 0, 0)
        
        # Numpy ile 20x20 piksellik bir kareyi anında boya (İşlemciyi hiç yormaz)
        # Y ekseni: 0'dan 20'ye. X ekseni: 0-20, 20-40, 40-60... şeklinde ilerler.
        frame[0:20, i*20:(i+1)*20] = color
        
    return frame

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
        
        # [DÜZELTME]: Claude'un belirttiği Jetson Nano'ya (TRT 8.2) özel Dinamik Shape Hatası Çözümü.
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            # Eğer modelde -1 (dinamik) boyut varsa, bunu çökmeye sebep olmaması için 1'e çeviriyoruz.
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
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence > CONF_THRES:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = (cx - w / 2.0 - dw) / ratio
                y1 = (cy - h / 2.0 - dh) / ratio
                x2 = (cx + w / 2.0 - dw) / ratio
                y2 = (cy + h / 2.0 - dh) / ratio
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
                
        indices = nms(boxes, scores, IOU_THRES)
        detections = [{"x1": boxes[i][0], "y1": boxes[i][1], "x2": boxes[i][2], "y2": boxes[i][3], "conf": scores[i], "cls": class_ids[i]} for i in indices]
        return detections

# =========================
# THREAD 1: KAMERA
# =========================
def capture_thread():
    global latest_stream_frame, latest_infer_frame, latest_frame_ts, latest_frame_seq ,running
    
    current_seq = 0  # [YENİ] Sayacımız 0'dan başlıyor

    cap = cv2.VideoCapture(USB_CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while running:
        ret, frame = cap.read()
        if not ret: continue
        
        ts = time.time()

        # [YENİ]: Barkod çizilmeden ÖNCE temiz bir kopya alıp yapay zeka için ayırıyoruz
        clean_frame = frame.copy()

        # Şimdi orijinal görüntünün üzerine barkodumuzu çiziyoruz
        barcode_frame = draw_barcode(frame, current_seq)
        
        # 1. Yayıncı kabinine (Stream) BARKODLU görüntüyü koyuyoruz
        with stream_lock:
            latest_stream_frame = barcode_frame
        
        # 2. Yapay zeka kabinine (Infer) TEMİZ görüntüyü koyuyoruz
        with infer_lock:
            latest_infer_frame = clean_frame
            latest_frame_ts = ts
            latest_frame_seq = current_seq  # Sıra numarasını yapay zekaya paslıyoruz

        # Sayacı 1 artır, 255'i geçince 0'a döndür
        current_seq = (current_seq + 1) % 256

# =========================
# THREAD 2: SRT VİDEO YAYINI
# =========================
def stream_thread():
    global latest_stream_frame, running
    
    gst_out = (
        "appsrc ! video/x-raw, format=BGR ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "videoconvert ! video/x-raw, format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        "nvv4l2h264enc bitrate={} maxperf-enable=0 preset-level=1 control-rate=1 ! "
        "video/x-h264, stream-format=byte-stream ! "
        "h264parse ! mpegtsmux alignment=7 ! "
        "srtsink uri=srt://{}:{} sync=false wait-for-connection=false"
    ).format(BITRATE, PC_IP, SRT_PORT)
    
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(FPS), (FRAME_W, FRAME_H), True)

    while running:
        frame_to_send = None
        with stream_lock:
            if latest_stream_frame is not None:
                frame_to_send = latest_stream_frame.copy()
        
        if frame_to_send is not None: out.write(frame_to_send)
        else: time.sleep(0.01)

# =========================
# THREAD 3: AI VE METADATA
# =========================
def inference_thread():
    global latest_infer_frame, latest_frame_ts, running
    
    # [DÜZELTME]: PyCUDA'yı BURADA, kendi Thread'i içinde başlatıyoruz! Çökmeleri engeller.
    cuda.init()
    ctx = cuda.Device(0).make_context()
    
    yolo = None
    try:
        yolo = YOLO_TRT(ENGINE_PATH)
        print("[AI] YOLO aktif.")
    except Exception as e:
        print("[ERR] AI Baslatilamadi:", e)
            
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # [DÜZELTME]: UDP soketinin veri tamponunu artırdık (Paket kaybını önlemek için)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
    
    frame_count = 0
    
    # [DÜZELTME]: Yanıp sönen kutuları engellemek için son tespitleri hafızada tutuyoruz (Cache).
    last_known_detections = []
    last_infer_ms = 0.0
    
    try:
        while running:
            frame = None
            frame_ts = 0.0
            
            with infer_lock:
                if latest_infer_frame is not None:
                    frame = latest_infer_frame.copy()
                    frame_ts = latest_frame_ts
                    frame_seq = latest_frame_seq  # [YENİ] Barkod numarasını aldık
                    
            if frame is None:
                time.sleep(0.01)
                continue
                
            if yolo and (frame_count % INFER_EVERY_N == 0):
                t0 = time.time()
                try:
                    last_known_detections = yolo.infer(frame)
                except Exception: pass
                last_infer_ms = (time.time() - t0) * 1000

            frame_count += 1
            
            # [DÜZELTME]: AI o karede çalışmamış olsa bile, hafızadaki son kutuları (last_known_detections) yolluyoruz.
            payload = {
                "ts": frame_ts,
                "seq": frame_seq,  # <--- İşte Yerdeki PC'nin arayacağı TC Kimlik Numarası!
                "infer_ms": last_infer_ms,
                "detections": last_known_detections,
                "det_count": len(last_known_detections),
                "trt_ok": True if yolo else False
            }
            
            try: sock.sendto(json.dumps(payload).encode('utf-8'), (PC_IP, META_PORT))
            except: pass
            
            time.sleep(0.01)
            
    finally:
        # Program kapanırken ekran kartı hafızasını temizler.
        ctx.pop()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    threads = [threading.Thread(target=capture_thread), threading.Thread(target=stream_thread), threading.Thread(target=inference_thread)]
    for t in threads: t.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        running = False
        for t in threads: t.join()