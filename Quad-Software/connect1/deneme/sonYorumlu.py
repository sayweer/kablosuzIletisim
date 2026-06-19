#!/usr/bin/env python3  # Bu script’in Python 3 ile çalışacağını belirtir (Linux shebang)
# -*- coding: utf-8 -*-  # Dosya encoding’i UTF-8 (Türkçe karakterler için)

"""
Jetson Nano Sender (Python 3.6)  # Docstring: programın açıklaması
- Kamera al (GStreamer / fallback V4L2)
- TensorRT engine ile tespit
- SRT (MPEG-TS/H264) video gönder
- UDP ile metadata gönder

Önemli düzeltmeler:
1) (1,7,8400) için varsayılan decode: noobj (YOLOv8/11 tarzı 4+nc)
2) Kutu kaybolması için stale ayarı genişletildi
3) Inference çökse de stream devam eder
"""

from __future__ import print_function  # Py2/3 print uyumu (Py3.6’da zararsız)
import os  # OS/env kontrolü, dosya var mı vs.
import cv2  # OpenCV: kamera, görüntü, VideoWriter/VideoCapture
import time  # zaman ölçümü, sleep, timestamp
import json  # metadata’yı JSON’a çevirme
import socket  # UDP metadata gönderme
import threading  # çoklu thread (capture/infer/stream/meta)
import numpy as np  # NMS, tensor şekillendirme, array işlemleri

# =========================
# KONFIG (JETSON)
# =========================
PC_IP = "192.168.1.50"  # Videoyu/metadata’yı göndereceğin PC’nin IP’si
SRT_PORT = 9000  # SRT video portu (PC listener bu portu dinler)
META_PORT = 5005  # UDP metadata portu (PC bu portu dinler)

WIDTH = 640  # Kamera/stream hedef genişlik
HEIGHT = 480  # Kamera/stream hedef yükseklik
FPS = 30  # Hedef FPS  !!!!!!!!!  (yük artar, ısı artar)
BITRATE_KBPS = 2500  # H264 bitrate (kbps), kalite/iş yükü dengesi
SRT_LATENCY_MS = 100  # SRT’nin ek gecikme buffer’ı (ms) (daha düşük = daha az gecikme, daha çok drop riski)

# Jetson üstünde çizip gönder
STREAM_ANNOTATED = True  # Kutuları Jetson’da çiz -> CPU yükü artar !!!!!!!!! (ısı + gecikme)

USE_CSI_CAMERA = False  # CSI kamera mı? (nvargus)
CAM_DEVICE = "/dev/video0"  # USB kamera device path

ENABLE_INFERENCE = True  # TensorRT inference aktif mi? !!!!!!!!! (ana ısı kaynağı)
ENGINE_PATH = "quad_yolov11n.engine"  # TensorRT engine dosyası
IMGSZ = 640  # Model input size (letterbox ile buraya ölçeklenir)
CONF_THRES = 0.20  # Confidence eşiği (düşürürsen daha çok bbox çıkar)
IOU_THRES = 0.45  # NMS IoU eşiği
INFER_EVERY_N = 1  # Her N frame’de bir infer  !!!!!!!!! (1 = en ağır yük, en çok ısı)
# noobj -> out = [x,y,w,h,cls1,cls2,...]
# obj   -> out = [x,y,w,h,obj,cls1,cls2,...]
# auto  -> iki yolu da dener, daha çok det çıkan yolu seçer
DECODE_MODE = "noobj"  # Decode varsayımı (senin engine formatına uygun)

DET_STALE_SEC = 2.0  # Detections bayat sayılmadan kaç sn tutulur (görsel kaybolmasın diye)
KEEP_LAST_DET_ALWAYS = True  # True ise stale’e bakmadan son det’leri kullan (takip hissi için)

LOG_EVERY_SEC = 3.0  # Kaç saniyede bir debug/istatistik log basılsın

# =========================
# GLOBALS
# =========================
running = True  # Thread’lerin çalışmasını kontrol eden ana flag
state_lock = threading.Lock()  # Paylaşılan state’i korumak için lock

latest_raw = None  # En son yakalanan frame (BGR numpy array)
latest_raw_seq = 0  # Frame sayaç (hangi sıradaki frame)

latest_dets = []  # En son bulunan bbox listesi
latest_det_ts = 0.0  # En son det’in timestamp’i
latest_infer_ms = 0.0  # Son inference süresi (ms)
latest_trt_ok = False  # TRT hazır mı/çalışıyor mu

latest_meta = {  # PC’ye giden metadata payload’u
    "ts": 0.0,  # meta timestamp
    "seq": 0,  # frame seq
    "infer_ms": 0.0,  # infer süresi
    "detections": [],  # bbox list
    "det_count": 0,  # kaç bbox
    "trt_ok": False  # TRT çalışıyor mu
}

# =========================
# TRT IMPORT
# =========================
HAS_TRT = False  # TRT import başarılı mı?
try:
    import tensorrt as trt  # TensorRT runtime
    import pycuda.driver as cuda  # CUDA memory copy / stream
    HAS_TRT = True  # TRT kullanılabilir
except Exception as e:
    print("[TRT] TensorRT/PyCUDA import yok:", e)  # TRT yoksa inference kapanır
    HAS_TRT = False  # TRT devre dışı

def opencv_has_gstreamer():  # OpenCV’nin GStreamer destekli olup olmadığını kontrol eder
    try:
        bi = cv2.getBuildInformation()  # OpenCV build info stringi
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)  # GStreamer var mı
    except Exception:
        return False  # bilgi alınamazsa yok varsay

# =========================
# CAMERA
# =========================
def build_camera_candidates():  # Farklı kamera pipeline seçeneklerini listeler
    cands = []  # aday pipeline listesi
    if USE_CSI_CAMERA:  # CSI kamera (nvargus)
        p = (  # nvargus -> NVMM -> BGR
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(w=WIDTH, h=HEIGHT, fps=FPS)  # pipeline format paramları
        cands.append(("csi_nvargus", p))  # aday ekle
    else:
        p1 = (  # USB kamera MJPEG ise
            "v4l2src device={dev} ! "
            "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)  # string format
        cands.append(("usb_mjpeg", p1))  # MJPEG pipeline

        p2 = (  # USB kamera raw (YUYV vs) ise
            "v4l2src device={dev} ! "
            "video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1 sync=false"
        ).format(dev=CAM_DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)  # string format
        cands.append(("usb_raw", p2))  # raw pipeline
    return cands  # adayları döndür

def open_camera():  # Kamera açmayı dener
    for name, pipe in build_camera_candidates():  # her aday pipeline’ı dene
        print("[CAM] Deneniyor:", name)  # hangi pipeline deneniyor
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)  # GStreamer pipeline ile capture aç
        if cap.isOpened():  # açıldı mı
            ok, fr = cap.read()  # bir frame dene
            if ok and fr is not None and fr.size > 0:  # frame valid mi
                print("[CAM] Acildi:", name)  # başarı
                return cap, name  # capture objesini döndür
        try:
            cap.release()  # başarısızsa release
        except Exception:
            pass  # release hatası umursanmaz

    print("[CAM] GStreamer acilmadi, V4L2 fallback...")  # GStreamer olmadıysa OpenCV V4L2
    cap = cv2.VideoCapture(CAM_DEVICE)  # direkt device ile aç
    if cap.isOpened():  # açıldı mı
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)  # width ayarla
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)  # height ayarla
        cap.set(cv2.CAP_PROP_FPS, FPS)  # fps ayarla
        ok, fr = cap.read()  # test frame
        if ok and fr is not None and fr.size > 0:  # valid mi
            print("[CAM] V4L2 fallback acildi.")  # başarı
            return cap, "opencv_v4l2"  # capture döndür

    try:
        cap.release()  # başarısızsa release
    except Exception:
        pass
    return None, None  # kamera açılamadı

# =========================
# SRT WRITER (NVENC)
# =========================
def build_srt_writer_pipeline():  # Jetson’dan PC’ye SRT H264 TS gönderecek pipeline
    uri = "srt://{}:{}?mode=caller&latency={}&transtype=live".format(  # SRT URI
        PC_IP, SRT_PORT, SRT_LATENCY_MS  # hedef ip/port/latency
    )
    return (  # appsrc -> convert -> NVENC -> h264parse -> mpegtsmux -> srtsink
        "appsrc is-live=true block=true do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        "queue leaky=downstream max-size-buffers=2 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2h264enc bitrate={br} insert-sps-pps=true iframeinterval=30 idrinterval=30 "
        "control-rate=1 preset-level=1 maxperf-enable=1 ! "  # !!!!!!!!! NVENC max perf -> ısı artar
        "h264parse config-interval=1 ! mpegtsmux alignment=7 ! queue ! "
        "srtsink uri=\"{uri}\" wait-for-connection=false sync=false async=false"
    ).format(w=WIDTH, h=HEIGHT, fps=FPS, br=BITRATE_KBPS * 1000, uri=uri)  # paramları koy

def open_writer():  # VideoWriter açmayı dener
    pipe = build_srt_writer_pipeline()  # pipeline string
    print("[SRT] Writer pipeline:\n", pipe)  # debug bas
    wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)  # GStreamer writer aç
    if wr.isOpened():  # açıldıysa
        print("[SRT] Writer acildi (NVENC/NVMM)")  # başarı
        return wr  # writer döndür
    try:
        wr.release()  # açılmadıysa release
    except Exception:
        pass
    return None  # writer açılamadı

# =========================
# HELPERS
# =========================
def letterbox_bgr(img, new_shape=640, color=(114, 114, 114)):  # YOLO letterbox (oran koruyarak pad)
    h, w = img.shape[:2]  # input görüntü boyutları
    if isinstance(new_shape, int):  # tek sayı geldiyse kare yap
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))  # scale ratio
    new_unpad = (int(round(w * r)), int(round(h * r)))  # ölçeklenmiş boyut

    dw = float(new_shape[1] - new_unpad[0]) / 2.0  # sağ-sol padding
    dh = float(new_shape[0] - new_unpad[1]) / 2.0  # üst-alt padding

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resize

    top = int(round(dh - 0.1))  # pad hesap (round stabil)
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(  # pad ekle
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return out, r, dw, dh  # letterboxed görüntü + ratio + offsetler

def nms_xyxy(boxes, scores, iou_thres):  # klasik NMS (CPU)  !!!!!!!!! (çok bbox olursa CPU’yu yer)
    if len(boxes) == 0:
        return []  # boşsa boş dön
    boxes = np.array(boxes, dtype=np.float32)  # numpy’a çevir
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]  # koordinatlar
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)  # alan
    order = scores.argsort()[::-1]  # skor desc sırala
    keep = []  # tutulacak indexler

    while order.size > 0:  # sırada eleman kaldıkça
        i = int(order[0])  # en yüksek skor
        keep.append(i)  # bunu tut

        xx1 = np.maximum(x1[i], x1[order[1:]])  # overlap hesapları
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)  # overlap width
        h = np.maximum(0.0, yy2 - yy1)  # overlap height
        inter = w * h  # intersection area
        union = areas[i] + areas[order[1:]] - inter + 1e-6  # union
        iou = inter / union  # IoU

        inds = np.where(iou <= iou_thres)[0]  # iou küçük olanları tut
        order = order[inds + 1]  # sırayı güncelle
    return keep  # final indeksler

def clip_box(x1, y1, x2, y2, w, h):  # bbox’ı görüntü sınırlarına kırp
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return x1, y1, x2, y2

def draw_dets(frame, dets, color=(0, 200, 255)):  # bbox çizimi (CPU) !!!!!!!!! (özellikle çok bbox varsa)
    for d in dets:  # her detection için
        x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))  # sol üst
        x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))  # sağ alt
        cls = int(d.get("cls", 0))  # class id
        conf = float(d.get("conf", 0.0))  # confidence

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # dikdörtgen çiz
        txt = "c{} {:.2f}".format(cls, conf)  # label string
        cv2.putText(frame, txt, (x1, max(20, y1 - 8)),  # text yaz
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame  # çizilmiş frame

# =========================
# TRT DETECTOR
# =========================
class TRTDetector(object):  # TensorRT inference wrapper’ı
    def __init__(self, engine_path):  # engine yükleme + buffer allocate
        if not HAS_TRT:
            raise RuntimeError("TensorRT/PyCUDA yok")  # TRT yoksa çık
        if not os.path.isfile(engine_path):
            raise RuntimeError("Engine yok: {}".format(engine_path))  # engine dosyası yoksa çık

        self.logger = trt.Logger(trt.Logger.WARNING)  # TRT logger seviyesi
        self.runtime = trt.Runtime(self.logger)  # TRT runtime

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())  # engine deserialize
        if self.engine is None:
            raise RuntimeError("Engine deserialize fail")  # bozuk engine

        self.context = self.engine.create_execution_context()  # execution context
        if self.context is None:
            raise RuntimeError("Context create fail")  # context oluşturulamadı

        self.input_idx = None  # input binding index
        for i in range(self.engine.num_bindings):  # tüm binding’leri tara
            if self.engine.binding_is_input(i):  # input mu
                self.input_idx = i  # index
                break
        if self.input_idx is None:
            raise RuntimeError("Input binding yok")  # input bulunamadı

        in_shape = tuple(self.context.get_binding_shape(self.input_idx))  # input shape
        if -1 in in_shape:  # dynamic shape varsa
            self.context.set_binding_shape(self.input_idx, (1, 3, IMGSZ, IMGSZ))  # sabitle

        self.bindings = [None] * self.engine.num_bindings  # device ptr listesi
        self.host_in = None; self.dev_in = None  # input host/device
        self.host_out = []; self.dev_out = []  # output host/device listeleri
        self.out_bind_idxs = []  # output binding index listesi

        for i in range(self.engine.num_bindings):  # her binding için allocate
            dtype = trt.nptype(self.engine.get_binding_dtype(i))  # numpy dtype
            shape = tuple(self.context.get_binding_shape(i))  # binding shape
            if -1 in shape:
                shape = tuple([1 if d < 0 else int(d) for d in shape])  # dynamic -> 1’e çek

            size = int(np.prod(shape))  # eleman sayısı
            host_mem = cuda.pagelocked_empty(size, dtype)  # pinned host mem (hızlı kopya)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)  # GPU mem allocate
            self.bindings[i] = int(dev_mem)  # binding ptr

            if self.engine.binding_is_input(i):  # input ise
                self.host_in = host_mem
                self.dev_in = dev_mem
            else:  # output ise
                self.out_bind_idxs.append(i)
                self.host_out.append(host_mem)
                self.dev_out.append(dev_mem)

        self.stream = cuda.Stream()  # CUDA stream !!!!!!!!! (infer + kopya işleri burada)
        self._shape_logged = False  # output shape debug bir kere basılsın
        self._dbg_t = 0.0  # debug timer

        print("[TRT] Engine yuklendi:", engine_path)  # bilgi
        for i in range(self.engine.num_bindings):
            print("[TRT] binding", i, self.engine.get_binding_name(i), self.context.get_binding_shape(i))  # binding log

    def preprocess(self, bgr):  # BGR -> letterbox -> RGB -> normalize -> NCHW
        img, ratio, dw, dh = letterbox_bgr(bgr, IMGSZ)  # letterbox
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # renk dönüşümü (CPU)
        x = rgb.astype(np.float32) / 255.0  # normalize
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)  # CHW -> NCHW (batch=1)
        return x, ratio, dw, dh  # input tensor + letterbox paramları

    def infer_raw(self, x):  # TRT inference (host->device, execute, device->host)
        np.copyto(self.host_in, x.ravel())  # host input buffer’a kopya !!!!!!!!! (her frame)
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)  # H2D async !!!!!!!!!
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)  # TRT execute !!!!!!!!! (GPU yük)
        for i in range(len(self.dev_out)):
            cuda.memcpy_dtoh_async(self.host_out[i], self.dev_out[i], self.stream)  # D2H async !!!!!!!!!
        self.stream.synchronize()  # tüm işi bitirene kadar bekle !!!!!!!!! (latency burada hissedilir)

        outs = []  # output numpy listesi
        for out_i, bind_i in enumerate(self.out_bind_idxs):
            shape = tuple(self.context.get_binding_shape(bind_i))  # output shape
            if -1 in shape:
                shape = tuple([1 if d < 0 else int(d) for d in shape])
            arr = np.array(self.host_out[out_i]).reshape(shape)  # host buffer -> numpy reshape
            outs.append(arr)
        return outs  # output list

    def _decode_cn(self, out_cn, orig_w, orig_h, ratio, dw, dh, use_obj):  # (C,N) decode
        C = int(out_cn.shape[0]); N = int(out_cn.shape[1])  # kanal ve anchor sayısı
        if C < 6 or N < 1:
            return []  # yeterli kanal yok

        xywh = out_cn[0:4, :].astype(np.float32)  # bbox paramları
        max_xywh = float(np.max(np.abs(xywh))) if xywh.size > 0 else 0.0  # normalize mi kontrol
        if max_xywh <= 2.0:
            xywh = xywh * float(IMGSZ)  # normalize ise piksele çevir

        score_mat = out_cn[4:, :].astype(np.float32)  # skor matrisi

        if use_obj and score_mat.shape[0] >= 2:  # obj+cls formatı
            obj = score_mat[0, :]  # obj
            cls_scores = score_mat[1:, :]  # class scores
            cls_ids = np.argmax(cls_scores, axis=0).astype(np.int32)  # en iyi sınıf
            conf = (obj * np.max(cls_scores, axis=0)).astype(np.float32)  # final conf (obj*cls)
        else:  # noobj formatı (YOLOv8/11 sık görülen)
            cls_ids = np.argmax(score_mat, axis=0).astype(np.int32)  # en iyi sınıf
            conf = np.max(score_mat, axis=0).astype(np.float32)  # conf = max class score

        idxs = np.where(conf >= CONF_THRES)[0]  # threshold filtre
        if idxs.size == 0:
            return []  # hiç bbox yok

        if idxs.size > 300:  # performans için limit
            top = np.argsort(conf[idxs])[::-1][:300]  # top-k
            idxs = idxs[top]

        boxes = []; scores = []; clses = []  # NMS için listeler

        for j in idxs:  # seçili bbox’lar
            cx = float(xywh[0, j]); cy = float(xywh[1, j])  # center
            w = float(xywh[2, j]); h = float(xywh[3, j])  # width/height

            x1 = cx - w / 2.0; y1 = cy - h / 2.0  # xyxy dönüşümü
            x2 = cx + w / 2.0; y2 = cy + h / 2.0

            x1 = (x1 - dw) / ratio; y1 = (y1 - dh) / ratio  # letterbox geri çöz
            x2 = (x2 - dw) / ratio; y2 = (y2 - dh) / ratio

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)  # sınır kırp

            boxes.append([x1, y1, x2, y2])  # bbox
            scores.append(float(conf[j]))  # score
            clses.append(int(cls_ids[j]))  # class

        keep = nms_xyxy(boxes, scores, IOU_THRES)  # NMS !!!!!!!!! (çok bbox varsa ısı/CPU)

        dets = []  # final det listesi
        for i in keep:
            b = boxes[i]
            dets.append({"cls": int(clses[i]), "conf": float(scores[i]),
                         "x1": float(b[0]), "y1": float(b[1]),
                         "x2": float(b[2]), "y2": float(b[3])})
        return dets  # detections

    def _decode_n6(self, out_n6, orig_w, orig_h, ratio, dw, dh):  # (N,6+) decode
        arr = out_n6.astype(np.float32)  # float32
        if arr.shape[1] < 6:
            return []  # format uygun değil

        max_xy = float(np.max(np.abs(arr[:, :4]))) if arr.shape[0] > 0 else 0.0  # normalize kontrol
        if max_xy <= 2.0:
            arr[:, :4] *= float(IMGSZ)  # normalize -> pixel

        boxes = []; scores = []; clses = []  # NMS listeleri

        for i in range(arr.shape[0]):  # her bbox
            conf = float(arr[i, 4])  # conf
            if conf < CONF_THRES:
                continue  # düşük conf atla

            x1 = float(arr[i, 0]); y1 = float(arr[i, 1])
            x2 = float(arr[i, 2]); y2 = float(arr[i, 3])
            cls = int(arr[i, 5]) if arr.shape[1] > 5 else 0  # cls

            x1 = (x1 - dw) / ratio; y1 = (y1 - dh) / ratio  # letterbox geri çöz
            x2 = (x2 - dw) / ratio; y2 = (y2 - dh) / ratio

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, orig_w, orig_h)  # kırp

            boxes.append([x1, y1, x2, y2]); scores.append(conf); clses.append(cls)  # listelere ekle

        keep = nms_xyxy(boxes, scores, IOU_THRES)  # NMS !!!!!!!!!
        dets = []
        for k in keep:
            b = boxes[k]
            dets.append({"cls": int(clses[k]), "conf": float(scores[k]),
                         "x1": float(b[0]), "y1": float(b[1]),
                         "x2": float(b[2]), "y2": float(b[3])})
        return dets

    def decode_auto(self, out, orig_w, orig_h, ratio, dw, dh):  # output formatını otomatik çöz
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]  # batch=1 ise squeeze
        out = np.squeeze(out)  # gereksiz boyutları at

        if not self._shape_logged:
            print("[TRT] output shape:", out.shape)  # bir kere output shape bas
            self._shape_logged = True

        if out.ndim != 2:
            return []  # 2D değilse decode yok

        cands = []  # aday decode sonuçları

        def add_cand(name, dets):  # aday ekleme helper
            if dets is None:
                return
            conf_sum = 0.0
            for d in dets:
                conf_sum += float(d.get("conf", 0.0))  # toplam conf
            cands.append((name, dets, len(dets), conf_sum))  # (isim, detler, count, sum)

        r, c = int(out.shape[0]), int(out.shape[1])  # shape

        if r >= 6 and c > r:  # (C,N) gibi duruyorsa
            if DECODE_MODE in ("auto", "noobj"):
                add_cand("cn_noobj", self._decode_cn(out, orig_w, orig_h, ratio, dw, dh, False))  # noobj
            if DECODE_MODE in ("auto", "obj"):
                add_cand("cn_obj", self._decode_cn(out, orig_w, orig_h, ratio, dw, dh, True))  # obj

        out_t = out.T  # transpose dene
        rt, ct = int(out_t.shape[0]), int(out_t.shape[1])
        if rt >= 6 and ct > rt:  # transpose (C,N) olabilir
            if DECODE_MODE in ("auto", "noobj"):
                add_cand("t_cn_noobj", self._decode_cn(out_t, orig_w, orig_h, ratio, dw, dh, False))
            if DECODE_MODE in ("auto", "obj"):
                add_cand("t_cn_obj", self._decode_cn(out_t, orig_w, orig_h, ratio, dw, dh, True))

        if DECODE_MODE in ("auto", "n6"):  # N,6 formatı da denensin istenirse
            if c >= 6:
                add_cand("n6", self._decode_n6(out, orig_w, orig_h, ratio, dw, dh))
            if out_t.shape[1] >= 6:
                add_cand("t_n6", self._decode_n6(out_t, orig_w, orig_h, ratio, dw, dh))

        if len(cands) == 0:
            return []  # hiç aday yok

        cands.sort(key=lambda x: (x[2], x[3]), reverse=True)  # en çok det, sonra conf_sum
        best = cands[0]  # en iyi aday

        now = time.time()
        if now - self._dbg_t >= LOG_EVERY_SEC:  # belli aralıkla debug
            brief = ", ".join(["{}:{}".format(x[0], x[2]) for x in cands[:4]])
            print("[TRTDBG] pick={} det={} | {}".format(best[0], best[2], brief))
            self._dbg_t = now

        return best[1]  # det listesi

    def predict(self, bgr):  # tek frame için infer + decode
        h, w = bgr.shape[:2]  # orijinal boyut
        x, ratio, dw, dh = self.preprocess(bgr)  # preprocess
        outs = self.infer_raw(x)  # TRT infer !!!!!!!!!
        if not outs:
            return []  # output yok
        return self.decode_auto(outs[0], w, h, ratio, dw, dh)  # decode

# =========================
# THREADS
# =========================
def capture_loop():  # kamera frame’lerini sürekli latest_raw’a yazar
    global running, latest_raw, latest_raw_seq
    cap = None  # capture objesi

    while running:  # ana loop
        if cap is None:  # kamera kapalıysa aç
            cap, _ = open_camera()
            if cap is None:
                print("[CAM] Kamera acilamadi, 2sn sonra...")
                time.sleep(2.0)
                continue

        ok, frame = cap.read()  # frame oku
        if not ok or frame is None:
            print("[CAM] Frame yok, kamera reset...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(0.4)
            continue

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:  # boyut uyuşmuyorsa resize
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        frame = np.ascontiguousarray(frame)  # contiguous yap (OpenCV/Writer için)

        with state_lock:  # shared state lock
            latest_raw = frame  # son frame
            latest_raw_seq += 1  # seq artır

    try:
        if cap is not None:
            cap.release()  # çıkışta release
    except Exception:
        pass

def infer_loop():  # inference yapan thread
    global running, latest_dets, latest_det_ts, latest_infer_ms, latest_trt_ok, latest_meta

    infer_enabled = bool(ENABLE_INFERENCE and HAS_TRT)  # inference koşulu !!!!!!!!!
    detector = None  # TRTDetector
    ctx = None  # CUDA context
    frame_counter = 0  # kaç frame işlendi

    if infer_enabled:
        try:
            cuda.init()  # CUDA init !!!!!!!!!
            ctx = cuda.Device(0).make_context()  # context yarat !!!!!!!!! (GPU işleri)
            detector = TRTDetector(ENGINE_PATH)  # engine yükle !!!!!!!!!
            latest_trt_ok = True  # TRT ok
            print("[TRT] Hazir.")
        except Exception as e:
            print("[TRT] Baslatilamadi, inference kapaniyor:", e)
            infer_enabled = False
            latest_trt_ok = False
            detector = None

    while running:
        with state_lock:
            frame = None if latest_raw is None else latest_raw.copy()  # frame kopya !!!!!!!!! (kopya maliyeti)
            seq = int(latest_raw_seq)  # seq al

        if frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1  # sayaç
        run_infer = infer_enabled and detector is not None and (frame_counter % INFER_EVERY_N == 0)  # infer şartı !!!!!!!!! (N=1 ağır)

        if run_infer:
            t0 = time.time()  # süre ölç
            dets = []
            try:
                dets = detector.predict(frame)  # TRT infer+decode !!!!!!!!! (en büyük gecikme/ısı)
            except Exception as e:
                print("[TRT] infer hata:", e)
                dets = []
            infer_ms = (time.time() - t0) * 1000.0  # ms cinsinden

            with state_lock:
                latest_dets = dets  # detleri kaydet
                latest_det_ts = time.time()  # det timestamp
                latest_infer_ms = float(infer_ms)  # infer süresi
                latest_meta = {  # metadata güncelle
                    "ts": time.time(),
                    "seq": int(seq),
                    "infer_ms": float(infer_ms),
                    "detections": dets,
                    "det_count": len(dets),
                    "trt_ok": bool(latest_trt_ok)
                }
        else:
            with state_lock:
                latest_meta = {  # infer yokken de meta akmaya devam etsin
                    "ts": time.time(),
                    "seq": int(seq),
                    "infer_ms": float(latest_infer_ms),
                    "detections": list(latest_dets),
                    "det_count": len(latest_dets),
                    "trt_ok": bool(latest_trt_ok)
                }

        time.sleep(0.001 if infer_enabled else 0.01)  # küçük sleep (CPU’yu %100 kilitlemesin)

    try:
        if ctx is not None:
            ctx.pop()  # context pop
            ctx.detach()  # context detach
    except Exception:
        pass

def stream_loop():  # SRT ile videoyu gönderen thread
    global running
    wr = None  # VideoWriter
    sent = 0  # gönderilen frame sayısı
    t_log = time.time()  # log timer

    while running:
        if wr is None:  # writer yoksa aç
            wr = open_writer()
            if wr is None:
                print("[SRT] Writer acilamadi, 2sn sonra...")
                time.sleep(2.0)
                continue

        with state_lock:
            frame = None if latest_raw is None else latest_raw.copy()  # en yeni frame kopyası !!!!!!!!! (kopya)
            dets = list(latest_dets)  # en son det listesi
            det_ts = float(latest_det_ts)  # det zamanı
            infer_ms = float(latest_infer_ms)  # infer süresi

        if frame is None:
            time.sleep(0.01)
            continue

        if STREAM_ANNOTATED and len(dets) > 0:  # kutu çizme açık ve det var
            if KEEP_LAST_DET_ALWAYS or ((time.time() - det_ts) <= DET_STALE_SEC):  # stale kontrol
                frame = draw_dets(frame, dets)  # kutuları çiz !!!!!!!!! (CPU yük)

        frame = np.ascontiguousarray(frame)  # contiguous

        try:
            wr.write(frame)  # pipeline’a yaz -> NVENC+SRT !!!!!!!!! (yük + ısı)
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
        if now - t_log >= LOG_EVERY_SEC:  # periyodik log
            print("[SRT] sent={} det={} infer_ms={:.1f}".format(sent, len(dets), infer_ms))
            t_log = now

    try:
        if wr is not None:
            wr.release()  # çıkışta release
    except Exception:
        pass

def meta_loop():  # UDP metadata gönderen thread
    global running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP socket
    print("[META] UDP -> {}:{}".format(PC_IP, META_PORT))  # hedef log

    while running:
        with state_lock:
            payload = dict(latest_meta)  # meta snapshot
        try:
            sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, META_PORT))  # UDP gönder
        except Exception:
            pass
        time.sleep(0.03)  # ~33Hz meta gönderimi

    try:
        sock.close()  # socket kapat
    except Exception:
        pass

def main():  # ana giriş noktası
    global running
    print("[SYS] OpenCV:", cv2.__version__)  # OpenCV sürümü
    print("[SYS] GStreamer:", opencv_has_gstreamer())  # GStreamer var mı
    print("[SYS] HAS_TRT:", HAS_TRT, "ENABLE_INFERENCE:", ENABLE_INFERENCE)  # TRT durumu
    print("[SYS] STREAM_ANNOTATED:", STREAM_ANNOTATED, "DECODE_MODE:", DECODE_MODE)  # config özeti

    if not opencv_has_gstreamer():  # GStreamer yoksa çık
        print("[ERR] OpenCV GStreamer yok. Bu haliyle calismaz.")
        return

    threads = [  # 4 thread’i başlatacağız
        threading.Thread(target=capture_loop, daemon=True),  # capture thread
        threading.Thread(target=infer_loop, daemon=True),  # inference thread
        threading.Thread(target=stream_loop, daemon=True),  # streaming thread
        threading.Thread(target=meta_loop, daemon=True),  # metadata thread
    ]

    for t in threads:
        t.start()  # thread start

    print("[SYS] Calisiyor. Cikmak icin Ctrl+C")  # kullanıcı bilgilendirme
    try:
        while True:
            time.sleep(0.5)  # main thread boşta bekler
    except KeyboardInterrupt:
        running = False  # Ctrl+C ile thread’lere dur sinyali
        time.sleep(1.0)  # kapanma için kısa bekleme
        print("[SYS] Bitti.")  # çıkış log

if __name__ == "__main__":
    main()  # script direkt çalıştırıldıysa main()
