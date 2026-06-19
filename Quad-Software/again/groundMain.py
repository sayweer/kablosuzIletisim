#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Station (Ubuntu) – SRT Viewer (MPEG-TS/H264)

AMAÇ:
- Jetson'dan SRT ile gelen MPEG-TS/H264 videoyu aç
- Decoder öncesi bitstream drop YAPMA (karınca/macroblock sebebi)
- Gecikme birikmesin diye appsink'te drop et (raw frame drop)

NOTLAR:
- drop=true + max-buffers=1 -> okuma yavaşlarsa eski frame atılır, birikmiş lag oluşmaz.
- sync=false -> timestamp'e göre beklemez, "en günceli ver" mantığı.
"""

import cv2
import time
import sys

SRT_PORT        = 9000
SRT_LATENCY_MS  = 300    # 150 düşük gecikme ama saha riskli olabilir; 300-600 daha stabil
SRT_PAYLOADSIZE = 1316

WINDOW          = "Ground Station - SRT Viewer"
DISPLAY_FPS     = 30
MAX_RETRY_DELAY = 8.0

def opencv_has_gstreamer():
    try:
        info = cv2.getBuildInformation()
        for line in info.splitlines():
            if "GStreamer" in line and "YES" in line:
                return True
        return False
    except Exception:
        return False

def build_pipe(decoder="avdec_h264"):
    """
    KRİTİK: Decoder öncesi queue'lar NON-LEAKY.
    - leaky=downstream olursa TS/H264 parçalanır -> macroblocking.
    - queue max-size-buffers küçük tutulur -> abartı buffer birikmesin.
      (0 demek "limitsiz", istemiyoruz.)

    appsink:
    - max-buffers=1 -> yalnızca 1 frame tutulur
    - drop=true -> yetişemezsek eski frame at
    - sync=false -> saat/timestamp bekleme
    """
    uri = "srt://:{port}?mode=listener&latency={lat}&transtype=live&payloadsize={ps}".format(
        port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE
    )

    return (
        'srtsrc uri="{uri}" ! '
        'queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
        'tsdemux ! '
        'queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
        'h264parse ! '
        '{dec} ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    ).format(uri=uri, dec=decoder)

def try_open(pipes):
    for name, pipe in pipes:
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            # İlk frame'i hızlıca çekmeye çalış
            ok, fr = cap.read()
            if ok and fr is not None and fr.size > 0:
                return cap, name
        try:
            cap.release()
        except Exception:
            pass
    return None, None

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok!")
        sys.exit(1)

    # Decoder denemeleri: CPU garantili, NV varsa o da denenebilir
    pipes = [
        ("avdec_h264", build_pipe("avdec_h264")),
        ("nvh264dec",  build_pipe("nvh264dec")),
    ]

    print("[PIPE] Candidates:")
    for n, p in pipes:
        print(" -", n, ":", p)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    frame_interval = 1.0 / float(DISPLAY_FPS)
    retry_delay = 1.0
    cap = None
    active = None
    fullscreen = False

    while True:
        if cap is None or not cap.isOpened():
            print("[VID] Bekleniyor... ({}s sonra yeniden)".format(int(retry_delay)))
            cap, active = try_open(pipes)
            if cap is None:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)
                continue
            retry_delay = 1.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("[VID] Baglandi! decoder={} res={}x{}".format(active, w, h))

        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Stream koptu, reconnect...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(0.5)
            continue

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, prop)

        # FPS rate limit: CPU boşuna yanmasın.
        # Bu sleep "lag biriktirmez", çünkü appsink drop=true max-buffers=1 var.
        elapsed = time.perf_counter() - t0
        s = frame_interval - elapsed
        if s > 0:
            time.sleep(s)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[SYS] Kapatildi.")

if __name__ == "__main__":
    main()