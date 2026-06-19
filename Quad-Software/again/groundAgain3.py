#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import sys

SRT_PORT        = 9000
SRT_LATENCY_MS  = 150        # Jetson tarafıyla eşleşmeli
SRT_PAYLOADSIZE = 1316

WINDOW          = "SRT Viewer - Yer Istasyonu"
MAX_RETRY_DELAY = 8.0        # Maksimum yeniden bağlanma bekleme süresi

def opencv_has_gstreamer() -> bool:
    try:
        info = cv2.getBuildInformation()
        return "GStreamer" in info and "YES" in info[info.index("GStreamer"):]
    except Exception:
        return False

def build_pipe() -> str:
    uri = "srt://:{port}?mode=listener&latency={lat}&transtype=live&payloadsize={ps}".format(
        port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE
    )
    # Claude'un kodundaki srcsrc hatası srtsrc olarak düzeltildi
    return (
        'srtsrc uri="{uri}" ! '
        'queue max-size-buffers=8 leaky=downstream ! '
        'tsdemux ! '
        'queue max-size-buffers=8 leaky=downstream ! '
        'h264parse ! '
        'avdec_h264 max-threads=2 ! '
        'videoconvert ! '
        'video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=2 sync=false'
    ).format(uri=uri)

def try_open(pipe: str):
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        return None
    return cap

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV bu sistemde GStreamer destegi olmadan derlenmis!")
        sys.exit(1)

    pipe = build_pipe()
    print("[PIPE]\n", pipe, "\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    retry_delay = 1.0
    cap = None
    fullscreen = False

    while True:
        # --- Bağlantı ve Yeniden Bağlanma (Back-off) ---
        if cap is None or not cap.isOpened():
            print(f"[VID] Baglanti bekleniyor... ({int(retry_delay)}s sonra yeniden)")
            cap = try_open(pipe)
            if cap is None:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)
                continue
            
            retry_delay = 1.0   # Başarılı bağlantıda bekleme süresini sıfırla
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[VID] Baglandi! Cozunurluk: {w}x{h}")

        # --- Görüntü Okuma ---
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Kare okunamadi – stream koptu.")
            try: cap.release()
            except Exception: pass
            cap = None
            time.sleep(1.0)
            continue

        cv2.imshow(WINDOW, frame)

        # --- Klavye Kontrolleri ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):           # 'q' veya ESC ile çıkış
            break
        elif key == ord('f'):               # 'f' ile tam ekran (Fullscreen)
            fullscreen = not fullscreen
            flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, flag)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[VID] Kapatildi.")

if __name__ == "__main__":
    main()