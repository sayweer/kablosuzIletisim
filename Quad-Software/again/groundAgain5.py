#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yer İstasyonu – SRT Viewer (Ubuntu)
Python 3.6+ uyumlu | v2 – Saha Testi Hazır

Tüm düzeltmeler:
  - appsink max-buffers=1  → pipeline hiç gecikme biriktirmez (Gemini'nin doğru tespiti)
  - appsink drop=true      → tampon dolduğunda eski kare atılır, yeni kare hemen gelir
  - FPS rate-limiter       → CPU gereksiz yere yanmaz (drop=true ile lag riski yok)
  - exponential back-off   → bağlantı koptuğunda akıllıca yeniden bağlanır
  - f-string yok           → Python 3.6 uyumlu
"""

import cv2
import time
import sys

SRT_PORT        = 9000
SRT_LATENCY_MS  = 150       # jetson_stream.py ile aynı olmalı!
SRT_PAYLOADSIZE = 1316

WINDOW          = "SRT Viewer - Yer Istasyonu"
DISPLAY_FPS     = 30        # görüntüleme döngüsü FPS sınırı
MAX_RETRY_DELAY = 8.0       # saniye – maksimum yeniden bağlanma bekleme süresi


def opencv_has_gstreamer():
    """OpenCV'nin GStreamer desteğiyle derlenip derlenmediğini kontrol eder."""
    try:
        build_info = cv2.getBuildInformation()
        # Satır satır kontrol – yanlış pozitifi önler
        for line in build_info.splitlines():
            if "GStreamer" in line and "YES" in line:
                return True
        return False
    except Exception:
        return False


def build_pipe():
    """SRT dinleyici GStreamer pipeline'ı döndürür."""
    uri = "srt://:{port}?mode=listener&latency={lat}&transtype=live&payloadsize={ps}".format(
        port=SRT_PORT, lat=SRT_LATENCY_MS, ps=SRT_PAYLOADSIZE
    )
    # CPU decode – Ubuntu'da tüm OpenCV kurulumlarında çalışır
    # avdec_h264 max-threads=2 ile CPU yükü sınırlandırıldı
    return (
        'srtsrc uri="{uri}" ! '
        'queue max-size-buffers=8 leaky=downstream ! '
        'tsdemux ! '
        'queue max-size-buffers=8 leaky=downstream ! '
        'h264parse ! '
        'avdec_h264 max-threads=2 ! '
        'videoconvert ! '
        'video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    ).format(uri=uri)


def try_open(pipe):
    """VideoCapture açmayı dener, başarısızsa None döner."""
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV bu sistemde GStreamer destegi olmadan derlenmis!")
        print("      Cozum: sudo apt install libgstreamer1.0-dev "
              "gstreamer1.0-plugins-bad gstreamer1.0-plugins-good")
        print("      Ardindan OpenCV'yi kaynaktan derleyin.")
        sys.exit(1)

    pipe = build_pipe()
    print("[PIPE]\n{}\n".format(pipe))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    frame_interval = 1.0 / DISPLAY_FPS
    retry_delay    = 1.0
    cap            = None
    fullscreen     = False

    while True:
        # ── Bağlantı / Yeniden bağlanma (exponential back-off) ──────────
        if cap is None or not cap.isOpened():
            print("[VID] Baglanti bekleniyor... ({}s sonra yeniden denenecek)".format(
                int(retry_delay)))
            cap = try_open(pipe)
            if cap is None:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)
                continue
            retry_delay = 1.0   # başarılı bağlantıda sıfırla
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("[VID] Baglandi! Cozunurluk: {}x{}".format(w, h))

        # ── Kare oku ────────────────────────────────────────────────────
        loop_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Kare okunamadi – stream koptu, yeniden baglaniliyor...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(1.0)
            continue

        cv2.imshow(WINDOW, frame)

        # ── Klavye kontrolleri ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):           # q veya ESC → çıkış
            break
        elif key == ord('f'):               # f → tam ekran geçiş
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, prop)

        # ── FPS rate-limiter (CPU kullanımını düşürür) ──────────────────
        elapsed = time.perf_counter() - loop_start
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Temizlik ────────────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[VID] Kapatildi.")


if __name__ == "__main__":
    main()