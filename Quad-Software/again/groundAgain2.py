#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time

SRT_PORT = 9000
SRT_LATENCY_MS = 100 # SENDER ile eşleşmeli. 400ms yerine 100ms lag sorununu çözer.
SRT_PAYLOADSIZE = 1316

WINDOW = "Ground Station - SRT Viewer"

def opencv_has_gstreamer():
    try:
        bi = cv2.getBuildInformation()
        return ("GStreamer: YES" in bi) or ("GStreamer:                   YES" in bi)
    except Exception:
        return False

def build_pipe():
    uri = "srt://:{}?mode=listener&latency={}&transtype=live&payloadsize={}".format(
        SRT_PORT, SRT_LATENCY_MS, SRT_PAYLOADSIZE
    )
    
    # Eger Ubuntu yer istasyonunda NVIDIA GPU varsa avdec_h264 yerine nvv4l2decoder kullanabilirsiniz.
    # Burada standart ve sorunsuz calisan CPU decoder birakilmistir.
    return (
        "srtsrc uri=\"{uri}\" ! "
        "queue max-size-buffers=4 leaky=downstream ! "
        "tsdemux ! queue max-size-buffers=4 leaky=downstream ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    ).format(uri=uri)

def main():
    if not opencv_has_gstreamer():
        print("[ERR] OpenCV GStreamer destegi yok! Lutfen gstreamer eklentili OpenCV kurun.")
        return

    pipe = build_pipe()
    print("[PIPE]\n", pipe)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    cap = None
    while True:
        if cap is None or not cap.isOpened():
            print("[VID] Jetson Nano'dan Baglanti Bekleniyor...")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                time.sleep(2.0)
                continue
            print("[VID] Baglandi!")

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Stream koptu veya frame hatali. Yeniden baglaniliyor...")
            try: cap.release()
            except Exception: pass
            cap = None
            time.sleep(1.0)
            continue

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()