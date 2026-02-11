#!/usr/bin/env bash
set -euo pipefail

# =========================
# AYARLAR
# =========================
PC_IP="${1:-192.168.1.50}"        # PC'nin IP'si (parametre ile geçebilirsin)
SRT_PORT="${2:-9000}"             # PC tarafında dinlenen SRT portu
WIDTH="${3:-1280}"
HEIGHT="${4:-720}"
FPS="${5:-30}"
BITRATE_KBPS="${6:-3000}"         # 3000 kbps başlangıç için iyi

echo "[INFO] PC_IP=$PC_IP PORT=$SRT_PORT ${WIDTH}x${HEIGHT}@${FPS} bitrate=${BITRATE_KBPS}kbps"

# =========================
# GStreamer pipeline
# libcamera-vid -> h264parse -> mpegtsmux -> srtsink
# =========================
# latency=120ms ve pbkeylen=16 saha testleri için makul.
# wait-for-connection=false ile alıcı yeniden açıldığında yayın pipeline'ı düşmez, devam eder.
exec gst-launch-1.0 -e \
    libcamerasrc ! \
    video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! \
    queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream ! \
    v4l2h264enc extra-controls="controls,video_bitrate=${BITRATE_KBPS}000;" ! \
    h264parse config-interval=1 ! \
    mpegtsmux ! \
    srtsink uri="srt://${PC_IP}:${SRT_PORT}?mode=caller&latency=120&pbkeylen=16&messageapi=false" \
    wait-for-connection=false sync=false async=false
