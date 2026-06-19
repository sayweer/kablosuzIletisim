#!/usr/bin/env bash
set -euo pipefail

# Parametreler
PC_IP="${1:-192.168.1.50}"   
SRT_PORT="${2:-9000}"
WIDTH="${3:-640}"
HEIGHT="${4:-480}"
FPS="${5:-30}"
BITRATE_KBPS="${6:-2000}"

echo "[INFO] RAW H264 Modu -> PC_IP=$PC_IP PORT=$SRT_PORT"

# MPEGTSMUX YOK. Doğrudan H264 parse edip gönderiyoruz.
exec gst-launch-1.0 -v \
    v4l2src device=/dev/video0 ! \
    image/jpeg,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! \
    jpegdec ! \
    videoconvert ! \
    x264enc bitrate=${BITRATE_KBPS} speed-preset=ultrafast tune=zerolatency key-int-max=30 ! \
    video/x-h264,profile=baseline,stream-format=byte-stream ! \
    h264parse config-interval=1 ! \
    srtsink uri="srt://${PC_IP}:${SRT_PORT}?mode=caller&latency=100" \
    wait-for-connection=false sync=false