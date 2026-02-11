import cv2
import time
import threading
from pymavlink import mavutil

# =========================
# KONFİGÜRASYON
# =========================
SRT_LISTEN_PORT = 9000

# MAVLink bağlantı örnekleri:
# 1) FC, companion üzerinden UDP yolluyorsa:
MAVLINK_CONN_STR = "udp:0.0.0.0:14550"

# 2) USB-serial direkt PC'ye bağlıysa şu format:
# MAVLINK_CONN_STR = "/dev/ttyACM0"
# MAVLINK_BAUD = 57600

MAVLINK_BAUD = 57600
MAVLINK_SOURCE_SYS = 255
MAVLINK_SOURCE_COMP = 190

PRINT_INTERVAL_SEC = 0.5  # Telemetriyi terminale basma aralığı


class MavlinkReceiver(threading.Thread):
    def __init__(self, conn_str, baud=57600):
        super().__init__(daemon=True)
        self.conn_str = conn_str
        self.baud = baud
        self._stop = threading.Event()

        self.master = None
        self.last_heartbeat = 0
        self.last_print = 0

        # Son telemetri
        self.data = {
            "mode": "N/A",
            "armed": "N/A",
            "lat": None,
            "lon": None,
            "alt_m": None,
            "groundspeed": None,
            "roll": None,
            "pitch": None,
            "yaw": None,
            "battery_v": None
        }
        self.lock = threading.Lock()

    def stop(self):
        self._stop.set()

    def connect(self):
        while not self._stop.is_set():
            try:
                print(f"[MAV] Bağlanıyor: {self.conn_str}")
                if self.conn_str.startswith("/dev/"):
                    self.master = mavutil.mavlink_connection(
                        self.conn_str,
                        baud=self.baud,
                        source_system=MAVLINK_SOURCE_SYS,
                        source_component=MAVLINK_SOURCE_COMP,
                        autoreconnect=True,
                        dialect="common"
                    )
                else:
                    self.master = mavutil.mavlink_connection(
                        self.conn_str,
                        source_system=MAVLINK_SOURCE_SYS,
                        source_component=MAVLINK_SOURCE_COMP,
                        autoreconnect=True,
                        dialect="common"
                    )

                # MAVLink2 zorla
                self.master.mav.set_proto_version(2)

                print("[MAV] Heartbeat bekleniyor...")
                self.master.wait_heartbeat(timeout=10)
                self.last_heartbeat = time.time()
                print("[MAV] Heartbeat alındı. Bağlantı tamam.")
                return
            except Exception as e:
                print(f"[MAV] Bağlanamadı: {e}. 2 sn sonra tekrar denenecek.")
                time.sleep(2)

    def request_streams(self):
        # ArduPilot/PX4 uyumu için iki farklı yöntem, hangisi desteklenirse o çalışır.
        try:
            # Legacy data stream request
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # Hz
                1
            )
        except Exception:
            pass

        try:
            # MAV_CMD_SET_MESSAGE_INTERVAL örnekleri (mikrosaniye)
            def set_interval(msg_id, hz):
                interval_us = int(1e6 / hz)
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,
                    interval_us,
                    0, 0, 0, 0, 0
                )
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20)
            set_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2)
        except Exception:
            pass

    def parse_msg(self, msg):
        mtype = msg.get_type()
        with self.lock:
            if mtype == "HEARTBEAT":
                self.last_heartbeat = time.time()
                try:
                    self.data["mode"] = mavutil.mode_string_v10(msg)
                except Exception:
                    pass
                self.data["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

            elif mtype == "GLOBAL_POSITION_INT":
                self.data["lat"] = msg.lat / 1e7
                self.data["lon"] = msg.lon / 1e7
                self.data["alt_m"] = msg.relative_alt / 1000.0
                self.data["groundspeed"] = (msg.vx**2 + msg.vy**2) ** 0.5 / 100.0

            elif mtype == "ATTITUDE":
                self.data["roll"] = msg.roll
                self.data["pitch"] = msg.pitch
                self.data["yaw"] = msg.yaw

            elif mtype == "SYS_STATUS":
                # mV -> V
                if msg.voltage_battery != 65535:
                    self.data["battery_v"] = msg.voltage_battery / 1000.0

    def print_status(self):
        with self.lock:
            d = dict(self.data)
        print(
            f"[TEL] mode={d['mode']} armed={d['armed']} "
            f"lat={d['lat']} lon={d['lon']} alt={d['alt_m']}m "
            f"spd={d['groundspeed']}m/s roll={d['roll']} pitch={d['pitch']} yaw={d['yaw']} "
            f"bat={d['battery_v']}V"
        )

    def run(self):
        while not self._stop.is_set():
            self.connect()
            if self.master is None:
                time.sleep(1)
                continue

            self.request_streams()

            while not self._stop.is_set():
                try:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    now = time.time()

                    if msg is not None:
                        self.parse_msg(msg)

                    # heartbeat timeout -> reconnect
                    if now - self.last_heartbeat > 5:
                        raise RuntimeError("Heartbeat timeout (5s)")

                    if now - self.last_print >= PRINT_INTERVAL_SEC:
                        self.last_print = now
                        self.print_status()

                except Exception as e:
                    print(f"[MAV] Bağlantı koptu / hata: {e}. Yeniden bağlanılıyor...")
                    try:
                        if self.master:
                            self.master.close()
                    except Exception:
                        pass
                    self.master = None
                    time.sleep(1)
                    break


def build_gst_srt_pipeline(port: int) -> str:
    # SRT listener: PC'de dinle
    # tsdemux + h264 decode + appsink -> OpenCV
    return (
        f"srtsrc uri=srt://0.0.0.0:{port}?mode=listener&latency=120 "
        f"! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true sync=false"
    )


def main():
    # MAVLink thread
    mav_thread = MavlinkReceiver(MAVLINK_CONN_STR, baud=MAVLINK_BAUD)
    mav_thread.start()

    # Video receiver
    pipeline = build_gst_srt_pipeline(SRT_LISTEN_PORT)
    print(f"[VID] Pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[VID] Video açılamadı. GStreamer/SRT pipeline kontrol et.")
        mav_thread.stop()
        return

    print("[VID] Video akışı başladı. Çıkış için 'q'.")

    last_frame_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        now = time.time()

        if not ok:
            # kısa bekleyip tekrar dene
            time.sleep(0.01)
            if now - last_frame_time > 3:
                cv2.putText(
                    frame if frame is not None else (255 * cv2.UMat(480, 640, cv2.CV_8UC3)).get(),
                    "VIDEO SIGNAL LOST - waiting...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            continue

        last_frame_time = now
        frame_count += 1

        # FPS hesap
        if frame_count >= 20:
            elapsed = now - (last_frame_time - 0.0001)  # stabilite için
            # daha basit bir hesap:
            # bu satır gerçekçi olsun diye ayrı ölçelim
            # ama pratikte gösterim için:
            fps = 20.0 / max((now - (last_frame_time - 0.67)), 1e-3)  # yaklaşık
            frame_count = 0

        # Ekrana telemetri overlay (opsiyonel)
        cv2.putText(frame, "SRT VIDEO + MAVLINK2 TELEMETRY", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Press 'q' to quit", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Ground Station", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mav_thread.stop()
    print("[SYS] Kapandı.")


if __name__ == "__main__":
    main()