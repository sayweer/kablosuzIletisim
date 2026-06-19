#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Jetson Nano Orkestratör
──────────────────────────────────
Başlatma sırası:
  1. Drone bağlan
  2. Vision thread'lerini başlat (kamera + TRT + SRT stream)
  3. Kalkış
  4. Control thread'ini başlat (PID + iniş)
  5. Her iki thread bitene kadar bekle, sonra temiz kapat

Kullanım:
  python3 main.py
"""

import threading
import time
import sys

from mavv22      import Drone
from shared_data import shared
from vision_jetson import run_vision
from control     import Control

# ═══════════════════════════════════════════════════════════════
#  MİSYON KONFIG — buradan ayarla
# ═══════════════════════════════════════════════════════════════
DRONE_IP   = "udpin:0.0.0.0:14550"   # Jetson'a gelen MAVLink portu
DRONE_BAUD = 115200

TAKEOFF_ALT = 10   # metre

# Hedef koordinatı [lat, lon, alt]
HEDEF_KONUM = [40.189881, 29.130876, 10]

# "red" → triangle (kırmızı işaret)
# "blue" → hexagon (mavi işaret)
HEDEF_COLOR = "blue"

# "yuk" | "altigen" | "ucgen"
ORT_CISIM = "altigen"

# [hız0, hız1, hız2, hız3]  — iniş hız listesi (m/s, negatif = iniş)
HIZ_LISTESI = [-0.4, -0.3, -0.2, -0.1]

SERVO_PWM = 2100   # yük bırakma servo pwm

# ═══════════════════════════════════════════════════════════════

def run_control(drone):
    ctrl = Control(
        dron        = drone,
        color       = HEDEF_COLOR,
        ort_cisim   = ORT_CISIM,
        hiz_listesi = HIZ_LISTESI,
        pwm         = SERVO_PWM,
        konum       = HEDEF_KONUM,
    )
    try:
        ctrl.control_target(shared)
    except StopIteration as e:
        print("[MAIN] Görev tamamlandı:", e)
    except Exception as e:
        print("[MAIN] Control hatası:", e)
    finally:
        # Görevi bitir, vision'ı da durdur
        shared.update_data([0, 0, True])
        print("[MAIN] Control thread bitti.")


def main():
    # ── Drone bağlantısı ──────────────────────────────────────
    print("[MAIN] Drone bağlanıyor:", DRONE_IP)
    try:
        drone = Drone(ip=DRONE_IP, baud=DRONE_BAUD)
    except Exception as e:
        print("[MAIN] Drone bağlantı hatası:", e)
        sys.exit(1)

    # ── Vision thread'lerini başlat ───────────────────────────
    vision_thread = threading.Thread(
        target=run_vision, daemon=True, name="vision")
    vision_thread.start()
    print("[MAIN] Vision thread başlatıldı.")

    # Vision'ın ilk frame'i yakalamasını bekle
    print("[MAIN] İlk frame bekleniyor...")
    timeout = time.time() + 15
    while time.time() < timeout:
        b, r, _ = shared.get_center()
        # Herhangi bir detection olmasa da frame geldi mi?
        # Basit yöntem: 2 saniye bekle
        time.sleep(2)
        break
    print("[MAIN] Vision hazır.")

    # ── Kalkış ───────────────────────────────────────────────
    drone.mode = "GUIDED"
    drone.arm_disarm(True)
    drone.takeoff(TAKEOFF_ALT)
    print("[MAIN] Kalkış tamamlandı. Alt:", drone.location.alt)

    # ── Control thread'ini başlat ─────────────────────────────
    ctrl_thread = threading.Thread(
        target=run_control, args=(drone,), daemon=False, name="control")
    ctrl_thread.start()
    print("[MAIN] Control thread başlatıldı.")

    # ── Bitişi bekle ──────────────────────────────────────────
    ctrl_thread.join()
    print("[MAIN] Görev tamamlandı. İniş yapılıyor...")

    try:
        drone.vehicle.set_mode(9)   # LAND
    except Exception as e:
        print("[MAIN] Land komutu hatası:", e)

    print("[MAIN] Program bitti.")


if __name__ == "__main__":
    main()
