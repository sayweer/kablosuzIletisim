import random
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import threading
from pymavlink import mavutil
import time
import datetime
import os
import shutil
import math
import tkinter as tk

import signal
import sys

from get_video import CAM


class attitude:
    def __init__(self,roll,pitch,yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class location:
    def __init__(self,lat,lon,alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

class Drone:
    def __init__(self,ip,baud=115200):
      self.vehicle = None

      os.environ["MAVLINK20"] = "1"  # MAVLink 2.0'ı zorunlu kıl
      self.vehicle = mavutil.mavlink_connection(str(ip),baud=baud)
      self.vehicle.wait_heartbeat() #drondan sinyal gelip gelmediğini kontrol ediyor.
      print("baglandi")
      time.sleep(1)

    def arm_disarm(self,arm):
        if arm:
            arm=1
        elif arm==0:
            arm=0
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            arm, 0, 0, 0, 0, 0, 0)
        if arm == 0:
            print("Disarm olana kadar bekleniyor...")
            self.vehicle.motors_disarmed_wait()#motors_disarmed_wait()
            print('Disarmed')
            time.sleep(1)
        if arm == 1:
            print("Arm olana kadar bekleniyor...")
            self.vehicle.motors_armed_wait()#motors_disarmed_wait()
            print('Armed')
            time.sleep(1)

    @property
    def mode(self):
        while True:
            msg = self.vehicle.recv_match(type = 'HEARTBEAT', blocking = True)
            if msg:
                msg = msg.to_dict()
                # Mode mapping sözlüğünü al
                mode_mapping = self.vehicle.mode_mapping()

                # Sözlüğü ters çevirerek mode_id -> mode_string eşlemesi oluştur
                reverse_mode_mapping = {v: k for k, v in mode_mapping.items()}

                # Belirli bir mode_id'nin karşılık gelen string modunu almak için:
                mode_id = msg.get("custom_mode")  # Örnek ID
                mode_string = reverse_mode_mapping.get(mode_id, "Unknown Mode")
                return mode_string
    
    @mode.setter
    def mode(self, new_mode):
        mode_mapping = self.vehicle.mode_mapping()

        if new_mode not in mode_mapping:
            print(f'Bilinmeyen mod: {new_mode}')
            print('Geçerli modlar:', list(mode_mapping.keys()))
            return

        mode_id = mode_mapping[new_mode]

        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
    
        # Mod değişikliğini bekle
        while self.mode != new_mode:
            print(f"{new_mode} moduna geçiş bekleniyor...")
            time.sleep(0.1)

        print(new_mode)
    
    def takeoff(self, altitude): 
        # Send the takeoff command
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,  # target_system
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,  # command
            0,  # confirmation
            0,  # param1 (pitch angle, 0 for default)
            0,  # param2 (unused)
            0,  # param3 (unused)
            0,  # param4 (unused)
            0,  # param5 (unused)
            0,  # param6 (unused)
            altitude  # param7 (target altitude)
        )
        time.sleep(1)
        # Wait for the drone to reach the target altitude
        while True:
            alt = self.location.alt
            if alt > altitude * 0.95:
                print("yüksekliğe ulaşıldı")
                break
            else:
                print("yükseklik", alt)
        print("Takeoff complete. Proceeding to the mission.")

    def move(self,v_x=0, v_y=0, v_z=0):
        self.vehicle.mav.set_position_target_local_ned_send(
        10,
        self.vehicle.target_component, self.vehicle.target_system,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b100111000111,
        0,0,0,
        v_y,v_x,-v_z,
        0,0,0,
        0,0)

    @property
    def channels(self):
        while True:
            message = self.vehicle.recv_match(type='RC_CHANNELS', blocking=True)
            if message:
                channels_s = [
                    0,
                    message.chan1_raw,
                    message.chan2_raw,
                    message.chan3_raw,
                    message.chan4_raw,
                    message.chan5_raw,
                    message.chan6_raw,
                    message.chan7_raw,
                    message.chan8_raw,
                ]
                
                return channels_s

    @property
    def attitude(self):
        while True:
            msg = self.vehicle.recv_match(type="ATTITUDE",blocking=True)
            if msg:
                msg = msg.to_dict()
                roll = msg.get("roll")
                pitch = msg.get("pitch")
                yaw = msg.get("yaw")
                return attitude(roll,pitch,yaw)

    @property
    def location(self):
        while True:
            msg = self.vehicle.recv_match(type="GLOBAL_POSITION_INT",blocking=True)
            if msg:
                msg = msg.to_dict()
                lat = msg.get("lat") / 1.0e7
                lon = msg.get("lon") / 1.0e7
                alt = msg.get("relative_alt")/ 1000
                return location(lat,lon,alt)

    def get_all_param(self):
        self.vehicle.mav.param_request_list_send(
            self.vehicle.target_system, self.vehicle.target_component
        )
        while True:
            time.sleep(0.01)
            try:
                message = self.vehicle.recv_match(type='PARAM_VALUE', blocking=True).to_dict()
                print('name: %s\tvalue: %d' % (message['param_id'],
                                            message['param_value']))
            except Exception as error:
                print(error)
                sys.exit(0)

    def set_param(self, param_name, param_value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
        """Belirtilen parametreyi geçici olarak RAM'e yazar."""
        self.vehicle.mav.param_set_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            param_name.encode(),  # Parametre adını byte olarak gönderiyoruz
            float(param_value),   # Değer float olarak gönderiliyor
            param_type            # Parametre tipi (varsayılan: float32)
        )
        print(f"{param_name} parametresi {param_value} olarak ayarlandı.")

    def readmission(self, aFileName):
        missionlist = []
        with open(aFileName) as f:
            for i, line in enumerate(f):
                if i == 0:
                    if not line.startswith('QGC WPL 110'):
                        raise Exception('File is not supported WP version')
                else:
                    linearray = line.split('\t')
                    missionlist.append({
                        'seq': int(linearray[0]),
                        'frame': int(linearray[2]),
                        'command': int(linearray[3]),
                        'param1': float(linearray[4]),
                        'param2': float(linearray[5]),
                        'param3': float(linearray[6]),
                        'param4': float(linearray[7]),
                        'x': float(linearray[8]),
                        'y': float(linearray[9]),
                        'z': float(linearray[10]),
                        'autocontinue': int(linearray[11].strip())
                    })
        return missionlist

    def upload_mission(self, aFileName):
        missionlist = self.readmission(aFileName)
        for item in missionlist:
            self.vehicle.mav.mission_item_send(
                self.vehicle.target_system,
                self.vehicle.target_component,
                item['seq'],
                item['frame'],
                item['command'],
                0,  # current
                item['autocontinue'],
                item['param1'],
                item['param2'],
                item['param3'],
                item['param4'],
                item['x'],
                item['y'],
                item['z']
            )
        print("Mission uploaded")

    def get_waypoints(self):

        self.vehicle.mav.mission_request_list_send(
            self.vehicle.target_system,
            self.vehicle.target_component
        )

        msg = None
        while msg is None:
            msg = self.vehicle.recv_match(type='MISSION_COUNT', blocking=True)
        
        waypoint_count = msg.count
        print(f"Toplam {waypoint_count} waypoint var.")
        waypoints = []


        for i in range(waypoint_count):
            self.vehicle.mav.mission_request_send(
                self.vehicle.target_system,
                self.vehicle.target_component,
                i
            )

            msg = None
            while msg is None:
                msg = self.vehicle.recv_match(type='MISSION_ITEM', blocking=True)

            if msg is None:
                print(f"Waypoint {i} alınamadı!")
                continue
       
            waypoints.append({
                'seq': msg.seq,
                'frame': msg.frame,
                'command': msg.command,
                'param1': msg.param1,
                'param2': msg.param2,
                'param3': msg.param3,
                'param4': msg.param4,
                'x': msg.x,
                'y': msg.y,
                'z': msg.z,
                'autocontinue': msg.autocontinue
            })
            time.sleep(0.1)

        return waypoints
    
    def download_waypoints(self, file_name):

        waypoints = self.get_waypoints()  
        with open(file_name, 'w') as f:
            f.write('QGC WPL 110\n')  
            for wp in waypoints:
                line = '\t'.join(map(lambda x: f"{x:.8f}" if isinstance(x, float) else str(x), wp.values()))  
                f.write(line + '\n')  
        print(f"Waypoint'ler {file_name} dosyasına kaydedildi.")

    @property
    def total_waypoints(self):
        return len(self.get_waypoints())
            
    def wait_mission_ready(self, timeout=10):
        self.vehicle.mav.mission_request_list_send(
            self.vehicle.target_system,
            self.vehicle.target_component
        )

        start_time = time.time()
        msg = None
        while msg is None:
            if time.time() - start_time > timeout:
                raise TimeoutError("Görev sayısı zaman aşımına uğradı!")
            
            msg = self.vehicle.recv_match(type='MISSION_COUNT', blocking=True)
            time.sleep(0.1)
        
        waypoint_count = msg.count
        print(f"Toplam {waypoint_count} görev var.")
        
        waypoints = []
        for i in range(waypoint_count):
            self.vehicle.mav.mission_request_send(
                self.vehicle.target_system,
                self.vehicle.target_component,
                i
            )
            msg = None
            while msg is None:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Görev öğesi zaman aşımına uğradı!")
                    
                msg = self.vehicle.recv_match(type='MISSION_ITEM', blocking=True)
                time.sleep(0.1)
            
            waypoints.append(msg)
        
        if len(waypoints) == waypoint_count:
            print("Görev listesi hazır!")
        else:
            print("Görev listesi tamamlanamadı!")


    def image_basla(self):
        self.image = CAM()
        threading.Thread(target=self.image.basla).start()
    
    def Control(self):
        print("Control başladı")
        self.random = 0
        self.vehicle.mode = "GUIDED"

        self.pid_x = PIDController(Kp=0.5, Ki=0.015, Kd=0.005)
        self.pid_y = PIDController(Kp=0.4, Ki=0.015, Kd=0.005)

        self.pid_x_values = []
        self.pid_target_x_values = []
        self.pid_y_values = []
        self.pid_target_y_values = []
        self.time_values = []

        threading.Thread(target=self.control_target).start()
        threading.Thread(target=self.force_quit).start()
        print("IP ile bağlantı sağlandı")
        time.sleep(1)



    def randomMove(self):
        self.randomNumber=random.randint(0,300)
        if self.randomNumber==3:
            print("Random calisti.........................................................")
            self.random = 1
            self.move(v_z=0,v_x=2,v_y=2)
            time.sleep(2)
            self.random = 0


    def control_target(self):
        self.pid_output_x = 0.1
        baslangic = time.time()
        print("baslangic",baslangic)
        x_vector = None
        y_vector = None
        while True:
            onceki_time = time.time()
            if self.image.quit:
                self.quit()
                break
            
            pitch = round(self.vehicle.attitude.pitch, 5)
            pitch = int(((pitch) * 180) / (3.14159265359))
            frame_center_y_error = int(np.interp(pitch, [-28, 28], [-540, +540]))

            roll = round(self.vehicle.attitude.roll, 5)
            roll = int(((roll) * 180) / (3.14159265359))
            frame_center_x_error = int(np.interp(roll, [-40, 40], [-960, +960]))


            # Image processing for detecting target object
            if self.image.w and self.image.h and self.image.center_point is not None:
                
                if self.random == 0:
                    threading.Thread(target=self.randomMove).start()
                
                self.image.frame_center_y_error = frame_center_y_error
                self.image.frame_center_x_error = frame_center_x_error

                frame_height, frame_width = self.image.height, self.image.width
                frame_center_x = (frame_width / 2) + frame_center_x_error
                frame_center_y = (frame_height / 2) + frame_center_y_error
                box_x, box_y = self.image.center_point[0], self.image.center_point[1]



                error_x = box_x - frame_center_x
                error_y = box_y - frame_center_y
                # For x_vector (horizontal movement), use the PID controller output
                er_x_degree = np.interp(error_x, [-960, 960], [-40, +40])
                x_vector = round(er_x_degree/5,2)


                # For y movement (vertical), we can use a similar approach if necessary
                er_y_degree = -(np.interp(error_y, [-540, 540], [-28, +28]))
                y_vector = round(er_y_degree/5,2)

                
                self.pid_output_x, self.pid_output_y = self.pid_x.calculate(x_vector), self.pid_y.calculate(y_vector)
                self.pid_output_x, self.pid_output_y = round(self.pid_output_x,1),round(self.pid_output_y,1)

                if self.random == 0:
                    if self.vehicle.location.global_relative_frame.alt > 10:
                        self.move_drone(0.5)
                    elif self.vehicle.location.global_relative_frame.alt > 5:
                        self.move_drone(0.3)
                    elif self.vehicle.location.global_relative_frame.alt <= 5 and self.vehicle.location.global_relative_frame.alt > 3:
                        #self.randomMove()
                        self.move_drone(0.2)
                    elif self.vehicle.location.global_relative_frame.alt <= 3 and self.vehicle.location.global_relative_frame.alt > 2:
                        self.move_drone(0.5)
                    else:
                        self.move(v_y=0, v_x=0, v_z=0)


                print("x_vector: ", x_vector, "\ty_vector: ", y_vector)
                print("pid_output_x: ", self.pid_output_x, "\tpid_output_y: ", self.pid_output_y, "\tyükseklik: ", self.vehicle.location.global_relative_frame.alt)
                time.sleep(0.05)
                sure = time.time()-onceki_time
                #print(1/sure)
            
            else:
                if x_vector and y_vector:
                    print("Nesne kayboldu, son görülen konuma gidiliyor...")
                    self.move(v_y=y_vector/2, v_x=x_vector/2, v_z=0)
                    time.sleep(0.05)
                """
                self.image.frame_center_y_error = 0
                self.image.frame_center_x_error = 0
                self.pid_x_values.append(-10)
                self.pid_target_x_values.append(-10)
                self.time_values.append(time.time() - baslangic)

                self.image.frame_center_y_error = 0
                self.image.roll = 0
                time.sleep(0.1)
                print("yaw")
                self.pid_x.last_time = time.time()
                """

    def move_drone(self, aralik):
        if abs(self.pid_output_x) < aralik and abs(self.pid_output_y) < aralik:
            if self.vehicle.location.global_relative_frame.alt > 10:
                self.move(v_y=self.pid_output_y, v_x=self.pid_output_x, v_z=-2)
            else:
                self.move(v_y=self.pid_output_y, v_x=self.pid_output_x, v_z=-1)
        else:
            self.move(v_y=self.pid_output_y, v_x=self.pid_output_x, v_z=0)


    def quit(self):
        x = datetime.datetime.now()
        x = str(x)
        if 'data_log' in os.listdir():
            pass
        else:
            os.mkdir('data_log')
            time.sleep(1)
        os.mkdir("data_log/"+x)
        f = open(f"data_log/{x}/myfile.txt", "w")
        for i in range(len(self.pid_x_values)):
            f.write(str(self.time_values[i]) + "," + str(self.pid_target_x_values[i]) + "," + str(self.pid_x_values[i]) + "\n")
        time.sleep(0.3)
        shutil.move("output3.mp4", f"data_log/{x}/output3.mp4")
        d = open(f"data_log/{x}/degerler.txt", "w")
        d.write(f"Kp: {self.pid_x.Kp}\nKi: {self.pid_x.Ki}\nKd: {self.pid_x.Kd}")
        self.image.quit = True
        self.vehicle.close()
    
    def force_quit(self):
        # RC_CHANNELS mesajlarını dinlemek için bir listener ekleniyor
        @self.vehicle.on_message('RC_CHANNELS')
        def RCIN_listener(_, name, message):
            self.channelsmy = [
                0,
                message.chan1_raw,
                message.chan2_raw,
                message.chan3_raw,
                message.chan4_raw,
                message.chan5_raw,
                message.chan6_raw,
                message.chan7_raw,
                message.chan8_raw,
            ]
            #print(self.channelsmy)
            if self.channelsmy[7] > 1500:
                self.win.quit()
                print("Kanal 7 kapatıldı")
                print("Kanal 7 kapatıldı")
                self.image.quit = True
                self.vehicle.close()
                
            else:
                #print("Kanal 7 kapalı")
                pass

class PIDController:
    def __init__(self, Kp, Ki, Kd, integral_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0
        self.previous_derivative = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit

    def calculate(self, error):
        current_time = time.time()
        dt = max(current_time - self.last_time, 1e-6)

        # Proportional term
        proportional = self.Kp * error

        # Integral term with windup protection
        self.integral += error * dt
        if self.integral_limit:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        integral = self.Ki * self.integral

        # Derivative term with noise filtering
        derivative = (error - self.previous_error) / dt
        self.previous_derivative = 0.3 * derivative + 0.9 * self.previous_derivative
        derivative = self.Kd * self.previous_derivative

        # PID output
        output = proportional + integral + derivative

        # Update state
        self.previous_error = error
        self.last_time = current_time

        return output