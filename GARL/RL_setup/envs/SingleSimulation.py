
from __future__ import print_function
import os, time, random, signal, subprocess, math
from pathlib import Path
# Ensure you have the correct gz-transport and gz-msgs versions installed
from dronekit import connect, VehicleMode
import gz.msgs10.world_control_pb2
from pymavlink import mavutil
import numpy as np
from envs.ComputerVision import CameraAprilTagDetector
from envs.create_gz_instance import GzWorldConfig 
#from ComputerVision import CameraAprilTagDetector
import shlex, subprocess, psutil   # psutil = pip install psutil
import dronekit_sitl

from gz.msgs10.vector3d_pb2 import Vector3d
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node
import gz.msgs10.boolean_pb2
from gz.msgs10.world_control_pb2 import WorldControl

# ---------------------------------------------------------
ACCEL_IMPACT_THRESH = 5.0    # [m/s²] delta from running mean that signals impact
EMA_ALPHA           = 0.1    # smoothing factor for running‐mean acceleration
IMU_STREAM_HZ       = 20   # request RAW_IMU at 100Hz for fast reaction
ARMING_CHECK_PARAM  = 1      # 0: Disable all checks, 1: Disable GPS check (faster arming)
SITL_STARTUP_DELAY  = 0      # Seconds to wait for SITL to start
GAZEBO_STARTUP_DELAY= 5    # Seconds to wait for Gazebo to start
CONNECT_RETRIES     = 5      # Number of attempts to connect to DroneKit
CONNECT_TIMEOUT     = 60     # Timeout for each connection attempt
CAMERA_TOPIC_NAME   = "/world/five_by_five_room/model/iris_with_lidar/model/gimbal/link/pitch_link/sensor/camera/image"
GIMBAL_PITCH_CMD_TOPIC = "/gimbal/cmd_pitch" # Topic to command gimbal pitch
# Desired pitch angle for the camera to face downwards (in radians)
# -90 degrees = -pi/2 radians
# The SDF comment says range is -135 to +45 degrees, -90 is within this.
DOWNWARD_PITCH_RADIANS = -math.pi / 2

class Simulation:
    def __init__(self, instance_id):

        self.instance_id = instance_id
        self.connection_str = f"udp:127.0.0.1:{14550}"
        self._acc_ema = None
        self.vehicle = None
        self.sitl_proc = None
        self.gazebo_proc = None
        self._simulation_running = False
        self._drone_ready = False
        self.latest_imu = None
        self.world = None
        self._imu_listener_ref = None # To store reference to the bound method for removal
        self.rocked = False
        self.gui_proc = None


    # ---------------------------------------------------------
    # 1️⃣   Spawn simulator processes
    # ---------------------------------------------------------
    def start_sim(self, world: str, headless):
        """Launch Gazebo Harmonic + ArduPilot SITL for ONE worker."""


        self.world = world
        # ── 1. ArduPilot SITL ----------------------------------------------------
        sitl_cmd = (
            "sim_vehicle.py -v ArduCopter -f gazebo-iris "
            "--model JSON "
            f"--out={self.connection_str} "
            "--add-param-file=$HOME/garl/ardupilot_gazebo/config/gazebo-iris-gimbal.parm " 
            "--no-rebuild --console --speedup 2"
        )
     #   print(sitl_cmd)

        self.sitl_proc = subprocess.Popen(
            sitl_cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

     #   print("SITL: ", self.sitl_proc.pid)
        # ── 2. Gazebo Harmonic (gz-sim) -----------------------------------------
        gazebo_env = os.environ.copy()
        gazebo_env.pop("QT_PLUGIN_PATH", None)
        gazebo_env["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"

        time.sleep(1)

        gazebo_cmd = ( 
                      f"gz sim -s -r -v4 {world} "
                        )
        print(gazebo_cmd)
        self.gazebo_proc = subprocess.Popen(
            gazebo_cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=gazebo_env,
        )
        print("GZ: ", self.gazebo_proc.pid)

        self.processes = [self.sitl_proc.pid, self.sitl_proc.pid + 1, self.gazebo_proc.pid]
        headless = True
        if not headless:
            print("STARTING GUI")
            time.sleep(2)
            gui_cmd = f"gz sim -g"
            self.gui_proc = subprocess.Popen(
            gui_cmd,
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=gazebo_env,
        )   
            time.sleep(2)
            print("GUI: ", self.gui_proc.pid)
            self.processes.append(self.gazebo_proc.pid+1)
            self.processes.append(self.gui_proc.pid)
            self.processes.append(self.gui_proc.pid + 1)
        
   #     print(self.processes)



        self._simulation_running = True
        self.crashed = False
     #   print(f"[sim {self.instance_id}] Harmonic server up on sim{self.instance_id}")

    # ─────────────────────────────────────────────────────────────────────────────
    def dronekit_connect(self, wait_ready: bool = True) -> bool:
        """
        Connect DroneKit to the SITL instance started in `start_sim`.
        """
        self.vehicle = None

        time.sleep(4)
        try:
            
            print("Connecting drone")
            self.vehicle = connect(
                str(self.connection_str),
                wait_ready=True,
                timeout=30,
                rate=10
            )
        except Exception as e:            

            print(f"[sim {self.instance_id}] DroneKit connect failed: {e}")
            self.vehicle = None
            return False

        print(f"[sim {self.instance_id}] Connected to {self.connection_str}")

        # ── optional per-drone init ------------------------------------------------
       # self._start_collision_monitor()
        #self._configure_highrate_imu()

        # channel overrides, GUIDED mode, arming … AIMING THE CAMERA DOWN, ONE CAN OVERRIDE THESE CHANNELS TO ALTER CAMERA DIRECTIONS
        self.vehicle.channels.overrides = {"6": 1500, "7": 1300, "8": 1500}

        self.vehicle.mode = VehicleMode("GUIDED")
        for _ in range(5):
            if self.vehicle.mode.name == "GUIDED":
                break
            time.sleep(1)
            self.vehicle.mode = VehicleMode("GUIDED")

        if self.vehicle.mode.name != "GUIDED":
            print(f"[sim {self.instance_id}] cannot enter GUIDED")
            return False

        print("Vehicle in GUIDED mode")
        try:
            self.vehicle.wait_for_armable()
        except Exception as e:
            print(f"[sim {self.instance_id}] not armable: {e}")
            return False
        
        self.vehicle.armed = True
        time.sleep(0.1)
        self.vehicle.simple_takeoff(2)
        print("VEHICLE TOOK OF")
        self.origin = self.vehicle.location.local_frame
        time.sleep(2)
        self._drone_ready = True

        return True



    def close_sim(self):

        for proc in self.processes:
                try:
                    os.killpg(os.getpgid(proc), signal.SIGKILL)  # Forcefully kill the process group
                    time.sleep(0.5)
                except:
                    continue

        self.gazebo_proc = None
        self.sitl_proc = None
        self.gui_proc = None
    


    def kill_simulation(self):
        if not self._simulation_running:
            print("[*] Simulation not running, skipping kill.")
            return

        self._simulation_running = False
        self._drone_ready = False
        print("[*] Shutting down simulation …")

        self.close_sim()
        if self.vehicle:
            print("[*] Closing DroneKit connection.")
            if self._imu_listener_ref: # Attempt to remove IMU listener
                try:
                    self.vehicle.remove_message_listener('RAW_IMU', self._imu_listener_ref)
                    print("[*] IMU listener removed.")
                except Exception as e:
                    print(f"Error removing IMU listener: {e}")
                self._imu_listener_ref = None # Clear reference
            try:
                self.vehicle.close()
                
            except Exception as e:
                print(f"Error closing DroneKit connection: {e}")
            self.vehicle = None
            time.sleep(1)

        print("[*] All simulator processes and connections cleaned up.")


    # ---------------------------------------------------------
    # 3️⃣   DroneKit connection and flight helpers
    # --------------------------------------------------------

    def reset_pose(self):

        print("PAUZING")

        pauze = Node()
        service_pauze = f"world/{self.world.name.split('.')[0]}/control"
        pauze_msgs = WorldControl()
        pauze_msgs.pause = True
        p = pauze.request(
            service=service_pauze,
            request=pauze_msgs,
            timeout=300,
            request_type=gz.msgs10.world_control_pb2.WorldControl,
            response_type=gz.msgs10.boolean_pb2.Boolean
        )        

        time.sleep(2)

        self.vehicle.mode = "ACRO"
        print(f"Service call succeeded: {p}")

        # teleporting gazebo Pose
        tp = Node()

        pos_msgs = Vector3d()
        pos_msgs.x = 0
        pos_msgs.y = 0
        pos_msgs.z = 0.25

        req = Pose()
        req.name = 'iris_with_lidar'
        req.position.CopyFrom(pos_msgs)
        service_name = f"world/{self.world.name.split('.')[0]}/set_pose"
        print("TPING")

        future = tp.request(
                service=service_name,
                request=req,
                timeout=300,
                request_type=gz.msgs10.pose_pb2.Pose,
                response_type=gz.msgs10.boolean_pb2.Boolean
        )

        time.sleep(2)
        pauze_msgs.pause = False

        print("PLAYING")


        p = pauze.request(
            service=service_pauze,
            request=pauze_msgs,
            timeout=300,
            request_type=gz.msgs10.world_control_pb2.WorldControl,
            response_type=gz.msgs10.boolean_pb2.Boolean
        )
        time.sleep(1)


        print(f"Service call succeeded: {p}")


    # ——— Simple velocity helper ———
    def send_ned_velocity(self, vx, vy, vz, duration_s=0.1):

        if not self.vehicle or not self._simulation_running:
             return False

        if not self.vehicle.armed or self.vehicle.mode.name != 'GUIDED':
            try:
                self.vehicle.arm(wait=True, timeout=10)
                self.vehicle.simple_takeoff(1)
                time.sleep(1)
                if abs(self.vehicle.attitude.pitch) > np.deg2rad(30) or abs(self.vehicle.attitude.roll) > np.deg2rad(30):    
                    self.rocked = True
                    return False
            except:
                 
                print("[!] Vehicle Got ROCKED")
                self.rocked = True
                return False
            return

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        try:
            self.vehicle.send_mavlink(msg)
            #self.vehicle.commands.upload
        except Exception as e:
            print(f"[!] Error sending MAVLink message: {e}")
            return
        

    def condition_yaw(self,heading, relative=False):
        if relative:
            is_relative=1 #yaw relative to direction of travel
        else:
            is_relative=0 #yaw is an absolute angle
        # create the CONDITION_YAW command using command_long_encode()
        if self.vehicle.attitude.yaw < 0:

            direction = 1
        else:
            direction = -1
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
            0, #confirmation
            heading,    # param 1, yaw in degrees
            30,          # param 2, yaw speed deg/s
            direction,          # param 3, direction -1 ccw, 1 cw
            is_relative, # param 4, relative offset 1, absolute angle 0
            0, 0, 0)    # param 5 ~ 7 not used
        # send command to vehicle
        self.vehicle.send_mavlink(msg)








"""
try:
    for i in range(3):
        di = Simulation(5)
        di.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", True)

        sim = Simulation(1)

        sim.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", True)

        dim = Simulation(4)

        dim.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", True)
        
        i = Simulation(3)
        i.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", False)


        
        si = Simulation(2)
        si.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", True)


        sim.dronekit_connect()
        si.dronekit_connect()
        dim.dronekit_connect()        
        i.dronekit_connect()
        di.dronekit_connect()

        sim.send_ned_velocity(5,0,0)
        si.send_ned_velocity(0,5,0)
        di.send_ned_velocity(-5,0,0)
        dim.send_ned_velocity(0,-5,0)
        i.send_ned_velocity(5,-5,0)
        
        s = Simulation(3)
        s.start_sim("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf")
        s.dronekit_connect()
   
        time.sleep(6)
        sim.kill_simulation()
        si.kill_simulation()
        di.kill_simulation()
        i.kill_simulation()
        dim.kill_simulation()
        
finally:
    sim.kill_simulation()
    si.kill_simulation()
    dim.kill_simulation()
    di.kill_simulation()
    i.kill_simulation()
     """