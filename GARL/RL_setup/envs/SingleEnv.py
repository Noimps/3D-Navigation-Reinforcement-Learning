from dronekit import VehicleMode

import numpy as np
from envs.SingleSimulation import Simulation
import gymnasium as gym 
import time
from envs.SingleSensorReading import SensorReading
#from envs.ComputerVision import CameraAprilTagDetector
#from dronekit import VehicleMode
from math import sin, cos
#import cv2
import random
from pathlib import Path

SAME_ACTION_REWARD = +0.01
CENTERING_SCALE      = +0.5
NEW_CELL_REWARD      = +100
SUCCESS_REWARD       = +1500 

STEP_COST            = -0.01      # per 0.1-s step
LIDAR_COLLISION_PEN  = -0.05
TIMEOUT_PEN          = -10
CRASH_PEN            = -1500

TARGET_ALTITUDE      = 2.0  # meters, or whatever is good for your room

EXPLORATION_SCALE = 0.75

ALTITUDE_DEVIATION_PENALTY_SCALE = -0.5 # A negative value, tune this!
PHASE_DIR = "phase_one"

script_dir = Path(__file__).resolve().parent
worlds_dir      = script_dir.parent.parent / f'ardupilot_gazebo/worlds'
worlds     = list(worlds_dir.glob('*.sdf'))

class SingleEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
 
    def __init__(
        self,
        headless: bool,
        instance_id: int ,     
        max_steps: int = 1000,
    ):
        super().__init__()
        self.instance_id   = instance_id
        self.headless = headless

        """
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1]),   # Example: vx, vy, vz, yaw_rate
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )"""

        # Discrete actions: forward, back, left, right, yaw_left, yaw_right, hover, land
        self.action_space = gym.spaces.Discrete(7)

        # Observations: previous action + 6×Lidar +  COS(YAW), sin(yaw) + tagfound, dtag_x, dtag_y + rel_height + 5x5 grid of 0.5 m cells + postiion
        base_low  = np.array([0] + [0]*6  + [ -1,  -1] + [0,0,0], dtype=np.float32)
        base_high = np.array([6] + [5]*6  + [ 1,  1] + [5,5,5],dtype=np.float32)

        # Extra features: 5×5x3 visited grid 
        low  = np.concatenate([base_low, np.zeros(75, dtype=np.float32)])
        high = np.concatenate([base_high, np.ones(75, dtype=np.float32)])

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Initialize variables
        self.max_steps = max_steps
        self.step_count = 0
        self.cold_start = True
        self.first_find = True
       

        self.visited: set[tuple[int, int, int]] = set()     # {(ix, iy, iz)}  :contentReference[oaicite:0]{index=0}
        self.CELL_SIZE = 1   
        self.GRID_SHAPE = (5, 5, 3)
        self.total_cells = 1
        for i in self.GRID_SHAPE:
            self.total_cells = self.total_cells * i

        self.world = random.choice(worlds)            
        self.drone = Simulation(self.instance_id)
        self.drone.start_sim(self.world, headless=self.headless) 
        c = 0
        
        if not self.drone.dronekit_connect():
            self.hard_recovery(True)

        print("SIMULATION WILL START NOW")








        
    def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)

            self.sensors  = SensorReading(instance_id=self.instance_id)
            self.sensors._parse_sdf(self.world) 
                    #self.sensors.start_touch_listeners()    
            self.sensors.start_lidars()
            self.sensors.start_touch_listeners()


            self.accumulated_reward = 0
            self.step_count = 0
            self.goal_reached = False     

            self.rel_height = 0.0

            self.drone.crashed = False
            self.sensors.crashed = False

            self.first_find = False # Reset for reward function
            self.lowest_x = float('inf') # Initialize for reward function
            self.lowest_y = float('inf') # Initialize for reward function
            self.prev_action = 0  # init once, e.g. in __init__
            self.current_action = 0  # init once, e.g. in __init__

            self.origin_down = 0
            self.origin_north = 0
            self.origin_east = 0
            info = {}


                        
            self.visited.clear()
            self.visited_grid = np.zeros(self.GRID_SHAPE, dtype=np.float32)
            self.explored = False
            



            obs  = self._get_obs(0)

          #  print("ORIGINS", self.origin_north, self.origin_east, self.origin_down )

            return obs, info     # ← tuple!

    def _pos_to_cell(self, north: float, east: float, down: float) -> tuple[int, int, int]:
        """Map local-frame N/E (m) to discrete grid coordinates (ix, iy)."""
        
        self.dy = (north - self.origin_north) / self.CELL_SIZE
        self.dx = (east  - self.origin_east ) / self.CELL_SIZE
        self.dz = (down) / self.CELL_SIZE
        iy = int(np.clip(np.round(self.dy), -self.GRID_SHAPE[0] + 1, self.GRID_SHAPE[0] - 1))  # row
        ix = int(np.clip(np.round(self.dx), -self.GRID_SHAPE[1] +1, self.GRID_SHAPE[1] - 1))  # col
        iz = int(np.clip(np.round(-self.dz), -self.GRID_SHAPE[2] +1, self.GRID_SHAPE[2] - 1))  # col
       # print(ix, iy, iz)
        return ix, iy, iz

    def step(self, action: np.ndarray):
        # Send action via MAVLink        
        
        self.step_count += 1

        self.last_rel_height = self.rel_height

    #    print(f"chosen: {action}")
        self._send_action(action)
        obs = self._get_obs(action)
        #print(obs)
        if self.step_count % 50 == 0:
            print(obs)
        
        terminated = self._terminated(obs)
        truncated = self.step_count >= self.max_steps
        
        reward = self._compute_reward(obs, terminated, truncated)
        info = {}

        return obs, reward, terminated, truncated, info    
        
    def _send_action(self, action: int):
        """
        Sends continuous NED velocity command to the drone.
        
        Parameters:
        - action: np.ndarray of shape (3,) or (4,) representing [vx, vy, vz] or [vx, vy, vz, yaw_rate]
                values are expected in range [-1, 1] and will be scaled to real units.
        """

        v = 0.5
        lookup = {
            0: ( 0.0,  0.0,  0.0),        # hover
            1: (+v ,  0.0,  0.0),         # forward  (+X)camera_detector
            2: (-v ,  0.0,  0.0),         # back     (-X)
            3: ( 0.0, +v ,  0.0),         # left     (+Y)
            4: ( 0.0, -v ,  0.0),         # right    (-Y)
            5: ( 0.0,  0.0, -v ),         # up       (-Z in NED)
            6: ( 0.0,  0.0, +v ),         # down     (+Z)
        }

        vx, vy, vz = lookup.get(int(action), (0.0, 0.0, 0.0))
        self.current_action = action
       # print("action gekozen")
        # Send MAVLink velocity command (GUIDED mode assumed)
        self.drone.send_ned_velocity(vx, vy, vz, 0.1)
        time.sleep(0.1)

    def _get_obs(self, action):
        
        self.lidar   = self.sensors.lidar_data        # already np.float32 (24,)
       # print(f"LIDAR:  {self.lidar}")

        lf = self.drone.vehicle.location.local_frame
        ix, iy, iz = self._pos_to_cell(lf.north, lf.east, lf.down) 

        if (ix, iy, iz) not in self.visited:
            print("NEW CELL REACHED: ", (ix, iy, iz))
            self.visited.add((ix, iy, iz))
            self.explored = True
            self.visited_grid[ix, iy, iz] = 1.0    

        yaw     = float(self.drone.vehicle.attitude.yaw or 0.0)   # [-π, π]
        yaw_enc = np.array([sin(yaw), cos(yaw)], dtype=np.float32)
        grid_flat = self.visited_grid.flatten()         # (125,)
     #   print("POSITION:", (ix, iy, iz))



        obs = np.concatenate((np.array([action]), self.lidar, yaw_enc, [ix,iy,iz], grid_flat), dtype=np.float32)
        return obs


        


    def _compute_reward(self, obs, terminated, truncated):
            r = STEP_COST   # ➊ small ticking cost2000

            if self.current_action == self.prev_action:
                r += SAME_ACTION_REWARD

            if self.explored:
                print(" exploration reward a niffo")
                r += NEW_CELL_REWARD   
            self.explored = False
            """

            # ➍ vision-based shaping
            tag_seen, err_x, err_y = self.camera_detector.camera_data
            if tag_seen == 1:
                r+=0.2
                if self.first_find:
                    r += 10                      # first sighting
                    self.first_find = False
                    self.lowest_x, self.lowest_y = err_x, err_y
                # quadratic reduction = positive reward when getting closer
                r += CENTERING_SCALE * (self.lowest_x**2 - err_x**2)
                r += CENTERING_SCALE * (self.lowest_y**2 - err_y**2)
                self.lowest_x, self.lowest_y = err_x, err_y
            """
            # ➎ episode-level terms
            if truncated:
                self.reset_pose()
                r += TIMEOUT_PEN
                self.accumulated_reward += r
                print(" \n")
                print("MAX STEPS REACHED")
                print(f"ACCUMULATED REWARD: {self.accumulated_reward}")
                print(" \n")

            elif terminated:
                if self.goal_reached:
                    r += SUCCESS_REWARD 
                    self.accumulated_reward += r
                    print(" \n")
                    print(f"SUCCEEDED IN: {self.step_count}")                    
                    print(f"ACCUMULATED REWARD: {self.accumulated_reward}")
                    print(" \n")
                else:                    
                    r += CRASH_PEN
                    self.accumulated_reward += r

                    print(" \n")
                    print(f"FAILED IN: {self.step_count}")
                    print(f"ACCUMULATED REWARD: {self.accumulated_reward}")
                    print(" \n")
                
            else:
                self.accumulated_reward += r
            self.prev_action = self.current_action
            return float(r)

    

    def reset_pose(self):
        self.drone.send_ned_velocity(0,0,0)
        self.drone.vehicle.mode = "ACRO"

        
        self.sensors.close()

        self.drone.reset_pose()
        time.sleep(5)
        self.drone.vehicle.mode = "GUIDED"
        time.sleep(0.1)
        self.drone.vehicle.armed = True
        time.sleep(0.1)
        self.drone.vehicle.simple_takeoff(2)
        time.sleep(3)
        self.sensors.crashed = False
        self.drone.crashed = False

    def _terminated(self,obs):        
        if self.drone.rocked:
            self.hard_recovery()
            return True
        
        if 0 in obs[1:7]:
            if obs[5] == 0:
                
                if self.step_count >= 2:
                    self.reset_pose()
                    print("Ground crash")
                    time.sleep(0.5)
                    if abs(self.drone.vehicle.location.local_frame.down) < 1:
                        print("THING IS FUCKED")
                        self.hard_recovery()
                    
                    return True
            else:
                self.reset_pose()
                print("Drone touch crash!")
                time.sleep(0.5)
                if abs(self.drone.vehicle.location.local_frame.down) < 1:
                        print("THING IS FUCKED")
                        self.hard_recovery()
                return True



            

        if abs(self.drone.vehicle.attitude.pitch) > np.deg2rad(25) or abs(self.drone.vehicle.attitude.roll) > np.deg2rad(25):    
            print("DRONE DONE TILTED IN A BAD BAD MANNER")
            print("Pitch: ",np.rad2deg(self.drone.vehicle.attitude.pitch))
            print("Roll: ",np.rad2deg(self.drone.vehicle.attitude.roll))
            print("Drone Title crash!")
            self.hard_recovery()
            return True   
        
        if self.sensors.crashed == True:
            print("DRONE DONE TOUCHED SOMETHING")
            self.reset_pose()
            time.sleep(0.5)
            if abs(self.drone.vehicle.location.local_frame.down) < 1:
                print("THING IS FUCKED #2")
                self.hard_recovery()


            return True

        if len(self.visited) > EXPLORATION_SCALE * self.total_cells:
            self.goal_reached = True
            return True 
        """
 





     
        
        elif not self.drone.vehicle.is_armable and not self.drone.vehicle.armed:
            return True

    """


    def hard_recovery(self, first=False):

        print("=========================")
        print("HARD RECOVERY STARTED")
        self.drone.kill_simulation()
        if not first:
            try:
                self.sensors.close()
            except:
                pass
        self.drone = None
        self.sensors = None
        time.sleep(3)

        print("NEW HARD RECOVERY WORLD")
        world = random.choice(worlds)            
        self.drone = Simulation(self.instance_id)

        self.drone.start_sim(world, headless=self.headless) 

        c = 0
        time.sleep(1)
        if not self.drone.dronekit_connect():
            self.hard_recovery()
        
        else:
            lf = self.drone.vehicle.location.local_frame    # north/east from EKF origin
            self.origin_north = lf.north 
            self.origin_east  = lf.east  
            self.origin_down = lf.down

            self.initial_yaw = self.drone.vehicle.attitude.yaw
            self.drone.condition_yaw(self.initial_yaw)

        print("HARD RECOVERY FINISHED ONTO NEXT SIMULATION")
        print("=============================")


    def close(self):
        self.drone.kill_simulation() # Kills SITL/Gazebo, closes DroneKit connection
        self.sensors.close() # Calls the new SensorReading.close() method
        self.camera_detector= None