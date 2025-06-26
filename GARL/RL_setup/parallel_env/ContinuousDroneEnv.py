
import numpy as np
from GARL.parallel_env.Simulation import Simulation
import gymnasium as gym 
import time
from GARL.parallel_env.SensorReading import SensorReading
#from envs.ComputerVision import CameraAprilTagDetector
#from dronekit import VehicleMode
from math import sin, cos
#import cv2
import random
from pathlib import Path

SAME_ACTION_REWARD = +0.1
CENTERING_SCALE      = +0.5
NEW_CELL_REWARD      = +100
SUCCESS_REWARD       = +1500 

STEP_COST            = -0.01      # per 0.1-s step
LIDAR_COLLISION_PEN  = -5
TIMEOUT_PEN          = -10
CRASH_PEN            = -1500

TARGET_ALTITUDE      = 2.0  # meters, or whatever is good for your room

EXPLORATION_SCALE = 0.75

ALTITUDE_DEVIATION_PENALTY_SCALE = -0.5 # A negative value, tune this!
PHASE_DIR = "phase_one"

script_dir = Path(__file__).resolve().parent
worlds_dir      = script_dir.parent.parent / f'ardupilot_gazebo/worlds/{PHASE_DIR}'
worlds     = list(worlds_dir.glob('*.sdf'))

class ContinuousDroneEnv(gym.Env):
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



        # Discrete actions: forward, back, left, right, yaw_left, yaw_right, hover, land
        self.action_space = gym.spaces.Discrete(7)
        """
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1]),   # Example: vx, vy, vz, yaw_rate
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )"""
        # Observations: [vx, vy, vz] + 6×Lidar +  COS(YAW), sin(yaw) + tagfound, dtag_x, dtag_y + rel_height + 5x5 grid of 0.5 m cells + postiion
        base_low  = np.array([0] + [0]*6 + [-1, -1] + [0,0,0], dtype=np.float32)
        base_high = np.array([6] + [5]*6 + [ 1,  1]  + [5,5,5],dtype=np.float32)

        # Extra features: 10×10 visited grid  +  (x_norm, y_norm)
        low  = np.concatenate([base_low, np.zeros(75, dtype=np.float32)])
        high = np.concatenate([base_high, np.ones(75, dtype=np.float32)])


        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.CELL_SIZE = 1          # metres
        # Initialize variables
        self.max_steps = max_steps
        self.step_count = 0
        self.cold_start = True
        self.first_find = True
        self.visited: set[tuple[int, int, int]] = set()     # {(ix, iy, iz)}  :contentReference[oaicite:0]{index=0}

        
        self.GRID_SHAPE = (5, 5, 3)
        self.total_cells = 1
        for i in self.GRID_SHAPE:
            self.total_cells = self.total_cells * i

        self.goal_reached = False     

        self.drone = Simulation(self.instance_id)
        self.sensors  = SensorReading(self.instance_id)
        
    def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)

            self.drone.kill_simulation() # This calls self.drone.vehicle.close() etc.
            self.sensors.close() # Call sensor-specific cleanup
            self.camera_detector = None
            time.sleep(3)
            self.drone = None
            self.sensors = None
            
            world = random.choice(worlds)
            #print(f"chosen ", world)
        


            
            self.drone = Simulation(self.instance_id)
            self.drone.start_sim(world, headless=self.headless) 
            c = 0
            while True:
                if self.drone.dronekit_connect():
                    break

                c += 1
                if c % 3 ==0:
                    self.drone.kill_simulation()
                    self.drone = None
                    time.sleep(2)
                    self.drone = Simulation(self.instance_id)
                    self.drone.start_sim(world, headless=self.headless)
                """
                self.drone.kill_simulation() # This calls self.drone.vehicle.close() etc.
                
                time.sleep(5)
                self.drone.start_sim(world, headless=False) 
                self.drone.dronekit_connect()
                self.world_instance = self.drone.gz_instance.copied_world_path
                """
            self.drone.vehicle.armed = True
            self.drone.vehicle.simple_takeoff(TARGET_ALTITUDE)
            
            self.world_instance = self.drone.gz_instance.copied_world_path

            # Parse SDF and start contact listeners (this will create new nodes, but old ones are cleaned up)

            self.sensors  = SensorReading(instance_id=self.instance_id)
            self.sensors._parse_sdf(self.world_instance) 
            self.sensors.start_touch_listeners()    
            self.sensors.start_lidars()
          

            self.step_count = 0

            self.rel_height = 0.0
            self.drone.crashed = False
            self.sensors.crashed = False
            self.first_find = True # Reset for reward function
            self.lowest_x = float('inf') # Initialize for reward function
            self.lowest_y = float('inf') # Initialize for reward function
            self.prev_action = 0  # init once, e.g. in __init__
            self.current_action = 0  # init once, e.g. in __init__



            # ─── at the very end of reset() after the vehicle is ready ─────────────────────

            info = {}



            camera_topic = f"/world/{self.sensors.world_name}/model/{self.drone.gz_instance.new_model_name}/model/gimbal/link/pitch_link/sensor/camera/image"
           #print("CameraTOPIC: ", camera_topic )

        #    self.camera_detector = CameraAprilTagDetector(camera_topic=camera_topic, simulation_state_checker=lambda: self.drone._simulation_running, instance_id=self.instance_id)
            time.sleep(3)       
            self.sensors.started = True
                        
            self.visited.clear()
            self.visited_grid = np.zeros(self.GRID_SHAPE, dtype=np.float32)
            self.explored = False
            
            lf = self.drone.vehicle.location.local_frame    # north/east from EKF origin
            self.origin_north = lf.north 
            self.origin_east  = lf.east  
            self.origin_down = lf.down
            self.visited.clear()

            obs  = self._get_obs(0)

          #  print("ORIGINS", self.origin_north, self.origin_east, self.origin_down )

            return obs, info     # ← tuple!

    def _pos_to_cell(self, north: float, east: float, down: float) -> tuple[int, int, int]:
        """Map local-frame N/E (m) to discrete grid coordinates (ix, iy)."""
        
        dy = (north - self.origin_north) / self.CELL_SIZE
        dx = (east  - self.origin_east ) / self.CELL_SIZE
        dz = (down) / self.CELL_SIZE
        iy = int(np.clip(np.round(dy), -self.GRID_SHAPE[0] + 1, self.GRID_SHAPE[0] - 1))  # row
        ix = int(np.clip(np.round(dx), -self.GRID_SHAPE[1] +1, self.GRID_SHAPE[1] - 1))  # col
        iz = int(np.clip(np.round(-dz), -self.GRID_SHAPE[2] +1, self.GRID_SHAPE[2] - 1))  # col
       # print(ix, iy, iz)
        return ix, iy, iz

    def step(self, action: np.ndarray):
        # Send action via MAVLink
        self.last_rel_height = self.rel_height
        #self.last_x = self.camera_detector.camera_data[1]
       # self.last_y = self.camera_detector.camera_data[2]
        
        self._send_action(action)

        obs = self._get_obs(action)
        if self.step_count % 10 == 0:
          #  print(self.visited_grid)
            pass

        self.step_count += 1
        
        terminated = self._terminated(obs)
        truncated = self.step_count >= self.max_steps
        #if truncated:
            #print("Max steps reached!")
        
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
        self.rel_height = self.drone.vehicle.location.global_relative_frame.alt 
        # Safety: Arm the drone if not already
       # print(self.rel_height, " affds")
        """   
        if self.rel_height == 0:                           # still on the pad
            start = time.time()

            # wait until either altitude > 5 cm or 7 s have passed
            while self.rel_height < 0.1:
                print(self.rel_height)
                self.drone.vehicle.armed = True
                self.drone.vehicle.simple_takeoff(TARGET_ALTITUDE)  
                time.sleep(1)     
                self.rel_height = self.drone.vehicle.location.global_relative_frame.alt 
                print(time.time() - start)
                if time.time() - start > 10:

                    self.sensors.crashed = True
                    return  # skip sending velocity this step


        # Scale actions
            
max_speed = 0.5  # m/s; change this to suit your environment
        
        alpha = 0.5  # Smoothing factor for action smoothing

        for i in range(len(action)):
            v_com = float(np.clip(action[i], -1, 1)) * max_speed  # Ensure action is within [-1, 1]
            action[i] = alpha * self.prev_action[i] + (1 - alpha) * v_com  # Smooth the action
        

        self.current_action = action
        # Update NED vector

        # Send MAVLink velocity command (assuming you're in GUIDED mode)
        self.drone.send_ned_velocity(action[0], action[1], action[2], 0.1)  # No yaw control in this example
                
        time.sleep(0.03)
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
        time.sleep(0.05)

    def _get_obs(self, action):
        # velocity (3)
        

        # lidar (24)
        lidar   = self.sensors.lidar_data        # already np.float32 (24,)

        # yaw → sin/cos  (2)
        yaw     = float(self.drone.vehicle.attitude.yaw or 0.0)   # [-π, π]
        yaw_enc = np.array([sin(yaw), cos(yaw)], dtype=np.float32)

        # computer-vision features (3)
        #cv_feat = np.asarray(self.camera_detector.camera_data,dtype=np.float32)                    # (3,)

        # relative height (1)
       # h_rel   = np.arra#y([self.drone.vehicle.location
                          #          .global_relative_frame.alt],
                           # dtype=np.float32)

        lf = self.drone.vehicle.location.local_frame
        ix, iy, iz = self._pos_to_cell(lf.north, lf.east, lf.down) 
        #print("Lidar", lidar)


        if (ix, iy, iz) not in self.visited:
            print("NEW CELL REACHED: ", (ix, iy, iz))
            self.visited.add((ix, iy, iz))
            self.explored = True
        
            self.visited_grid[ix, iy, iz] = 1.0             
        grid_flat = self.visited_grid.flatten()         # (125,)
        
        #print("POSITION:", (ix, iy, iz))
        #print(self.visited_grid)

        obs = np.concatenate((np.array([action]), lidar, yaw_enc, [ix,iy,iz], grid_flat), dtype=np.float32)
        return obs


        


    def _compute_reward(self, obs, terminated, truncated):
            r = STEP_COST   # ➊ small ticking cost2000

            if self.current_action == self.prev_action:
                r += SAME_ACTION_REWARD
            # In your _compute_reward method, after STEP_COST
            #altitude_error = self.drone.vehicle.location.global_relative_frame.alt - TARGET_ALTITUDE
            #r += ALTITUDE_DEVIATION_PENALTY_SCALE * (altitude_error**2)


            #penalise imminent collisions
            if np.any(self.sensors.lidar_data < 0.20):
                r += LIDAR_COLLISION_PEN

            # reward exploring new cells
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
                r += TIMEOUT_PEN
            elif terminated:
                if self.goal_reached:
                    print(f"SUCCEEDED IN: {self.step_count}")
                    r += SUCCESS_REWARD 
                else:
                    print(f"FAILED IN: {self.step_count}")

                    r += CRASH_PEN

            self.prev_action = self.current_action
            return float(r)

    
    def _discrete_cell(self):
        # current NED metres relative to the origin
        lf = self.drone.vehicle.location.local_frame
        dx = lf.north  - self.drone.origin.north
        dy = lf.east   - self.drone.origin.east
        dz = lf.down   - self.drone.origin.down
        # quantise
        ix = int(np.floor(dx / self.CELL_SIZE))
        iy = int(np.floor(dy / self.CELL_SIZE))
        iz = int(np.floor(dz / self.CELL_SIZE))   # or drop z if you only care 2-D
        
        return ix, iy, iz
        
        
        """
        elif self.drone.crashed:
            print("Drone impact crash!")
            return True
        """   

    def _terminated(self,obs):
        if self.sensors.crashed:
            
            print("Drone touch crash!")
            return True
        

        elif not self.drone.vehicle.is_armable and not self.drone.vehicle.armed:
            return True
        
        elif abs(self.drone.vehicle.attitude.pitch) > np.deg2rad(30) or abs(self.drone.vehicle.attitude.roll) > np.deg2rad(30):    
            print("Pitch: ",np.rad2deg(self.drone.vehicle.attitude.pitch))
            print("Roll: ",np.rad2deg(self.drone.vehicle.attitude.roll))
            print("Drone Title crash!")
            return True
        """
        elif abs(self.last_x) < 50 and abs(self.last_y) <50:
            self.drone.vehicle.mode = VehicleMode("LAND")
            time.sleep(5)
            print("Drone landed!")
            self.goal_reached = True
            return True
        
        return False"""
        if len(self.visited) > EXPLORATION_SCALE * self.total_cells:
            self.goal_reached = True
            return True
            
    def close(self):
        self.drone.kill_simulation() # Kills SITL/Gazebo, closes DroneKit connection
        self.sensors.close() # Calls the new SensorReading.close() method
        self.camera_detector= None