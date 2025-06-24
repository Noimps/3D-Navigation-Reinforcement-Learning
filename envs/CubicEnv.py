import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Assuming these are defined globally or passed as init parameters
NN_SIZES = [[32,32], [64,64], [128,128], [256,256]]
FINISH_PERCENTAGE = 0.84
SPOT_GOAL_HEIGTH = 5 # This variable is currently unused for exploration, can remove if desired

class GridAgent(gym.Env):

    def __init__(
        self,
        grid=None,
        max_steps = 2000,
        width: int = 20,
        depth: int = 20,
        height: int = 12,
        cell_size: float = 0.25,
        local_map_length = 4,
        room_path = None,
        render_mode: str = None,
        crash_penalty: float = -2.0
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.height = height
        self.cell_size = cell_size
        self.local_map_length = local_map_length
        self.max_steps = max_steps
        self.rooms = None
        self.valid_facings = {
            0: "north",
            1: "east",
            2: "south",
            3: "west"
        }
        self.crash_penalty = crash_penalty


        self.action_map = {
            0: [0], # move Forward
            1: [+1], # move Right
            2: [+2], # move Back
            3: [+3], # move Left
            4: [0, 0, 1], # Move Up
            5: [0, 0, -1], # Move Down
        }

        self.action_space = spaces.Discrete(6)
        # Define observation space for 80-length vector, allowing for normalization
        self.observation_space = spaces.Box(
            low=np.full(80, -1.0, dtype=np.float32), # Generic lower bound, adjust for precise normalization
            high=np.full(80, 1.0, dtype=np.float32), # Generic upper bound, adjust for precise normalization
            dtype=np.float32
        )

        if room_path != None:
            room_path = Path(room_path)
            self.rooms = list(room_path.glob('*.txt'))

        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # --- Initialization for tracking total_free_cells ---
        # This will be set in load_room, but initialize here to prevent errors if reset isn't called first
        self.total_free_cells = 1 # Avoid division by zero before load_room sets it properly


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        self.x, self.y, self.z = self.load_room()

        self.internal_grid = np.full((self.width, self.depth, self.height), -1, dtype=int)
        self.internal_grid[self.x, self.y, self.z] = 1 # Mark start cell as visited

        self.visited_count = 1
        self.step_count = 0
        self.bump_count = 0
        self.dicovery_streak = 0
        self.facing = 0

        self.last_action = 0

        self.done = False
        self.explored = False # Flag for immediate discovery in current step
        self.bumped = False
        self.last_bump = False
        self.near_wall = False
        self.was_near_wall = False
        self.cells_insight_down = 0
        self.oscilation = False

        if self.render_mode == "human" and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        return self.get_obs(), {}
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.near_wall:
            self.was_near_wall = True
            self.near_wall = False

        self.step_count += 1        
        truncated = self.step_count >= self.max_steps
        
        current_action_for_reward = action 

        self.do_action(action)
       
        obs = self.get_obs() # Pass no action, since last_action is handled internally
        reward = self.compute_reward(current_action_for_reward, truncated=truncated, obs=obs)
        self.last_action = action # Update last_action AFTER computing reward for current step

        terminated = self.done
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated,  info
    
    def do_action(self, action):
        directions_relative_to_facing = {
            0: [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)], # Forward (N, E, S, W)
            1: [(1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)], # Right (N, E, S, W)
            2: [(0, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0)], # Backward (N, E, S, W)
            3: [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)], # Left (N, E, S, W)
        }

        action = int(action)
        vx, vy, vz = 0, 0, 0 # Initialize movement vector

        if action < 4: # Horizontal movement
            vx, vy, vz = directions_relative_to_facing[action][self.facing]
            # Update facing based on cardinal direction moved (optional, but consistent with your code)
            if vy == 1: self.facing = 0 # Moved North
            elif vx == 1: self.facing = 1 # Moved East
            elif vy == -1: self.facing = 2 # Moved South
            else: self.facing = 3 # Moved West
        else: # Vertical movement
            vx, vy, vz = self.action_map[action] # [0,0,1] or [0,0,-1]

        # Attempt to move
        if self._mark_visited(self.x + vx, self.y + vy, self.z + vz):
            self.x += vx
            self.y += vy
            self.z += vz        
        else:
            self.bumped = True
        
        # Mark current cell as visited (even if already visited, increment count)
        # Only increment count if it's not a wall, and within bounds.
        if 0 <= self.x < self.width and 0 <= self.y < self.depth and 0 <= self.z < self.height and self.grid[self.x, self.y, self.z] != -2:
             self.internal_grid[self.x][self.y][self.z] += 1
            

    def compute_reward(self, current_action: int, truncated: bool, obs) -> float:
        """
        Computes the rewards given to the model at every step after action is implemented and new observation is made.
        This version is tuned for pure exploration, aiming to visit a high percentage of free cells.
        """

        r = -0.05 # Smaller base penalty to stimulate speed, but not too aggressive

        # Penalize for re-visiting the same cell multiple times (discourages looping)
        # Cap the penalty to avoid extreme negative rewards from staying in one spot
        visited_val = self.internal_grid[self.x, self.y, self.z]
        r -= min(visited_val * 0.02, 0.5) # Penalty maxes out at 0.5 if visited_val is 25

        # --- CRASH PENALTIES and AVOIDANCE REWARDS ---
        if self.bumped:
            self.bumped = False # Reset flag
            self.last_bump = True # Indicate a bump just occurred
            self.bump_count += 1
            r += self.crash_penalty # Penalty for bumping into a wall (e.g., -2.0)
        else:        
            self.last_bump = False # Reset flag if no bump this step
            # Reward for moving away from a wall if previously near one
            if self.was_near_wall:
                self.was_near_wall = False # Reset flag
                r += 0.15 # Small bonus for successfully maneuvering away from a wall

            # --- MOVEMENT EFFICIENCY REWARDS ---
            # Bonus for consecutive actions (stimulates following straight paths)
            # Exclude vertical actions from this straight-path bonus if they break a horizontal path
            if self.last_action != 2 and current_action == self.last_action and self.last_action < 4:
                r += 0.05 # Small bonus for repeating the last horizontal action

            # Penalize oscillating (e.g., moving backward and then immediately backward again)
            if self.last_action == 2 and current_action == 2:
                r -= 0.5 # Moderate penalty for unproductive backward oscillation
            

        # --- EXPLORATION REWARDS ---
        if self.explored:
            self.explored = False # Reset flag
            r += +1.0 # Significant bonus for discovering a new, previously unknown cell (0 or -1 -> 1)
    
        # --- FINISHING TRACKERS (Exploration Goal) ---
        current_exploration_percentage = self.visited_count / self.total_free_cells
        if current_exploration_percentage >= FINISH_PERCENTAGE:
            self.done = True
            r += 100.0 # LARGE TERMINAL REWARD for achieving the exploration goal!
            # print moved to training script for cleaner logging, or keep here if desired for immediate feedback
            print(f"Goal Reached! Explored {current_exploration_percentage*100:.2f}% after {self.step_count} Steps.")
            
        if truncated:
            # Small penalty for running out of steps before achieving exploration goal
            r += -5.0 # Penalize for not completing the task within max_steps
            print(f"Truncated after: {self.step_count} Steps, with: {self.bump_count} Bumps and {current_exploration_percentage*100:.2f}% of cells discovered")

        return r

    # --- REMAINING METHODS (get_obs, do_action, _mark_visited, _sense_direction, get_position, load_room, render, close) ---
    # Will go here as updated in previous responses or remain as they were in your original code.

    def _get_3d_local_map(self, map_size: int = 4) -> np.ndarray:
        """
        Extracts a 3D cuboid local map from the self.internal_grid centered around the agent.
        The map_size determines the dimensions (map_size x map_size x map_size).
        Values are: -2 (wall), -1 (unknown), 0 (free, known), >=1 (visited count)
        """
        local_map_cuboid = np.full((map_size, map_size, map_size), -1, dtype=np.float32)
        half_size = map_size // 2

        for dx in range(-half_size, map_size - half_size):
            for dy in range(-half_size, map_size - half_size):
                for dz in range(-half_size, map_size - half_size):
                    gx, gy, gz = self.x + dx, self.y + dy, self.z + dz
                    lx, ly, lz = dx + half_size, dy + half_size, dz + half_size

                    if 0 <= gx < self.width and \
                       0 <= gy < self.depth and \
                       0 <= gz < self.height:
                        local_map_cuboid[lx, ly, lz] = self.internal_grid[gx, gy, gz]

        local_map_cuboid[half_size, half_size, half_size] = self.internal_grid[self.x, self.y, self.z]

        return local_map_cuboid.flatten()


    def get_obs(self): # No action parameter needed if last_action is internal
        # Step 1: Perform directional sensing to update internal_grid.
        directions_but_relative = {
            "up": (0, 0, 1), "down": (0, 0, -1),
            "forward": [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)][self.facing],
            "backward": [(0, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0)][self.facing],
            "left": [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)][self.facing],
            "right": [(1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)][self.facing]
        }
        
        for direction_name in ["forward", "left", "right", "backward", "up", "down"]:
            dx, dy, dz = directions_but_relative[direction_name]
            self._sense_direction(dx, dy, dz, view_steps=self.local_map_length) # Only for side effect on internal_grid


        # Step 2: Extract the 3D local voxel map from the *updated* internal_grid
        local_voxel_map_flattened = self._get_3d_local_map(map_size=4) 
        # Normalize to [0, 1] range. Max value can be `self.max_steps` theoretically, but cap for obs
        # Max reasonable visited count in a local map could be ~20-50, but let's assume max is 20 for normalization.
        max_visited_count_for_obs = 20.0 # Cap value for normalization
        local_voxel_map_flattened = np.clip(local_voxel_map_flattened, -2, max_visited_count_for_obs)
        local_voxel_map_flattened = (local_voxel_map_flattened + 2) / (max_visited_count_for_obs + 2)


        # Step 3: Agent's Orientation (2D facing)
        facing_one_hot = np.zeros(4, dtype=np.float32)
        facing_one_hot[self.facing] = 1.0


        # Step 4: Other scalar information for exploration guidance
        last_action_norm = float(self.last_action) / (self.action_space.n - 1) # Normalize 0-5 to 0-1
        was_near_wall_val = float(self.was_near_wall) # 0 or 1
        last_bump_val = float(self.last_bump) # 0 or 1
        cells_insight_down_norm = float(self.cells_insight_down) / self.local_map_length # Normalize


        # Step 5: Current Exploration Percentage (NEW!)
        current_exploration_percentage = self.visited_count / self.total_free_cells


        # Step 6: Concatenate all features
        obs_parts = [
            local_voxel_map_flattened, # 64 values
            facing_one_hot,            # 4 values
            np.array([last_action_norm, was_near_wall_val, last_bump_val, cells_insight_down_norm], dtype=np.float32), # 4 values
            np.array([current_exploration_percentage], dtype=np.float32) # 1 value
        ]
        
        obs = np.concatenate(obs_parts)
        
        # Pad with zeros to exactly 80.
        padding_needed = 80 - obs.shape[0]
        if padding_needed > 0:
            obs = np.pad(obs, (0, padding_needed), 'constant', constant_values=0)
        elif padding_needed < 0:
             raise ValueError(f"Observation length exceeded 80. Current: {obs.shape[0]}. Check component sizes.")

        assert obs.shape[0] == 80, f"Observation length is {obs.shape[0]}, expected 80."
        return obs

    # Other methods like _is_blocked, _mark_visited, _sense_direction, get_position, load_room, render, close
    # remain as in your provided Venv.py or previous suggestions.
    # Make sure to update the load_room method to correctly set self.total_free_cells.
    def _is_blocked(self, x: int, y: int, z: int) -> bool: # Added based on your provided file
        # This function seems to be unused, but keeping it for completeness if needed elsewhere.
        # It checks if a cell in internal_grid is marked as blocked (-2).
        return self.internal_grid[x][y][z] == -2
    
    def _mark_visited(self, x: int, y: int, z: int) -> bool: # Added based on your provided file
        """
        Attempts to mark a cell as visited.
        Returns True if movement to (x,y,z) is valid (not a wall or out of bounds), False otherwise.
        Updates internal_grid and visited_count.
        """
        if not (0 <= x < self.width and 0 <= y < self.depth and 0 <= z < self.height):
            return False # Out of bounds

        if self.grid[x][y][z] == -2: # If the target cell is a permanent wall (from self.grid)
            return False

        # If it's a valid, non-wall cell
        if self.internal_grid[x][y][z] == 0: # If it's an unknown free cell (discovered but not visited)
            self.internal_grid[x][y][z] += 1 # Mark as visited
            self.visited_count += 1
            self.explored = True

        elif self.internal_grid[x][y][z] > 0: # If it was previously unknown
            self.internal_grid[x][y][z] += 1 # Mark as visited

        return True # Movement is valid

    def _sense_direction(self, dx: int, dy: int, dz: int, view_steps: int) -> tuple[float, list[int]]: # Added based on your provided file
        """
        This function simulates range sensorn output into some direction.
        The count_non_block will return how many steps until the agent is against a wall into that direction

        -- IF AGAINST THE WALL COUNT_NON_BLOCK == 0
        -- IF ONE STEP AWAY FROM BEING AGAINST A WALL COUNT_NON_BLOCK == 1

        In the return statement the step amount gets normalized with the cell size, 
        therefor we get DISTANCE IN METER UNTIL BLOCKED CELL.
        
        """
        cx, cy, cz = self.x, self.y, self.z 
        count_non_block = 0
        wall_hit = False
        local_map = []
        local_map = [-1] * view_steps  # Pre-allocate the list with default value -1
        for step in range(1, view_steps + 1):
            nx = cx + dx * step
            ny = cy + dy * step
            nz = cz + dz * step

            # Check for out of bounds
            if not (0 <= nx < self.width and 0 <= ny < self.depth and 0 <= nz < self.height):
                break



            if not wall_hit:
                if self.grid[nx][ny][nz] == -2: # If it's a permanent wall from the true grid
                    self.internal_grid[nx][ny][nz] = -2 # Mark as blocked in internal grid
                    local_map[step - 1] = -2
                    wall_hit = True
                    if step == 1: # If we hit a wall on the first step, we are near a wall
                        self.near_wall = True

                # Count how far we can sense down, to use in the reward function
                else:
                    count_non_block += 1

                    # If it's an unknown free cell, mark it as free but not visited (0)
                    if self.internal_grid[nx][ny][nz] == -1: 
                        self.internal_grid[nx][ny][nz] = 0



            local_map[step - 1] = self.internal_grid[nx][ny][nz]


        if dz == -1:
            self.cells_insight_down = count_non_block
            
        return count_non_block, local_map

    def get_position(self) -> tuple[int, int, int]: # Added based on your provided file
        return (self.x, self.y, self.z)

    def load_room(self): # Added based on your provided file
        sx, sy, sz = None, None, None
        self.gx, self.gy, self.gz = None, None, None # Keep gx,gy,gz for room loading, even if not used for goal
        # Generate room defined in txt file, randomly chosen
        if self.rooms != None:
            room = random.choice(self.rooms)            
            with open(room, 'r') as f:
                lines = f.readlines()
            row_index = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                elif line.startswith("Start position"):
                    sx, sy, sz = map(int, line.split("=")[1].split(","))

                elif line.startswith("Goal"): # Keep parsing goal, but not used for reward
                    self.gx, self.gy, self.gz = map(int, line.split("=")[1].split(","))

                elif line.startswith("Size"):
                    
                    dimensions = line.split('=')[1].split(',')
                    self.width = int(dimensions[0])
                    self.depth = int(dimensions[1])
                    self.height = int(dimensions[2])
                    self.grid = np.zeros((self.width, self.depth, self.height), dtype= int)

                elif line.startswith("Layer"):
                    z_index = int(line.split('=')[1])
                    row_index = 0                
                else:
                    values = list(map(int, line.split()))
                    values = [v if v != 2 else -2 for v in values]  # Convert 2 to -2 for free cells
                    if len(values) != self.width:
                        raise ValueError(f"Line '{line}' has {len(values)} values, but width is {self.width} for layer {z_index}, row {row_index}.")
                    self.grid[:, row_index, z_index] = values
                    row_index += 1

        else: # just create an empty room with walls, ceiling, floor
            self.grid = np.zeros((self.width, self.depth, self.height), dtype= int)

            self.grid[0, :, :] = -2
            self.grid[-1, :, :] = -2
            self.grid[:, 0, :] = -2
            self.grid[:, -1, :] = -2
            self.grid[:, :, 0] = -2
            self.grid[:, :, -1] = -2
        
        possible_start_pose = []
        self.total_free_cells = 0 # This will be correctly calculated and used
        for x in range(1, self.width - 1):
            for y in range(1, self.depth - 1):
                for z in range(1, self.height - 1):
                    if self.grid[x][y][z] != -2:
                        self.total_free_cells += 1
                        possible_start_pose.append((x,y,z))
        
        self.max_steps = self.total_free_cells # Max steps based on total free cells

        if sx is None or sy is None or sz is None:
            sx, sy, sz = random.choice(possible_start_pose)
        
        if self.grid[sx][sy][sz] == -2:
            print(f"Warning: Provided start position ({sx},{sy},{sz}) is a wall. Choosing a random valid start position.")
            sx, sy, sz = random.choice(possible_start_pose)

        # No need to pick a goal position if not used for reward
        if self.gx is None or self.gy is None or self.gz is None:
             # Just set to dummy values if not loaded from file, won't be used for reward
             self.gx, self.gy, self.gz = 0,0,0 

        return sx, sy, sz

    def render(self): # Added based on your provided file
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "matplotlib":
            self._render_matplotlib()
        else:
            pass # No rendering for other modes

    def _render_text(self): # Added based on your provided file
        print(f"--- Step: {self.step_count}, Pos: ({self.x}, {self.y}, {self.z}), Facing: {self.valid_facings[self.facing]} ---")
        display_grid = np.copy(self.internal_grid[:, :, self.z])
        
        display_grid[self.x, self.y] = 9

        for y_coord in range(self.depth):
            row_str = ""
            for x_coord in range(self.width):
                cell_value = display_grid[x_coord, y_coord]
                if cell_value == 9: row_str += "A "
                elif cell_value == -2: row_str += "# "
                elif cell_value >= 1: row_str += ". "
                elif cell_value == 0: row_str += "o "
                else: row_str += "? "
            print(row_str)
        print("-" * (self.width * 2))

    def _render_matplotlib(self): # Added based on your provided file
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        self.ax.set_title(f"3D Grid Exploration (Step: {self.step_count})")
        self.ax.set_xlim(-1, self.width)
        self.ax.set_ylim(-1, self.depth)
        self.ax.set_zlim(-1, self.height)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        griddy = self.internal_grid[:, :, 1:]
        wall_x, wall_y, wall_z = np.where(griddy == -2)
        self.ax.scatter(wall_x, wall_y, wall_z +1, c='black', marker='s', s=100, label='Walls (Known)') # +1 adjust for layer z=0 walls
        
        self.ax.scatter(self.x, self.y, self.z, c='red', marker='^', s=200, label='Agent')

        dx_facing, dy_facing, dz_facing = 0, 0, 0
        if self.facing == 0: dy_facing = 1
        elif self.facing == 1: dx_facing = 1
        elif self.facing == 2: dy_facing = -1
        elif self.facing == 3: dx_facing = -1
        
        self.ax.quiver(self.x, self.y, self.z, dx_facing, dy_facing, dz_facing, length=1, color='red', linewidth=3, arrow_length_ratio=0.3)

        plt.legend()
        plt.draw()
        plt.pause(0.01)

    def close(self): # Added based on your provided file
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
