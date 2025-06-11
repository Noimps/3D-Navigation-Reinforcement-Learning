import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
NN_SIZES = [[32,32], [64,64], [128,128], [256,256]]
FINISH_PERCENTAGE = 0.8
SPOT_GOAL_HEIGTH = 5

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
        render_mode: str = None # Added render_mode
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
        if grid:
            self.grid = grid


        self.action_map = {
            0: [0], # move Forward
            1: [+1], # move Right
            2: [+2], # move Back
            3: [+3], # move Left
            4: [0, 0, 1], # Move Up
            5: [0, 0, -1], # Move Down
            # Actions 0-3 now include a move forward based on facing
        }

        # SB3 compatibility: define action and observation space
        self.action_space = spaces.Discrete(6)
        # Original obs space: 6 * local_map_length (for local_map) + 6 (for distances)
        # New obs space: Add 1 for the last_action (integer 0-5)
        # Total observation size will be (6 * local_map_length) + 6 + 1
        self.observation_space = spaces.Box(
            # Local map values: -1 (unknown), 0 (free, known), 1 (visited), 2 (wall)
            # Distances: >=0
            # Last action: 0-5
            low=np.array([-1] * (6 * self.local_map_length) + [0] * 6 + [0], dtype=np.float32),
            high=np.array([2] * (6 * self.local_map_length) + [float('inf')] * 6 + [5], dtype=np.float32),
            dtype=np.float32
        )

        if room_path != None:
            room_path = Path(room_path)
            self.rooms = list(room_path.glob('*.txt'))

        # Render mode setup
        self.render_mode = render_mode
        self.fig = None
        self.ax = None


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed) # Call super.reset for Gymnasium compatibility

        #print("RESETTING")
        self.x, self.y, self.z = self.load_room()

        self.internal_grid = np.full((self.width, self.depth, self.height), -1, dtype=int)  # -1 = unknown
        self.internal_grid[self.x, self.y, self.z] = 1

        self.visited_count = 1      # Exploration counter
        self.step_count = 0         # step counter
        self.bump_count = 0         # crash counter
        self.dicovery_streak = 0    # Discovery multiplier
        self.facing = 0             # North

        # Initialize last_action to a value outside the normal action space or a neutral value
        # For example, -1 indicates no previous action
        self.last_action = 0 # Or a random valid action to start with, e.g., self.action_space.sample()

        self.done = False
        self.explored = False
        self.bumped = False

        # Reset rendering for matplotlib if it was initialized
        if self.render_mode == "human" and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
    
    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        self.step_count += 1        
        truncated = self.step_count >= self.max_steps
        """
        print(f"Initial POSITION: ", self.x, self.y, self.z)
        print(f"INITIAL FACING: ", self.facing)

        if action == 0:
            print("moving FORWARD ")
        elif action == 1:
            print("moving RIGHT ")

        elif action == 2:
            print("moving BACKWARD ")
        elif action == 3:
            print("moving LEFT ")
        elif action == 4:
            print("moving Up ")
        else:
            print("moving DOWN ")
        """
        # Store current action for observation in the next step and reward calculation
        current_action_for_reward = action 

        self.do_action(action)
       # print("NEW POSITION: ", self.x, self.y, self.z)
       # print("NEW FACING: ", self.facing)
        # Update last_action for the *next* observation
        self.last_action = action 
    
        obs = self.get_obs()        
        reward = self.compute_reward(current_action_for_reward, truncated=truncated)
        
      #  print("REWARD: ", reward)
        #time.sleep(10)
        terminated = self.done
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated,  info
    
    def do_action(self, action):
        directions_but_relative = {
            0: [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)][self.facing], # MOVE FORWARD RELATIVE TO INITIAL FACING   North, East, South, West    
            1: [(1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)][self.facing], # MOVE RIGHT RELATIVE TO INITIAL FACING  
            2: [(0, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0)][self.facing], # MOVE BACKWARD RELATIVE TO INITIAL FACING  
            3: [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)][self.facing], # MOVE Left RELATIVE TO INITIAL FACING  
        }

        action = int(action)  # Convert numpy array or other type to int
        a = self.action_map[action]
        if action < 4:
            vx, vy, vz = directions_but_relative[action]
            if vy == 1:
                self.facing = 0
            elif vx == 1 :
                self.facing = 1
            elif vy == -1:
                self.facing = 2
            else:
                self.facing = 3
        else:
            vx, vy, vz = a


        action = int(action)  # Convert numpy array or other type to int
        a = self.action_map[action]
  

        if self._mark_visited(self.x + vx, self.y + vy, self.z + vz):
            self.x += vx
            self.y += vy
            self.z += vz
        else:
            self.bumped = True
        # Attempt to move


    def compute_reward(self, current_action: int, truncated: bool) -> float:

        r = -0.1
        if self.bumped:
            self.bumped = False
            self.bump_count += 1
            r += -10
        
        # Reward for consecutive actions
        if self.last_action != 2 and current_action == self.last_action and self.last_action < 4:
            r += 0.05 # Small bonus for repeating the last action

        for i in range(0, SPOT_GOAL_HEIGTH):
            if (self.x, self.y, self.z - i) == (self.gx, self.gy, self.gz):
                self.done = True
            #    print('\n')
                print(f"Finished after: {self.step_count} Steps : {self.bump_count} bumps")
                r+= +100.0
         
        if truncated:
            r += -0
            print(f"Truncated after: {self.step_count} Steps, with: {self.bump_count} Bumps and {self.visited_count} cells discovered")
        
        
        if self.explored:
            self.explored = False
            r += +1.0

        return r

    def get_obs(self):
      #  print("getting obs")
        map_local = []
        distances = []
        
        directions_but_relative = {
            "up": (0, 0, 1),
            "down": (0, 0, -1),
            "forward": [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)][self.facing], # North, East, South, West
            "backward": [(0, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0)][self.facing], # North, East, South, West
            "left": [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)][self.facing], # North, East, South, West
            "right": [(1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)][self.facing] # North, East, South, West
        }
        
        for direction in ["forward",  "left",  "right", "backward","up", "down"]:
            dx, dy, dz = directions_but_relative[direction]
            distance_to_blockage, local_map = self._sense_direction(dx, dy, dz, view_steps=self.local_map_length) # Changed view_steps to local_map_length
            distances.append(distance_to_blockage)
            map_local.append(local_map) # local_map is already padded to local_map_length in _sense_direction
        """
        print('\n')
        print("LOCAL MAP: ")
        print("\n")
        i = self.local_map_length -1
        while i > -1:
            print("       ",map_local[0][i])
            i -= 1
        i = self.local_map_length -1
        while i > -1:
            print(map_local[1][i], end=" ")
            i -= 1
        
        print("A", end= " ")

        for i in map_local[2]:
            print(i, end=" ")

        print("\n")

        for i in map_local[3]:
            print("       ",i)

        """
        # Concatenate local map info, distances, and the last action
        # Ensure last_action is cast to float32 \for consistency
        obs = np.concatenate(map_local + [distances] + [np.array([float(self.last_action)], dtype=np.float32)])
        return np.array(obs, dtype=np.float32) # Ensure final obs is float32


    def _is_blocked(self, x: int, y: int, z: int) -> bool:
        # This function seems to be unused, but keeping it for completeness if needed elsewhere.
        # It checks if a cell in internal_grid is marked as blocked (2).
        return self.internal_grid[x][y][z] == 2

    def _mark_visited(self, x: int, y: int, z: int) -> bool:
        """
        Attempts to mark a cell as visited.
        Returns True if movement to (x,y,z) is valid (not a wall or out of bounds), False otherwise.
        Updates internal_grid and visited_count.
        """
        if not (0 <= x < self.width and 0 <= y < self.depth and 0 <= z < self.height):
            return False # Out of bounds

        if self.grid[x][y][z] == 2: # If the target cell is a permanent wall (from self.grid)
            return False

        # If it's a valid, non-wall cell
        if self.internal_grid[x][y][z] == 0: # If it's an unknown free cell (discovered but not visited)
            self.internal_grid[x][y][z] = 1 # Mark as visited
            self.visited_count += 1
            self.explored = True

        elif self.internal_grid[x][y][z] == -1: # If it was previously unknown
            self.internal_grid[x][y][z] = 1 # Mark as visited
            self.visited_count += 1
            self.explored = True

        # If self.internal_grid[x][y][z] == 1 (already visited), do nothing but still allow movement.
        
        return True # Movement is valid


    def _sense_direction(self, dx: int, dy: int, dz: int, view_steps: int) -> tuple[float, list[int]]:
        cx, cy, cz = self.x, self.y, self.z 
        count_non_block = 0
        local_map = []
        for step in range(1, view_steps + 1):
            nx = cx + dx * step
            ny = cy + dy * step
            nz = cz + dz * step

            # Check for out of bounds
            if not (0 <= nx < self.width and 0 <= ny < self.depth and 0 <= nz < self.height):
                # Mark the cell just *before* out of bounds as a known wall (2) if it's not already.
                # This helps the agent learn boundaries.
                if step > 1: # Only if we moved at least one step
                    prev_nx, prev_ny, prev_nz = cx + dx * (step-1), cy + dy * (step-1), cz + dz * (step-1)
                    if self.internal_grid[prev_nx][prev_ny][prev_nz] != 2: # Don't overwrite existing wall
                        self.internal_grid[prev_nx][prev_ny][prev_nz] = 2 # Mark last known valid cell as blocked
                local_map.append(2) # Treat as blocked at this step
                break

            if self.grid[nx][ny][nz] == 2: # If it's a permanent wall from the true grid
                self.internal_grid[nx][ny][nz] = 2 # Mark as blocked in internal grid
                local_map.append(2)
                break

            count_non_block += 1
            # If it's an unknown free cell, mark it as free but not visited (0)
            if self.internal_grid[nx][ny][nz] == -1: 
                self.internal_grid[nx][ny][nz] = 0 
            
            local_map.append(self.internal_grid[nx][ny][nz])

        # Pad with -1 (unknown) if the sensed path is shorter than view_steps
        while len(local_map) < view_steps:
            local_map.append(-1)
            
        return round(count_non_block * self.cell_size, 2), local_map
    


    def get_position(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


    def load_room(self):
        sx, sy, sz = None, None, None
        self.gx, self.gy, self.gz = None, None, None
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
                    sx, sy, sz = map(int, line.split("=")[1].split(",")) # Convert to int

                elif line.startswith("Goal"):
                    self.gx, self.gy, self.gz = map(int, line.split("=")[1].split(",")) # Convert to int

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
                    # Ensure values length matches width for this z-layer
                    if len(values) != self.width:
                        raise ValueError(f"Line '{line}' has {len(values)} values, but width is {self.width} for layer {z_index}, row {row_index}.")
                    self.grid[:, row_index, z_index] = values
                    row_index += 1
        
        else: # just create an empty room with walls, ceiling, floor
            self.grid = np.zeros((self.width, self.depth, self.height), dtype= int)

            self.grid[0, :, :] = 2
            self.grid[-1, :, :] = 2
            self.grid[:, 0, :] = 2
            self.grid[:, -1, :] = 2
            self.grid[:, :, 0] = 2
            self.grid[:, :, -1] = 2
        

        #print(f"{self.grid.shape}")

        # Collect all possible starting positions an count all free cells

        possible_start_pose = []
        self.total_free_cells = 0
        for x in range(1, self.width - 1):
            for y in range(1, self.depth - 1):
                for z in range(1, self.height - 1):
                    if self.grid[x][y][z] != 2:
                        self.total_free_cells += 1
                        possible_start_pose.append((x,y,z))
        
       # print(f"TOTAL FREE: {self.total_free_cells} for map: {room}")
        self.max_steps = self.total_free_cells
        if sx is None or sy is None or sz is None: # Check if sx,sy,sz are still None
            sx, sy, sz = random.choice(possible_start_pose)
        
        # Ensure the chosen start position is not a wall in the true grid
        if self.grid[sx][sy][sz] == 2:
            # If the provided start position is a wall, try to find a valid one
            print(f"Warning: Provided start position ({sx},{sy},{sz}) is a wall. Choosing a random valid start position.")
            sx, sy, sz = random.choice(possible_start_pose)

        if self.gx is None or self.gy is None or self.gz is None: # Check if sx,sy,sz are still None
            self.gx, self.gy, self.gz = random.choice(possible_start_pose)
        
        # Ensure the chosen start position is not a wall in the true grid
        if self.grid[self.gx][self.gy][self.gz] == 2:
            # If the provided start position is a wall, try to find a valid one
            print(f"Warning: Provided GOAL position is a wall. Choosing a random valid start position.")
            self.gx, self.gy, self.gz = random.choice(possible_start_pose)

        return sx, sy, sz

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "matplotlib":
            self._render_matplotlib()
        else:
            pass # No rendering for other modes

    def _render_text(self):
        # Render a 2D slice at the agent's current Z-level
        print(f"--- Step: {self.step_count}, Pos: ({self.x}, {self.y}, {self.z}), Facing: {self.valid_facings[self.facing]} ---")
        display_grid = np.copy(self.internal_grid[:, :, self.z]) # Get the current Z-layer
        
        # Mark the agent's position
        display_grid[self.x, self.y] = 9 # Agent marker

        for y_coord in range(self.depth):
            row_str = ""
            for x_coord in range(self.width):
                cell_value = display_grid[x_coord, y_coord]
                if cell_value == 9:
                    row_str += "A " # Agent
                elif cell_value == 2:
                    row_str += "# " # Wall
                elif cell_value == 1:
                    row_str += ". " # Visited
                elif cell_value == 0:
                    row_str += "o " # Free (discovered but not visited)
                else: # -1 (unknown)
                    row_str += "? " # Unknown
            print(row_str)
        print("-" * (self.width * 2))

    def _render_matplotlib(self):
        if self.fig is None:
            plt.ion() # Turn on interactive mode
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        self.ax.set_title(f"3D Grid Exploration (Step: {self.step_count})")
        self.ax.set_xlim(0, self.width - 1)
        self.ax.set_ylim(0, self.depth - 1)
        self.ax.set_zlim(0, self.height - 1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Visualize the grid based on internal_grid values
        # Walls (2)
        wall_x, wall_y, wall_z = np.where(self.internal_grid == 2)
        self.ax.scatter(wall_x, wall_y, wall_z, c='black', marker='s', s=100, label='Walls (Known)')

        # Visited cells (1)
        visited_x, visited_y, visited_z = np.where(self.internal_grid == 1)
        self.ax.scatter(visited_x, visited_y, visited_z, c='green', marker='o', s=50, alpha=0.6, label='Visited')

        # Free but not visited (0)
        free_x, free_y, free_z = np.where(self.internal_grid == 0)
        self.ax.scatter(free_x, free_y, free_z, c='blue', marker='.', s=30, alpha=0.3, label='Free (Known)')

        # Unknown cells (-1) - optional, can be too noisy
        # unknown_x, unknown_y, unknown_z = np.where(self.internal_grid == -1)
        # self.ax.scatter(unknown_x, unknown_y, unknown_z, c='gray', marker='x', s=10, alpha=0.1, label='Unknown')


        # Agent's current position
        self.ax.scatter(self.x, self.y, self.z, c='red', marker='^', s=200, label='Agent')

        # Add an arrow for agent's facing direction
        dx_facing, dy_facing, dz_facing = 0, 0, 0
        if self.facing == 0: # North (positive Y)
            dy_facing = 1
        elif self.facing == 1: # East (positive X)
            dx_facing = 1
        elif self.facing == 2: # South (negative Y)
            dy_facing = -1
        elif self.facing == 3: # West (negative X)
            dx_facing = -1
        
        self.ax.quiver(self.x, self.y, self.z, dx_facing, dy_facing, dz_facing, length=1, color='red', linewidth=3, arrow_length_ratio=0.3)


        plt.legend()
        plt.draw()
        plt.pause(0.01) # Pause for a short time to update the plot

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None