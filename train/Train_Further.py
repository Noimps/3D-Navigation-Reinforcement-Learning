"""
Incremental multi-phase PPO-LSTM training on GridAgent with intermediate checkpoints and reproducible seeding.

This script takes existing models from a specified directory, continues their training for a
pre-defined number of steps in a new phase (Phase 2), and saves the resulting models to a
new directory with a "_P2" suffix.
"""

import os
import random
import re
import glob
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.CubicEnv import GridAgent  # Assuming envs.Kenv is the correct path
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# ---------------------------------------
# Seeding for reproducibility
# ---------------------------------------
BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
NUM_ENVS = 8
# Note: LOCAL_MAP_LENGTHS is now determined from the filename of the loaded model.
CRASH_PENALTIES = [-2.0]

# ---------------------------------------
# Directory Configuration
# ---------------------------------------
LOAD_DIR = "./exp3_architectures/best_P1_empty_r10_cp-2.0"          # Directory with existing .zip models
SAVE_DIR = "./exp3_architectures/P2"      # Where to save after more training
os.makedirs(SAVE_DIR, exist_ok=True)

EVAL_FREQ = 100_000
best_model_dir = os.path.join(SAVE_DIR, "best_exp3_P2")
os.makedirs(best_model_dir, exist_ok=True)

# ---------------------------------------
# Experiment configuration for Phase 2
# ---------------------------------------
PHASES = [
    ("P2_small", "./rooms/P2_training", "./rooms/P2_evaluate"),
]
# Training steps for this continuation phase
STEPS_PHASE = {
    "P2_small": 20_000_000,
}

# NOTE: Architectures, LSTM sizes, and PPO hyperparameters are loaded from the
# saved model files, so their explicit definition here is not needed for training continuation.

# ---------------------------------------
# Environment wrapper
# ---------------------------------------
class GridEnv(gym.Env):
    """A wrapper for the GridAgent to be compatible with SB3."""
    def __init__(self, room_path, local_map_length, crash_penalty=-2.0):
        super().__init__()
        self.agent = GridAgent(room_path=room_path,
                               local_map_length=local_map_length)
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space

    def reset(self, *, seed=None, options=None):
        return self.agent.reset(seed=seed, options=options)

    def step(self, action):
        return self.agent.step(action)

    def render(self, mode='human'):
        if mode == 'human':
            pos = self.agent.get_position()
            print(f"Agent Pos: {pos}, Visited: {self.agent.visited_count}/{self.agent.total_free_cells} ({self.agent.visited_count/self.agent.total_free_cells*100:.2f}%)")
        else:
            super().render(mode=mode)

def make_env_fn(room_path, local_map_length, seed_offset):
    """Factory for vectorized environment workers with unique but reproducible seeds."""
    def _init():
        env = GridEnv(room_path, local_map_length)
        env.reset(seed=BASE_SEED + seed_offset)
        return Monitor(env)
    return _init

# ---------------------------------------
# Main continuation training loop
# ---------------------------------------
if __name__ == "__main__":
    # --- Verify that Phase 2 room directories exist ---
    try:
        if not os.path.exists(PHASES[0][1]) or not os.path.exists(PHASES[0][2]):
            raise FileNotFoundError("Phase 2 training or evaluation directory not found.")
        print("Phase 2 directories verified.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # Find all model files to continue training
    model_files = glob.glob(os.path.join(LOAD_DIR, "*.zip"))

    if not model_files:
        print(f"No .zip models found in {LOAD_DIR}. Exiting.")
        exit()

    print(f"Found {len(model_files)} models to continue training.")

    # Loop through each model file
    for model_path in model_files:
        model_filename = os.path.basename(model_path)
        print(f"\n{'='*40}\nProcessing model: {model_filename}\n{'='*40}")

        # --- Parse ray_len from the filename ---
        match = re.search(r'_view(\d+)\.zip$', model_filename)
        if not match:
            print(f"Warning: Could not parse 'ray_len' from filename: {model_filename}. Skipping this model.")
            continue
        ray_len = int(match.group(1))
        print(f"Successfully parsed ray_len: {ray_len}")

        # --- Set up for the next phase ---
        phase_name, train_path, eval_path = PHASES[0]
        steps_this_phase = STEPS_PHASE[phase_name]

        print(f"STARTING PHASE: {phase_name} for {model_filename}")

        # Create evaluation environment for this model
        eval_env = SubprocVecEnv([
            make_env_fn(eval_path, ray_len, i + NUM_ENVS)
            for i in range(NUM_ENVS)
        ])

        # Create training environment for this model
        train_env = SubprocVecEnv([
            make_env_fn(train_path, ray_len, i) for i in range(NUM_ENVS)
        ])

        # --- Load the model and set the new environment ---
        print("Loading existing model...")
        model = RecurrentPPO.load(model_path, env=train_env, verbose=1)
        print(f"Model loaded. Continuing training on {phase_name}.")

        # --- Training segments ---
        segment_size = steps_this_phase // 10
        segments = [segment_size] * 10
        remainder = steps_this_phase - sum(segments)
        if remainder > 0:
            segments[-1] += remainder

        # Cumulative steps are tracked by the model's internal counter, which is preserved on load.
        # We don't reset it.
        for i, seg_steps in enumerate(segments):
            print(f"\n--- Training {phase_name} segment {i + 1}/{len(segments)} ({seg_steps} steps) ---")
            
            eval_callback = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=best_model_dir,
                log_path=best_model_dir,
                eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            )
            
            # Continue learning without resetting the timestep counter
            model.learn(total_timesteps=seg_steps, reset_num_timesteps=False, callback=eval_callback)
            
            # --- Save the final model with the new naming convention ---
            base_name, extension = os.path.splitext(model_filename)
            save_filename = f"{base_name}_P2{extension}_i"
            save_path = os.path.join(SAVE_DIR, save_filename)
            
            model.save(save_path)
            print(f"Checkpoint saved to: {save_path}")

        # --- Clean up environments for this model ---
        print(f"Finished training for {model_filename}.")
        train_env.close()
        eval_env.close()

    print("\nAll continuation training experiments completed.")