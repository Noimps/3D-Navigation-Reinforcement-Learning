"""
Incremental multi-phase PPO-LSTM training on GridAgent with intermediate checkpoints and reproducible seeding.

This script trains a Recurrent PPO model through multiple phases, saving checkpoints at every 10%
interval of the total timesteps for each phase.

For each configuration:
  1. Train on Phase 1 (e.g., empty rooms)
  2. Continue on Phase 2 (e.g., small mazes)
  3. Continue on Phase 3 (e.g., large furnished rooms)

Requires:
    pip install stable-baselines3 sb3-contrib gymnasium torch numpy
"""

import os
import random
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# Corrected import path based on your updated Venv.py location
from envs.Kenv import GridAgent
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path # Still useful for path manipulation if needed, but not for creating dummy files

# ---------------------------------------
# Seeding for reproducibility
# ---------------------------------------
BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
NUM_ENVS = 8
LOCAL_MAP_LENGTHS = [ 10] # This affects how far _sense_direction looks in GridAgent
CRASH_PENALTIES = [-2.0]  # These are now set within GridAgent, but can be overridden if passed to init

SAVE_DIR = "./exp3_architectures"
os.makedirs(SAVE_DIR, exist_ok=True)


EVAL_FREQ = 100_000
best_model_dir = os.path.join(SAVE_DIR, f"best_exp3_P1")
os.makedirs(best_model_dir, exist_ok=True)

# ---------------------------------------
# Experiment configuration
# ---------------------------------------
PHASES = [
    ("P1_empty", "./rooms/P1_training", "./rooms/P1_evaluate"),
    # Uncomment and define these if you add more phases later
    # ("P2_small", "./rooms/P2_training", "./rooms/P2_evaluate"),
    # ("P3_large", "./rooms/P3_training", "./rooms/P3_evaluate"),
]
# Training steps per phase
STEPS_PHASE = {
    #"P1_empty": 20_000_000,
     "P2_small": 20_000_000,
    # "P3_large": 2_000_000,
}


# MLP architectures for Actor and Critic networks (after LSTM)
# MlpLstmPolicy uses 'net_arch' for layers after the LSTM output.
# The initial feature extractor (before LSTM) defaults to a Flatten layer if observation is flat,
# or a NatureCNN if observation is image. Here, it will be a Flatten for the 80-vector.
architectures = [
    #dict(pi=[128, 128], vf=[128, 128]),
    #dict(pi=[256, 256], vf=[256, 256]),
    # You could add deeper architectures if needed
    dict(pi=[256, 256, 128], vf=[256, 256, 128]),
]

# LSTM configurations
lstm_sizes = [
    dict(lstm_hidden_size=256, n_lstm_layers=1), # Explicitly set shared_lstm
    # dict(lstm_hidden_size=128, n_lstm_layers=2, shared_lstm=True), # Example with 2 layers
    # dict(lstm_hidden_size=256, n_lstm_layers=1, shared_lstm=False), # Example with separate LSTMs
]

# PPO hyperparameters
ppo_hparam_sets = [
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            vf_coef=0.5, clip_range=0.2, n_epochs=10)
    # Add other sets if you have them from your commented section
]


# ---------------------------------------
# Environment wrapper
# ---------------------------------------
class GridEnv(gym.Env):
    """A wrapper for the GridAgent to be compatible with SB3."""
    def __init__(self, room_path, local_map_length, crash_penalty=-2.0):
        super().__init__()
        # Pass crash_penalty to GridAgent if you want to control it from the training script
        # Otherwise, GridAgent uses its internal default.
        # For simplicity, sticking to GridAgent's internal default for crash_penalty
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
            # print only a summary to avoid flooding console during training
            print(f"Agent Pos: {pos}, Visited: {self.agent.visited_count}/{self.agent.total_free_cells} ({self.agent.visited_count/self.agent.total_free_cells*100:.2f}%)")
        else:
            super().render(mode=mode)

def make_env_fn(room_path, local_map_length, seed_offset):
    """Factory for vectorized environment workers with unique but reproducible seeds."""
    def _init():
        env = GridEnv(room_path, local_map_length)
        env.reset(seed=BASE_SEED + seed_offset)
        return Monitor(env) # Monitor wraps env to provide episode stats
    return _init

# ---------------------------------------
# Main experiment loop
# ---------------------------------------
if __name__ == "__main__":
    # --- IMPORTANT: Ensure room directories exist and contain valid .txt files ---
    # The dummy room creation code has been removed.
    # Verify that paths like "./rooms/P1_training" and "./rooms/P1_evaluate"
    # exist and are populated with your actual room files.

    # Sanity check the environment to catch issues early
    try:
        # Pass a specific room path for initial check
        check_env(GridEnv(PHASES[0][1], LOCAL_MAP_LENGTHS[0]), warn=True)
        print("Environment check passed.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        # Print detailed traceback for debugging
        import traceback
        traceback.print_exc()
        exit()


    for ray_len in LOCAL_MAP_LENGTHS: # Iterating over local_map_length
        for hp_i, ppo_hp in enumerate(ppo_hparam_sets):
            # When architectures are dictionaries for pi and vf, make arch_str more descriptive
            for arch in architectures:
                arch_str = f"pi{arch['pi']}_vf{arch['vf']}"
                for lstm_kwargs in lstm_sizes:
                    
                    # Ensure shared_lstm is correctly reflected in the ID if it's a variable
                    shared_lstm_str = "shared" if lstm_kwargs.get('shared_lstm', True) else "separate"
                    lstm_str = f"h{lstm_kwargs['lstm_hidden_size']}l{lstm_kwargs['n_lstm_layers']}_{shared_lstm_str}"
                    
                    base_id = f"r{ray_len}_arch{arch_str}_lstm{lstm_str}"
                    model = None
                    
                    cumulative_steps_total = 0

                    for phase_name, train_path, eval_path in PHASES:
                        print(f"\n{'='*20}\nSTARTING PHASE: {phase_name} for {base_id}\n{'='*20}")

                        # Create a NEW evaluation environment for each phase
                        eval_env = SubprocVecEnv([
                            make_env_fn(eval_path, ray_len, i + NUM_ENVS) # Offset seed for eval envs
                            for i in range(NUM_ENVS) # Using NUM_ENVS for eval for consistency, can be different
                        ])

                        steps_this_phase = STEPS_PHASE[phase_name]
                        if steps_this_phase <= 0:
                            print(f"Skipping phase {phase_name} as it has 0 steps.")
                            eval_env.close() # Ensure eval env is closed if phase is skipped
                            continue

                        segment_size = steps_this_phase // 10
                        if segment_size == 0:
                            segments = [steps_this_phase]
                        else:
                            segments = [segment_size] * 10
                            remainder = steps_this_phase - sum(segments)
                            if remainder > 0:
                                segments[-1] += remainder
                        
                        # Create a NEW training environment for each phase
                        env_fns = [make_env_fn(train_path, ray_len, i) for i in range(NUM_ENVS)]
                        train_env = SubprocVecEnv(env_fns)

                        # Instantiate a new model for the first phase, or set the new environment for continuation
                        if model is None:
                            print("Instantiating new model.")
                            # Pass policy_kwargs correctly (net_arch and lstm_kwargs)
                            policy_kwargs = dict(net_arch=arch, **lstm_kwargs)
                            model = RecurrentPPO(
                                policy="MlpLstmPolicy",
                                env=train_env, # Set the environment here
                                verbose=1,
                                policy_kwargs=policy_kwargs,
                                **ppo_hp, # Apply PPO hyperparameters
                            )
                        else:
                            print(f"Continuing training on {phase_name}. Setting new environment.")
                            model.set_env(train_env) # Update the environment for the existing model

                        # Training segments with intermediate saves
                        for i, seg_steps in enumerate(segments):
                            cumulative_steps_total += seg_steps

                            print(f"\n--- Training {phase_name} segment {i + 1}/{len(segments)} ({seg_steps} steps) ---")
                            print(f"--- Cumulative steps for this model: {cumulative_steps_total} ---")

                            # EvalCallback setup
                            eval_callback = EvalCallback(
                                eval_env=eval_env,
                                best_model_save_path=best_model_dir,
                                log_path=best_model_dir,
                                eval_freq=max(EVAL_FREQ // NUM_ENVS, 1), # Eval freq is per env step
                                n_eval_episodes=10, # Number of episodes for evaluation
                                deterministic=True,
                                render=False, # Don't render eval during training unless specifically needed
                            )
                            
                            model.learn(total_timesteps=seg_steps, reset_num_timesteps=False, callback=eval_callback)
                            
                            # Define a unique save path for the checkpoint
                            # Using hp_i to differentiate hyperparameter sets
                            save_path  = f"{SAVE_DIR}/rppo_hp{hp_i+1}_arch_{arch_str}_lstm_{lstm_str}_s{cumulative_steps_total}_view{ray_len}.zip"
                            model.save(save_path)
                            print(f"Checkpoint saved to: {save_path}")

                        # IMPORTANT: Close environments ONLY after completing all segments for the current phase
                        train_env.close()
                        eval_env.close() # Also close the evaluation environment for this phase

    print("\nAll incremental exploration experiments with checkpoints completed.")