"""
Train a recurrent (LSTM) PPO agent on GridAgent.

Requires:
    pip install stable-baselines3 sb3-contrib gymnasium
"""

import gymnasium as gym
from sb3_contrib import RecurrentPPO               # ← LSTM-capable PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces
import numpy as np

from envs.simpleEnv import GridAgent

# ---------------------------------------
# Config
# ---------------------------------------
NN_SIZES = [
    #[64, 64],
    #[128, 128],
    [256, 256],
    [128, 64, 128],
    [256, 256, 128],
]

STEPS          = 500_000
ROOMS          = "./rooms"        # Grid map (or leave None for random rooms)
NUM_ENVS       = 8          # parallel workers
SAVE_DIR       = "./lstms"  # where to save checkpoints

# ---------------------------------------
# Environment wrapper
# ---------------------------------------
class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.agent             = GridAgent(room_path=ROOMS)
        self.action_space      = self.agent.action_space
        self.observation_space = self.agent.observation_space

    def reset(self, seed=None, options=None):
        self.agent.reset()
        return self.agent.get_obs(), {}

    def step(self, action):
        return self.agent.step(action)

    def render(self):
        print("Agent Position:", self.agent.get_position())

def make_env():
    """Factory for SubprocVecEnv."""
    def _init():
        return GridEnv()
    return _init


    """    dict(learning_rate=3e-4, n_steps=2048, batch_size=64,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            vf_coef=0.5, clip_range=0.2, n_epochs=10),
    """
# ---------------------------------------
# PPO hyper-parameters (unchanged)
# ---------------------------------------
ppo_hparam_sets = [
    # --- SET 1 ---

    # --- SET 2 ---
    dict(learning_rate=1e-4, n_steps=1024, batch_size=32,
         gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
        vf_coef=0.5, clip_range=0.1, n_epochs=10),

    # --- SET 3 ---
    dict(learning_rate=5e-4, n_steps=4096, batch_size=128,
         gamma=0.995, gae_lambda=0.98, ent_coef=0.01,
         vf_coef=0.6, clip_range=0.2, n_epochs=15),
]


# ---------------------------------------
# LSTM-specific policy kwargs
#   • lstm_hidden_size – size of each LSTM layer
#   • n_lstm_layers   – number of stacked LSTM layers
#   • shared_lstm     – share the LSTM between policy and value nets
# ---------------------------------------
lstm_policy_sets = [
    # LIGHTWEIGHT
    dict(lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False),

    # BALANCED
    dict(lstm_hidden_size=256, n_lstm_layers=2, shared_lstm=True, enable_critic_lstm=False),

    # LARGE
    dict(lstm_hidden_size=512, n_lstm_layers=2, shared_lstm=True,enable_critic_lstm=False),
]

# ---------------------------------------
# Training
# ---------------------------------------
if __name__ == "__main__":

    # Quick env sanity check (single instance)
    check_env(GridEnv(), warn=True)

    for hp_i, ppo_hp in enumerate(ppo_hparam_sets, start=1):
        print(f"\n=== PPO Hyper-Param Set {hp_i}/{len(ppo_hparam_sets)} ===")
        print(ppo_hp)

        for arch in NN_SIZES:
            for lstm_i, lstm_kwargs in enumerate(lstm_policy_sets, start=1):

                print(f"  • net_arch={arch} | LSTM set {lstm_i}: {lstm_kwargs}")

                # Build vectorised environment
                env_fns = [make_env() for _ in range(NUM_ENVS)]
                vec_env = SubprocVecEnv(env_fns)

                # Merge feed-forward and LSTM kwargs
                policy_kwargs = dict(net_arch=arch, **lstm_kwargs)

                # Instantiate recurrent PPO
                model = RecurrentPPO(
                    policy="MlpLstmPolicy",
                    env=vec_env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    **ppo_hp,                     # unpack PPO hyper-params
                )

                # Train
                model.learn(total_timesteps=STEPS, progress_bar=True)

                # Save
                net_name   = "_".join(map(str, arch))
                lstm_name  = f"{lstm_kwargs['lstm_hidden_size']}x{lstm_kwargs['n_lstm_layers']}"
                save_path  = f"{SAVE_DIR}/rppo_hp{hp_i}_arch{net_name}_lstm{lstm_name}_{STEPS}"
                model.save(save_path)

                vec_env.close()

    print("\nAll training runs completed.")
