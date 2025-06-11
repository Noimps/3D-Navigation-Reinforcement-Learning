import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces
import numpy as np

from envs.simpleEnv import GridAgent

#ROOMS = "/home/noimps/drl/rooms"
NN_SIZES = [
   # [64, 64],
    #[128, 128],
    [256, 256],
    [128, 64, 128],
    [256, 256, 128]
]

STEPS = 500_000
ROOMS = None

# Wrapper to turn GridAgent into a Gymnasium-compatible Env
class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.agent = GridAgent(room_path=ROOMS)
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space

    def reset(self, seed=None, options=None):
        self.agent.reset()
        return self.agent.get_obs(), {}

    def step(self, action):
        return self.agent.step(action)

    def render(self):
        print("Agent Position:", self.agent.get_position())

# Helper to create environments for SubprocVecEnv
def make_env():
    def _init():
        env = GridEnv()
        return env
    return _init

if __name__ == '__main__':
    # Number of parallel environments
    num_envs = 8

    # --- Hyperparameter Sets for PPO ---
    # These are some common starting points and variations.
    # You might want to iterate on these or add more.
    hyperparameter_sets = [
        #SET 1
        {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "clip_range": 0.2,
            "n_epochs": 10
        },
        #SET 2
        {
            "learning_rate": 0.0001, # Slightly lower learning rate
            "n_steps": 1024,        # Smaller rollout buffer
            "batch_size": 32,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.005,      # Reduced entropy to encourage exploitation
            "vf_coef": 0.5,
            "clip_range": 0.1,      # Tighter clipping
            "n_epochs": 10
        },
        # SET 3
        {
            "learning_rate": 0.0005, # Slightly higher learning rate
            "n_steps": 4096,        # Larger rollout buffer
            "batch_size": 128,
            "gamma": 0.995,         # Higher gamma for longer horizons
            "gae_lambda": 0.98,     # Higher GAE lambda for smoother advantage
            "ent_coef": 0.01,
            "vf_coef": 0.6,         # Slightly more emphasis on value function
            "clip_range": 0.2,
            "n_epochs": 15          # More epochs for policy updates
        }
    ]
    # --- End Hyperparameter Sets ---

    for i, params in enumerate(hyperparameter_sets):
        print(f"\n--- Starting training for Hyperparameter Set {i+1}/{len(hyperparameter_sets)} ---")
        print(f"Hyperparameters: {params}")

        for layer_size in NN_SIZES:
            env_fns = [make_env() for _ in range(num_envs)]
            vec_env = SubprocVecEnv(env_fns)

            # Optional: Validate a single instance of the environment
            check_env(GridEnv(), warn=True)

            policy_kwargs = dict(
                net_arch=layer_size,
                # For MlpPolicy, lstm_hidden_size and n_lstm_layers are not applicable.
                # If you intend to use an LSTM policy (e.g., LstmPolicy), you'd specify them.
                # For MlpPolicy, these should be removed or set appropriately.
                # Since your original code had them, I'm commenting them out for MlpPolicy.
                # If GridAgent uses LSTM, then you'd switch to "LstmPolicy" and uncomment these.
                # lstm_hidden_size=512,
                # n_lstm_layers=2
            )
                        
            # Initialize PPO model with the vectorized environment and hyperparameters
            model = PPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                clip_range=params["clip_range"],
                n_epochs=params["n_epochs"]
            )

            # Train model
            model.learn(total_timesteps=STEPS, progress_bar=True)

            # Save model
            model.save(f"./models/ppo_set{i+1}_size{'_'.join(map(str, layer_size))}_{STEPS}steps_lastaction")

            # Close environment after training
            vec_env.close()

    print("Parallel training complete for all hyperparameter sets.")