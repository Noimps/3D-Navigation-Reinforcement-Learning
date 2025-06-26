import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.SingleEnv import SingleEnv

# --------------------
# Configuration
# --------------------
save_dir    = "./checkpoints"
vecnormalize_dir = "./vecnormalize_saves"
TOTAL_TIMESTEPS = 100000  # total training timesteps
SAVE_FREQ       = 10000    # save model every N steps

os.makedirs(save_dir, exist_ok=True)
os.makedirs(vecnormalize_dir, exist_ok=True)

# --------------------
# Environment Setup
# --------------------
def make_env(instance_id):
    return SingleEnv(instance_id=instance_id, headless=True)
    

env = DummyVecEnv([lambda: make_env(1)])  # Create a single environment instance
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# --------------------
# Callback for saving model checkpoints
# --------------------
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                           name_prefix='ppo_model')

# --------------------
# Create the PPO model
# --------------------
model = PPO("MlpPolicy", env, verbose=1)

# --------------------
# Train the model
# --------------------
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=checkpoint_callback)

# Save the final model and VecNormalize statistics
model.save(os.path.join(save_dir, "ppo_final_model"))
env.save(os.path.join(vecnormalize_dir, "vecnormalize_final.pkl"))

env.close()