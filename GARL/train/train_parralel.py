import os
import glob
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import sys
from GARL.parallel_env.ContinuousDroneEnv import ContinuousDroneEnv 
from pathlib import Path
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback # Import BaseCallback for custom callback
from envs.NewDroneEnv import NewDroneEnv
N_ENVS = 4          

# --- Custom Callback to Save VecNormalize ---
class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize object along with the model.
    :param save_path: (str) Path to the folder where the model and VecNormalize will be saved.
    :param name_prefix: (str) Prefix for the saved model/VecNormalize filename.
    :param save_freq: (int) Save the model and VecNormalize every `save_freq` steps.
    """
    def __init__(self, save_path: str, name_prefix: str = "model_grid", save_freq: int = 1000, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_freq = save_freq

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            current_timesteps = self.num_timesteps
            # Build paths for model and VecNormalize
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{current_timesteps}_steps.zip")
            vec_normalize_path = os.path.join(self.save_path, f"{self.name_prefix}_{current_timesteps}_steps_vec_normalize.pkl")

            # Save the model
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")

            # Save the VecNormalize (if present)
            # self.training_env is the VecEnv wrapper
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(vec_normalize_path)
                if self.verbose > 0:
                    print(f"Saving VecNormalize statistics to {vec_normalize_path}")
            else:
                if self.verbose > 0:
                    print("VecNormalize not found in training environment (self.training_env is not VecNormalize), skipping VecNormalize save for this checkpoint.")
        return True
# --- End Custom Callback ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps for this run')
    parser.add_argument('--save-path', type=str, default='./models/ppo_lstm_drone_final', help='Base path for the final model and VecNormalize (e.g., ../models/ppo_lstm_drone_final)')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/', help='Directory to save model and VecNormalize checkpoints during training')
    parser.add_argument('--load-model', type=str, help='Path to an existing model to load and continue training from (e.g., ./checkpoints/ppo_model_13000_steps.zip)')
    parser.add_argument('--nenvs', default=1, type=int, help="Amount of environments to train parralel")
    parser.add_argument('--headless', type=bool, default=False, help='Makes the simulation run headless')

    # Removed --load-vec-normalize as it's now handled by deriving from --load-model
    args = parser.parse_args()

    os.makedirs(args.checkpoint_path, exist_ok=True) 
    custom_checkpoint_callback = SaveVecNormalizeCallback(
        save_path=args.checkpoint_path, # Use the dedicated checkpoint path
        name_prefix="ppo_model",
        save_freq=10000, #
        verbose=1 # Set to 1 to see saving messages
    )

    def make_env(rank: int):

        def _init():
            return NewDroneEnv(
                instance_id=rank,
                headless=args.headless
            )
        return _init

    env = SubprocVecEnv([make_env(i+1) for i in range(args.nenvs)], start_method="spawn")
        
    # --- VecNormalize Handling ---
    vec_normalize_file = None
    if args.load_model:
        model_dir = os.path.dirname(args.load_model)
        model_name_without_ext = os.path.splitext(os.path.basename(args.load_model))[0]
        vec_normalize_file = os.path.join(model_dir, f"{model_name_without_ext}_vec_normalize.pkl")
        
    if vec_normalize_file and os.path.exists(vec_normalize_file):
        print(f"Loading VecNormalize statistics from: {vec_normalize_file}")
        env = VecNormalize.load(vec_normalize_file, env)
    else:

        print("Creating new VecNormalize instance (first training run or no VecNormalize found/specified).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    try:
        if args.load_model:
            print(f"Loading model from {args.load_model} to continue training...")
            model = RecurrentPPO.load(args.load_model, env=env)
            print("Model loaded successfully. Resuming training.")
        else:

            policy_kwargs = dict(
            # size of the hidden state that the LSTM carries between timesteps
            lstm_hidden_size = 512,   # <— change this
            n_lstm_layers    = 2,     # <— and/or this
            # MLP layers that process the observation *before* it goes into the LSTM
            net_arch = [128, 128],    # shared net for both actor & critic
            activation_fn = nn.Tanh   # any `torch.nn` activation
        )
            print("No --load-model specified. Starting new training run.")
            model = RecurrentPPO(
                'MlpLstmPolicy',
                env,
                policy_kwargs=policy_kwargs,
                verbose=1
            )

        print(f"Training for an additional {args.timesteps} timesteps...")
        # Pass the custom callback to model.learn()
        model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=custom_checkpoint_callback)
        
        # --- Final Model and VecNormalize Saving (after normal completion) ---
        # Ensure the final save directory exists
        final_model_save_dir = os.path.dirname(args.save_path)
        os.makedirs(final_model_save_dir, exist_ok=True)
        
        model_final_path = f"{args.save_path}.zip"
        vec_normalize_final_path = f"{args.save_path}_vec_normalize.pkl"

        model.save(model_final_path)
        env.save(vec_normalize_final_path) # Save VecNormalize statistics

        print(f"\nFinal model saved to {model_final_path}")
        print(f"VecNormalize statistics saved to {vec_normalize_final_path}")
        # --- End Final Saving ---

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # It's important to close the environment cleanly.
        env.close()
        if 'env' in locals() and env is not None:
            env.close()
            print("\nEnvironment closed.")