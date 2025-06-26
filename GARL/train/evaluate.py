import os
import glob
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import sys
from DDRLM.envs import ContinuousDroneEnv # Assuming this is your environment
import numpy as np # Import numpy for array manipulation if needed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, default='../models/ppo_lstm_drone_final.zip', help='Path to load the trained model')
    parser.add_argument('--n-eval-episodes', type=int, default=10, help='Number of episodes to evaluate the model')
    parser.add_argument('--render', action='store_true', default=True, help='Whether to render the environment (visualize in Gazebo)')
    args = parser.parse_args()

    world_files = glob.glob(os.path.join(args.world_dir, '*.sdf'))

    def make_env():
        return ContinuousDroneEnv(world_files) 

    env = DummyVecEnv([make_env])

    vec_normalize_file = None
    if args.load_model:
        model_dir = os.path.dirname(args.load_model)
        model_name_without_ext = os.path.splitext(os.path.basename(args.load_model))[0]
        # Attempt to derive the VecNormalize path from the loaded model's path
        vec_normalize_file = os.path.join(model_dir, f"{model_name_without_ext}_vec_normalize.pkl")
        
    if vec_normalize_file and os.path.exists(vec_normalize_file):
        print(f"Loading VecNormalize statistics from: {vec_normalize_file}")
        env = VecNormalize.load(vec_normalize_file, env)
    else:
        # Only create a new VecNormalize if no model is being loaded OR
        # if a model is loaded but its corresponding VecNormalize file was not found.
        if args.load_model:
            print(f"WARNING: Model '{args.load_model}' loaded, but NO corresponding VecNormalize file found at '{vec_normalize_file}'.")
            print("This will create a NEW VecNormalize instance, which will likely lead to poor performance when resuming training.")
            print("Consider finding the correct VecNormalize .pkl file or restarting training from scratch.")
        else:
            print("Creating new VecNormalize instance (first training run or no VecNormalize found/specified).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    try:
        print(f"Loading model from {args.load_model}")
        model = RecurrentPPO.load(args.load_model, env=env)
        print("Model loaded successfully.")

        print(f"Starting evaluation for {args.n_eval_episodes} episodes...")
        for episode in range(args.n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            # Initialize LSTM states at the beginning of each episode
            # model.predict automatically handles the initial state if you pass `state=None`
            # For RecurrentPPO, states are (hidden_state, cell_state)
            # Each is of shape (n_layers, n_envs, hidden_size)
            # Since DummyVecEnv has n_envs=1, it will be (n_layers, 1, hidden_size)
            # RecurrentPPO's predict method expects `state` as the previous hidden state tuple.
            # If `state` is None, it initializes them to zeros.
            lstm_states = None 
            
            # `dones` array for `model.predict` is typically used in vectorized environments
            # where some sub-environments might finish while others continue.
            # For DummyVecEnv (single environment), we just need to ensure the state is reset
            # when `done` becomes True.
            # The `predict` method of RecurrentPPO, when called with `state`, expects a `dones`
            # array if it's a vectorized environment to reset the states of specific environments.
            # However, for a single environment (DummyVecEnv), this argument is usually not
            # explicitly passed, or handled by the `vec_env` internally.
            
            # The error `BaseAlgorithm.predict() got an unexpected keyword argument 'done'`
            # confirms that `predict` method itself doesn't directly take `done`.
            # Instead, the `done` information is used by the `model.predict` method's internal
            # logic when it's provided with an `env` object during its creation or when
            # it's wrapped by a `VecEnv` that handles this.

            print(f"--- Episode {episode + 1}/{args.n_eval_episodes} ---")
            while not done:
                # `predict` method returns action and new_lstm_states (if policy is recurrent)
                # It handles the internal logic of resetting states based on 'done' information
                # that would come from a true VecEnv. For a single environment, when `done` is True,
                # you manually set `lstm_states = None` before the next episode's `reset`.
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=np.array([done]))
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0] # Assuming reward is a list/array for VecEnv

               #` if args.render:
                    # If your GazeboDroneEnv has a render method, you can call it here.
                    # Otherwise, Gazebo itself will be running and showing the drone.
                    # env.render() # Uncomment if your env has a separate render method

                if done:
                    print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
                    # After an episode is done, reset the LSTM states for the next episode
                    # This is crucial for recurrent policies when the episode terminates.
                    lstm_states = None # Reset LSTM states for the beginning of the next episode


    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed.")