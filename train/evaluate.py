import os
import numpy as np
from stable_baselines3 import PPO
from envs.simpleEnv import GridAgent # Assuming simpleEnv.py is in the same directory or accessible
import gymnasium as gym
import time # Added for pausing in rendering

# --- Configuration ---
MODELS_DIR = "./lstms"
EVAL_EPISODES = 50
RESULTS_FILE = "evaluation_with_remember_action.txt"
ROOMS = "./r" # Set to your room path if you used one during training
RENDER_EVAL_EPISODES = 0 # Number of episodes to render (set to 0 for no rendering)
RENDER_PAUSE_SEC = 0.00 # Pause duration between frames for rendering

# --- Environment Wrapper ---
class GridEnv(gym.Env):
    """A wrapper for the GridAgent to make it compatible with Gymnasium."""
    def __init__(self, room_path=None, render_mode="human"): # Pass render_mode to wrapper
        super().__init__()
        # It's important that the environment parameters match those used during training
        self.agent = GridAgent(
            render_mode=render_mode, # Pass the render_mode down to the GridAgent
            room_path=room_path,
            max_steps=2000,
            width=20,
            depth=20,
            height=12,
            cell_size=0.25,
            local_map_length=4
        )
        self.agent.reset()
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space
        self.render_mode = render_mode # Store render_mode in wrapper

    def reset(self, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        # Gymnasium's reset now takes optional seed and options.
        # You should pass these if your underlying agent needs them for reproducibility.
        # For simpleEnv, we'll just call the agent's reset.
        super().reset(seed=seed) # Call super() reset first if you plan to use Gymnasium's seeding
        
        # Ensure agent reset handles the return (observation, info) if you're using Gymnasium 0.29+
        # Current simpleEnv.py reset returns obs, info. If it doesn't, you might need to adjust.
        # For now, assuming agent.reset() implicitly sets up for get_obs()
        self.agent.reset() 
        return self.agent.get_obs(), {} # Return observation and an empty info dict

    def step(self, action):
        """Takes an action and returns the transition."""
        return self.agent.step(action)

    def render(self):
        """Renders the environment by calling the agent's render method."""
        if self.render_mode is not None:
            self.agent.render()
            if self.render_mode == "matplotlib":
                time.sleep(RENDER_PAUSE_SEC) # Add a small pause for visualization

    def close(self):
        """Closes any rendering windows."""
        self.agent.close()


def evaluate_models(models_dir: str, num_episodes: int, results_filename: str, render_episodes: int):
    """
    Loads all models from a directory, evaluates them, and writes the results to a file.
    Includes an option to render a specified number of evaluation episodes.

    Args:
        models_dir (str): The directory containing the saved model .zip files.
        num_episodes (int): The number of episodes to run for each model.
        results_filename (str): The name of the file to save the evaluation results.
        render_episodes (int): The number of episodes to render for visual inspection.
    """
    print(f"Starting evaluation of models in '{models_dir}'...")

    # Ensure the models directory exists
    if not os.path.isdir(models_dir):
        print(f"Error: Directory not found at '{models_dir}'")
        return

    # Find all model files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]

    if not model_files:
        print(f"No .zip models found in '{models_dir}'")
        return

    # Create and open the results file to write the header
    with open(results_filename, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"{'Model Name':<40} | {'Avg Score':>12} | {'Avg Bumps':>12} | {'Finished (%)':>15} | {'Avg Discovered':>18}\n")
        f.write("-" * 105 + "\n")

    # Evaluate each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        print(f"\n--- Evaluating model: {model_name} ---")

        # Create the environment for each model evaluation to ensure clean state
        # The render_mode is passed based on whether this episode should be rendered.
        env = GridEnv(room_path=ROOMS, render_mode="matplotlib" if render_episodes > 0 else None)

        # Load the trained model
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Could not load model {model_name}. Error: {e}")
            env.close() # Ensure environment is closed even if model fails to load
            continue

        # --- Statistics for the current model ---
        total_scores = []
        total_bumps = []
        total_finishes = 0
        total_discovered_cells = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_score = 0

            # Only render for the first 'render_episodes'
            current_episode_renders = episode < render_episodes
            if current_episode_renders:
                print(f"  Rendering Episode {episode + 1}/{num_episodes} for {model_name}")
                env.render() # Initial render

            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_score += reward

                if current_episode_renders:
                    env.render() # Render each step

            # After the episode, collect stats from the agent
            total_scores.append(episode_score)
            total_bumps.append(env.agent.bump_count)
            total_discovered_cells.append(env.agent.visited_count)
            if env.agent.done: # `env.agent.done` is True only if it finished, not truncated
                total_finishes += 1

            print(f"  Episode {episode + 1}/{num_episodes} finished. Score: {episode_score:.2f}")
            
            # Ensure final render update for the episode
            if current_episode_renders:
                # Give a moment to see the final state before closing/resetting
                time.sleep(1.0) # Longer pause at end of rendered episode

        # Close the environment for the current model after all episodes
        env.close() 

        # Calculate averages
        avg_score = np.mean(total_scores)
        avg_bumps = np.mean(total_bumps)
        finish_percentage = (total_finishes / num_episodes) * 100
        avg_discovered = np.mean(total_discovered_cells)

        # Append results to the file
        with open(results_filename, 'a') as f:
            f.write(f"{model_name:<40} | {avg_score:>12.2f} | {avg_bumps:>12.2f} | {finish_percentage:>14.1f}% | {avg_discovered:>18.2f}\n")

        print(f"--- Results for {model_name} ---")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Bumps: {avg_bumps:.2f}")
        print(f"  Times Finished: {total_finishes}/{num_episodes} ({finish_percentage:.1f}%)")
        print(f"  Average Cells Discovered: {avg_discovered:.2f}")


if __name__ == '__main__':
    # Make sure to create a 'checkpoints' directory and place your models there
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory '{MODELS_DIR}'. Please place your trained models (e.g., 'ppo_model.zip') in this folder.")
    else:
        evaluate_models(
            models_dir=MODELS_DIR,
            num_episodes=EVAL_EPISODES,
            results_filename=RESULTS_FILE,
            render_episodes=RENDER_EVAL_EPISODES # Pass the rendering control
        )
        print(f"\nEvaluation complete. Results saved to '{RESULTS_FILE}'.")