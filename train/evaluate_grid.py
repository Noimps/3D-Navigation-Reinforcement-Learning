import os
import numpy as np
from stable_baselines3 import PPO
from envs.Venv import GridAgent # Assuming simpleEnv.py is in the same directory or accessible
import gymnasium as gym
import time # Added for pausing in rendering
import pandas as pd # Import pandas for CSV output
from sb3_contrib import RecurrentPPO

# --- Configuration ---
MODELS_DIR = "./exp3_architectures" # Directory where your trained models are saved
EVAL_EPISODES = 10
RESULTS_TXT_FILE = "exp3_viewDistance.txt" # For aggregated results (existing format)
RESULTS_CSV_FILE = "exp3_viewDistance.csv" # New file for individual trial data
ROOMS = "./rooms/P1_evaluate" # Set to your room path if you used one during training
RENDER_EVAL_EPISODES = 0 # Number of episodes to render (set to 0 for no rendering)
RENDER_PAUSE_SEC = 0 # Pause duration between frames for rendering

EVAL_ROOMS = {
    "P1_empty": "./rooms/P1_evaluate",
    "P2_small": "./rooms/P2_evaluate",
    "P3_large": "./rooms/P3_evaluate",
}



# --- Environment Wrapper ---
class GridEnv(gym.Env):
    """A wrapper for the GridAgent to make it compatible with Gymnasium."""
    def __init__(self, local_map_length,room_path=None, crash_penalty=-2.0, render_mode="human"): # Pass render_mode to wrapper
        super().__init__()
        # It's important that the environment parameters match those used during training
        self.agent = GridAgent(
            render_mode=render_mode, # Pass the render_mode down to the GridAgent
            room_path=room_path,
            local_map_length=local_map_length,
            crash_penalty=crash_penalty, # Ensure this matches your training setup
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

        self.render_mode = "matplotlib"
        if self.render_mode is not None:
            self.agent.render()
            if self.render_mode == "matplotlib":
                time.sleep(RENDER_PAUSE_SEC) # Add a small pause for visualization

    def close(self):
        """Closes any rendering windows."""
        self.agent.close()


def evaluate_models(models_dir: str, num_episodes: int, results_txt_filename: str, results_csv_filename: str, render_episodes: int):
    """
    Loads all models from a directory, evaluates them, and writes the results to two files:
    - A .txt file for aggregated results (existing format).
    - A .csv file for individual trial data, necessary for statistical analysis.

    Args:
        models_dir (str): The directory containing the saved model .zip files.
        num_episodes (int): The number of episodes to run for each model.
        results_txt_filename (str): The name of the file to save the aggregated evaluation results.
        results_csv_filename (str): The name of the file to save individual trial results.
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

    # Create and open the aggregated results file to write the header
    with open(results_txt_filename, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"{'Model Name':<40} | {'Avg Score':>12} | {'Avg Bumps':>12} | {'Finished (%)':>15} | {'Avg Discovered':>18} | {'Avg Steps':>12}\n")
        f.write("-" * 120 + "\n")

    # List to store all individual episode results for CSV output
    all_episode_results = []

    # Evaluate each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        print(f"\n--- Evaluating model: {model_name} ---")

       # phase = "_".join([model_name.split('_')[0], model_name.split('_')[1]])  # Extract phase from model name



        comps = model_name.split('arch')
       # print(f"  Model components: {comps}") # Debugging line to see how the model name is split
        hp_set = int(comps[0].split('_')[1].strip('hp')) # Extract hyperparameter set
        #hp_set = 1 # Extract hyperparameter set

        print(f"  Hyperparameter set: {hp_set}")
        arch = comps[1].split('lstm')[0].strip('_')
        print(arch)
        lstm = comps[1].split('lstm')[1].split('_')[0]
        print(lstm)


        trained_steps = int(comps[1].split('lstm')[1].split('_')[1].strip('s')) # Extract trained steps
       # print(comps[1].split('steps')[1].split('_')[1])
       # trained_steps = int(comps[1].split('steps')[1].split('_')[1]) # Extract trained steps

        
        view_distance = int(comps[1].split('view')[1].split('_')[0])        

      #  view_distance = 16      
        print(view_distance)
        print(f"  Trained steps: {trained_steps}")
        crash_penalty = float(comps[1].split('crash')[1].split('_')[0])
       # print(crash_penalty)
        if trained_steps <= 1000000:
            phase = "P1_empty"  # Use P1_empty for models trained with less than or equal to 500k steps
        elif trained_steps <= 21000000:
            phase = "P2_small"
        else:
            phase = "P3_large"
        room_path = EVAL_ROOMS[phase]


        #crash_penalty = -2.0 # Set a default crash penalty, adjust as needed
        print(f"  Using phase: {phase}")
        print(f"  Room path: {room_path}")


        # Create the environment for each model evaluation to ensure clean state
        env = GridEnv(room_path=room_path, local_map_length=view_distance, render_mode="matplotlib" if render_episodes > 0 else None)

        # Load the trained model
        try:
            model = RecurrentPPO.load(model_path, env=env)
        except Exception as e:
            print(f"Could not load model {model_name}. Error: {e}")
            env.close() # Ensure environment is closed even if model fails to load
            continue
        

        print(f"LOADED MODEL:  Model Architecture: {arch}, View Distance: {view_distance}, LSTM Size: {lstm}, Steps Trained: {trained_steps}")

        # --- Statistics for the current model ---
        total_scores = []
        total_bumps = []
        total_finishes = 0
        total_discovered_cells = []
        total_steps = 0

        for episode_num in range(num_episodes):
            state, episode_start = None, True

            obs, _ = env.reset()
            done = False
            truncated = False

            episode_score = 0
            episode_steps = 0 # Track steps for current episode
            episode_bumps = 0 # Track bumps for current episode
            episode_discovered_cells = 0 # Track discovered cells for current episode

            # Only render for the first 'render_episodes'
            current_episode_renders = episode_num < render_episodes
            if current_episode_renders:
                print(f"  Rendering Episode {episode_num + 1}/{num_episodes} for {model_name}")
            env.render() # Initial render

            while not (done or truncated):
                action, state = model.predict(
                    obs,
                    state=state,
                    episode_start=[episode_start],
                    deterministic=True,
                )
                episode_start = False
                obs, reward, done, truncated, info = env.step(action)
                episode_score += reward
                episode_steps += 1 # Increment step count for the current episode
                if current_episode_renders and env.agent.bump_count > 10:
                    env.render() # Render each step
                    time.sleep(RENDER_PAUSE_SEC) # Pause for rendering


            # After the episode, collect stats from the agent
            episode_bumps = env.agent.bump_count
            episode_discovered_cells = env.agent.visited_count
            finished_episode = env.agent.done # True if finished, False if truncated or max steps

            # Store individual episode results   P1_empty_r6_arch128-128_lstm128x1_s250000



            all_episode_results.append({
                'Model_Name': model_name,
                'hp_set': hp_set,
                'Architecture': arch,
                'LSTM_Size': lstm,
                'Trained_Steps': trained_steps,
                'View_Distance': view_distance,
                'Crash_Penalty': crash_penalty,
                'Episode_Number': episode_num + 1,
                'Score': episode_score,
                'Bumps': episode_bumps,
                'Finished': finished_episode,
                'Discovered_Cells': episode_discovered_cells,
                'Steps_Taken': episode_steps,

            })


            total_scores.append(episode_score)
            total_bumps.append(episode_bumps)
            total_discovered_cells.append(episode_discovered_cells)
            total_steps += episode_steps # Sum up total steps for average calculation
            if finished_episode:
                total_finishes += 1

            print(f"  Episode {episode_num + 1}/{num_episodes} finished. Score: {episode_score:.2f}")

            # Ensure final render update for the episode
            if current_episode_renders:
                # Give a moment to see the final state before closing/resetting
                time.sleep(1.0) # Longer pause at end of rendered episode

        # Close the environment for the current model after all episodes
        env.close()

        # Calculate averages for the aggregated TXT file
        avg_score = np.mean(total_scores)
        avg_bumps = np.mean(total_bumps)
        finish_percentage = (total_finishes / num_episodes) * 100
        avg_discovered = np.mean(total_discovered_cells)
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0

        # Append aggregated results to the TXT file
        with open(results_txt_filename, 'a') as f:
            f.write(f"{model_name:<40} | {avg_score:>12.2f} | {avg_bumps:>12.2f} | {finish_percentage:>14.1f}% | {avg_discovered:>18.2f} | {avg_steps:>12.2f}\n")

        print(f"--- Aggregated Results for {model_name} ---")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Bumps: {avg_bumps:.2f}")
        print(f"  Times Finished: {total_finishes}/{num_episodes} ({finish_percentage:.1f}%)")
        print(f"  Average Cells Discovered: {avg_discovered:.2f}")
        print(f"  Average Steps Taken: {avg_steps:.2f}")

    # After evaluating all models, save all individual episode results to CSV
    if all_episode_results:
        df_results = pd.DataFrame(all_episode_results)
        df_results.to_csv(results_csv_filename, index=False)
        print(f"\nIndividual trial results saved to '{results_csv_filename}'.")
    else:
        print("\nNo models were evaluated, so no individual trial results were saved.")


if __name__ == '__main__':
    # Make sure to create a 'checkpoints' directory and place your models there
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory '{MODELS_DIR}'. Please place your trained models (e.g., 'ppo_model.zip') in this folder.")
    else:
        evaluate_models(
            models_dir=MODELS_DIR,
            num_episodes=EVAL_EPISODES,
            results_txt_filename=RESULTS_TXT_FILE,
            results_csv_filename=RESULTS_CSV_FILE,
            render_episodes=RENDER_EVAL_EPISODES # Pass the rendering control
        )
        print(f"\nEvaluation complete. Aggregated results saved to '{RESULTS_TXT_FILE}'.")