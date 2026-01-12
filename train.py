import gymnasium as gym
import numpy as np
import pandas as pd
from envs.intraday_option_env import IntradayOptionEnv
import config
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def plot_results(log_folder, save_folder, title='Learning Curve'):
    """
    Reads the monitor.csv file and plots the training results.
    """
    x, y = [], []
    try:
        # PPO logs are saved in monitor.csv in the log_folder
        # Stable Baselines3 Monitor saves as <timestamp>.monitor.csv or monitor.csv
        import glob
        monitor_files = glob.glob(os.path.join(log_folder, "*.monitor.csv"))
        if not monitor_files:
            print("No monitor files found.")
            return

        # Combine all monitor files
        dfs = []
        for file in monitor_files:
            with open(file, 'rt') as f:
                first_line = f.readline()
                if not first_line or first_line[0] != '#': continue
                dfs.append(pd.read_csv(f, index_col=None))
        
        if not dfs:
            print("No valid monitor data found.")
            return
            
        df = pd.concat(dfs).sort_values('t')
            
        x = df['t'].values  
        rewards = df['r'].values
        
        # Calculate rolling average (window = 50 episodes)
        rolling_window = 50
        rolling_avg = pd.Series(rewards).rolling(window=rolling_window).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        plt.plot(rolling_avg, color='red', label=f'{rolling_window}-Episode Moving Average')
        plt.title(title)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(save_folder, 'training_curve.png')
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    # Configuration
    MODE = 'add'  # 'add' or 'ratio'
    AGENT = 'PPO' # Agent Name
    SAVE_DIR = os.path.join('model', AGENT)
    SAVE_PATH = os.path.join(SAVE_DIR, f'{AGENT.lower()}_intraday_model')
    LOG_DIR = os.path.join(SAVE_DIR, 'logs')
    DATA_PATH = 'data/train.csv'
    START_DATE = "2020-01-01"  # Format: 'YYYY-MM-DD'
    END_DATE = "2020-12-31"    # Format: 'YYYY-MM-DD'
    NUM_ENVS = 1 # Number of parallel environments (adjust based on CPU cores)

    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Starting {NUM_ENVS} Parallel Environments with Mode: {MODE}")
    print(f"Using SubprocVecEnv for multiprocessing speedup.")

    try:
        # Create Vectorized Environment with SubprocVecEnv
        # We need to pass lambda function to make_vec_env
        env_kwargs = {
            'data_path': DATA_PATH, 
            'mode': MODE, 
            'start_date': START_DATE, 
            'end_date': END_DATE
        }
        
        env = make_vec_env(
            IntradayOptionEnv,
            n_envs=NUM_ENVS,
            seed=0,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs,
            monitor_dir=LOG_DIR  # Automatically wraps with Monitor
        )

        print("Training PPO Agent...")
        # User-customized training parameters for 1-minute data
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=10000 // NUM_ENVS, batch_size=1000, gamma=0.99, tensorboard_log=LOG_DIR, device='cpu')
        model.learn(total_timesteps=1000000)
        print("Training Finished.")
        
        model.save(SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")
        
        # Plotting Results
        print("Generating training plots...")
        plot_results(LOG_DIR, SAVE_DIR)
        
    except ImportError:
        print("Stable-Baselines3 not found. Falling back to Random Agent.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()