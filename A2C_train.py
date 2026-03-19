import gymnasium as gym
import json
import numpy as np
import pandas as pd
from envs.intraday_option_env import IntradayOptionEnv
import config
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def plot_results(log_folder, save_folder, title='Learning Curve'):
    """
    Reads the monitor.csv file and plots the training results.
    """
    x, y = [], []
    try:
        # A2C logs are saved in monitor.csv in the log_folder
        # Stable Baselines3 Monitor saves as <timestamp>.monitor.csv or monitor.csv
        import glob
        monitor_files = glob.glob(os.path.join(log_folder, "**", "*.monitor.csv"), recursive=True)
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
    import datetime
    import random
    import torch
    
    # Set Random Seeds for Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    CURRENT_RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # MODE = 'add'  # 'add' or 'ratio'
    AGENT = 'A2C' # Agent Name
    
    # Path to a saved model to retrain/continue training
    # If None, a new model is created.
    RETRAIN_MODEL_PATH = None 

    if RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH):
        SAVE_DIR = os.path.dirname(RETRAIN_MODEL_PATH)
        TIMESTAMP = os.path.basename(SAVE_DIR)
        print(f"Resuming training in existing directory: {SAVE_DIR}")
    else:
        TIMESTAMP = CURRENT_RUN_ID
        # Structure: model/A2C/{TIMESTAMP}/
        BASE_DIR = os.path.join('model', AGENT)
        SAVE_DIR = os.path.join(BASE_DIR, TIMESTAMP)
    
    # Artifacts inside the timestamped folder
    SAVE_PATH = os.path.join(SAVE_DIR, f'{AGENT.lower()}_intraday_model')
    LOG_DIR = os.path.join(SAVE_DIR, 'logs')
    MONITOR_DIR = os.path.join(LOG_DIR, f"monitor_{CURRENT_RUN_ID}")
    EVAL_MONITOR_DIR = os.path.join(SAVE_DIR, 'eval_monitor', f"eval_{CURRENT_RUN_ID}")
    
    DATA_PATH = 'data/train.csv'
    VIX_DATA_PATH = 'data/INDIA_VIX.csv'
    START_DATE = "2021-01-01"  # Training: 6 Years
    END_DATE = "2022-12-31"    # Validation is 2021 (set below)
    NUM_ENVS = 4  # Number of parallel environments (adjust based on CPU cores)

    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)
    os.makedirs(EVAL_MONITOR_DIR, exist_ok=True)

    print(f"Starting {NUM_ENVS} Parallel Environments.")
    print(f"Using SubprocVecEnv for multiprocessing speedup.")

    try:
        # Create Vectorized Environment with SubprocVecEnv
        # We need to pass lambda function to make_vec_env
        env_kwargs = {
            'data_path': DATA_PATH, 
            'vix_data_path': VIX_DATA_PATH,
            'start_date': START_DATE, 
            'end_date': END_DATE
        }
        
        env = make_vec_env(
            IntradayOptionEnv,
            n_envs=NUM_ENVS,
            seed=42,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs,
            monitor_dir=MONITOR_DIR  # Automatically wraps with Monitor, saved in unique run subfolder
        )

        # Create Validation Environment (Prevent Overfitting)
        # We use a separate time period (e.g., 2021) to evaluate the model
        # The 'best_model' will be saved based on performance in THIS environment, not the training one.
        VAL_START_DATE = "2023-01-01"
        VAL_END_DATE = "2023-12-31" # 1 year validation
        
        print(f"Creating Validation Environment ({VAL_START_DATE} to {VAL_END_DATE})...")
        eval_env_kwargs = {
            'data_path': DATA_PATH, 
            'vix_data_path': VIX_DATA_PATH,
            'start_date': VAL_START_DATE, 
            'end_date': VAL_END_DATE
        }
        # Eval env doesn't need to be vectorized, but Monitor is crucial for EvalCallback to read stats
        # We use SubprocVecEnv to ensure strict type matching with the training env (VecEnv)
        eval_env = make_vec_env(
            IntradayOptionEnv,
            n_envs=1,
            seed=42,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=eval_env_kwargs,
            monitor_dir=EVAL_MONITOR_DIR
        )

        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

        # Callbacks for Model Management
        # eval_freq: Evaluate every 5000 steps (approx 13 trading days of 375 steps)
        # best_model_save_path: Where to save the model driven by validation performance
        eval_callback = EvalCallback(eval_env, best_model_save_path=SAVE_DIR,
                                     log_path=LOG_DIR, eval_freq=5000,
                                     deterministic=True, render=False)
        
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=SAVE_DIR,
                                                 name_prefix=f'{AGENT.lower()}_intraday_checkpoint')

        print(f"Training {AGENT} Agent...")
        # User-customized training parameters for 1-minute data
        learning_rate = 0.0007  # Default for A2C
        n_steps = 5  # Default for A2C
        gamma = 0.99
        ent_coef = 0.001
        total_timesteps = 2_000_000
        
        if RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH):
            print(f"Loading existing model from: {RETRAIN_MODEL_PATH}")
            model = A2C.load(RETRAIN_MODEL_PATH, env=env, tensorboard_log=LOG_DIR, device='cuda')
        else:
            print(f"Creating a NEW {AGENT} Agent with defined parameters...")
            model = A2C("MlpPolicy", env, seed=42, verbose=1, 
                        learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, 
                        ent_coef=ent_coef, tensorboard_log=LOG_DIR, device='cuda')

        
        # training with callbacks
        reset_num_timesteps = False if (RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH)) else True
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback], progress_bar=True, reset_num_timesteps=reset_num_timesteps)
        print("Training Finished.")

        # Save final model as well
        final_model_path = os.path.join(SAVE_DIR, f'{AGENT.lower()}_intraday_model_final')
        model.save(final_model_path)
        print(f"Final Model saved to {final_model_path}")
        
        # Save Metadata
        metadata = {
            "timestamp": TIMESTAMP,
            "agent": AGENT,
            "training_data": {
                "path": DATA_PATH,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
            "validation_data": {
                "start_date": VAL_START_DATE,
                "end_date": VAL_END_DATE,
            },
            "environment_parameters": env_kwargs,
            "evaluation_environment_parameters": eval_env_kwargs,
            "model_parameters": {
                "policy": "MlpPolicy",
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "gamma": gamma,
                "ent_coef": ent_coef,
                "total_timesteps": total_timesteps
            },
            "reward_settings": {
                "PNL_LAMBDA": config.PNL_LAMBDA,
                "STEP_PNL_LAMBDA": config.STEP_PNL_LAMBDA,
                "DRAWDOWN_LAMBDA": config.DRAWDOWN_LAMBDA,
                "TRADE_PENALTY_LAMBDA": config.TRADE_PENALTY_LAMBDA,
                "TURNOVER_LAMBDA": config.TURNOVER_LAMBDA,
                "FORCED_EXIT_LAMBDA": config.FORCED_EXIT_LAMBDA
            },
            "config_parameters": {
                "INITIAL_CAPITAL": config.INITIAL_CAPITAL,
                "MAX_LOTS": config.MAX_LOTS,
                "TRANSACTION_COST_PCT": config.TRANSACTION_COST_PCT,
                "SLIPPAGE_PCT": config.SLIPPAGE_PCT,
                "START_TIME": config.START_TIME,
                "END_TIME": config.END_TIME,
                "STRADDLE_STRIKE_GAP": config.STRADDLE_STRIKE_GAP,
                "LOT_SIZE": config.LOT_SIZE,
                "STRATEGY_TYPE": config.STRATEGY_TYPE,
                "STRIKE_SELECTION_METHOD": config.STRIKE_SELECTION_METHOD,
                "BOLLINGER_STD": config.BOLLINGER_STD,
                "ATR_MULTIPLIER": config.ATR_MULTIPLIER,
                "EXPIRY_DAY_OF_WEEK": config.EXPIRY_DAY_OF_WEEK,
                "WINDOW_SIZE": config.WINDOW_SIZE,
                "RISK_FREE_RATE": config.RISK_FREE_RATE
            }
        }
        
        metadata_path = os.path.join(SAVE_DIR, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")
        
        # Plotting Results
        print("Generating training plots...")
        plot_results(LOG_DIR, SAVE_DIR)
        
        print("-" * 50)
        print(f"TRAINING COMPLETE. Model ID: {TIMESTAMP}")
        print(f"Model saved at: {SAVE_DIR}")
        print("-" * 50)
        
    except ImportError as e:
        print(f"ImportError details: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
