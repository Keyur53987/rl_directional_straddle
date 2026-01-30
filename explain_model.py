import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from envs.intraday_option_env import IntradayOptionEnv
import config
import os
import torch

def get_feature_names():
    """Returns the list of 55 feature names corresponding to the observation space."""
    return [
        "vol_5min", "vol_15min", "vol_30min", "vol_60min", 
        "parkinson_vol", "garman_klass_vol", "vol_percentile", "vol_of_vol", "vol_trend",
        "iv_spread", "reserved_1", "reserved_2",
        "close_norm", "return_1min", "return_5min", "return_15min", "return_30min",
        "vwap_dist", "dist_day_high", "dist_day_low", "momentum", "reserved_3",
        "ce_delta", "pe_delta", "ce_gamma", "ce_vega", "ce_theta", "ce_vanna", "ce_volga",
        "net_delta", "gamma_exposure", "vega_exposure",
        "rsi", "macd_line", "macd_signal", "macd_hist", "bollinger_pos", "atr",
        "ce_lots", "pe_lots", "ce_pe_ratio",
        "premium_deployed",
        "ce_strike_dist", "pe_strike_dist",
        "ce_entry_price", "pe_entry_price",
        "current_pnl", "pnl_premium_ratio",
        "min_since_open", "min_to_close", "time_sine", "time_cosine",
        "peak_drawdown", "max_drawdown",
        "sharpe_ratio"
    ]

def main():
    # --- Configuration ---
    AGENT = 'PPO'
    DATA_PATH = 'data/test.csv'
    N_SAMPLES = 2000 # Number of observations to collect
    
    # Auto-detect latest model
    BASE_MODEL_DIR = os.path.join('model', AGENT)
    if not os.path.exists(BASE_MODEL_DIR):
        print("No models found.")
        return
        
    subdirs = [d for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
    subdirs.sort()
    MODEL_ID = subdirs[-1]
    MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_ID)
    
    # Load Model
    model_path = os.path.join(MODEL_DIR, f'{AGENT.lower()}_intraday_model_final')
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(MODEL_DIR, 'best_model')
        
    print(f"Loading Model: {model_path}")
    model = PPO.load(model_path)
    
    # Create Env
    env = IntradayOptionEnv(data_path=DATA_PATH, force_entry_prob=0.0)
    
    print(f"Collecting {N_SAMPLES} observations from environment...")
    obs, _ = env.reset()
    observations = []
    
    for _ in range(N_SAMPLES):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs.copy())
        
        if done:
            obs, _ = env.reset()
            
    observations = np.array(observations) # Shape: (N, 55)
    
    print("Running Permutation Importance on Value Function...")
    # Convert to torch tensor for the policy network
    obs_tensor = torch.as_tensor(observations).to(model.device)
    
    # 1. Get Baseline Values (Original predictions)
    with torch.no_grad():
        # Access the Value Network (Critic)
        # PPO policy object has .predict_values()
        baseline_values = model.policy.predict_values(obs_tensor).flatten().cpu().numpy()
        
    feature_names = get_feature_names()
    importances = {}
    
    # 2. Permutation Loop
    for i, feature_name in enumerate(feature_names):
        # Create a copy with shuffled column
        shuffled_obs = observations.copy()
        np.random.shuffle(shuffled_obs[:, i])
        
        shuffled_tensor = torch.as_tensor(shuffled_obs).to(model.device)
        
        with torch.no_grad():
            shuffled_values = model.policy.predict_values(shuffled_tensor).flatten().cpu().numpy()
            
        # Calculate MAE (Mean Absolute Error) impact
        mae = np.mean(np.abs(baseline_values - shuffled_values))
        importances[feature_name] = mae
        
    # 3. Visualization
    df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Important Features:")
    print(df_imp.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp.head(20), palette='viridis')
    plt.title(f'Feature Importance (Permutation on Value Function) - {MODEL_ID}')
    plt.xlabel('Mean Absolute Error Impact')
    plt.tight_layout()
    
    save_path = os.path.join(MODEL_DIR, 'feature_importance.png')
    plt.savefig(save_path)
    print(f"\nFeature Importance Chart saved to: {save_path}")

if __name__ == "__main__":
    main()
