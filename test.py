import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.intraday_option_env import IntradayOptionEnv
import config
import os
import datetime

def main():
    # Configuration
    AGENT = 'PPO'
    SAVE_DIR = os.path.join('model', AGENT)
    MODEL_PATH = os.path.join(SAVE_DIR, f'{AGENT.lower()}_intraday_model')
    RESULTS_DIR = os.path.join(SAVE_DIR, 'results')
    DATA_PATH = 'data/test.csv'
    MODE = 'add'
    START_DATE = "2025-01-01"  # User can filter range
    END_DATE = "2025-12-31"    # User can filter range

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading Environment with Mode: {MODE}")
    env = IntradayOptionEnv(data_path=DATA_PATH, mode=MODE, start_date=START_DATE, end_date=END_DATE)

    # Check if model exists
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model file {MODEL_PATH}.zip not found. Please train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    daily_results = []
    
    total_dates = len(env.dates)
    print(f"Starting Testing on {total_dates} Days...")

    for i in range(total_dates):
        # Sequential Testing
        obs, _ = env.reset(options={'date_index': i})
        done = False
        total_reward = 0
        steps = 0
        
        # Track episode stats
        initial_capital = env.cash
        prev_action = 0 # Hold
        trade_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Count Adjustments as trades
            if np.sum(action) > 0: # Check if any action taken (Add=1, Offload=2)
                trade_count += 1
            
        # End of Episode Metrics
        current_date_obj = env.dates[env.current_date_idx]
        final_pnl = env.pnl_curve[-1]
        peak_pnl = env.peak_pnl
        max_dd = env.max_drawdown
        
        # Investment (margin used approx or max premium)
        # Dynamic: Max Premium Deployed tracked by Env
        if 'max_investment' in info:
            investment = info['max_investment']
            if investment == 0: investment = config.INITIAL_CAPITAL # Edge case: no trade
        else:
            investment = config.INITIAL_CAPITAL 
        return_pct = (final_pnl / investment) * 100
        
        daily_results.append({
            'episode': i + 1,
            'date': current_date_obj,
            'pnl': round(final_pnl, 2),
            'max_drawdown': round(max_dd, 2),
            'peak_pnl': round(peak_pnl, 2),
            'trades': trade_count,
            'investment': investment,
            'total_reward': round(total_reward, 4),
            'return_pct': round(return_pct, 4)
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{total_dates} days...")

    # --- 1. Daily Results CSV ---
    df_daily = pd.DataFrame(daily_results)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    daily_csv_path = os.path.join(RESULTS_DIR, 'daily_results.csv')
    df_daily.to_csv(daily_csv_path, index=False)
    print(f"Saved Daily Results to {daily_csv_path}")

    # --- 2. Monthly Results CSV ---
    df_daily['month'] = df_daily['date'].dt.to_period('M')
    
    monthly_stats = df_daily.groupby('month').agg({
        'pnl': 'sum',
        'max_drawdown': 'max', # Worst drawdown in the month
        'trades': 'sum',
        'investment': 'first', # Capital base
    }).reset_index()
    
    monthly_stats['avg_pnl'] = df_daily.groupby('month')['pnl'].mean().values
    monthly_stats['roi_pct'] = (monthly_stats['pnl'] / monthly_stats['investment']) * 100
    
    # Format
    monthly_stats['pnl'] = monthly_stats['pnl'].round(2)
    monthly_stats['avg_pnl'] = monthly_stats['avg_pnl'].round(2)
    monthly_stats['roi_pct'] = monthly_stats['roi_pct'].round(2)
    
    monthly_csv_path = os.path.join(RESULTS_DIR, 'monthly_results.csv')
    monthly_stats.to_csv(monthly_csv_path, index=False)
    print(f"Saved Monthly Results to {monthly_csv_path}")

    # --- 3. Overall Results CSV ---
    total_pnl = df_daily['pnl'].sum()
    overall_roi = (total_pnl / config.INITIAL_CAPITAL) * 100
    avg_daily_pnl = df_daily['pnl'].mean()
    worst_drawdown = df_daily['max_drawdown'].max()
    
    wins = df_daily[df_daily['pnl'] > 0]
    win_rate = (len(wins) / len(df_daily)) * 100
    
    avg_trades = df_daily['trades'].mean()
    total_trades = df_daily['trades'].sum()
    
    # Sharpe Ratio (Daily Returns)
    daily_returns = df_daily['pnl'] / config.INITIAL_CAPITAL
    if daily_returns.std() != 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    overall_res = [{
        'total_pnl': round(total_pnl, 2),
        'overall_roi_pct': round(overall_roi, 2),
        'avg_daily_pnl': round(avg_daily_pnl, 2),
        'max_drawdown': round(worst_drawdown, 2),
        'win_rate_pct': round(win_rate, 2),
        'avg_trades_per_day': round(avg_trades, 1),
        'total_trades': total_trades,
        'sharpe_ratio': round(sharpe, 4)
    }]
    
    df_overall = pd.DataFrame(overall_res)
    overall_csv_path = os.path.join(RESULTS_DIR, 'overall_results.csv')
    df_overall.to_csv(overall_csv_path, index=False)
    print(f"Saved Overall Results to {overall_csv_path}")
    
    print("\nTest Complete.")
    print(df_overall.to_string(index=False))

if __name__ == "__main__": 
    main()
