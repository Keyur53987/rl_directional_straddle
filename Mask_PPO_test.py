"""
MaskablePPO Testing Script for Directional Straddle v2 - Multithreaded
======================================================================
Tests a trained MaskablePPO model on held-out data using multiprocessing.
Generates daily, monthly, and overall results CSVs.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import os
import datetime
import random
import torch
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.intraday_option_env_maskPPO import IntradayOptionEnvV2
import Mask_PPO_config as config

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

# ── Worker Globals and Init ──
w_model = None
w_env = None
w_raw_env = None

def init_worker(model_path, data_path, vix_data_path, start_date, end_date):
    global w_model, w_env, w_raw_env
    # Load on CPU for multiprocessing
    w_model = MaskablePPO.load(model_path, device='cpu')
    w_raw_env = IntradayOptionEnvV2(
        data_path=data_path,
        vix_data_path=vix_data_path,
        start_date=start_date,
        end_date=end_date,
    )
    w_env = ActionMasker(w_raw_env, mask_fn)

def worker_process_episode(ep_idx):
    global w_model, w_env, w_raw_env
    
    obs, _ = w_env.reset(options={'date_index': ep_idx})
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action_masks = w_env.action_masks() if hasattr(w_env, 'action_masks') else w_raw_env.action_masks()
        action, _states = w_model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, truncated, info = w_env.step(action)
        total_reward += reward
        steps += 1

    # Episode metrics
    trade_count = len(w_raw_env.trade_logs)
    current_date_obj = w_raw_env.dates[w_raw_env.current_date_idx]
    final_pnl = w_raw_env.pnl_curve[-1]
    peak_pnl = w_raw_env.peak_pnl
    max_dd = w_raw_env.max_drawdown

    investment = w_raw_env.premium_deployed if w_raw_env.premium_deployed > 0 else config.INITIAL_CAPITAL
    return_pct = (final_pnl / investment) * 100

    # Position side used (from trade logs)
    sides_used = set(t.get('side', '') for t in w_raw_env.trade_logs if t.get('side'))
    position_type = '/'.join(sides_used) if sides_used else 'FLAT'

    daily_result = {
        'episode': ep_idx + 1,
        'date': current_date_obj,
        'pnl': round(final_pnl, 2),
        'max_drawdown': round(max_dd, 2),
        'peak_pnl': round(peak_pnl, 2),
        'trades': trade_count,
        'investment': round(investment, 2),
        'total_reward': round(total_reward, 4),
        'return_pct': round(return_pct, 4),
        'position_type': position_type,
        'final_margin': round(info.get('current_margin', 0), 2),
    }

    return daily_result

def main():
    # Configuration
    AGENT = 'MaskablePPO'
    DATA_PATH = 'data/NIFTY50_2025.csv'
    VIX_DATA_PATH = 'data/INDIA_VIX.csv'
    START_DATE = "2025-01-01"
    END_DATE = "2025-12-31"

    # Set to a specific model ID or None for latest
    MODEL_ID = "20260417_142918"

    # Seeds
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Model Selection ──
    BASE_MODEL_DIR = os.path.join('model', AGENT)

    if MODEL_ID is None:
        if not os.path.exists(BASE_MODEL_DIR):
            print(f"No models found in {BASE_MODEL_DIR}")
            return
        subdirs = [d for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
        if not subdirs:
            print(f"No model folders found in {BASE_MODEL_DIR}")
            return
        subdirs.sort()
        MODEL_ID = subdirs[-1]
        print(f"Auto-selected Latest Model ID: {MODEL_ID}")
    else:
        print(f"Using Specified Model ID: {MODEL_ID}")

    MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_ID)

    # Generate results dir name from data file
    data_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    RESULTS_DIR = os.path.join(MODEL_DIR, f'results_{data_name}_{START_DATE[:4]}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find model file
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model')
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f'{AGENT.lower()}_final')

    if os.path.exists(BEST_MODEL_PATH + ".zip"): LOAD_PATH = BEST_MODEL_PATH + ".zip"
    elif os.path.exists(FINAL_MODEL_PATH + ".zip"): LOAD_PATH = FINAL_MODEL_PATH + ".zip"
    else:
        print(f"No model file found in {MODEL_DIR}.")
        return

    # We need a quick dummy env to get dates for multiprocessing
    dummy_env = IntradayOptionEnvV2(data_path=DATA_PATH, vix_data_path=VIX_DATA_PATH, start_date=START_DATE, end_date=END_DATE)
    total_dates = len(dummy_env.dates)
    
    print(f"Starting Multiprocessed Testing on {total_dates} Episodes ({START_DATE} -> {END_DATE})...")
    print(f"Position Mode: {config.POSITION_MODE}")
    
    daily_results = []
    
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Launching {num_workers} parallel workers...")

    # Strip .zip from LOAD_PATH for the load function
    load_path_no_ext = LOAD_PATH[:-4]

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(load_path_no_ext, DATA_PATH, VIX_DATA_PATH, START_DATE, END_DATE)
    ) as pool:
        # Use imap to get progress bar
        for res in tqdm(pool.imap(worker_process_episode, range(total_dates)), total=total_dates, desc="Testing episodes"):
            daily_results.append(res)
            
    # Sort results by episode to ensure chronological order despite multiprocessing completion times
    daily_results.sort(key=lambda x: x['episode'])

    # ── Results ──

    # 1. Daily
    df_daily = pd.DataFrame(daily_results)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    daily_csv_path = os.path.join(RESULTS_DIR, 'daily_results.csv')
    df_daily.to_csv(daily_csv_path, index=False)
    print(f"\nSaved Daily Results to {daily_csv_path}")

    # 2. Monthly
    df_daily['month'] = df_daily['date'].dt.to_period('M')
    monthly_stats = df_daily.groupby('month').agg({
        'pnl': 'sum',
        'max_drawdown': 'max',
        'trades': 'sum',
        'investment': 'first',
    }).reset_index()
    monthly_stats['avg_pnl'] = df_daily.groupby('month')['pnl'].mean().values
    monthly_stats['roi_pct'] = (monthly_stats['pnl'] / monthly_stats['investment']) * 100
    monthly_stats['pnl'] = monthly_stats['pnl'].round(2)
    monthly_stats['avg_pnl'] = monthly_stats['avg_pnl'].round(2)
    monthly_stats['roi_pct'] = monthly_stats['roi_pct'].round(2)
    monthly_csv_path = os.path.join(RESULTS_DIR, 'monthly_results.csv')
    monthly_stats.to_csv(monthly_csv_path, index=False)
    print(f"Saved Monthly Results to {monthly_csv_path}")

    # 3. Overall
    total_pnl = df_daily['pnl'].sum()
    overall_roi = (total_pnl / config.INITIAL_CAPITAL) * 100
    avg_daily_pnl = df_daily['pnl'].mean()
    worst_drawdown = df_daily['max_drawdown'].max()

    wins = df_daily[df_daily['pnl'] > 0]
    win_rate = (len(wins) / len(df_daily)) * 100 if len(df_daily) > 0 else 0

    avg_trades = df_daily['trades'].mean()
    total_trades = df_daily['trades'].sum()
    flat_episodes = len(df_daily[df_daily['trades'] == 0])
    flat_pct = (flat_episodes / len(df_daily)) * 100 if len(df_daily) > 0 else 0

    daily_returns = df_daily['pnl'] / config.INITIAL_CAPITAL
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(52)

    downside_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / (downside_returns.std() + 1e-8)) * np.sqrt(52) if len(downside_returns) > 0 else 0.0

    gross_profit = df_daily.loc[df_daily['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(df_daily.loc[df_daily['pnl'] < 0, 'pnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    losses = df_daily[df_daily['pnl'] <= 0]
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
    win_prob = len(wins) / len(df_daily) if len(df_daily) > 0 else 0.0
    loss_prob = len(losses) / len(df_daily) if len(df_daily) > 0 else 0.0
    expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)

    overall_res = [{
        'total_pnl': round(total_pnl, 2),
        'overall_roi_pct': round(overall_roi, 2),
        'avg_weekly_pnl': round(avg_daily_pnl, 2),
        'max_drawdown': round(worst_drawdown, 2),
        'win_rate_pct': round(win_rate, 2),
        'avg_trades_per_episode': round(avg_trades, 1),
        'total_trades': total_trades,
        'flat_episodes_pct': round(flat_pct, 1),
        'sharpe_ratio': round(sharpe, 4),
        'sortino_ratio': round(sortino, 4),
        'profit_factor': round(profit_factor, 4),
        'expectancy': round(expectancy, 2),
        'position_mode': config.POSITION_MODE,
    }]

    df_overall = pd.DataFrame(overall_res)
    overall_csv_path = os.path.join(RESULTS_DIR, 'overall_results.csv')
    df_overall.to_csv(overall_csv_path, index=False)
    print(f"Saved Overall Results to {overall_csv_path}")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")
    print(df_overall.to_string(index=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
