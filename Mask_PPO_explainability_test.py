"""
MaskablePPO Explainability Test Script - Multithreaded
======================================================
Runs model inference and captures rich step-level + trade-level data
for comprehensive explainability analysis. Uses multiprocessing for speed.

Outputs:
  - step_level_data.csv    : Every minute-bar decision with market context
  - trade_level_data.csv   : Every entry/exit with regime + timing info
  - episode_summary.csv    : Enriched episode-level summary
  - shap_importance.csv    : SHAP-based feature importance
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import os
import datetime
import random
import torch
import json
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.intraday_option_env_maskPPO_v3 import IntradayOptionEnvV3
import Mask_PPO_config as config

# ── Constants ──
ACTION_NAMES = {
    0: 'HOLD', 1: 'ENTER_LONG', 2: 'ENTER_SHORT', 3: 'EXIT_ALL',
    4: 'ADD_CE', 5: 'ADD_PE', 6: 'REDUCE_CE', 7: 'REDUCE_PE'
}

FEATURE_NAMES = [
    'vol_5min', 'vol_15min', 'vol_30min', 'vol_60min', 'vol_1day', 'vol_7day', 'vol_15day',
    'parkinson_vol', 'garman_klass_vol', 'vol_percentile', 'vol_of_vol', 'vol_trend', 'vrp', 'current_iv', 'vol_pad',
    'close_normalized', 'return_1min', 'return_5min', 'return_15min', 'return_30min', 'vwap_distance', 'dist_day_high', 'dist_day_low', 'price_momentum', 'price_pad',
    'ce_delta', 'pe_delta', 'ce_gamma', 'pe_gamma', 'ce_vega', 'pe_vega', 'ce_theta', 'pe_theta', 'ce_vanna', 'pe_vanna', 'ce_volga', 'pe_volga', 'net_delta', 'gamma_exposure', 'vega_exposure',
    'rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'bollinger_position', 'atr',
    'ce_lots_norm', 'pe_lots_norm', 'ce_pe_ratio', 'premium_deployed_norm', 'ce_strike_dist', 'pe_strike_dist', 'entry_ce_ratio', 'entry_pe_ratio', 'pnl_norm', 'pnl_premium_ratio',
    'minutes_since_open', 'minutes_to_close', 'time_sine', 'time_cosine',
    'current_drawdown', 'max_drawdown_norm', 'rolling_sharpe',
    'time_to_expiry_norm', 'is_overnight',
    'position_side_enc', 'position_duration',
    'ce_unrealized_pnl', 'pe_unrealized_pnl',  # v3 features
]

MARKET_EVENTS = {
    '2020-03-23': 'COVID Crash Low',
    '2020-03-24': 'COVID Recovery Begin',
    '2021-02-01': 'Union Budget 2021',
    '2021-05-10': 'COVID 2nd Wave Selloff',
    '2022-02-01': 'Union Budget 2022',
    '2022-02-24': 'Russia-Ukraine War',
    '2022-06-16': 'Fed 75bps Hike',
    '2023-02-01': 'Union Budget 2023',
    '2023-03-13': 'SVB Banking Crisis',
    '2024-02-01': 'Interim Budget 2024',
    '2024-06-04': 'Election Results Crash',
    '2024-08-05': 'Japan Carry Trade Unwind',
    '2025-02-01': 'Union Budget 2025',
    '2025-04-02': 'Trump Tariff Crash',
}

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

def classify_market_regime(day_return_pct, vix_value):
    if day_return_pct > 0.5: trend = 'BULLISH'
    elif day_return_pct < -0.5: trend = 'BEARISH'
    else: trend = 'SIDEWAYS'
    
    if vix_value is not None and vix_value > 0:
        if vix_value > 20: vol_regime = 'HIGH_VOL'
        elif vix_value > 14: vol_regime = 'MED_VOL'
        else: vol_regime = 'LOW_VOL'
    else: vol_regime = 'UNKNOWN'
    return trend, vol_regime

# ── Worker Globals and Init ──
w_model = None
w_env = None
w_raw_env = None

def init_worker(model_path, data_path, vix_data_path, start_date, end_date):
    global w_model, w_env, w_raw_env
    # Load on CPU for multiprocessing
    w_model = MaskablePPO.load(model_path, device='cpu')
    w_raw_env = IntradayOptionEnvV3(
        data_path=data_path,
        vix_data_path=vix_data_path,
        start_date=start_date,
        end_date=end_date,
    )
    w_env = ActionMasker(w_raw_env, mask_fn)

def worker_process_episode(args):
    ep_idx, data_label = args
    global w_model, w_env, w_raw_env
    
    step_records = []
    trade_records = []
    
    obs, _ = w_env.reset(options={'date_index': ep_idx})
    done = False
    total_reward = 0
    step_num = 0
    episode_date = w_raw_env.dates[w_raw_env.current_date_idx]

    prev_trade_count = 0
    entry_context = {}  
    trade_id_counter = 0

    while not done:
        current_row = w_raw_env.day_data.iloc[w_raw_env.current_step]
        spot = current_row['close']
        timestamp = current_row['datetime']
        vix_val = current_row.get('vix', None)
        if pd.isna(vix_val): vix_val = None

        day_return_pct = ((spot - w_raw_env.day_open) / w_raw_env.day_open) * 100 if w_raw_env.day_open > 0 else 0
        action_mask = w_raw_env.action_masks()

        action, _ = w_model.predict(obs, deterministic=True, action_masks=action_mask)
        action = int(action)

        obs_next, reward, done, truncated, info = w_env.step(action)
        total_reward += reward
        step_num += 1

        trend, vol_regime = classify_market_regime(day_return_pct, vix_val if vix_val else 0)

        step_records.append({
            'dataset': data_label, 'episode': ep_idx + 1, 'date': str(episode_date),
            'step': step_num, 'timestamp': str(timestamp), 'spot_price': round(spot, 2),
            'day_return_pct': round(day_return_pct, 4), 'iv': round(obs[13], 4) if len(obs) > 13 else 0,
            'vix': round(vix_val, 2) if vix_val else None, 'rsi': round(obs[40], 4) if len(obs) > 40 else 0,
            'macd_histogram': round(obs[43], 6) if len(obs) > 43 else 0, 'bollinger_position': round(obs[44], 4) if len(obs) > 44 else 0,
            'vol_percentile': round(obs[9], 4) if len(obs) > 9 else 0, 'action': action,
            'action_name': ACTION_NAMES.get(action, 'UNKNOWN'),
            'valid_actions': str([i for i, m in enumerate(action_mask) if m]),
            'position_side': w_raw_env.position_side or 'FLAT', 'ce_lots': w_raw_env.ce_lots, 'pe_lots': w_raw_env.pe_lots,
            'step_pnl': round(info.get('step_pnl', 0), 2), 'total_pnl': round(info.get('total_pnl', 0), 2),
            'reward': round(reward, 6), 'trend_regime': trend, 'vol_regime': vol_regime,
            'hour': timestamp.hour if hasattr(timestamp, 'hour') else pd.to_datetime(timestamp).hour,
            'minute': timestamp.minute if hasattr(timestamp, 'minute') else pd.to_datetime(timestamp).minute,
            'day_of_week': timestamp.weekday() if hasattr(timestamp, 'weekday') else pd.to_datetime(timestamp).weekday(),
            'max_drawdown': round(info.get('max_drawdown', 0), 2),
        })

        new_trades = w_raw_env.trade_logs[prev_trade_count:]
        if new_trades:
            for trade in new_trades:
                trade_id_counter += 1
                trade_action = trade.get('action', '')
                is_entry = trade_action in ('BUY', 'SELL') and action in (1, 2, 4, 5)
                is_exit = trade_action in ('SELL', 'BUY_BACK') and action in (3, 6, 7)

                if is_entry:
                    entry_key = f"{trade.get('leg')}_{trade.get('strike')}"
                    entry_context[entry_key] = {
                        'entry_time': str(timestamp), 'entry_spot': spot,
                        'entry_iv': obs[13] if len(obs) > 13 else 0, 'entry_vix': vix_val,
                        'entry_rsi': obs[40] if len(obs) > 40 else 0,
                        'entry_vol_regime': vol_regime, 'entry_trend': trend, 'entry_step': step_num,
                    }

                trade_record = {
                    'dataset': data_label, 'episode': ep_idx + 1, 'date': str(episode_date),
                    'trade_id': trade_id_counter, 'env_action': action, 'env_action_name': ACTION_NAMES.get(action, 'UNKNOWN'),
                    'trade_action': trade_action, 'side': trade.get('side', ''), 'leg': trade.get('leg', ''),
                    'strike': trade.get('strike', 0), 'price': round(trade.get('price', 0), 2),
                    'qty': trade.get('qty', 0), 'pnl': round(trade.get('pnl', 0), 2),
                    'timestamp': str(trade.get('timestamp', '')), 'spot_at_trade': round(spot, 2),
                    'iv_at_trade': round(obs[13], 4) if len(obs) > 13 else 0, 'vix_at_trade': round(vix_val, 2) if vix_val else None,
                    'rsi_at_trade': round(obs[40], 4) if len(obs) > 40 else 0, 'trend_regime': trend, 'vol_regime': vol_regime,
                    'day_return_pct': round(day_return_pct, 4), 
                    'hour': timestamp.hour if hasattr(timestamp, 'hour') else pd.to_datetime(timestamp).hour,
                    'minute': timestamp.minute if hasattr(timestamp, 'minute') else pd.to_datetime(timestamp).minute,
                    'day_of_week': timestamp.weekday() if hasattr(timestamp, 'weekday') else pd.to_datetime(timestamp).weekday(),
                    'is_entry': is_entry, 'is_exit': is_exit,
                }

                if is_exit:
                    entry_key = f"{trade.get('leg')}_{trade.get('strike')}"
                    ctx = entry_context.get(entry_key, {})
                    if ctx:
                        trade_record.update({
                            'entry_time': ctx.get('entry_time', ''), 'entry_spot': ctx.get('entry_spot', 0),
                            'entry_iv': round(ctx.get('entry_iv', 0), 4), 'entry_vix': ctx.get('entry_vix'),
                            'holding_steps': step_num - ctx.get('entry_step', 0),
                            'spot_move_pct': round(((spot - ctx['entry_spot']) / ctx['entry_spot']) * 100, 4) if ctx.get('entry_spot', 0) > 0 else 0,
                            'iv_change': round((obs[13] if len(obs) > 13 else 0) - ctx.get('entry_iv', 0), 4)
                        })

                trade_records.append(trade_record)
            prev_trade_count = len(w_raw_env.trade_logs)

        obs = obs_next

    # Episode summary logic
    final_pnl = w_raw_env.pnl_curve[-1] if w_raw_env.pnl_curve else 0
    trade_count = len(w_raw_env.trade_logs)
    investment = w_raw_env.premium_deployed if w_raw_env.premium_deployed > 0 else config.INITIAL_CAPITAL
    return_pct = (final_pnl / investment) * 100

    sides_used = set(t.get('side', '') for t in w_raw_env.trade_logs if t.get('side'))
    position_type = '/'.join(sides_used) if sides_used else 'FLAT'

    first_row = w_raw_env.day_data.iloc[0]
    episode_vix = first_row.get('vix', None) if not pd.isna(first_row.get('vix', None)) else None
    episode_spot_open = w_raw_env.day_open
    episode_spot_close = w_raw_env.day_data.iloc[-1]['close'] if len(w_raw_env.day_data) > 0 else episode_spot_open
    episode_return = ((episode_spot_close - episode_spot_open) / episode_spot_open) * 100 if episode_spot_open > 0 else 0

    ep_trend, ep_vol_regime = classify_market_regime(episode_return, episode_vix if episode_vix else 0)
    market_event = MARKET_EVENTS.get(str(episode_date), '')

    action_counts = {}
    for s in step_records:
        a = s['action_name']
        action_counts[a] = action_counts.get(a, 0) + 1

    episode_record = {
        'dataset': data_label, 'episode': ep_idx + 1, 'date': str(episode_date),
        'day_of_week': episode_date.weekday(), 'day_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][episode_date.weekday()],
        'month': episode_date.month, 'year': episode_date.year, 'pnl': round(final_pnl, 2),
        'return_pct': round(return_pct, 4), 'max_drawdown': round(w_raw_env.max_drawdown, 2),
        'peak_pnl': round(w_raw_env.peak_pnl, 2), 'trades': trade_count, 'investment': round(investment, 2),
        'total_reward': round(total_reward, 4), 'position_type': position_type,
        'final_margin': round(info.get('current_margin', 0), 2) if info else 0,
        'spot_open': round(episode_spot_open, 2), 'spot_close': round(episode_spot_close, 2),
        'spot_return_pct': round(episode_return, 4), 'vix': round(episode_vix, 2) if episode_vix else None,
        'trend_regime': ep_trend, 'vol_regime': ep_vol_regime, 'market_event': market_event,
        'total_steps': step_num, 'hold_pct': round(action_counts.get('HOLD', 0) / max(step_num, 1) * 100, 2),
        'enter_long_count': action_counts.get('ENTER_LONG', 0), 'enter_short_count': action_counts.get('ENTER_SHORT', 0),
        'exit_count': action_counts.get('EXIT_ALL', 0), 'add_ce_count': action_counts.get('ADD_CE', 0),
        'add_pe_count': action_counts.get('ADD_PE', 0), 'reduce_ce_count': action_counts.get('REDUCE_CE', 0),
        'reduce_pe_count': action_counts.get('REDUCE_PE', 0),
    }

    return step_records, trade_records, episode_record

def compute_shap_importance(model, env, raw_env, num_samples=200):
    print("  Computing SHAP feature importance (perturbation-based)...")
    obs_dim = model.observation_space.shape[0]
    # Use feature names up to obs_dim (handles both 67 and 69)
    feature_names = FEATURE_NAMES[:obs_dim]
    importances = np.zeros(obs_dim)
    obs_samples = []

    total_dates = len(raw_env.dates)
    sample_indices = np.random.choice(total_dates, min(num_samples // 50, total_dates), replace=False)

    for date_idx in sample_indices:
        obs, _ = env.reset(options={'date_index': int(date_idx)})
        done = False
        step_count = 0
        while not done and step_count < 50:
            obs_samples.append(obs.copy())
            action_masks = raw_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _, done, _, _ = env.step(action)
            step_count += 1

    obs_array = np.array(obs_samples[:num_samples])
    print(f"    Collected {len(obs_array)} observation samples (obs_dim={obs_dim})")

    if len(obs_array) == 0:
        return pd.DataFrame({'feature': feature_names, 'importance': np.zeros(obs_dim)})

    policy = model.policy

    for feat_idx in range(obs_dim):
        changes = []
        for obs in obs_array[:min(100, len(obs_array))]:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                orig_dist = policy.get_distribution(obs_tensor)
                orig_probs = orig_dist.distribution.probs.cpu().numpy().flatten()
            
            perturbed = obs.copy()
            perturbed[feat_idx] = 0.0
            pert_tensor = torch.FloatTensor(perturbed).unsqueeze(0).to(model.device)
            with torch.no_grad():
                pert_dist = policy.get_distribution(pert_tensor)
                pert_probs = pert_dist.distribution.probs.cpu().numpy().flatten()
            
            kl = np.sum(orig_probs * np.log((orig_probs + 1e-10) / (pert_probs + 1e-10)))
            changes.append(abs(kl))
            
        importances[feat_idx] = np.mean(changes) if changes else 0.0

    total = importances.sum()
    if total > 0: importances = importances / total

    df_shap = pd.DataFrame({
        'feature': feature_names, 'importance': importances,
        'rank': np.argsort(-importances) + 1
    }).sort_values('importance', ascending=False)
    return df_shap

def compute_shap_importance_gradient(model, env, raw_env, num_samples=200):
    print("  Computing SHAP feature importance (gradient-based)...")
    obs_dim = model.observation_space.shape[0]
    feature_names = FEATURE_NAMES[:obs_dim]
    importances = np.zeros(obs_dim)
    obs_samples = []

    total_dates = len(raw_env.dates)
    sample_indices = np.random.choice(total_dates, min(num_samples // 50, total_dates), replace=False)

    for date_idx in sample_indices:
        obs, _ = env.reset(options={'date_index': int(date_idx)})
        done = False
        step_count = 0
        while not done and step_count < 50:
            obs_samples.append(obs.copy())
            action_masks = raw_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _, done, _, _ = env.step(action)
            step_count += 1

    obs_array = np.array(obs_samples[:num_samples])
    print(f"    Collected {len(obs_array)} observation samples (obs_dim={obs_dim})")

    if len(obs_array) == 0:
        return pd.DataFrame({'feature': feature_names, 'importance': np.zeros(obs_dim)})

    policy = model.policy

    grads_list = []
    for obs in obs_array[:min(100, len(obs_array))]:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
        obs_tensor.requires_grad_(True)
        
        dist = policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs
        max_prob = probs.max()
        
        if obs_tensor.grad is not None:
            obs_tensor.grad.zero_()
            
        max_prob.backward()
        
        grad = obs_tensor.grad.cpu().numpy().flatten()
        # Scale gradient by feature value (Gradient * Input) as a proxy for attribution
        attribution = np.abs(grad * obs)
        grads_list.append(attribution)
        
    importances = np.mean(grads_list, axis=0)

    total = importances.sum()
    if total > 0: importances = importances / total

    df_shap = pd.DataFrame({
        'feature': feature_names, 'importance': importances,
        'rank': np.argsort(-importances) + 1
    }).sort_values('importance', ascending=False)
    return df_shap


def main():
    AGENT = 'MaskablePPO'
    VIX_DATA_PATH = 'data/INDIA_VIX.csv'
    MODEL_ID = "20260424_142141" 
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    BASE_MODEL_DIR = os.path.join('model', AGENT)
    MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_ID)
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model')
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f'{AGENT.lower()}_final')

    if os.path.exists(BEST_MODEL_PATH + ".zip"): LOAD_PATH = BEST_MODEL_PATH
    elif os.path.exists(FINAL_MODEL_PATH + ".zip"): LOAD_PATH = FINAL_MODEL_PATH
    else:
        print(f"No model found in {MODEL_DIR}.")
        return

    datasets = [
        {'data_path': 'data/NIFTY50_2025.csv', 'label': 'test', 'start_date': '2025-01-01', 'end_date': '2025-12-31'}
        # {'data_path': 'data/train.csv', 'label': 'train', 'start_date': '2020-01-01', 'end_date': '2023-12-31'},
    ]

    for ds in datasets:
        print(f"\n{'='*60}\n  Explainability Test: {ds['label']}\n  Data: {ds['start_date']} -> {ds['end_date']}\n{'='*60}")
        results_dir = os.path.join(MODEL_DIR, f"explainability_{ds['label']}")
        os.makedirs(results_dir, exist_ok=True)
        
        # We need a quick read to know total dates
        dummy_env = IntradayOptionEnvV3(data_path=ds['data_path'], start_date=ds['start_date'], end_date=ds['end_date'])
        total_dates = len(dummy_env.dates)
        print(f"  Total epochs to process: {total_dates}")
        
        step_records, trade_records, episode_records = [], [], []
        
        num_workers = max(1, mp.cpu_count() - 1)
        print(f"  Launching {num_workers} parallel workers...")
        
        # Parallel Execution
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(LOAD_PATH, ds['data_path'], VIX_DATA_PATH, ds['start_date'], ds['end_date'])
        ) as pool:
            
            args = [(i, ds['label']) for i in range(total_dates)]
            
            # Use imap to get progress bar
            for s_res, t_res, e_res in tqdm(pool.imap(worker_process_episode, args), total=total_dates, desc="Processing episodes"):
                step_records.extend(s_res)
                trade_records.extend(t_res)
                episode_records.append(e_res)
        
        # Save results
        df_steps = pd.DataFrame(step_records)
        df_trades = pd.DataFrame(trade_records)
        df_episodes = pd.DataFrame(episode_records)

        step_path = os.path.join(results_dir, 'step_level_data.csv')
        trade_path = os.path.join(results_dir, 'trade_level_data.csv')
        episode_path = os.path.join(results_dir, 'episode_summary.csv')

        df_steps.to_csv(step_path, index=False)
        df_trades.to_csv(trade_path, index=False)
        df_episodes.to_csv(episode_path, index=False)

        print(f"  OK Saved step features: {len(df_steps)} rows")
        print(f"  OK Saved trade actions: {len(df_trades)} rows")
        print(f"  OK Saved episode summaries: {len(df_episodes)} rows")
        
        compute_shap = (ds['label'] == 'test')
        if compute_shap:
            # We do SHAP purely on Main process
            print("\n  Loading local env in main process for SHAP...")
            model = MaskablePPO.load(LOAD_PATH, device='cpu')
            env = ActionMasker(dummy_env, mask_fn)
            df_shap = compute_shap_importance(model, env, dummy_env, num_samples=100000)
            df_shap.to_csv(os.path.join(results_dir, 'shap_importance.csv'), index=False)
            print(f"  OK SHAP importance saved")
            
            df_shap_grad = compute_shap_importance_gradient(model, env, dummy_env, num_samples=100000)
            df_shap_grad.to_csv(os.path.join(results_dir, 'shap_importance_gradient.csv'), index=False)
            print(f"  OK SHAP importance (gradient) saved")

    print(f"\n{'='*60}\n  ALL EXPLAINABILITY TESTS COMPLETE\n{'='*60}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
