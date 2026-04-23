"""
Walk-Forward Validation for RL Options Trader
==============================================
Method: Anchored Expanding Window
  Fold 1: Train 2015-2018 → Test 2019
  Fold 2: Train 2015-2019 → Test 2020
  ...
  Fold 5: Train 2015-2022 → Test 2023

Each fold trains a fresh model for N_TRAIN_STEPS then evaluates on
the held-out test year, logging Sharpe, max drawdown, win rate, etc.
Results saved to model/walk_forward/results.csv and results.json.
"""

import os
import json
import numpy as np
import pandas as pd
import datetime

import config
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from envs.intraday_option_env import IntradayOptionEnv

# MaskablePPO support (v2 env)
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from envs.intraday_option_env_v2 import IntradayOptionEnvV2
    import Mask_PPO_config as mask_config
    HAS_MASKABLE = True
except ImportError:
    HAS_MASKABLE = False

# ──────────────────────────────────────────────
# CONFIGURATION   ← Edit here before running
# ──────────────────────────────────────────────
DATA_PATH      = 'data/train.csv'
VIX_DATA_PATH  = 'data/INDIA_VIX.csv'
AGENT          = 'PPO'          # 'PPO', 'A2C', or 'MaskablePPO'
N_TRAIN_STEPS  = 500_000        # per fold (set 100_000 for a quick sanity check)
N_EVAL_EPISODES = 50            # evaluation episodes per fold
NUM_ENVS       = 4              # parallel training envs
SAVE_DIR       = 'model/walk_forward'

# Anchored expanding folds: (train_start, train_end, test_start, test_end)
FOLDS = [
    ('2015-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
    ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2015-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2015-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2015-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
]
# ──────────────────────────────────────────────


if AGENT == 'MaskablePPO':
    assert HAS_MASKABLE, 'sb3-contrib not installed. Run: pip install sb3-contrib'
    AgentCls = MaskablePPO
    def mask_fn(env):
        return env.action_masks()
elif AGENT == 'PPO':
    AgentCls = PPO
else:
    AgentCls = A2C


def _get_reward_metadata():
    if AGENT == 'MaskablePPO':
        return {
            "STEP_PNL_LAMBDA": mask_config.STEP_PNL_LAMBDA,
            "DRAWDOWN_LAMBDA": mask_config.DRAWDOWN_LAMBDA,
            "WIN_BONUS": mask_config.WIN_BONUS,
            "NO_TRADE_PENALTY": mask_config.NO_TRADE_PENALTY,
            "REWARD_NORMALIZER": mask_config.REWARD_NORMALIZER,
        }
    return {
        "PNL_LAMBDA": config.PNL_LAMBDA,
        "STEP_PNL_LAMBDA": config.STEP_PNL_LAMBDA,
        "DRAWDOWN_LAMBDA": config.DRAWDOWN_LAMBDA,
        "TRADE_PENALTY_LAMBDA": config.TRADE_PENALTY_LAMBDA,
        "TURNOVER_LAMBDA": config.TURNOVER_LAMBDA,
    }

def _get_config_metadata():
    cfg = mask_config if AGENT == 'MaskablePPO' else config
    d = {
        "INITIAL_CAPITAL": cfg.INITIAL_CAPITAL,
        "MAX_LOTS": cfg.MAX_LOTS,
        "TRANSACTION_COST_PCT": cfg.TRANSACTION_COST_PCT,
        "SLIPPAGE_PCT": cfg.SLIPPAGE_PCT,
        "START_TIME": cfg.START_TIME,
        "END_TIME": cfg.END_TIME,
        "STRADDLE_STRIKE_GAP": cfg.STRADDLE_STRIKE_GAP,
        "LOT_SIZE": cfg.LOT_SIZE,
        "STRIKE_SELECTION_METHOD": cfg.STRIKE_SELECTION_METHOD,
        "EXPIRY_DAY_OF_WEEK": cfg.EXPIRY_DAY_OF_WEEK,
        "MAX_EPISODE_DAYS": cfg.MAX_EPISODE_DAYS,
        "WINDOW_SIZE": cfg.WINDOW_SIZE,
        "RISK_FREE_RATE": cfg.RISK_FREE_RATE,
    }
    if AGENT == 'MaskablePPO':
        d["POSITION_MODE"] = mask_config.POSITION_MODE
    else:
        d["STRATEGY_TYPE"] = config.STRATEGY_TYPE
    return d


def make_env_kwargs(data_path, vix_path, start, end):
    return {
        'data_path': data_path,
        'vix_data_path': vix_path,
        'start_date': start,
        'end_date': end,
    }


def evaluate_fold(model, test_start, test_end, n_episodes):
    """Run n_episodes evaluation episodes, return fold metrics."""
    if AGENT == 'MaskablePPO':
        raw_env = IntradayOptionEnvV2(
            data_path=DATA_PATH, vix_data_path=VIX_DATA_PATH,
            start_date=test_start, end_date=test_end,
        )
        env = ActionMasker(raw_env, mask_fn)
    else:
        env = IntradayOptionEnv(
            data_path=DATA_PATH, vix_data_path=VIX_DATA_PATH,
            start_date=test_start, end_date=test_end,
        )
        raw_env = env

    episode_rewards = []
    episode_pnls    = []
    episode_trades  = []
    wins = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0.0
        final_info   = {}

        while not (done or truncated):
            if AGENT == 'MaskablePPO':
                masks = raw_env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            final_info = info

        episode_rewards.append(total_reward)
        pnl = final_info.get('realized_pnl', 0.0)
        episode_pnls.append(pnl)
        episode_trades.append(final_info.get('total_trades', 0))
        if pnl > 0:
            wins += 1

    rewards = np.array(episode_rewards)
    pnls    = np.array(episode_pnls)

    # Sharpe over episodes (each episode = ~1 week of returns)
    sharpe = (rewards.mean() / (rewards.std() + 1e-8)) * np.sqrt(52)

    return {
        'mean_reward':       float(rewards.mean()),
        'std_reward':        float(rewards.std()),
        'sharpe':            float(sharpe),
        'mean_pnl':          float(pnls.mean()),
        'max_drawdown_mean': float(np.array([env.max_drawdown for _ in [0]]).mean()),  # last episode only
        'win_rate':          float(wins / n_episodes),
        'mean_trades':       float(np.array(episode_trades).mean()),
        'n_episodes':        n_episodes,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_results = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(FOLDS):
        fold_num = fold_idx + 1
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num}/{len(FOLDS)}")
        print(f"  Train: {train_start} → {train_end}")
        print(f"  Test:  {test_start} → {test_end}")
        print(f"{'='*60}")

        fold_dir = os.path.join(SAVE_DIR, f'fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)

        # ── Train ──────────────────────────────────────
        env_kwargs = make_env_kwargs(DATA_PATH, VIX_DATA_PATH, train_start, train_end)

        if AGENT == 'MaskablePPO':
            def make_masked_env(rank, seed=42):
                def _init():
                    e = IntradayOptionEnvV2(**env_kwargs)
                    e = ActionMasker(e, mask_fn)
                    e.reset(seed=seed + rank)
                    return e
                return _init
            train_env = SubprocVecEnv([make_masked_env(i) for i in range(NUM_ENVS)])
        else:
            train_env = make_vec_env(
                IntradayOptionEnv,
                n_envs=NUM_ENVS,
                seed=42,
                vec_env_cls=SubprocVecEnv,
                env_kwargs=env_kwargs,
            )

        model_kwargs = dict(
            seed=42, verbose=0,
            tensorboard_log=os.path.join(fold_dir, 'logs'),
            device='cuda',
        )
        if AGENT == 'MaskablePPO':
            model_kwargs.update(dict(
                learning_rate=mask_config.LEARNING_RATE,
                n_steps=mask_config.N_STEPS,
                batch_size=mask_config.BATCH_SIZE,
                gamma=mask_config.GAMMA,
                ent_coef=mask_config.ENT_COEF,
            ))

        model = AgentCls('MlpPolicy', train_env, **model_kwargs)
        print(f"  Training for {N_TRAIN_STEPS:,} steps ...")
        model.learn(total_timesteps=N_TRAIN_STEPS, progress_bar=True)
        model.save(os.path.join(fold_dir, 'model'))
        train_env.close()
        print("  Training done.")

        # ── Evaluate ───────────────────────────────────
        print(f"  Evaluating {N_EVAL_EPISODES} episodes on test set ...")
        metrics = evaluate_fold(model, test_start, test_end, N_EVAL_EPISODES)

        fold_result = {
            'fold':        fold_num,
            'train_start': train_start,
            'train_end':   train_end,
            'test_start':  test_start,
            'test_end':    test_end,
            **metrics,
        }
        all_results.append(fold_result)

        print(f"  ✓ Sharpe={metrics['sharpe']:.2f}  WinRate={metrics['win_rate']:.1%}  "
              f"MeanPnL={metrics['mean_pnl']:.0f}  MeanReward={metrics['mean_reward']:.3f}")

    # ── Save Results ───────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path   = os.path.join(SAVE_DIR, 'results.csv')
    json_path  = os.path.join(SAVE_DIR, 'results.json')

    results_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)

    # ── Save Metadata ──────────────────────────────────
    # Match the structure of PPO_train.py / A2C_train.py
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "agent": AGENT,
        "run_type": "walk_forward_validation",
        "data_path": DATA_PATH,
        "vix_data_path": VIX_DATA_PATH,
        "n_train_steps_per_fold": N_TRAIN_STEPS,
        "n_eval_episodes_per_fold": N_EVAL_EPISODES,
        "folds": FOLDS,
        "reward_settings": _get_reward_metadata(),
        "config_parameters": _get_config_metadata(),
    }
    
    metadata_path = os.path.join(SAVE_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n{'='*60}")
    print("  WALK-FORWARD VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(results_df[['fold', 'test_start', 'test_end', 'sharpe', 'win_rate', 'mean_pnl']].to_string(index=False))
    print(f"\nFull results saved to:\n  {csv_path}\n  {json_path}\n  {metadata_path}")


if __name__ == '__main__':
    main()
