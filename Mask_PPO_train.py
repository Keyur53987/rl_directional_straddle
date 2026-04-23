"""
MaskablePPO Training Script for Directional Straddle v2
========================================================
Uses sb3-contrib MaskablePPO with action masking.
Trains on intraday_option_env_v2 (Discrete(8), no forced entry).
"""

import gymnasium as gym
import json
import numpy as np
import pandas as pd
import os
import datetime
import random
import torch
import matplotlib.pyplot as plt
import glob

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.intraday_option_env_maskPPO import IntradayOptionEnvV2
import Mask_PPO_config as config


# ─────────────────────────────────────────────────
# PnL TRACKING CALLBACK
# ─────────────────────────────────────────────────

class PnLTrackingCallback(BaseCallback):
    """
    Logs ACTUAL ₹ PnL during training (not shaped reward).
    Prints summary every `print_freq` episodes.
    """
    def __init__(self, print_freq=50, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_pnls = []
        self.episode_trades = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                pnl = info.get("realized_pnl", 0.0)
                trades = info.get("total_trades", 0)
                self.episode_pnls.append(pnl)
                self.episode_trades.append(trades)

                if len(self.episode_pnls) % self.print_freq == 0:
                    recent_pnl = self.episode_pnls[-self.print_freq:]
                    recent_trades = self.episode_trades[-self.print_freq:]
                    avg_pnl = sum(recent_pnl) / len(recent_pnl)
                    win_rate = sum(1 for p in recent_pnl if p > 0) / len(recent_pnl) * 100
                    avg_trades = sum(recent_trades) / len(recent_trades)
                    flat_pct = sum(1 for t in recent_trades if t == 0) / len(recent_trades) * 100

                    print(f"[PnL Monitor] Ep {len(self.episode_pnls):>5d} | "
                          f"AvgPnL=₹{avg_pnl:>8.0f} | WinRate={win_rate:>5.1f}% | "
                          f"AvgTrades={avg_trades:>4.1f} | Flat%={flat_pct:>4.1f}%")
        return True


# ─────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────

def plot_results(log_folder, save_folder, title='Learning Curve'):
    """Read monitor.csv and plot training results."""
    try:
        monitor_files = glob.glob(os.path.join(log_folder, "*.monitor.csv"))
        if not monitor_files:
            print("No monitor files found.")
            return

        dfs = []
        for file in monitor_files:
            with open(file, 'rt') as f:
                first_line = f.readline()
                if not first_line or first_line[0] != '#':
                    continue
                dfs.append(pd.read_csv(f, index_col=None))

        if not dfs:
            print("No valid monitor data found.")
            return

        df = pd.concat(dfs).sort_values('t')
        rewards = df['r'].values

        rolling_window = 50
        rolling_avg = pd.Series(rewards).rolling(window=rolling_window).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        plt.plot(rolling_avg, color='red', label=f'{rolling_window}-Episode Moving Average')
        plt.title(title)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(save_folder, 'training_curve.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training plot saved to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Error plotting results: {e}")


# ─────────────────────────────────────────────────
# ACTION MASKER WRAPPER
# ─────────────────────────────────────────────────

def mask_fn(env: gym.Env) -> np.ndarray:
    """Extract action masks from the environment."""
    return env.action_masks()


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    # Seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CURRENT_RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    AGENT = 'MaskablePPO'

    # Paths
    DATA_PATH = 'data/train.csv'
    VIX_DATA_PATH = 'data/INDIA_VIX.csv'
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"

    RETRAIN_MODEL_PATH = None

    if RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH):
        SAVE_DIR = os.path.dirname(RETRAIN_MODEL_PATH)
        TIMESTAMP = os.path.basename(SAVE_DIR)
        print(f"Resuming training in existing directory: {SAVE_DIR}")
    else:
        TIMESTAMP = CURRENT_RUN_ID
        BASE_DIR = os.path.join('model', AGENT)
        SAVE_DIR = os.path.join(BASE_DIR, TIMESTAMP)

    LOG_DIR = os.path.join(SAVE_DIR, 'logs')
    MONITOR_DIR = os.path.join(LOG_DIR, f"monitor_{CURRENT_RUN_ID}")
    EVAL_MONITOR_DIR = os.path.join(SAVE_DIR, 'eval_monitor', f"eval_{CURRENT_RUN_ID}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)
    os.makedirs(EVAL_MONITOR_DIR, exist_ok=True)

    print(f"=" * 60)
    print(f"  MaskablePPO Training — Directional Straddle v2")
    print(f"  Position Mode: {config.POSITION_MODE}")
    print(f"  Data: {START_DATE} → {END_DATE}")
    print(f"  Run ID: {TIMESTAMP}")
    print(f"=" * 60)

    try:
        # ── Training Environment ──
        env_kwargs = {
            'data_path': DATA_PATH,
            'vix_data_path': VIX_DATA_PATH,
            'start_date': START_DATE,
            'end_date': END_DATE
        }

        print(f"Starting {config.NUM_ENVS} Parallel Environments...")

        # Create vectorized env with ActionMasker
        def make_env(rank, seed=0):
            def _init():
                env = IntradayOptionEnvV2(**env_kwargs)
                env = ActionMasker(env, mask_fn)
                env = Monitor(env, os.path.join(MONITOR_DIR, str(rank)))
                env.reset(seed=seed + rank)
                return env
            return _init

        env = SubprocVecEnv([make_env(i, config.SEED) for i in range(config.NUM_ENVS)])

        # ── Validation Environment ──
        VAL_START_DATE = "2024-01-01"
        VAL_END_DATE = "2024-06-30"
        print(f"Validation Environment: {VAL_START_DATE} → {VAL_END_DATE}")

        eval_env_kwargs = {
            'data_path': DATA_PATH,
            'vix_data_path': VIX_DATA_PATH,
            'start_date': VAL_START_DATE,
            'end_date': VAL_END_DATE
        }

        def make_eval_env():
            def _init():
                env = IntradayOptionEnvV2(**eval_env_kwargs)
                env = ActionMasker(env, mask_fn)
                env = Monitor(env, EVAL_MONITOR_DIR)
                return env
            return _init

        eval_env = SubprocVecEnv([make_eval_env()])

        # ── Callbacks ──
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=SAVE_DIR,
            log_path=LOG_DIR,
            eval_freq=5000,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=SAVE_DIR,
            name_prefix=f'{AGENT.lower()}_checkpoint'
        )

        pnl_callback = PnLTrackingCallback(print_freq=50)

        # ── Model ──
        print(f"Training MaskablePPO Agent...")
        print(f"  lr={config.LEARNING_RATE}, n_steps={config.N_STEPS}, "
              f"batch_size={config.BATCH_SIZE}, ent_coef={config.ENT_COEF}")

        if RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH):
            print(f"Loading existing model from: {RETRAIN_MODEL_PATH}")
            model = MaskablePPO.load(RETRAIN_MODEL_PATH, env=env,
                                     tensorboard_log=LOG_DIR, device='cuda')
        else:
            model = MaskablePPO(
                "MlpPolicy", env,
                seed=config.SEED,
                verbose=1,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE,
                gamma=config.GAMMA,
                ent_coef=config.ENT_COEF,
                tensorboard_log=LOG_DIR,
                device='cuda'
            )

        # ── Train ──
        reset_num_timesteps = not (RETRAIN_MODEL_PATH and os.path.exists(RETRAIN_MODEL_PATH))
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback, pnl_callback],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )
        print("Training Finished.")

        # Save final model
        final_model_path = os.path.join(SAVE_DIR, f'{AGENT.lower()}_final')
        model.save(final_model_path)
        print(f"Final Model saved to {final_model_path}")

        # ── Metadata ──
        metadata = {
            "timestamp": TIMESTAMP,
            "agent": AGENT,
            "environment_version": "v2",
            "training_data": {
                "path": DATA_PATH,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
            "validation_data": {
                "start_date": VAL_START_DATE,
                "end_date": VAL_END_DATE,
            },
            "model_parameters": {
                "policy": "MlpPolicy",
                "learning_rate": config.LEARNING_RATE,
                "n_steps": config.N_STEPS,
                "batch_size": config.BATCH_SIZE,
                "gamma": config.GAMMA,
                "ent_coef": config.ENT_COEF,
                "total_timesteps": config.TOTAL_TIMESTEPS,
            },
            "reward_settings": {
                "REWARD_NORMALIZER": config.REWARD_NORMALIZER,
                "STEP_PNL_LAMBDA": config.STEP_PNL_LAMBDA,
                "DRAWDOWN_LAMBDA": config.DRAWDOWN_LAMBDA,
                "WIN_BONUS": config.WIN_BONUS,
                "NO_TRADE_PENALTY": config.NO_TRADE_PENALTY,
            },
            "margin_settings": {
                "SPAN_STDEV_COVERAGE": config.SPAN_STDEV_COVERAGE,
                "SPAN_MPOR_DAYS": config.SPAN_MPOR_DAYS,
                "SPAN_VOL_STRESS": config.SPAN_VOL_STRESS,
                "EXPOSURE_MARGIN_PCT": config.EXPOSURE_MARGIN_PCT,
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
                "POSITION_MODE": config.POSITION_MODE,
                "STRIKE_SELECTION_METHOD": config.STRIKE_SELECTION_METHOD,
                "EXPIRY_DAY_OF_WEEK": config.EXPIRY_DAY_OF_WEEK,
                "MAX_EPISODE_DAYS": config.MAX_EPISODE_DAYS,
                "WINDOW_SIZE": config.WINDOW_SIZE,
                "RISK_FREE_RATE": config.RISK_FREE_RATE,
            },
            "pnl_tracking": {
                "total_episodes": len(pnl_callback.episode_pnls),
                "final_50_avg_pnl": float(np.mean(pnl_callback.episode_pnls[-50:])) if len(pnl_callback.episode_pnls) >= 50 else None,
                "final_50_win_rate": float(sum(1 for p in pnl_callback.episode_pnls[-50:] if p > 0) / 50) if len(pnl_callback.episode_pnls) >= 50 else None,
            }
        }

        metadata_path = os.path.join(SAVE_DIR, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")

        # Plotting
        print("Generating training plots...")
        plot_results(MONITOR_DIR, SAVE_DIR)

        print("-" * 60)
        print(f"TRAINING COMPLETE. Model ID: {TIMESTAMP}")
        print(f"Model saved at: {SAVE_DIR}")
        print("-" * 60)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
