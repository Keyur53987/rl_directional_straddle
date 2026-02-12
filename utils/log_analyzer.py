import os
import glob
import pandas as pd
import argparse
import numpy as np

def get_log_stats(log_folder):
    """
    Reads all monitor.csv files in the folder and returns statistics.
    Returns: dict of stats or None if no data found.
    """
    # Find all monitor files
    monitor_files = glob.glob(os.path.join(log_folder, "*.monitor.csv"))
    if not monitor_files:
        # Try checking for numbered logs directly in case folder structure implies it
        # Sometimes logs are in log_folder/logs/
        monitor_files = glob.glob(os.path.join(log_folder, "logs", "*.monitor.csv"))
        
    if not monitor_files:
        print(f"No monitor files found in {log_folder}")
        return None

    dfs = []
    for file in monitor_files:
        try:
            with open(file, 'rt') as f:
                first_line = f.readline()
                if not first_line: continue
                # valid monitor files start with #
                if first_line.startswith('#'):
                     df = pd.read_csv(f, index_col=None)
                else:
                     # Maybe no header comment? Try reading directly
                     f.seek(0)
                     df = pd.read_csv(f, index_col=None)
                     
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        print("No valid data found.")
        return None

    # Combine
    full_df = pd.concat(dfs, ignore_index=True)
    
    if 'r' not in full_df.columns:
        print("Column 'r' (reward) not found in logs.")
        return None
        
    rewards = full_df['r']
    
    # Calculate Statistics
    stats = {
        "mean_reward": float(rewards.mean()),
        "median_reward": float(rewards.median()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "total_episodes": len(rewards)
    }
    
    # Optional: windowed mean for the last 100 episodes
    if len(rewards) > 0:
        stats["mean_reward_last_100"] = float(rewards.tail(100).mean())
        
    return stats

def analyze_logs(log_folder):
    stats = get_log_stats(log_folder)
    
    if not stats:
        return

    print("-" * 40)
    print(f"Log Analysis for: {log_folder}")
    print("-" * 40)
    print(f"Total Episodes:   {stats['total_episodes']}")
    print(f"Mean Reward:      {stats['mean_reward']:.6f}")
    print(f"Median Reward:    {stats['median_reward']:.6f}")
    print(f"Std Deviation:    {stats['std_reward']:.6f}")
    print(f"Min Reward:       {stats['min_reward']:.6f}")
    print(f"Max Reward:       {stats['max_reward']:.6f}")
    print("-" * 40)
    
    if "mean_reward_last_100" in stats:
        print(f"Mean Reward (Last 100 Episodes): {stats['mean_reward_last_100']:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RL Logs to get precise reward metrics.")
    parser.add_argument("folder", type=str, help="Path to the model directory (containing logs folder or monitor files)")
    args = parser.parse_args()

    analyze_logs(args.folder)
