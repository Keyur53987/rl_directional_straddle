import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_combined_results(model_dir, save_path='combined_training_curve.png'):
    """
    Finds all monitor.csv files in the given model/logs directory recursively,
    sorts them by the chronological runs (original + retrainings), and plots
    a combined, continuous learning curve.
    """
    log_dir = os.path.join(model_dir, "logs")
    monitor_files = glob.glob(os.path.join(log_dir, "**", "*.monitor.csv"), recursive=True)
    
    if not monitor_files:
        print(f"No monitor files found in {model_dir}")
        return
        
    print(f"Found {len(monitor_files)} monitor files. Combining...")
    
    dfs = []
    for file in monitor_files:
        try:
            with open(file, 'rt') as f:
                first_line = f.readline()
                if not first_line or not first_line.startswith('#'):
                    continue
                df = pd.read_csv(f, index_col=None)
                
                # Extract folder name to use for chronological sorting 
                # (e.g. monitor_20260319_103000)
                folder_name = os.path.basename(os.path.dirname(file))
                df['run_folder'] = folder_name
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not dfs:
        print("No valid monitor data found.")
        return
        
    # Concatenate all dataframes
    df = pd.concat(dfs)
    
    # Sort by the run folder name (which contains the timestamp) and then by standard 't'
    # This ensures original training episodes precede the retraining episodes.
    df = df.sort_values(by=['run_folder', 't'])
    
    # Reset index to convert rows to chronological cumulative episodes
    df = df.reset_index(drop=True)
    
    rewards = df['r'].values
    
    rolling_window = 50
    # ensure enough data points for the window
    if len(rewards) < rolling_window:
        rolling_window = max(1, len(rewards) // 5)
        
    rolling_avg = pd.Series(rewards).rolling(window=rolling_window).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(rolling_avg, color='red', label=f'{rolling_window}-Episode Moving Average')
    
    # Draw vertical lines where a new training run started (e.g., retrainings)
    run_changes = df['run_folder'].ne(df['run_folder'].shift())
    change_indices = run_changes[run_changes].index.tolist()
    
    for idx in change_indices:
        if idx > 0:  # don't draw line at episode 0
            plt.axvline(x=idx, color='orange', linestyle='--', alpha=0.7, 
                        label='Retraining Starts' if idx == change_indices[1] else "")
            
    plt.title(f'Combined Learning Curve\n({len(rewards)} Total Episodes)')
    plt.xlabel('Cumulative Episodes (Original + Retraining)')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    out_file = os.path.join(model_dir, save_path)
    plt.savefig(out_file)
    print(f"Combined training plot saved to {out_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot combined original and retraining logs.")
    parser.add_argument("model_dir", type=str, help="Path to the model directory (e.g., model/PPO/20231027_120000)")
    args = parser.parse_args()
    
    plot_combined_results(args.model_dir)
