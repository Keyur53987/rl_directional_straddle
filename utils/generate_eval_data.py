import os
import glob
import json
import pandas as pd
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.log_analyzer import get_log_stats

def generate_eval_data(model_path):
    """
    Generates results.json and metadata.json for a single model path.
    """
    timestamp = os.path.basename(model_path)
    print(f"Processing model: {timestamp} at {model_path}...")
    
    # Paths
    daily_results_path = os.path.join(model_path, "results", "daily_results.csv")
    overall_results_path = os.path.join(model_path, "results", "overall_results.csv")
    metadata_path = os.path.join(model_path, "metadata.json")
    results_json_path = os.path.join(model_path, "results.json")
    
    # Check if results exist
    if not os.path.exists(daily_results_path):
        print(f"  Skipping {timestamp}: No daily_results.csv found at {daily_results_path}")
        return
        
    # 1. Handle Metadata (Read First to include in Results)
    metadata_data = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata_data = json.load(f)
        except Exception as e:
            print(f"  Error reading existing metadata.json: {e}")
    
    # Ensure timestamp is in metadata and matches the folder name (Primary Key)
    metadata_data["timestamp"] = timestamp
        
    # Save metadata.json
    with open(metadata_path, 'w') as f:
        json.dump(metadata_data, f, indent=4)
    print(f"  Verified/Saved metadata.json")

    # 2. Gather Results Data
    results_data = {
        "timestamp": timestamp
    }
    
    # Add Hyperparameter Section from Metadata
    results_data["Hyperparameter"] = {
        "training_data_range": metadata_data.get("training_data"),
        "validation_data_range": metadata_data.get("validation_data"),
        "model_parameters": metadata_data.get("model_parameters"),
        "config_parameters": metadata_data.get("config_parameters", {}),
        "reward_lambda": metadata_data.get("config_parameters", {}).get("REWARD_LAMBDA")
    }
    
    try:
        # Read Daily Results for Range
        daily_df = pd.read_csv(daily_results_path)
        if 'date' in daily_df.columns:
            results_data["testing_data_range"] = {
                "start_date": str(daily_df['date'].min()),
                "end_date": str(daily_df['date'].max()),
                "total_days": int(len(daily_df))
            }
        
        # Read Overall Results
        if os.path.exists(overall_results_path):
            overall_df = pd.read_csv(overall_results_path)
            # Convert the first row to a dictionary
            if not overall_df.empty:
                # Convert numpy types to native types for JSON serialization
                results_data["overall_result_information"] = overall_df.iloc[0].to_dict()
                for k, v in results_data["overall_result_information"].items():
                    if pd.isna(v): results_data["overall_result_information"][k] = None
                    elif hasattr(v, 'item'): results_data["overall_result_information"][k] = v.item()

        # Get Training Log Stats
        # Logs might be in 'logs' subdir or main dir
        log_stats = get_log_stats(model_path)
        if log_stats:
            results_data["training_mean_reward"] = log_stats.get("mean_reward")
            results_data["training_stats"] = log_stats
            
    except Exception as e:
        print(f"  Error processing results for {timestamp}: {e}")
        return

    # Save results.json
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"  Saved results.json")

    # Save results.txt
    results_txt_path = os.path.join(model_path, "results.txt")
    with open(results_txt_path, 'w') as f:
        for key, value in results_data.items():
            f.write(f"{key}:\n")
            f.write(json.dumps(value, indent=4))
            f.write("\n\n")
    print(f"  Saved results.txt")

if __name__ == "__main__":
    # Default to PPO folder
    model_id = "20260203_203412"
    ppo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "PPO", model_id)
    print(ppo_dir)
    generate_eval_data(ppo_dir)
