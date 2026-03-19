import pandas as pd
from envs.intraday_option_env import IntradayOptionEnv
import config
from datetime import datetime

def test_iv_logging():
    print("Initializing environment...")
    
    # Initialize environment with both intraday and VIX data
    # We use India VIX data file which is open in your editor
    env = IntradayOptionEnv(
        data_path='data/train.csv',  # Your main options/index data
        vix_data_path='data/INDIA_VIX.csv'     # Your new India VIX data
    )
    
    print("\nEnvironment initialized successfully. Merged data shape:", env.df.shape)
    if 'vix' in env.df.columns:
        print(f"VIX column successfully merged! Sample VIX values: {env.df['vix'].dropna().head().tolist()}")
    else:
        print("Warning: VIX column not found after merge.")
        
    print("\nStarting episode for 1 day (Testing date with 15-day history)...")
    # Pass date_index=20 to ensure we actually have 15-days of prior historical data!
    # On date_index=0, we had ZERO history, so it was falling back to 30-minute volatility.
    obs, info = env.reset(options={'date_index': 20})
    
    done = False
    truncated = False
    step_count = 0
    
    print(f"\n{'Time':<20} | {'Underlying Price':<18} | {'Implied Vol (IV Proxy)':<25} | {'Source'}")
    print("-" * 85)
    
    while not (done or truncated):
        # We can extract the time and IV directly from the environment state
        current_time = env.day_data.iloc[env.current_step]['datetime']
        current_price = env.day_data.iloc[env.current_step]['close']
        
        # Get the IV that Black-Scholes is using right now
        current_iv = env._get_current_volatility()
        
        # Determine if it's using VIX or Realized Volatility
        if 'vix' in env.day_data.columns and pd.notna(env.day_data.iloc[env.current_step]['vix']):
            source = "India VIX"
        else:
            # Determine actual fallback being used based on available history
            current_global_idx = env.global_start_idx + env.current_step
            if current_global_idx >= 5850:
                source = "Realized Vol (15-Day) [SMOOTH]"
            elif current_global_idx >= 2730:
                source = "Realized Vol (7-Day)"
            elif current_global_idx >= 390:
                source = "Realized Vol (1-Day)"
            else:
                source = "Realized Vol (30-Min) [JUMPY!]"
                
        print(f"{str(current_time):<20} | {current_price:<18.2f} | {current_iv:<25.6f} | {source}")
        
        # Take a dummy action (e.g., Hold both CE and PE)
        # Action space is MultiDiscrete([3, 3]), 0 = Hold
        action = [0, 0] 
        
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

    print("-" * 85)
    print(f"Episode finished! Total minutes processed: {step_count}")
    print(f"Final Realized PnL: {info.get('realized_pnl', 0):.2f}")

if __name__ == "__main__":
    test_iv_logging()
