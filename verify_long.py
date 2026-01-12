
import config
from envs.intraday_option_env import IntradayOptionEnv
import numpy as np

def test_long_strategy():
    config.STRATEGY_TYPE = 'LONG'
    env = IntradayOptionEnv(data_path='data/train.csv')
    
    # Reset
    obs, info = env.reset()
    print(f"\n--- Strategy: {config.STRATEGY_TYPE} ---")
    print(f"Initial: CE Lots: {env.ce_lots}, PE Lots: {env.pe_lots}")
    print(f"Entry Prices: CE {env.entry_price_ce:.2f}, PE {env.entry_price_pe:.2f}")
    
    # 1. Action: Add to CALL (Buy), Hold PUT
    # MultiDiscrete Action: [1, 0]
    # 1 = Add, 0 = Hold
    print("\n--- Step 1: Add Call (Buy) ---")
    action = np.array([1, 0]) 
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}")
    print(f"CE Lots: {env.ce_lots} (Exp: 2), PE Lots: {env.pe_lots} (Exp: 1)")
    print(f"Current PnL: {env.pnl_curve[-1]:.2f}")
    
    # 2. Action: Offload PUT (Sell), Hold CALL
    # MultiDiscrete Action: [0, 2]
    # 2 = Offload
    print("\n--- Step 2: Offload Put (Sell) ---")
    action = np.array([0, 2])
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}")
    print(f"CE Lots: {env.ce_lots} (Exp: 2), PE Lots: {env.pe_lots} (Exp: 0)")
    print(f"Realized PnL: {env.realized_pnl:.2f}")
    
    # 3. Action: Hold All
    print("\n--- Step 3: Hold ---")
    action = np.array([0, 0])
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}")
    print(f"CE Lots: {env.ce_lots}, PE Lots: {env.pe_lots}")

if __name__ == "__main__":
    try:
        test_long_strategy()
    except Exception as e:
        print(f"Error: {e}")
