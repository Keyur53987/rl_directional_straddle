"""
Configuration for Directional Straddle v2 — MaskablePPO Environment
===================================================================
Standalone config. Independent of old config.py.
"""

# ── Account ──
INITIAL_CAPITAL = 1_000_000.0
MAX_LOTS = 10
TRANSACTION_COST_PCT = 0.05 / 100   # 0.05% per trade (as decimal)
SLIPPAGE_PCT = 0.02 / 100           # 0.02% slippage (as decimal)

# ── Market ──
START_TIME = "09:15"
END_TIME = "15:30"
STRADDLE_STRIKE_GAP = 100           # NIFTY strike gap
LOT_SIZE = 75                       # NIFTY lot size
EXPIRY_DAY_OF_WEEK = 1              # 0=Mon, 1=Tue (NIFTY50 weekly expiry)
MAX_EPISODE_DAYS = 5                # Safety cap: max calendar days per episode

# ── Strategy ──
POSITION_MODE = 'BOTH'         # 'LONG_ONLY', 'SHORT_ONLY', 'BOTH'
STRIKE_SELECTION_METHOD = 'ATM'     # 'ATM', 'BOLLINGER', 'ATR'
BOLLINGER_STD = 2.0
ATR_MULTIPLIER = 1.5

# ── Margin Settings (Dynamic, IV-based) ──
# LONG:  margin = premium = BS_price × LOT_SIZE
# SHORT: margin = Dynamic SPAN + Fixed Exposure
#   Price scan = Spot × daily_vol × √MPOR × COVERAGE
#   Stressed price = BS(spot ± scan, strike, tte, IV × (1 + VOL_STRESS))
#   SPAN margin = (stressed_price - current_price) × LOT_SIZE
#   Exposure = EXPOSURE_PCT × spot × LOT_SIZE
SPAN_STDEV_COVERAGE = 3.5           # σ coverage for price scan (~99.95%)
SPAN_MPOR_DAYS = 2                  # Minimum Period of Risk (days)
SPAN_VOL_STRESS = 0.25              # Volatility stressed +25%
EXPOSURE_MARGIN_PCT = 0.03          # Fixed regulatory exposure (3% of notional)

# ── Reward (3 terms only) ──
REWARD_NORMALIZER = 25_000.0        # ~1 lot ATM straddle premium (75 × ~₹300)
STEP_PNL_LAMBDA = 1.0               # Primary: minute-by-minute PnL change
DRAWDOWN_LAMBDA = 0.01              # Light penalty for current drawdown
WIN_BONUS = 2.0                     # Terminal: bonus if episode PnL > 0
NO_TRADE_PENALTY = 1.0              # Terminal: penalty if 0 trades all episode

# ── Environment ──
WINDOW_SIZE = 150                   # Lookback window (150 minutes = 2.5 hours)
RISK_FREE_RATE = 0.06               # Annual risk-free rate

# ── Training Hyperparameters ──
LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.99
ENT_COEF = 0.05                     # Higher entropy for 8-action exploration
TOTAL_TIMESTEPS = 1_000
NUM_ENVS = 4
SEED = 42
