"""
Configuration for Directional Straddle v3 — MaskablePPO Environment
===================================================================
v3 changes from v2:
  - Multi-horizon PnL reward (1-step / 30-step / 120-step)
  - Separate CE/PE PnL signal in reward
  - Observation space: 69 features (67 from v2 + ce_unrealized_pnl + pe_unrealized_pnl)
  - Horizon weights tuned for weekly episodes (~1,875 steps)
"""

# ── Account ──
INITIAL_CAPITAL = 1_000_000.0
MAX_LOTS = 10
TRANSACTION_COST_PCT = 0.5 / 100   # 0.05% per trade (as decimal)
SLIPPAGE_PCT = 0.5 / 100           # 0.02% slippage (as decimal)

# ── Market ──
START_TIME = "09:15"
END_TIME = "15:30"
STRADDLE_STRIKE_GAP = 50           # NIFTY strike gap
SHORT_STRIKE_OFFSET = 100           # OTM offset for SHORT strangle (CE=ATM+offset, PE=ATM-offset)
LOT_SIZE = 65                       # NIFTY lot size
EXPIRY_DAY_OF_WEEK = 1              # 0=Mon, 1=Tue (NIFTY50 weekly expiry)
MAX_EPISODE_DAYS= 7                # Safety cap: max calendar days per episode

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

# ── Reward (Multi-Horizon & Separate CE/PE PnL) ──
# Horizons tuned for weekly episodes (~1,875 steps total)
REWARD_NORMALIZER = 5_000.0        # ~1 lot ATM straddle premium (75 × ~₹300)

PNL_W1   = 0.15                     # 1-step:   news shock detection (low weight, high reactivity)
PNL_W30  = 0.60                     # 30-step:  primary signal (30-min half-session trend)
PNL_W120 = 0.25                     # 120-step: half-day regime confirmation

DRAWDOWN_LAMBDA = 1.0               # Delta-drawdown: penalize only when DD deepens
WIN_BONUS = 2.0                     # Terminal: bonus if episode PnL > 0
NO_TRADE_PENALTY = 1.0              # Terminal: penalty if 0 trades all episode

# ── Environment ──
WINDOW_SIZE = 150                   # Lookback window (150 minutes = 2.5 hours)
RISK_FREE_RATE = 0.06               # Annual risk-free rate

# ── Training Hyperparameters ──
LEARNING_RATE = 7e-4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.9995
ENT_COEF = 0.05
TOTAL_TIMESTEPS = 1_500_000
NUM_ENVS = 4
SEED = 42
