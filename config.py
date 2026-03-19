
# Configuration for Intraday RL Trading System

# Account Settings

INITIAL_CAPITAL = 1000000.0
MAX_LOTS = 10
TRANSACTION_COST_PCT = 0.05  # 0.05% per trade
SLIPPAGE_PCT = 0.02          # 0.02% slippage

# Strategy Settings
# Strategy Settings
START_TIME = "09:30"
END_TIME = "15:15"
STRADDLE_STRIKE_GAP = 100      # NIFTY strike gap
LOT_SIZE = 75                  # NIFTY lot size

# Strategy Settings
STRATEGY_TYPE = 'LONG'         # 'LONG' (Buy) or 'SHORT' (Sell)

# Strike Selection Settings
STRIKE_SELECTION_METHOD = 'ATM' # 'ATM', 'BOLLINGER', 'ATR'
BOLLINGER_STD = 2.0            # Standard deviation for Bollinger Bands
ATR_MULTIPLIER = 1.5           # Multiplier for ATR-based gap
EXPIRY_DAY_OF_WEEK = 1         # 0=Mon, 1=Tue (NIFTY50 weekly expiry), 2=Wed, 3=Thu, 4=Fri
MAX_EPISODE_DAYS = 5           # Safety cap: max calendar days per episode (1 full week)

# RL Reward Settings
PNL_LAMBDA = 2.0                # Weight for cumulative PnL (ROI) — primary learning signal
STEP_PNL_LAMBDA = 0.5           # Weight for step-wise PnL (immediate feedback, was too small before)
DRAWDOWN_LAMBDA = 0.5           # Penalty weight for current drawdown
TRADE_PENALTY_LAMBDA = 0.01     # Penalty per lot traded / MAX_LOTS (was 0.2 — way too large with new norm)
TURNOVER_LAMBDA = 0.05          # Penalty weight for notional churned
# FORCED_EXIT_LAMBDA removed: episodes now run to expiry, no forced EOD close

# Environment Settings (1-minute data)
WINDOW_SIZE = 150              # Lookback window for state features (150 minutes = 2.5 hours)
RISK_FREE_RATE = 0.06          # Annual risk-free rate
