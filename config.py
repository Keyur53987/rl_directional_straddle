
# Configuration for Intraday RL Trading System

# Account Settings

INITIAL_CAPITAL = 1000000.0
MAX_LOTS = 10
TRANSACTION_COST_PCT = 0.0005  # 0.05% per trade
SLIPPAGE_PCT = 0.0002          # 0.02% slippage

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
EXPIRY_DAY_OF_WEEK = 1         # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri

# RL Reward Settings
REWARD_TYPE = "CUMULATIVE_ROI + DRAWDOWN_PENALTY + TRADE_PENALTY"
REWARD_LAMBDA = 0.05            # Penalty weight for drawdown
TRADE_PENALTY_LAMBDA = 0.01     # Penalty weight for trade count (reduces overtrading)
FORCED_EXIT_PENALTY = 1000.0   # Penalty for not exiting manually before EOD

# Environment Settings (1-minute data)
WINDOW_SIZE = 150              # Lookback window for state features (150 minutes = 2.5 hours)
RISK_FREE_RATE = 0.06          # Annual risk-free rate
IV_ESTIMATE = 0.15             # Constant IV for simulation if not provided # Change it with Newton-Raplphson method
