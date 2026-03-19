
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config
from utils.black_scholes import black_scholes, calculate_position_greeks
from utils.feature_calculator import FeatureCalculator

class IntradayOptionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path='data/train.csv', vix_data_path='data/INDIA_VIX.csv', start_date=None, end_date=None):
        super(IntradayOptionEnv, self).__init__()
        
        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        # Fast in-memory merge of VIX data if provided
        if vix_data_path is not None:
            vix_df = pd.read_csv(vix_data_path)
            vix_df['datetime'] = pd.to_datetime(vix_df['datetime'])
            
            # Assuming standard naming ('close' or 'vix')
            vix_col = 'vix'
            if 'vix' not in vix_df.columns:
                vix_col = 'close' if 'close' in vix_df.columns else vix_df.columns[1]
                
            vix_df = vix_df[['datetime', vix_col]].rename(columns={vix_col: 'vix'})
            
            # Merge left so we keep the exact timestamps of main intraday data
            self.df = pd.merge(self.df, vix_df, on='datetime', how='left')
            # Keeping missing VIX as strictly NaN as requested
        
        # Filter by date range if provided
        if start_date:
            self.df = self.df[self.df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            self.df = self.df[self.df['datetime'] <= pd.to_datetime(end_date)]
            
        all_dates = self.df['datetime'].dt.date.unique()
        # Episode start dates = Wednesdays only.
        # NIFTY50 weekly expiry is Tuesday. Wednesday is the first day of a NEW expiry cycle.
        # This ensures every episode spans a consistent full week (Wed→Tue), giving the agent
        # the same ~5 days of multi-day experience regardless of which date is picked.
        # If no Wednesday is available (e.g., holiday), we fall back to all dates.
        wednesdays = [d for d in all_dates if d.weekday() == 2]  # 2 = Wednesday
        self.dates = np.array(wednesdays) if len(wednesdays) > 10 else all_dates
        
        # Action Space: MultiDiscrete([3, 3])
        # [Call Action, Put Action]
        # 0=Hold, 1=Add(Buy), 2=Offload(Sell)
        self.action_space = spaces.MultiDiscrete([3, 3])
        
        # Observation Space: 63 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32)
        
        # Feature calculator
        self.feature_calc = FeatureCalculator(window_size=config.WINDOW_SIZE)
        
        self.current_step = 0
        self.current_date_idx = 0
        self.day_data = None
        
        # Position State
        self.ce_lots = 0
        self.pe_lots = 0
        self.ce_strike = 0
        self.pe_strike = 0
        self.entry_price_ce = 0
        self.entry_price_pe = 0
        self.cash = config.INITIAL_CAPITAL
        # Track three variables in the environment state:
        # self.total_pnl: Cumulative Realized + Unrealized PnL from the start of the episode.
        # self.peak_pnl: The highest total_pnl seen so far in the episode.
        # self.max_drawdown: The largest drop from peak_pnl seen so far (always positive).
        # self.premium_deployed: The maximum capital/margin used.
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.premium_deployed = 0.0
        
        self.pnl_curve = []
        self.realized_pnl = 0.0
        self.total_trades = 0  # Track total lots traded (buys + sells)
        self.total_turnover = 0.0  # Track turnover (abs_profit + abs_loss)
        
        # Track day's OHLC for features
        self.day_open = 0.0
        self.day_high = 0.0
        self.day_low = float('inf')
        
        # IV cache: recomputed once per step, reused in step() and _get_observation()
        self._cached_iv = 0.15  # fallback bootstrap value until first computation
        self._cached_iv_step = -1  # which step we cached at
        
    def _get_current_volatility(self, vol_risk_premium=1.20):
        """
        Get current Implied Volatility using India VIX.
        Result is cached per step to avoid repeated expensive computation.
        Falls back to 15-day realized vol if VIX is unavailable.
        """
        # Return cached value if we already computed it this step
        if self._cached_iv_step == self.current_step:
            return self._cached_iv
        # 1. Use actual India VIX from the fast in-memory merged dataset
        # Use VIX[T-1] (previous minute tick) to prevent look-ahead leakage.
        # At decision time T you only know the LAST published VIX, not the simultaneous one.
        if 'vix' in self.day_data.columns:
            vix_lookup_step = max(0, self.current_step - 1)
            vix_val = self.day_data.iloc[vix_lookup_step]['vix']
            if pd.notna(vix_val) and vix_val > 0:
                self._cached_iv = (vix_val / 100.0) * vol_risk_premium
                self._cached_iv_step = self.current_step
                return self._cached_iv

        # 2. Fallback to historical calculation if VIX isn't found
        # Use the pre-built incremental arrays (already bounded to 6000 rows) — no DataFrame slice needed
        closes = np.array(self._hist_closes, dtype=np.float32)
        highs  = np.array(self._hist_highs,  dtype=np.float32)
        lows   = np.array(self._hist_lows,   dtype=np.float32)
        opens  = np.array(self._hist_opens,  dtype=np.float32)
          
        if len(closes) < 5:
            # Not enough data — use a safe market-standard value (15% annualized)
            return 0.15
        
        # Calculate Volatility Features
        vol_features = self.feature_calc.calculate_volatility_features(
            closes, highs, lows, opens, closes
        )
        
        # Use 15-day realized volatility as primary proxy if VIX is unavailable
        vol = vol_features.get('rolling_vol_15day', 0.0)
        
        # Additional fallbacks just in case we don't have 15 days of history mapped yet
        if vol < 1e-4:
             vol = vol_features.get('rolling_vol_7day', 0.0)
        if vol < 1e-4:
             vol = vol_features.get('rolling_vol_1day', 0.0)
        if vol < 1e-4:
             vol = vol_features.get('rolling_vol_30min', 0.0)
             
        # Store in cache then return
        self._cached_iv = vol_risk_premium * vol
        self._cached_iv_step = self.current_step
        return self._cached_iv

        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a date
        if options and 'date_index' in options:
            self.current_date_idx = options['date_index']
        else:
            self.current_date_idx = np.random.randint(0, len(self.dates))
        date = self.dates[self.current_date_idx]
        
        # --- Multi-day episode: load data from picked date up to next Tuesday (NIFTY expiry) ---
        import datetime as dt_lib
        expiry_day_num = config.EXPIRY_DAY_OF_WEEK          # 1 = Tuesday
        days_ahead = (expiry_day_num - date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # if picked day IS expiry day, go to NEXT week's expiry
        expiry_date = date + dt_lib.timedelta(days=days_ahead)
        # Safety cap to avoid runaway episodes
        max_end = date + dt_lib.timedelta(days=config.MAX_EPISODE_DAYS)
        expiry_date = min(expiry_date, max_end)
        self.episode_expiry_dt = pd.to_datetime(f"{expiry_date} {config.END_TIME}")
        
        # Load all minute rows from start date through expiry date
        mask = (self.df['datetime'].dt.date >= date) & (self.df['datetime'].dt.date <= expiry_date)
        self.day_data = self.df[mask].reset_index(drop=True)
        
        # Track global index for historical window access (uses first day of episode)
        day_indices = self.df.index[self.df['datetime'].dt.date == date].tolist()
        self.global_start_idx = day_indices[0] if day_indices else 0
        
        # Pre-load historical OHLC window (up to 15 days prior) into NumPy arrays at episode start.
        # _get_observation() will append 1 row per step instead of re-slicing the full DataFrame.
        pre_end   = self.global_start_idx         # exclusive: rows before today
        pre_start = max(0, pre_end - 6000)        # up to ~15 trading days
        pre_hist  = self.df.iloc[pre_start:pre_end]
        self._hist_closes = pre_hist['close'].values.tolist()  # grow by 1 each step
        self._hist_highs  = pre_hist['high'].values.tolist()
        self._hist_lows   = pre_hist['low'].values.tolist()
        self._hist_opens  = pre_hist['open'].values.tolist()
        
        # Fast forward to 9:30 AM
        start_time = pd.to_datetime(f"{date} {config.START_TIME}")
        start_idx = self.day_data[self.day_data['datetime'] >= start_time].index
        
        if len(start_idx) == 0:
            return self.reset() # Skip empty days
            
        self.current_step = start_idx[0]
        
        # Track day's OHLC
        self.day_open = self.day_data.iloc[self.current_step]['open']
        self.day_high = self.day_data.iloc[self.current_step]['high']
        self.day_low = self.day_data.iloc[self.current_step]['low']
        
        # Initialize Position (Straddle/Strangle)
        current_price = self.day_data.iloc[self.current_step]['close']
        
        # Determine Strikes based on Config
        if config.STRIKE_SELECTION_METHOD == 'BOLLINGER':
            # Get raw indicators from cached data if possible, or recalculate
            # We need history for indicators.
            hist_data = self.day_data.iloc[:self.current_step]
            closes = hist_data['close'].values
            highs = hist_data['high'].values
            lows = hist_data['low'].values
            
            raw_indicators = self.feature_calc.get_raw_indicators(closes, highs, lows)
            
            # CE Strike >= Upper Band
            ce_raw = raw_indicators.get('bb_upper', current_price)
            # PE Strike <= Lower Band
            pe_raw = raw_indicators.get('bb_lower', current_price)
            
            # Round to nearest strike
            self.ce_strike = round(ce_raw / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            self.pe_strike = round(pe_raw / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            
            # Ensure OTM/ATM logic holds (Call >= Price, Put <= Price for strangle-ish)
            if self.ce_strike < current_price: self.ce_strike = round(current_price / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            if self.pe_strike > current_price: self.pe_strike = round(current_price / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            
        elif config.STRIKE_SELECTION_METHOD == 'ATR':
            hist_data = self.day_data.iloc[:self.current_step]
            closes = hist_data['close'].values
            highs = hist_data['high'].values
            lows = hist_data['low'].values
            
            raw_indicators = self.feature_calc.get_raw_indicators(closes, highs, lows)
            atr = raw_indicators.get('atr', 0.0)
            
            gap = atr * config.ATR_MULTIPLIER
            
            self.ce_strike = round((current_price + gap) / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            self.pe_strike = round((current_price - gap) / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            
        else: # Default 'ATM'
            atm_strike = round(current_price / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            self.ce_strike = atm_strike
            self.pe_strike = atm_strike
        
        # Initial Allocation: 1 lot each (Straddle)
        self.ce_lots = 1
        self.pe_lots = 1
        
        # Use Black-Scholes for option pricing
        # Dynamic IV Calculation
        current_iv = self._get_current_volatility()
        
        time_to_expiry = self._get_time_to_expiry(self.day_data.iloc[self.current_step]['datetime'])
        ce_bs = black_scholes(current_price, self.ce_strike, time_to_expiry, 
                              config.RISK_FREE_RATE, current_iv, 'call')
        pe_bs = black_scholes(current_price, self.pe_strike, time_to_expiry, 
                              config.RISK_FREE_RATE, current_iv, 'put')
        
        self.entry_price_ce = ce_bs['price']
        self.entry_price_pe = pe_bs['price']
        
        self.cash = config.INITIAL_CAPITAL
        self.pnl_curve = [0.0]
        self.realized_pnl = 0.0  # Track realized PnL
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.premium_deployed = 0.0 # Track peak capital usage
        self.total_trades = 0  # Reset trade count
        self.total_turnover = 0.0  # Reset turnover
        self.trade_logs = [] # Track trade details
        
        return self._get_observation(), {}

    def step(self, action):
        # Unpack actions
        # action is now [ce_action, pe_action]
        # 0=Hold, 1=Add(Buy), 2=Offload(Sell)
        ce_action = action[0]
        pe_action = action[1]
        
        # --- Ratio Maintenance Logic (Prevent Naked Positions) ---
        # Calculate tentative future lots
        future_ce_lots = self.ce_lots
        if ce_action == 1: future_ce_lots += 1 #Buy
        elif ce_action == 2 and self.ce_lots > 0: future_ce_lots -= 1 #sell
        
        future_pe_lots = self.pe_lots
        if pe_action == 1: future_pe_lots += 1 #Buy
        elif pe_action == 2 and self.pe_lots > 0: future_pe_lots -= 1 #sell
        
        # Check constraints
        is_naked_call = (future_ce_lots > 0 and future_pe_lots == 0)
        is_naked_put = (future_pe_lots > 0 and future_ce_lots == 0)
        
        if is_naked_call:
            # Revert action that caused naked call
            if self.pe_lots == 0 and pe_action != 1: 
                # Case: Trying to buy CE but not PE. Block CE buy.
                if ce_action == 1: ce_action = 0 # Force Hold
            elif self.pe_lots > 0 and future_pe_lots == 0:
                 # Case: Selling last PE while keeping CE. Block PE sell.
                 if pe_action == 2: pe_action = 0 # Force Hold
                 
            # Double check if still naked (e.g. if we reverted one but other condition persists?)
            # Actually, simpler logic:
            # If Result is Naked Call -> Cancel the CE Buy OR Cancel the PE Sell
        
        if is_naked_put:
             if self.ce_lots == 0 and ce_action != 1:
                 if pe_action == 1: pe_action = 0
             elif self.ce_lots > 0 and future_ce_lots == 0:
                 if ce_action == 2: ce_action = 0
                 
        # Re-calc check for safety/simplicity in one block:
        # If [1, 0] from [0, 0] -> Future [1, 0] (Naked Call). 
        #   Action causing it is CE=1. Revert CE=1 to 0.
        # If [2, 0] from [1, 1] -> Future [0, 1] (Naked Put).
        #   Action causing it is CE=2. Revert CE=2 to 0.
        
        # Final Robust Implementation:
        future_ce_lots = self.ce_lots + (1 if ce_action==1 else (-1 if (ce_action==2 and self.ce_lots>0) else 0))
        future_pe_lots = self.pe_lots + (1 if pe_action==1 else (-1 if (pe_action==2 and self.pe_lots>0) else 0))
        
        if future_ce_lots > 0 and future_pe_lots == 0:
            # Block the move creating asymmetry
            # If we bought CE, cancel buy
            if ce_action == 1 and self.ce_lots == future_ce_lots - 1: ce_action = 0
            # If we sold PE, cancel sell
            if pe_action == 2 and self.pe_lots == future_pe_lots + 1: pe_action = 0
            
        if future_pe_lots > 0 and future_ce_lots == 0:
            if pe_action == 1 and self.pe_lots == future_pe_lots - 1: pe_action = 0
            if ce_action == 2 and self.ce_lots == future_ce_lots + 1: ce_action = 0
        # ---------------------------------------------------------
        
        current_step_row = self.day_data.iloc[self.current_step]
        current_price = current_step_row['close']
        current_time = current_step_row['datetime']
        time_to_expiry = self._get_time_to_expiry(current_time)
        
        # Dynamic IV
        current_iv = self._get_current_volatility()
        
        # Calculate current option prices
        ce_bs_curr = black_scholes(current_price, self.ce_strike, time_to_expiry, config.RISK_FREE_RATE, current_iv, 'call')
        pe_bs_curr = black_scholes(current_price, self.pe_strike, time_to_expiry, config.RISK_FREE_RATE, current_iv, 'put')
        ce_price_curr = ce_bs_curr['price']
        pe_price_curr = pe_bs_curr['price']
        
        # Calculate Current Investment (Margin Used)
        current_investment = (self.entry_price_ce * self.ce_lots * config.LOT_SIZE) + \
                             (self.entry_price_pe * self.pe_lots * config.LOT_SIZE)
        
        # Available Capital = Initial + Realized PnL - Current Investment
        # (This is a simplified cash tracking. Real available cash is Initial - Net Cash Outflow + Net Cash Inflow)
        # Better: Available = Total Equity - Current Used
        # Total Equity = Initial + Realized PnL
        total_equity = config.INITIAL_CAPITAL + self.realized_pnl
        available_cash = total_equity - current_investment
        
        # --- EXECUTE CALL ACTION ---
        # ce_action == 0 (HOLD)
        if ce_action == 1: # ADD (BUY)
            cost = ce_price_curr * 1 * config.LOT_SIZE
            transaction_cost = cost * config.TRANSACTION_COST_PCT
            cost += transaction_cost
            if self.ce_lots < config.MAX_LOTS and cost <= available_cash:
                # Weighted Average Price
                total_cost = (self.entry_price_ce * self.ce_lots) + (ce_price_curr * 1)
                self.ce_lots += 1
                self.total_trades += 1  # Track CE buy
                self.entry_price_ce = total_cost / self.ce_lots
                
                # Update Available Cash immediately for next leg check
                available_cash -= cost
                self.realized_pnl -= transaction_cost # Transaction cost is immediate loss
                
                self.trade_logs.append({
                    'timestamp': current_time,
                    'leg': 'CE',
                    'action': 'BUY',
                    'strike': self.ce_strike,
                    'expiry': self._get_expiry_date_str(current_time),
                    'price': ce_price_curr,
                    'qty': config.LOT_SIZE,
                    'pnl': 0.0
                })
            # Else: Ignore action (Hold) due to margin/limit
            
        elif ce_action == 2: # OFFLOAD (SELL)
            if self.ce_lots > 0:
                pnl_per_lot = (ce_price_curr - self.entry_price_ce) * 1 * config.LOT_SIZE
                if config.STRATEGY_TYPE == 'SHORT': 
                     pnl_per_lot = (self.entry_price_ce - ce_price_curr) * 1 * config.LOT_SIZE
                
                transaction_cost = (ce_price_curr * 1 * config.LOT_SIZE) * config.TRANSACTION_COST_PCT
                pnl_per_lot -= transaction_cost
                
                self.realized_pnl += pnl_per_lot
                self.total_turnover += abs(pnl_per_lot)  # Track CE turnover
                self.ce_lots -= 1
                self.total_trades += 1  # Track CE sell
                if self.ce_lots == 0: self.entry_price_ce = 0
                
                # Update Available Cash (released margin + pnl)
                # Recovered = Entry Cost (+/-) PnL -> No, just current value sold
                # For long: we get back (Price * Size)
                proceeds = ce_price_curr * config.LOT_SIZE
                available_cash += proceeds
                
                # Log Trade
                self.trade_logs.append({
                    'timestamp': current_time,
                    'leg': 'CE',
                    'action': 'SELL',
                    'strike': self.ce_strike,
                    'expiry': self._get_expiry_date_str(current_time),
                    'price': ce_price_curr,
                    'qty': config.LOT_SIZE,
                    'pnl': pnl_per_lot
                })

        # --- EXECUTE PUT ACTION ---
        # pe_action == 0 (HOLD)
        if pe_action == 1: # ADD (BUY)
            cost = pe_price_curr * 1 * config.LOT_SIZE
            transaction_cost = cost * config.TRANSACTION_COST_PCT
            cost += transaction_cost
            if self.pe_lots < config.MAX_LOTS and cost <= available_cash:
                total_cost = (self.entry_price_pe * self.pe_lots) + (pe_price_curr * 1)
                self.pe_lots += 1
                self.total_trades += 1  # Track PE buy
                self.entry_price_pe = total_cost / self.pe_lots
                available_cash -= cost
                self.realized_pnl -= transaction_cost
                
                self.trade_logs.append({
                    'timestamp': current_time,
                    'leg': 'PE',
                    'action': 'BUY',
                    'strike': self.pe_strike,
                    'expiry': self._get_expiry_date_str(current_time),
                    'price': pe_price_curr,
                    'qty': config.LOT_SIZE,
                    'pnl': 0.0
                })
                
        elif pe_action == 2: # OFFLOAD (SELL)
            if self.pe_lots > 0:
                pnl_per_lot = (pe_price_curr - self.entry_price_pe) * 1 * config.LOT_SIZE
                if config.STRATEGY_TYPE == 'SHORT':
                     pnl_per_lot = (self.entry_price_pe - pe_price_curr) * 1 * config.LOT_SIZE
                
                transaction_cost = (pe_price_curr * 1 * config.LOT_SIZE) * config.TRANSACTION_COST_PCT
                pnl_per_lot -= transaction_cost

                self.realized_pnl += pnl_per_lot
                self.total_turnover += abs(pnl_per_lot)  # Track PE turnover
                self.pe_lots -= 1
                self.total_trades += 1  # Track PE sell
                if self.pe_lots == 0: self.entry_price_pe = 0
                
                proceeds = pe_price_curr * config.LOT_SIZE
                available_cash += proceeds
                
                self.trade_logs.append({
                    'timestamp': current_time,
                    'leg': 'PE',
                    'action': 'SELL',
                    'strike': self.pe_strike,
                    'expiry': self._get_expiry_date_str(current_time),
                    'price': pe_price_curr,
                    'qty': config.LOT_SIZE,
                    'pnl': pnl_per_lot
                })

        # Recalculate Investment for Max Tracking
        new_investment = (self.entry_price_ce * self.ce_lots * config.LOT_SIZE) + \
                         (self.entry_price_pe * self.pe_lots * config.LOT_SIZE)
        if new_investment > self.premium_deployed:
            self.premium_deployed = new_investment

        # --- ADVANCE TIME ---
        self.current_step += 1
        done = False
        truncated = False
        
        # Append current row into incremental history (trim to 6000 rows to bound memory)
        row = self.day_data.iloc[self.current_step - 1]
        self._hist_closes.append(float(row['close']))
        self._hist_highs.append(float(row['high']))
        self._hist_lows.append(float(row['low']))
        self._hist_opens.append(float(row['open']))
        if len(self._hist_closes) > 6000:
            self._hist_closes.pop(0)
            self._hist_highs.pop(0)
            self._hist_lows.pop(0)
            self._hist_opens.pop(0)
        
        # Update day high/low
        if self.current_step < len(self.day_data):
            next_row = self.day_data.iloc[self.current_step]
            self.day_high = max(self.day_high, next_row['high'])
            self.day_low = min(self.day_low, next_row['low'])
            
            # Recalculate prices for PnL
            current_time = next_row['datetime']
            current_price = next_row['close']
            time_to_expiry = self._get_time_to_expiry(current_time)
            
            # Dynamic IV for PnL update
            # Note: _get_current_volatility depends on self.current_step, which was just incremented.
            # So this will use the volatility including the new step.
            current_iv = self._get_current_volatility()
            
            ce_price = black_scholes(current_price, self.ce_strike, time_to_expiry, config.RISK_FREE_RATE, current_iv, 'call')['price']
            pe_price = black_scholes(current_price, self.pe_strike, time_to_expiry, config.RISK_FREE_RATE, current_iv, 'put')['price']
        else:
            # End of data
            done = True
            ce_price = ce_price_curr
            pe_price = pe_price_curr
        
        # Episode ends at weekly expiry time or end of data — NOT at daily EOD
        if done or current_time >= self.episode_expiry_dt:
            done = True
            # Graceful close at expiry — treated as a normal trade with transaction costs (no penalty)
            pnl_ce = (ce_price - self.entry_price_ce) * self.ce_lots * config.LOT_SIZE
            pnl_pe = (pe_price - self.entry_price_pe) * self.pe_lots * config.LOT_SIZE
            if config.STRATEGY_TYPE == 'SHORT':
                pnl_ce = (self.entry_price_ce - ce_price) * self.ce_lots * config.LOT_SIZE
                pnl_pe = (self.entry_price_pe - pe_price) * self.pe_lots * config.LOT_SIZE
            cost_ce = (ce_price * self.ce_lots * config.LOT_SIZE) * config.TRANSACTION_COST_PCT
            cost_pe = (pe_price * self.pe_lots * config.LOT_SIZE) * config.TRANSACTION_COST_PCT
            pnl_ce -= cost_ce
            pnl_pe -= cost_pe
            self.realized_pnl += (pnl_ce + pnl_pe)
            self.total_turnover += abs(pnl_ce) + abs(pnl_pe)
            if self.ce_lots > 0 or self.pe_lots > 0:
                self.total_trades += self.ce_lots + self.pe_lots
            self.ce_lots = 0
            self.pe_lots = 0
            
        # --- CALCULATE REWARD ---
        # Unrealized PnL
        unrealized_ce = (ce_price - self.entry_price_ce) * self.ce_lots * config.LOT_SIZE
        unrealized_pe = (pe_price - self.entry_price_pe) * self.pe_lots * config.LOT_SIZE
        
        if config.STRATEGY_TYPE == 'SHORT':
             unrealized_ce = (self.entry_price_ce - ce_price) * self.ce_lots * config.LOT_SIZE
             unrealized_pe = (self.entry_price_pe - pe_price) * self.pe_lots * config.LOT_SIZE
        
        prev_total_pnl = self.total_pnl  # Save previous PnL for step_pnl
        self.total_pnl = self.realized_pnl + unrealized_ce + unrealized_pe
        self.pnl_curve.append(self.total_pnl)
        
        # Step PnL: change in total PnL from last step
        step_pnl = self.total_pnl - prev_total_pnl
        
        # Track Peak PnL and Max Drawdown
        if self.total_pnl > self.peak_pnl: 
            self.peak_pnl = self.total_pnl
            
        current_drawdown = self.peak_pnl - self.total_pnl
        if current_drawdown > self.max_drawdown: 
            self.max_drawdown = current_drawdown
        
        # Reward Formula (all terms normalised by INITIAL_CAPITAL for stable scale)
        # ─────────────────────────────────────────────────────────────────────────
        # denom: fixed at INITIAL_CAPITAL so reward scale is consistent across all
        # steps and episodes (avoids exploding rewards at step 0 when no capital deployed yet)
        denom = config.INITIAL_CAPITAL
        
        # 1. Cumulative ROI: encourages profit growth across the full episode
        roi_term = config.PNL_LAMBDA * (self.total_pnl / denom)
        
        # 2. Step PnL: immediate feedback — teaches which moves help right now
        step_term = config.STEP_PNL_LAMBDA * (step_pnl / denom)
        
        # 3. Current drawdown penalty (NOT max): agent can recover from a dip without
        #    being permanently punished all episode. Discourages staying in a losing position.
        current_drawdown = self.peak_pnl - self.total_pnl
        dd_term = config.DRAWDOWN_LAMBDA * (current_drawdown / denom)
        
        # 4. Overtrading penalty: normalised by MAX_LOTS so it scales correctly
        #    regardless of episode length (works for both 375-step and 1500-step episodes)
        trade_term = config.TRADE_PENALTY_LAMBDA * (self.total_trades / max(config.MAX_LOTS, 1))
        
        # 5. Turnover penalty: notional capital churned, NOT abs(pnl)
        #    Discourages churning the book without adding value
        notional_traded = self.total_turnover  # already tracked as raw notional in step()
        to_term = config.TURNOVER_LAMBDA * (notional_traded / denom)
        
        reward = roi_term + step_term - dd_term - trade_term - to_term
        
        # Info
        info = {
            'premium_deployed': self.premium_deployed,
            'realized_pnl': self.realized_pnl,
            'roi': self.total_pnl / denom,
            'step_pnl': step_pnl,
            'current_drawdown_pct': current_drawdown / denom,
            'max_drawdown_pct': self.max_drawdown / denom,
            'total_trades': self.total_trades,
            'turnover': notional_traded
        }
        
        return self._get_observation(), reward, done, truncated, info

    def _get_expiry_date_str(self, current_time):
        """Helper to get expiry date string"""
        current_day_idx = current_time.weekday()
        expiry_day_idx = config.EXPIRY_DAY_OF_WEEK
        delta_days = 0
        if current_day_idx < expiry_day_idx:
            delta_days = expiry_day_idx - current_day_idx
        elif current_day_idx > expiry_day_idx:
            delta_days = 7 - (current_day_idx - expiry_day_idx)
        
        expiry_date = current_time.date() + timedelta(days=delta_days)
        return str(expiry_date)

    def _get_time_to_expiry(self, current_time):
        """
        Calculate time to expiry in years.
        Supports REAL weekly expiry logic.
        """
        # 1. Intra-day time to close (minutes)
        end_time_today = pd.to_datetime(f"{current_time.date()} {config.END_TIME}")
        if current_time > end_time_today:
            minutes_to_close_today = 0
        else:
            minutes_to_close_today = (end_time_today - current_time).total_seconds() / 60
        
        # 2. Days to Expiry Calculation
        current_day_idx = current_time.weekday() # Mon=0, Sun=6
        expiry_day_idx = config.EXPIRY_DAY_OF_WEEK
        
        delta_days = 0
        if current_day_idx == expiry_day_idx:
            # It's expiry day!
            delta_days = 0
        elif current_day_idx < expiry_day_idx:
            # e.g. Mon (0) -> Thu (3) = 3 days
            delta_days = expiry_day_idx - current_day_idx
        else:
            # e.g. Fri (4) -> Thu (3) = 6 days (next week)
            delta_days = 7 - (current_day_idx - expiry_day_idx)
            
        # Total minutes remaining
        # (days - 1 full days) * minutes_per_day + today's remaining minutes + expiry day full minutes?
        # Simpler approach:
        # If delta_days = 0: just minutes_to_close_today
        # If delta_days > 0: minutes_to_close_today + (delta_days * 1440)? NO, trading minutes (390)
        
        # We model "annualized trading time". 
        # T = (Minutes Remaining) / (252 * 390)
        # But for overnight, do we count full time or just trading time?
        # BSM usually takes "time to maturity" in years. 
        # Standard practice: Use calendar days or trading days.
        # Let's stick to 'Trading Minutes' for consistency with intraday vol.
        
        total_minutes_remaining = minutes_to_close_today + (delta_days * 390.0) # approx 6.5 hours per extra day
        
        # Convert to years (252 trading days * 390 minutes)
        # Note: If delta_days=0, it's 0DTE behavior as before.
        time_to_expiry = max(total_minutes_remaining / (252 * 390), 1e-6)
        
        return time_to_expiry

    def _get_observation(self):
        """
        Calculate all 65 features from real OHLC data.
        NO simulated data - everything derived from actual market data.
        """
        obs = np.zeros(65, dtype=np.float32)
        idx = 0
        
        # Use pre-built incremental history arrays — O(1) access vs O(n) DataFrame slice
        closes = np.array(self._hist_closes, dtype=np.float32)
        highs  = np.array(self._hist_highs,  dtype=np.float32)
        lows   = np.array(self._hist_lows,   dtype=np.float32)
        opens  = np.array(self._hist_opens,  dtype=np.float32)
        
        if len(closes) == 0:
            return obs
        
        current_price = closes[-1]
        current_time = self.day_data.iloc[self.current_step - 1]['datetime']
        
        # 1. Volatility Features (12 features) - CRITICAL
        vol_features = self.feature_calc.calculate_volatility_features(
            closes, highs, lows, opens, closes
        )
        obs[idx] = vol_features.get('rolling_vol_5min', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_15min', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_30min', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_60min', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_1day', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_7day', 0.0); idx += 1
        obs[idx] = vol_features.get('rolling_vol_15day', 0.0); idx += 1
        obs[idx] = vol_features.get('parkinson_vol', 0.0); idx += 1
        obs[idx] = vol_features.get('garman_klass_vol', 0.0); idx += 1
        obs[idx] = vol_features.get('vol_percentile', 0.5); idx += 1
        obs[idx] = vol_features.get('vol_of_vol', 0.0); idx += 1
        obs[idx] = vol_features.get('vol_trend', 0.0); idx += 1
        # IV vs Realized spread (Variance Risk Premium)
        realized_vol = vol_features.get('rolling_vol_30min', 0.0)
        current_iv = self._get_current_volatility()
        # If IV < Realized, options are relatively cheap (good for long straddle)
        obs[idx] = current_iv - realized_vol; idx += 1
        # Feed the actual IV (VIX) to the agent
        obs[idx] = current_iv; idx += 1
        # Padding to maintain 55 features
        obs[idx] = 0.0; idx += 1
        
        # 2. Price & Returns (10 features)
        price_features = self.feature_calc.calculate_price_features(
            closes, highs, lows, self.day_open, self.day_high, self.day_low
        )
        obs[idx] = price_features.get('close_normalized', 0.0); idx += 1
        obs[idx] = price_features.get('return_1min', 0.0); idx += 1
        obs[idx] = price_features.get('return_5min', 0.0); idx += 1
        obs[idx] = price_features.get('return_15min', 0.0); idx += 1
        obs[idx] = price_features.get('return_30min', 0.0); idx += 1
        obs[idx] = price_features.get('vwap_distance', 0.0); idx += 1
        obs[idx] = price_features.get('distance_from_day_high', 0.0); idx += 1
        obs[idx] = price_features.get('distance_from_day_low', 0.0); idx += 1
        obs[idx] = price_features.get('price_momentum', 0.0); idx += 1
        obs[idx] = 0.0; idx += 1  # Padding
        
        # 3. Greeks (10 features) - Using Black-Scholes
        time_to_expiry = self._get_time_to_expiry(current_time)
        
        # Use Dynamic IV for consistent Greeks
        current_iv = self._get_current_volatility()
        
        ce_greeks = black_scholes(current_price, self.ce_strike, time_to_expiry,
                                 config.RISK_FREE_RATE, current_iv, 'call')
        pe_greeks = black_scholes(current_price, self.pe_strike, time_to_expiry,
                                 config.RISK_FREE_RATE, current_iv, 'put')
        
        position_greeks = calculate_position_greeks(
            ce_greeks, pe_greeks, self.ce_lots, self.pe_lots, config.LOT_SIZE
        )
        
        obs[idx] = ce_greeks['delta']; idx += 1
        obs[idx] = pe_greeks['delta']; idx += 1
        obs[idx] = ce_greeks['gamma']; idx += 1
        obs[idx] = pe_greeks['gamma']; idx += 1
        obs[idx] = ce_greeks['vega']; idx += 1
        obs[idx] = pe_greeks['vega']; idx += 1
        obs[idx] = ce_greeks['theta']; idx += 1
        obs[idx] = pe_greeks['theta']; idx += 1
        obs[idx] = ce_greeks['vanna']; idx += 1  
        obs[idx] = pe_greeks['vanna']; idx += 1  
        obs[idx] = ce_greeks['volga']; idx += 1  
        obs[idx] = pe_greeks['volga']; idx += 1  
        obs[idx] = position_greeks['net_delta'] / 10000.0; idx += 1  # Normalized
        obs[idx] = position_greeks['gamma_exposure'] / 1000.0; idx += 1  # Normalized
        obs[idx] = position_greeks['vega_exposure'] / 10000.0; idx += 1  # Normalized
        
        # 4. Technical Indicators (6 features)
        tech_features = self.feature_calc.calculate_technical_indicators(
            closes, highs, lows
        )
        obs[idx] = tech_features.get('rsi', 0.5); idx += 1
        obs[idx] = tech_features.get('macd_line', 0.0); idx += 1
        obs[idx] = tech_features.get('macd_signal', 0.0); idx += 1
        obs[idx] = tech_features.get('macd_histogram', 0.0); idx += 1
        obs[idx] = tech_features.get('bollinger_position', 0.0); idx += 1
        obs[idx] = tech_features.get('atr', 0.0); idx += 1
        
        # 5. Position Features (10 features)
        obs[idx] = self.ce_lots / config.MAX_LOTS; idx += 1  # Normalized
        obs[idx] = self.pe_lots / config.MAX_LOTS; idx += 1  # Normalized
        ce_pe_ratio = self.ce_lots / (self.pe_lots + 1e-8)
        obs[idx] = np.clip(ce_pe_ratio / 5.0, -1, 1); idx += 1  # Normalized
        
        # Premium deployed
        premium = (ce_greeks['price'] * self.ce_lots + 
                  pe_greeks['price'] * self.pe_lots) * config.LOT_SIZE
        obs[idx] = premium / config.INITIAL_CAPITAL; idx += 1
        
        # Strike distances
        obs[idx] = (self.ce_strike - current_price) / current_price; idx += 1
        obs[idx] = (current_price - self.pe_strike) / current_price; idx += 1
        
        # Entry prices (normalized)
        obs[idx] = self.entry_price_ce / current_price; idx += 1
        obs[idx] = self.entry_price_pe / current_price; idx += 1
        
        # PnL
        current_pnl = self.pnl_curve[-1] if self.pnl_curve else 0.0
        obs[idx] = current_pnl / config.INITIAL_CAPITAL; idx += 1
        obs[idx] = (current_pnl / premium) if premium > 1e-8 else 0.0; idx += 1
        
        # 6. Time Features (4 features)
        start_time = pd.to_datetime(f"{current_time.date()} {config.START_TIME}")
        end_time = pd.to_datetime(f"{current_time.date()} {config.END_TIME}")
        time_features = self.feature_calc.calculate_time_features(
            current_time, start_time, end_time
        )
        obs[idx] = time_features['minutes_since_open']; idx += 1
        obs[idx] = time_features['minutes_to_close']; idx += 1
        obs[idx] = time_features['time_sine']; idx += 1
        obs[idx] = time_features['time_cosine']; idx += 1
        
        # 7. Risk Metrics (3 features)
        obs[idx] = (self.peak_pnl - current_pnl) / config.INITIAL_CAPITAL; idx += 1
        obs[idx] = self.max_drawdown / config.INITIAL_CAPITAL; idx += 1
        
        # Rolling Sharpe (simplified)
        if len(self.pnl_curve) >= 30:
            pnl_returns = np.diff(self.pnl_curve[-30:])
            sharpe = (np.mean(pnl_returns) / (np.std(pnl_returns) + 1e-8)) * np.sqrt(390)
            obs[idx] = np.clip(sharpe / 5.0, -1, 1); idx += 1
        else:
            obs[idx] = 0.0; idx += 1
        
        # Replace NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 8. Multi-day episode features (2 features) [indices 63-64]
        # days_to_expiry: normalized 0-1 (1.0 = full week remaining, 0.0 = at expiry)
        seconds_remaining = max((self.episode_expiry_dt - current_time).total_seconds(), 0)
        total_episode_seconds = config.MAX_EPISODE_DAYS * 24 * 3600
        obs[idx] = float(seconds_remaining / total_episode_seconds); idx += 1
        
        # is_overnight: 1.0 if we are outside market hours (between 15:15 and 09:15)
        hour = current_time.hour
        is_overnight = 1.0 if (hour >= 15 or hour < 9) else 0.0
        obs[idx] = is_overnight; idx += 1
        
        return obs
