"""
Intraday Option Environment v2 — Directional Straddle with Action Masking
=========================================================================
Key differences from v1:
  - Agent starts FLAT (no forced straddle entry)
  - Discrete(8) action space with action masking (MaskablePPO)
  - Supports LONG, SHORT, and BOTH position modes
  - Dynamic SPAN-like margin for SHORT positions (IV-based)
  - Clean reward: step_pnl + drawdown + terminal (3 terms only)
  - 67 observation features (65 from v1 + position_side + position_duration)
  - Weekly episodes Wed→Tue, natural expiry (no forced exit)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import Mask_PPO_config as config
from utils.black_scholes import black_scholes, calculate_position_greeks
from utils.feature_calculator import FeatureCalculator


class IntradayOptionEnvV2(gym.Env):
    """
    RL environment for directional straddle management.

    Action Space: Discrete(8)
        0 = HOLD
        1 = ENTER LONG   (buy 1 CE + 1 PE, from flat only)
        2 = ENTER SHORT  (sell 1 CE + 1 PE, from flat only)
        3 = EXIT ALL     (close entire position)
        4 = ADD CE       (add 1 CE lot)
        5 = ADD PE       (add 1 PE lot)
        6 = REDUCE CE    (remove 1 CE lot, keep >= 1)
        7 = REDUCE PE    (remove 1 PE lot, keep >= 1)
    """

    metadata = {'render.modes': ['human']}

    # Action constants
    HOLD = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    EXIT_ALL = 3
    ADD_CE = 4
    ADD_PE = 5
    REDUCE_CE = 6
    REDUCE_PE = 7

    def __init__(self, data_path='data/train.csv', vix_data_path='data/INDIA_VIX.csv',
                 start_date=None, end_date=None):
        super(IntradayOptionEnvV2, self).__init__()

        # ── Data Loading (identical to v1) ──
        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        if vix_data_path is not None:
            vix_df = pd.read_csv(vix_data_path)
            vix_df['datetime'] = pd.to_datetime(vix_df['datetime'])
            vix_col = 'vix'
            if 'vix' not in vix_df.columns:
                vix_col = 'close' if 'close' in vix_df.columns else vix_df.columns[1]
            vix_df = vix_df[['datetime', vix_col]].rename(columns={vix_col: 'vix'})
            self.df = pd.merge(self.df, vix_df, on='datetime', how='left')

        if start_date:
            self.df = self.df[self.df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            self.df = self.df[self.df['datetime'] <= pd.to_datetime(end_date)]

        all_dates = self.df['datetime'].dt.date.unique()
        wednesdays = [d for d in all_dates if d.weekday() == 2]
        self.dates = np.array(wednesdays) if len(wednesdays) > 10 else all_dates

        # ── Action & Observation Space ──
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32)

        # ── Feature Calculator ──
        self.feature_calc = FeatureCalculator(window_size=config.WINDOW_SIZE)

        # ── State Variables ──
        self.current_step = 0
        self.current_date_idx = 0
        self.day_data = None

        # Position state
        self.position_side = None  # None = flat, 'LONG', 'SHORT'
        self.ce_lots = 0
        self.pe_lots = 0
        self.ce_strike = 0
        self.pe_strike = 0
        self.entry_price_ce = 0.0
        self.entry_price_pe = 0.0

        # PnL tracking
        self.cash = config.INITIAL_CAPITAL
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.premium_deployed = 0.0
        self.pnl_curve = []
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.total_turnover = 0.0
        self.trade_logs = []

        # Margin tracking
        self._current_margin_used = 0.0

        # Flat/duration tracking
        self.flat_steps = 0
        self.entry_step = 0

        # Day OHLC
        self.day_open = 0.0
        self.day_high = 0.0
        self.day_low = float('inf')

        # IV cache
        self._cached_iv = 0.15
        self._cached_iv_step = -1

    # ─────────────────────────────────────────────────
    # VOLATILITY
    # ─────────────────────────────────────────────────

    def _get_current_volatility(self, vol_risk_premium=1.20):
        """Get IV using India VIX. Cached per step. Falls back to realized vol."""
        if self._cached_iv_step == self.current_step:
            return self._cached_iv

        # Use VIX[T-1] to prevent look-ahead
        if 'vix' in self.day_data.columns:
            vix_lookup_step = max(0, self.current_step - 1)
            vix_val = self.day_data.iloc[vix_lookup_step]['vix']
            if pd.notna(vix_val) and vix_val > 0:
                self._cached_iv = (vix_val / 100.0) * vol_risk_premium
                self._cached_iv_step = self.current_step
                return self._cached_iv

        # Fallback: historical realized vol
        closes = np.array(self._hist_closes, dtype=np.float32)
        highs = np.array(self._hist_highs, dtype=np.float32)
        lows = np.array(self._hist_lows, dtype=np.float32)
        opens = np.array(self._hist_opens, dtype=np.float32)

        if len(closes) < 5:
            return 0.15

        vol_features = self.feature_calc.calculate_volatility_features(
            closes, highs, lows, opens, closes
        )
        vol = vol_features.get('rolling_vol_15day', 0.0)
        if vol < 1e-4:
            vol = vol_features.get('rolling_vol_7day', 0.0)
        if vol < 1e-4:
            vol = vol_features.get('rolling_vol_1day', 0.0)
        if vol < 1e-4:
            vol = vol_features.get('rolling_vol_30min', 0.0)

        self._cached_iv = vol_risk_premium * vol
        self._cached_iv_step = self.current_step
        return self._cached_iv

    # ─────────────────────────────────────────────────
    # TIME TO EXPIRY
    # ─────────────────────────────────────────────────

    def _get_time_to_expiry(self, current_time):
        """Calculate time to expiry in years (trading-time based)."""
        end_time_today = pd.to_datetime(f"{current_time.date()} {config.END_TIME}")
        if current_time > end_time_today:
            minutes_to_close_today = 0
        else:
            minutes_to_close_today = (end_time_today - current_time).total_seconds() / 60

        current_day_idx = current_time.weekday()
        expiry_day_idx = config.EXPIRY_DAY_OF_WEEK

        if current_day_idx == expiry_day_idx:
            delta_days = 0
        elif current_day_idx < expiry_day_idx:
            delta_days = expiry_day_idx - current_day_idx
        else:
            delta_days = 7 - (current_day_idx - expiry_day_idx)

        total_minutes_remaining = minutes_to_close_today + (delta_days * 390.0)
        time_to_expiry = max(total_minutes_remaining / (252 * 390), 1e-6)
        return time_to_expiry

    def _get_expiry_date_str(self, current_time):
        """Helper to get expiry date string."""
        current_day_idx = current_time.weekday()
        expiry_day_idx = config.EXPIRY_DAY_OF_WEEK
        if current_day_idx < expiry_day_idx:
            delta_days = expiry_day_idx - current_day_idx
        elif current_day_idx > expiry_day_idx:
            delta_days = 7 - (current_day_idx - expiry_day_idx)
        else:
            delta_days = 0
        expiry_date = current_time.date() + timedelta(days=delta_days)
        return str(expiry_date)

    # ─────────────────────────────────────────────────
    # MARGIN CALCULATION
    # ─────────────────────────────────────────────────

    def _calculate_long_margin_per_lot(self, bs_price):
        """LONG margin = premium paid."""
        return bs_price * config.LOT_SIZE

    def _calculate_short_margin_per_lot(self, spot, strike, iv, time_to_expiry, option_type):
        """
        Dynamic SPAN-like margin for SHORT positions.
        Uses current IV for price scan → higher IV = higher margin.
        """
        # Daily vol from annualized IV
        daily_vol = max(iv / np.sqrt(252), 1e-6)

        # Price scan: worst-case move
        price_scan = spot * daily_vol * np.sqrt(config.SPAN_MPOR_DAYS) * config.SPAN_STDEV_COVERAGE

        # Stressed spot
        if option_type == 'call':
            stressed_spot = spot + price_scan
        else:
            stressed_spot = max(spot - price_scan, 1.0)

        # Stressed IV
        stressed_iv = iv * (1 + config.SPAN_VOL_STRESS)

        # Worst-case option price
        stressed_price = black_scholes(
            stressed_spot, strike, time_to_expiry,
            config.RISK_FREE_RATE, stressed_iv, option_type
        )['price']

        current_price = black_scholes(
            spot, strike, time_to_expiry,
            config.RISK_FREE_RATE, iv, option_type
        )['price']

        # SPAN = worst-case loss
        span_margin = max(stressed_price - current_price, 0) * config.LOT_SIZE

        # Exposure = fixed regulatory
        exposure_margin = config.EXPOSURE_MARGIN_PCT * spot * config.LOT_SIZE

        # Minimum = premium received
        return max(span_margin + exposure_margin, current_price * config.LOT_SIZE)

    def _calculate_margin_for_lot(self, spot, strike, iv, tte, option_type, side):
        """Calculate margin for 1 lot given position side."""
        bs = black_scholes(spot, strike, tte, config.RISK_FREE_RATE, iv, option_type)
        if side == 'LONG':
            return self._calculate_long_margin_per_lot(bs['price'])
        else:
            return self._calculate_short_margin_per_lot(spot, strike, iv, tte, option_type)

    def _recalculate_total_margin(self, spot, iv, tte):
        """Recalculate total margin for current position."""
        if self.position_side is None or (self.ce_lots == 0 and self.pe_lots == 0):
            self._current_margin_used = 0.0
            return

        ce_margin = self._calculate_margin_for_lot(
            spot, self.ce_strike, iv, tte, 'call', self.position_side
        ) * self.ce_lots
        pe_margin = self._calculate_margin_for_lot(
            spot, self.pe_strike, iv, tte, 'put', self.position_side
        ) * self.pe_lots
        self._current_margin_used = ce_margin + pe_margin

    def _get_available_cash(self):
        """Available cash = equity - margin used."""
        total_equity = config.INITIAL_CAPITAL + self.realized_pnl
        return total_equity - self._current_margin_used

    # ─────────────────────────────────────────────────
    # ACTION MASKS
    # ─────────────────────────────────────────────────

    def action_masks(self):
        """Return boolean mask of valid actions for current state."""
        mask = np.zeros(8, dtype=bool)
        mask[self.HOLD] = True  # Always valid

        if self.current_step >= len(self.day_data):
            return mask  # End of data — only HOLD

        current_row = self.day_data.iloc[self.current_step]
        spot = current_row['close']
        iv = self._get_current_volatility()
        tte = self._get_time_to_expiry(current_row['datetime'])
        atm = round(spot / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
        avail = self._get_available_cash()

        if self.ce_lots == 0 and self.pe_lots == 0:
            # FLAT — can enter
            if config.POSITION_MODE in ('LONG_ONLY', 'BOTH'):
                ce_price = black_scholes(spot, atm, tte, config.RISK_FREE_RATE, iv, 'call')['price']
                pe_price = black_scholes(spot, atm, tte, config.RISK_FREE_RATE, iv, 'put')['price']
                long_margin = self._calculate_long_margin_per_lot(ce_price) + \
                              self._calculate_long_margin_per_lot(pe_price)
                if long_margin <= avail:
                    mask[self.ENTER_LONG] = True

            if config.POSITION_MODE in ('SHORT_ONLY', 'BOTH'):
                short_margin = self._calculate_short_margin_per_lot(spot, atm, iv, tte, 'call') + \
                               self._calculate_short_margin_per_lot(spot, atm, iv, tte, 'put')
                if short_margin <= avail:
                    mask[self.ENTER_SHORT] = True
        else:
            # IN POSITION
            mask[self.EXIT_ALL] = True

            # ADD CE
            if self.ce_lots < config.MAX_LOTS:
                add_margin = self._calculate_margin_for_lot(
                    spot, self.ce_strike, iv, tte, 'call', self.position_side
                )
                if add_margin <= avail:
                    mask[self.ADD_CE] = True

            # ADD PE
            if self.pe_lots < config.MAX_LOTS:
                add_margin = self._calculate_margin_for_lot(
                    spot, self.pe_strike, iv, tte, 'put', self.position_side
                )
                if add_margin <= avail:
                    mask[self.ADD_PE] = True

            # REDUCE (keep >= 1)
            if self.ce_lots > 1:
                mask[self.REDUCE_CE] = True
            if self.pe_lots > 1:
                mask[self.REDUCE_PE] = True

        return mask

    # ─────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick date
        if options and 'date_index' in options:
            self.current_date_idx = options['date_index']
        else:
            self.current_date_idx = np.random.randint(0, len(self.dates))
        date = self.dates[self.current_date_idx]

        # Multi-day episode: Wed → Tue
        import datetime as dt_lib
        expiry_day_num = config.EXPIRY_DAY_OF_WEEK
        days_ahead = (expiry_day_num - date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        expiry_date = date + dt_lib.timedelta(days=days_ahead)
        max_end = date + dt_lib.timedelta(days=config.MAX_EPISODE_DAYS)
        expiry_date = min(expiry_date, max_end)
        self.episode_expiry_dt = pd.to_datetime(f"{expiry_date} {config.END_TIME}")

        # Load data
        mask = (self.df['datetime'].dt.date >= date) & (self.df['datetime'].dt.date <= expiry_date)
        self.day_data = self.df[mask].reset_index(drop=True)

        # Historical window
        day_indices = self.df.index[self.df['datetime'].dt.date == date].tolist()
        self.global_start_idx = day_indices[0] if day_indices else 0
        pre_end = self.global_start_idx
        pre_start = max(0, pre_end - 6000)
        pre_hist = self.df.iloc[pre_start:pre_end]
        self._hist_closes = pre_hist['close'].values.tolist()
        self._hist_highs = pre_hist['high'].values.tolist()
        self._hist_lows = pre_hist['low'].values.tolist()
        self._hist_opens = pre_hist['open'].values.tolist()

        # Fast forward to START_TIME
        start_time = pd.to_datetime(f"{date} {config.START_TIME}")
        start_idx = self.day_data[self.day_data['datetime'] >= start_time].index

        if len(start_idx) == 0:
            return self.reset()

        self.current_step = start_idx[0]

        # Day OHLC
        self.day_open = self.day_data.iloc[self.current_step]['open']
        self.day_high = self.day_data.iloc[self.current_step]['high']
        self.day_low = self.day_data.iloc[self.current_step]['low']

        # Start FLAT
        self.position_side = None
        self.ce_lots = 0
        self.pe_lots = 0
        self.ce_strike = 0
        self.pe_strike = 0
        self.entry_price_ce = 0.0
        self.entry_price_pe = 0.0

        # PnL
        self.cash = config.INITIAL_CAPITAL
        self.pnl_curve = [0.0]
        self.realized_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.premium_deployed = 0.0
        self.total_trades = 0
        self.total_turnover = 0.0
        self.trade_logs = []

        # Margin
        self._current_margin_used = 0.0

        # Tracking
        self.flat_steps = 0
        self.entry_step = 0
        self.total_episode_steps = len(self.day_data) - self.current_step

        # IV cache reset
        self._cached_iv = 0.15
        self._cached_iv_step = -1

        return self._get_observation(), {}

    # ─────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────

    def step(self, action):
        action = int(action)

        current_row = self.day_data.iloc[self.current_step]
        spot = current_row['close']
        current_time = current_row['datetime']
        tte = self._get_time_to_expiry(current_time)
        iv = self._get_current_volatility()

        # ── EXECUTE ACTION ──

        if action == self.HOLD:
            pass

        elif action == self.ENTER_LONG:
            self._execute_enter(spot, iv, tte, current_time, 'LONG')

        elif action == self.ENTER_SHORT:
            self._execute_enter(spot, iv, tte, current_time, 'SHORT')

        elif action == self.EXIT_ALL:
            self._execute_exit_all(spot, iv, tte, current_time)

        elif action == self.ADD_CE:
            self._execute_add(spot, iv, tte, current_time, 'call')

        elif action == self.ADD_PE:
            self._execute_add(spot, iv, tte, current_time, 'put')

        elif action == self.REDUCE_CE:
            self._execute_reduce(spot, iv, tte, current_time, 'call')

        elif action == self.REDUCE_PE:
            self._execute_reduce(spot, iv, tte, current_time, 'put')

        # ── ADVANCE TIME ──
        self.current_step += 1
        done = False
        truncated = False

        # Append to history
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

        # Update day OHLC
        if self.current_step < len(self.day_data):
            next_row = self.day_data.iloc[self.current_step]
            self.day_high = max(self.day_high, next_row['high'])
            self.day_low = min(self.day_low, next_row['low'])
            current_time = next_row['datetime']
            spot = next_row['close']
            tte = self._get_time_to_expiry(current_time)
            iv = self._get_current_volatility()
        else:
            done = True

        # ── EPISODE END CHECK ──
        if not done and current_time >= self.episode_expiry_dt:
            done = True

        if done and self.position_side is not None:
            # Natural expiry — mark PnL at BS prices, NO transaction costs
            ce_price = black_scholes(spot, self.ce_strike, max(tte, 1e-6),
                                     config.RISK_FREE_RATE, iv, 'call')['price']
            pe_price = black_scholes(spot, self.pe_strike, max(tte, 1e-6),
                                     config.RISK_FREE_RATE, iv, 'put')['price']

            if self.position_side == 'LONG':
                final_pnl = (ce_price - self.entry_price_ce) * self.ce_lots * config.LOT_SIZE + \
                            (pe_price - self.entry_price_pe) * self.pe_lots * config.LOT_SIZE
            else:
                final_pnl = (self.entry_price_ce - ce_price) * self.ce_lots * config.LOT_SIZE + \
                            (self.entry_price_pe - pe_price) * self.pe_lots * config.LOT_SIZE

            self.realized_pnl += final_pnl
            self.total_turnover += abs(final_pnl)
            self.total_trades += self.ce_lots + self.pe_lots
            self.ce_lots = 0
            self.pe_lots = 0
            self.position_side = None
            self._current_margin_used = 0.0

        # ── CALCULATE PnL ──
        prev_total_pnl = self.total_pnl
        unrealized = 0.0

        if self.position_side is not None and self.ce_lots > 0:
            ce_price = black_scholes(spot, self.ce_strike, max(tte, 1e-6),
                                     config.RISK_FREE_RATE, iv, 'call')['price']
            pe_price = black_scholes(spot, self.pe_strike, max(tte, 1e-6),
                                     config.RISK_FREE_RATE, iv, 'put')['price']

            if self.position_side == 'LONG':
                unrealized = (ce_price - self.entry_price_ce) * self.ce_lots * config.LOT_SIZE + \
                             (pe_price - self.entry_price_pe) * self.pe_lots * config.LOT_SIZE
            else:
                unrealized = (self.entry_price_ce - ce_price) * self.ce_lots * config.LOT_SIZE + \
                             (self.entry_price_pe - pe_price) * self.pe_lots * config.LOT_SIZE

            # Recalculate margin for SHORT (changes with IV/spot)
            if self.position_side == 'SHORT':
                self._recalculate_total_margin(spot, iv, tte)

        self.total_pnl = self.realized_pnl + unrealized
        self.pnl_curve.append(self.total_pnl)

        step_pnl = self.total_pnl - prev_total_pnl

        # Drawdown tracking
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        current_drawdown = self.peak_pnl - self.total_pnl
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Flat step tracking
        if self.ce_lots == 0 and self.pe_lots == 0:
            self.flat_steps += 1
        else:
            self.flat_steps = 0

        # ── REWARD (3 terms) ──
        step_term = config.STEP_PNL_LAMBDA * (step_pnl / config.REWARD_NORMALIZER)
        dd_term = config.DRAWDOWN_LAMBDA * (current_drawdown / config.REWARD_NORMALIZER)

        terminal = 0.0
        if done:
            if self.total_pnl > 0:
                terminal = config.WIN_BONUS
            elif self.total_trades == 0:
                terminal = -config.NO_TRADE_PENALTY

        reward = step_term - dd_term + terminal

        # ── INFO ──
        info = {
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'step_pnl': step_pnl,
            'position_side': self.position_side,
            'ce_lots': self.ce_lots,
            'pe_lots': self.pe_lots,
            'total_trades': self.total_trades,
            'flat_steps': self.flat_steps,
            'max_drawdown': self.max_drawdown,
            'premium_deployed': self.premium_deployed,
            'current_margin': self._current_margin_used,
        }

        return self._get_observation(), reward, done, truncated, info

    # ─────────────────────────────────────────────────
    # ACTION EXECUTORS
    # ─────────────────────────────────────────────────

    def _execute_enter(self, spot, iv, tte, current_time, side):
        """Enter a new straddle position (LONG or SHORT)."""
        # Calculate ATM strikes
        atm = round(spot / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
        self.ce_strike = atm
        self.pe_strike = atm

        # BS prices
        ce_bs = black_scholes(spot, self.ce_strike, tte, config.RISK_FREE_RATE, iv, 'call')
        pe_bs = black_scholes(spot, self.pe_strike, tte, config.RISK_FREE_RATE, iv, 'put')

        self.entry_price_ce = ce_bs['price']
        self.entry_price_pe = pe_bs['price']
        self.ce_lots = 1
        self.pe_lots = 1
        self.position_side = side
        self.entry_step = self.current_step
        self.total_trades += 2

        # Margin
        if side == 'LONG':
            margin = self._calculate_long_margin_per_lot(self.entry_price_ce) + \
                     self._calculate_long_margin_per_lot(self.entry_price_pe)
        else:
            margin = self._calculate_short_margin_per_lot(spot, self.ce_strike, iv, tte, 'call') + \
                     self._calculate_short_margin_per_lot(spot, self.pe_strike, iv, tte, 'put')
        self._current_margin_used = margin

        # Premium deployed
        premium = (self.entry_price_ce + self.entry_price_pe) * config.LOT_SIZE
        if premium > self.premium_deployed:
            self.premium_deployed = premium

        # Transaction costs
        txn_cost = premium * config.TRANSACTION_COST_PCT
        self.realized_pnl -= txn_cost

        # Trade logs
        action_str = 'BUY' if side == 'LONG' else 'SELL'
        for leg, strike, price in [('CE', self.ce_strike, self.entry_price_ce),
                                   ('PE', self.pe_strike, self.entry_price_pe)]:
            self.trade_logs.append({
                'timestamp': current_time, 'leg': leg, 'action': action_str,
                'strike': strike, 'expiry': self._get_expiry_date_str(current_time),
                'price': price, 'qty': config.LOT_SIZE, 'pnl': 0.0, 'side': side
            })

    def _execute_exit_all(self, spot, iv, tte, current_time):
        """Close entire position."""
        ce_price = black_scholes(spot, self.ce_strike, tte, config.RISK_FREE_RATE, iv, 'call')['price']
        pe_price = black_scholes(spot, self.pe_strike, tte, config.RISK_FREE_RATE, iv, 'put')['price']

        # PnL
        if self.position_side == 'LONG':
            pnl_ce = (ce_price - self.entry_price_ce) * self.ce_lots * config.LOT_SIZE
            pnl_pe = (pe_price - self.entry_price_pe) * self.pe_lots * config.LOT_SIZE
        else:
            pnl_ce = (self.entry_price_ce - ce_price) * self.ce_lots * config.LOT_SIZE
            pnl_pe = (self.entry_price_pe - pe_price) * self.pe_lots * config.LOT_SIZE

        # Transaction costs
        txn_cost = (ce_price * self.ce_lots + pe_price * self.pe_lots) * \
                   config.LOT_SIZE * config.TRANSACTION_COST_PCT
        total_pnl_close = pnl_ce + pnl_pe - txn_cost

        self.realized_pnl += total_pnl_close
        self.total_turnover += abs(pnl_ce) + abs(pnl_pe)
        self.total_trades += self.ce_lots + self.pe_lots

        # Log
        action_str = 'SELL' if self.position_side == 'LONG' else 'BUY_BACK'
        for leg, strike, price, pnl in [('CE', self.ce_strike, ce_price, pnl_ce),
                                        ('PE', self.pe_strike, pe_price, pnl_pe)]:
            self.trade_logs.append({
                'timestamp': current_time, 'leg': leg, 'action': action_str,
                'strike': strike, 'expiry': self._get_expiry_date_str(current_time),
                'price': price, 'qty': config.LOT_SIZE * (self.ce_lots if leg == 'CE' else self.pe_lots),
                'pnl': pnl, 'side': self.position_side
            })

        # Reset
        self.ce_lots = 0
        self.pe_lots = 0
        self.position_side = None
        self.ce_strike = 0
        self.pe_strike = 0
        self.entry_price_ce = 0.0
        self.entry_price_pe = 0.0
        self._current_margin_used = 0.0

    def _execute_add(self, spot, iv, tte, current_time, option_type):
        """Add 1 lot to a leg."""
        strike = self.ce_strike if option_type == 'call' else self.pe_strike
        bs = black_scholes(spot, strike, tte, config.RISK_FREE_RATE, iv, option_type)
        new_price = bs['price']

        if option_type == 'call':
            # Weighted average entry
            total_cost = self.entry_price_ce * self.ce_lots + new_price
            self.ce_lots += 1
            self.entry_price_ce = total_cost / self.ce_lots
        else:
            total_cost = self.entry_price_pe * self.pe_lots + new_price
            self.pe_lots += 1
            self.entry_price_pe = total_cost / self.pe_lots

        self.total_trades += 1

        # Transaction cost
        txn_cost = new_price * config.LOT_SIZE * config.TRANSACTION_COST_PCT
        self.realized_pnl -= txn_cost

        # Update margin
        self._recalculate_total_margin(spot, iv, tte)

        # Premium tracking
        premium = (self.entry_price_ce * self.ce_lots + self.entry_price_pe * self.pe_lots) * config.LOT_SIZE
        if premium > self.premium_deployed:
            self.premium_deployed = premium

        # Log
        action_str = 'BUY' if self.position_side == 'LONG' else 'SELL'
        leg = 'CE' if option_type == 'call' else 'PE'
        self.trade_logs.append({
            'timestamp': current_time, 'leg': leg, 'action': action_str,
            'strike': strike, 'expiry': self._get_expiry_date_str(current_time),
            'price': new_price, 'qty': config.LOT_SIZE, 'pnl': 0.0, 'side': self.position_side
        })

    def _execute_reduce(self, spot, iv, tte, current_time, option_type):
        """Reduce 1 lot from a leg (keep >= 1)."""
        strike = self.ce_strike if option_type == 'call' else self.pe_strike
        entry_price = self.entry_price_ce if option_type == 'call' else self.entry_price_pe
        bs = black_scholes(spot, strike, tte, config.RISK_FREE_RATE, iv, option_type)
        current_price = bs['price']

        # PnL for 1 lot
        if self.position_side == 'LONG':
            pnl = (current_price - entry_price) * config.LOT_SIZE
        else:
            pnl = (entry_price - current_price) * config.LOT_SIZE

        # Transaction cost
        txn_cost = current_price * config.LOT_SIZE * config.TRANSACTION_COST_PCT
        pnl -= txn_cost

        self.realized_pnl += pnl
        self.total_turnover += abs(pnl + txn_cost)  # Gross PnL for turnover
        self.total_trades += 1

        if option_type == 'call':
            self.ce_lots -= 1
        else:
            self.pe_lots -= 1

        # Update margin
        self._recalculate_total_margin(spot, iv, tte)

        # Log
        action_str = 'SELL' if self.position_side == 'LONG' else 'BUY_BACK'
        leg = 'CE' if option_type == 'call' else 'PE'
        self.trade_logs.append({
            'timestamp': current_time, 'leg': leg, 'action': action_str,
            'strike': strike, 'expiry': self._get_expiry_date_str(current_time),
            'price': current_price, 'qty': config.LOT_SIZE, 'pnl': pnl, 'side': self.position_side
        })

    # ─────────────────────────────────────────────────
    # OBSERVATION (67 features)
    # ─────────────────────────────────────────────────

    def _get_observation(self):
        """Calculate all 67 features."""
        obs = np.zeros(67, dtype=np.float32)
        idx = 0

        closes = np.array(self._hist_closes, dtype=np.float32)
        highs = np.array(self._hist_highs, dtype=np.float32)
        lows = np.array(self._hist_lows, dtype=np.float32)
        opens = np.array(self._hist_opens, dtype=np.float32)

        if len(closes) == 0:
            return obs

        current_price = closes[-1]
        current_time = self.day_data.iloc[min(self.current_step, len(self.day_data) - 1)]['datetime']

        # ── 1. Volatility Features (15) ──
        vol_features = self.feature_calc.calculate_volatility_features(closes, highs, lows, opens, closes)
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
        realized_vol = vol_features.get('rolling_vol_30min', 0.0)
        current_iv = self._get_current_volatility()
        obs[idx] = current_iv - realized_vol; idx += 1  # VRP
        obs[idx] = current_iv; idx += 1
        obs[idx] = 0.0; idx += 1  # Padding

        # ── 2. Price & Returns (10) ──
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

        # ── 3. Greeks (15) ──
        time_to_expiry = self._get_time_to_expiry(current_time)

        if self.ce_strike > 0 and self.pe_strike > 0:
            ce_greeks = black_scholes(current_price, self.ce_strike, time_to_expiry,
                                     config.RISK_FREE_RATE, current_iv, 'call')
            pe_greeks = black_scholes(current_price, self.pe_strike, time_to_expiry,
                                     config.RISK_FREE_RATE, current_iv, 'put')
            position_greeks = calculate_position_greeks(
                ce_greeks, pe_greeks, self.ce_lots, self.pe_lots, config.LOT_SIZE
            )
        else:
            # Flat — use ATM for informational Greeks
            atm = round(current_price / config.STRADDLE_STRIKE_GAP) * config.STRADDLE_STRIKE_GAP
            ce_greeks = black_scholes(current_price, atm, time_to_expiry,
                                     config.RISK_FREE_RATE, current_iv, 'call')
            pe_greeks = black_scholes(current_price, atm, time_to_expiry,
                                     config.RISK_FREE_RATE, current_iv, 'put')
            position_greeks = {'net_delta': 0.0, 'gamma_exposure': 0.0, 'vega_exposure': 0.0}

        obs[idx] = np.clip(ce_greeks['delta'], -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['delta'], -1, 1); idx += 1
        obs[idx] = np.clip(ce_greeks['gamma'] * 100, -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['gamma'] * 100, -1, 1); idx += 1
        obs[idx] = np.clip(ce_greeks['vega'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['vega'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(ce_greeks['theta'] * 100, -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['theta'] * 100, -1, 1); idx += 1
        obs[idx] = np.clip(ce_greeks['vanna'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['vanna'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(ce_greeks['volga'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(pe_greeks['volga'] / 10, -1, 1); idx += 1
        obs[idx] = np.clip(position_greeks['net_delta'] / 10000.0, -1, 1); idx += 1
        obs[idx] = np.clip(position_greeks['gamma_exposure'] / 1000.0, -1, 1); idx += 1
        obs[idx] = np.clip(position_greeks['vega_exposure'] / 10000.0, -1, 1); idx += 1

        # ── 4. Technical Indicators (6) ──
        tech_features = self.feature_calc.calculate_technical_indicators(closes, highs, lows)
        obs[idx] = tech_features.get('rsi', 0.5); idx += 1
        obs[idx] = tech_features.get('macd_line', 0.0); idx += 1
        obs[idx] = tech_features.get('macd_signal', 0.0); idx += 1
        obs[idx] = tech_features.get('macd_histogram', 0.0); idx += 1
        obs[idx] = tech_features.get('bollinger_position', 0.0); idx += 1
        obs[idx] = tech_features.get('atr', 0.0); idx += 1

        # ── 5. Position Features (10) ──
        obs[idx] = self.ce_lots / config.MAX_LOTS; idx += 1
        obs[idx] = self.pe_lots / config.MAX_LOTS; idx += 1
        ce_pe_ratio = self.ce_lots / (self.pe_lots + 1e-8)
        obs[idx] = np.clip(ce_pe_ratio / 5.0, -1, 1); idx += 1

        # Premium deployed
        if self.ce_strike > 0 and self.pe_strike > 0 and (self.ce_lots > 0 or self.pe_lots > 0):
            premium = (ce_greeks['price'] * self.ce_lots +
                      pe_greeks['price'] * self.pe_lots) * config.LOT_SIZE
        else:
            premium = 0.0
        obs[idx] = premium / config.INITIAL_CAPITAL; idx += 1

        # Strike distances
        if self.ce_strike > 0:
            obs[idx] = (self.ce_strike - current_price) / current_price; idx += 1
            obs[idx] = (current_price - self.pe_strike) / current_price; idx += 1
            obs[idx] = self.entry_price_ce / current_price; idx += 1
            obs[idx] = self.entry_price_pe / current_price; idx += 1
        else:
            obs[idx] = 0.0; idx += 1
            obs[idx] = 0.0; idx += 1
            obs[idx] = 0.0; idx += 1
            obs[idx] = 0.0; idx += 1

        # PnL
        current_pnl = self.pnl_curve[-1] if self.pnl_curve else 0.0
        obs[idx] = current_pnl / config.REWARD_NORMALIZER; idx += 1
        obs[idx] = (current_pnl / premium) if premium > 1e-8 else 0.0; idx += 1

        # ── 6. Time Features (4) ──
        start_time = pd.to_datetime(f"{current_time.date()} {config.START_TIME}")
        end_time = pd.to_datetime(f"{current_time.date()} {config.END_TIME}")
        time_features = self.feature_calc.calculate_time_features(current_time, start_time, end_time)
        obs[idx] = time_features['minutes_since_open']; idx += 1
        obs[idx] = time_features['minutes_to_close']; idx += 1
        obs[idx] = time_features['time_sine']; idx += 1
        obs[idx] = time_features['time_cosine']; idx += 1

        # ── 7. Risk Metrics (3) ──
        obs[idx] = (self.peak_pnl - current_pnl) / config.REWARD_NORMALIZER; idx += 1
        obs[idx] = self.max_drawdown / config.REWARD_NORMALIZER; idx += 1

        if len(self.pnl_curve) >= 30:
            pnl_returns = np.diff(self.pnl_curve[-30:])
            sharpe = (np.mean(pnl_returns) / (np.std(pnl_returns) + 1e-8)) * np.sqrt(390)
            obs[idx] = np.clip(sharpe / 5.0, -1, 1); idx += 1
        else:
            obs[idx] = 0.0; idx += 1

        # ── 8. Multi-day Features (2) ──
        seconds_remaining = max((self.episode_expiry_dt - current_time).total_seconds(), 0)
        total_episode_seconds = config.MAX_EPISODE_DAYS * 24 * 3600
        obs[idx] = float(seconds_remaining / total_episode_seconds); idx += 1

        hour = current_time.hour
        is_overnight = 1.0 if (hour >= 15 or hour < 9) else 0.0
        obs[idx] = is_overnight; idx += 1

        # Replace NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # ── 9. NEW: Position State Features (2) ──
        # position_side: +1=LONG, -1=SHORT, 0=FLAT
        if self.position_side == 'LONG':
            obs[65] = 1.0
        elif self.position_side == 'SHORT':
            obs[65] = -1.0
        else:
            obs[65] = 0.0

        # position_duration: normalized steps since entry
        if self.position_side is not None and self.total_episode_steps > 0:
            obs[66] = min((self.current_step - self.entry_step) / self.total_episode_steps, 1.0)
        else:
            obs[66] = 0.0

        return obs
