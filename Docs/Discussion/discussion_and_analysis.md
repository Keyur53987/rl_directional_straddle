# Chapter 5: Discussion and Analysis

This chapter interprets the empirical results of Chapter 4, contextualizes them within the broader landscape of algorithmic trading research, and critically examines the system's design trade-offs.

## 5.1 Why the Reward Function Had to Evolve

The iterative failure of V1 and V2 reward functions was not merely a hyperparameter tuning issue — it exposed a fundamental structural incompatibility between standard RL reward design and multi-leg derivative trading.

A straddle is delta-neutral at inception: the Call leg's unrealized gain is offset almost exactly by the Put leg's unrealized loss (and vice versa). Consequently, a monolithic total-PnL reward signal hovers near zero for the majority of a position's lifetime, regardless of whether the agent has made a profitable directional call. The agent cannot distinguish between "the market hasn't moved" and "I'm winning on CE but losing equally on PE." This ambiguity caused V1 to converge to a degenerate no-trade policy and V1.1 to churn aimlessly.

The V3 per-leg decomposition resolves this by providing the agent with two independent credit-assignment channels. When the CE leg profits ₹2,000 while the PE leg loses ₹1,800, the agent receives a net-positive signal rather than a near-zero one. This granularity, combined with multi-horizon blending (1-min reactive + 30-min trend + 120-min regime), creates a reward surface with enough gradient for PPO to learn meaningful entry/exit timing.

## 5.2 Action Masking vs. Penalty-Based Constraints

A recurring question in constrained RL is whether to prevent illegal actions architecturally or penalize them in the reward. Our experiments provide strong empirical evidence for the architectural approach.

V1.1 attempted to discourage invalid actions (e.g., reducing a leg that doesn't exist) through negative reward penalties. This created a confounding problem: the agent could not differentiate between "I took a bad market action" and "I attempted an operationally impossible action." Both produced negative reward, corrupting the policy gradient. MaskablePPO eliminates this entirely — illegal actions receive zero probability mass before the softmax, so the agent never wastes exploration budget on impossible states. The result was faster convergence and a cleaner policy that focuses exclusively on market-relevant decisions.

## 5.3 What the Agent Actually Learned

The SHAP explainability analysis reveals that the agent's decision-making is overwhelmingly temporal rather than momentum-driven. The top three features by importance are:

- **time_to_expiry_norm (21.0%)** — proximity to weekly expiry
- **vol_trend (17.5%)** — direction of volatility change  
- **minutes_to_close (9.9%)** — time remaining in the trading session

This is a theoretically sound result. Options are decaying assets whose Theta accelerates non-linearly as expiry approaches. An agent that primarily keys on "how much time is left" is implicitly learning to manage Theta risk — the single largest determinant of weekly option profitability. The dominance of `vol_trend` further confirms the agent has learned to anticipate IV crush/expansion rather than react to it.

Notably, traditional momentum indicators (RSI, MACD, Bollinger) rank near the bottom of SHAP importance. This suggests the agent has discovered that for short-dated options, time and volatility dynamics matter far more than price momentum — a finding consistent with options pricing theory but rarely demonstrated empirically in RL literature.

## 5.4 Regime-Dependent Strategy Switching

The episode-level data reveals a clear regime-dependent behavioral pattern. In the single HIGH_VOL episode (VIX > 20), the agent adopted a pure SHORT position with 99.3% HOLD rate and just 11 trades — selling inflated premium and patiently waiting for IV to crush. In LOW_VOL episodes, the agent trades more actively (avg 80 trades/episode) with frequent LONG/SHORT switches, capturing small directional moves where Theta alone is insufficient to generate returns.

This adaptive behavior emerges entirely from training without explicit regime-switching logic in the code. The agent infers the regime from its 69-dimensional observation vector and adjusts accordingly — validating the feature engineering design.

## 5.5 The Sim-to-Real Gap

The paper trading episode (29 Apr – 05 May 2026) provides a preliminary assessment of the sim-to-real transfer. The agent recovered from an initial ₹-3,578 drawdown to finish at ₹+3,496 profit, utilizing up to ~₹8.5 Lakhs of its ₹10 Lakh capital limit across 8 concurrent short lots.

However, the paper trading environment still uses simulated fills at mid-price with fixed 0.5% slippage. In a live order book, large lot orders (520 qty) would face variable market impact, potentially widening the effective transaction cost. The gap between paper and live performance remains an open question that can only be resolved through extended live deployment.

---

## 5.6 Limitations

1. **Fixed Slippage Model**: The training environment applies a constant 0.5% transaction cost regardless of market conditions. In reality, slippage is dynamic and correlates with volatility — precisely when the agent trades most aggressively, real slippage can exceed 1–2%, potentially eroding the backtested edge.

2. **Discrete Action Space**: The 8-action discrete space restricts adjustments to ±1 lot (65 units) per timestep. If the optimal rebalance requires multiple lots, the agent must take several sequential steps, during which the market may move adversely. This prevents instantaneous Delta neutralization.

3. **1-Minute Temporal Resolution**: The agent acts on 1-minute candle closes. Significant price moves around macroeconomic events or RBI announcements can occur within seconds, creating a systematic disadvantage against higher-frequency participants.

4. **Bid-Ask Spread Ignored**: Options are priced using theoretical Black-Scholes mid-price with assumed immediate fills. In real markets, every contract has a bid-ask spread (₹1–3 for ATM, up to ₹5–10 for OTM/illiquid strikes) that the agent must cross, adding a hidden execution cost never encountered during training.

## 5.7 Future Work

1. **Continuous Action Spaces (SAC/TD3)**: Transitioning to continuous-action algorithms would allow the agent to output exact lot quantities per leg in a single step, enabling proportional position sizing and instantaneous Delta neutralization — directly addressing Limitation #2.

2. **Order Book Integration**: Incorporating Level-2 LOB data (bid/ask prices, depth, order flow imbalance) into the observation space would allow the agent to learn passive limit-order execution strategies, reducing market impact and addressing the bid-ask spread limitation.

3. **Sequence Models (LSTM/Transformer)**: Replacing the MLP with a recurrent or attention-based architecture would give the agent temporal memory. Since SHAP already shows temporal features dominate decisions, a sequence model could reason about feature *trajectories* rather than just instantaneous values.

4. **Multi-Asset Portfolio Hedging**: Extending the environment to simultaneously manage NIFTY, BANKNIFTY, and FINNIFTY options through a multi-agent RL (MARL) framework would enable cross-asset correlation hedging and diversify the system's risk profile.

5. **Adaptive Transaction Cost Modeling**: Replacing the fixed slippage with a market-impact function that scales with order size, volatility, and book depth would produce policies inherently more conservative during illiquid periods and more aggressive when the order book is deep.
