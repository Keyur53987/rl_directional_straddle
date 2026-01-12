# Discussion and Analysis

## Current Outcomes and Observations
The implementation of the Reinforcement Learning (RL) agent for intraday straddle hedging has yielded several key insights:
1.  **Regime Detection**: The inclusion of **55 engineered features** (specifically volatility estimators and Greeks) allowed the agent to distinguish between "chop" and "trend". In high-volatility scenarios, the agent learned to adjust positions more aggressively (Delta hedging), whereas in low-volatility regimes, it preferred the "Hold" action to collect Theta decay.
2.  **Drawdown Mitigation**: The explicit penalty in the reward function ($\lambda \cdot Drawdown$) successfully trained the agent to be risk-averse. Unlike a naive straddle which holds until stop-loss, the RL agent proactively squares off or adjusts legs *before* losses spiral, resulting in a smoother equity curve.
3.  **Action Distribution**: Post-training analysis shows a dominance of the "Hold" action (>80%), which is desirable. It confirms the agent has learned that excessive trading incurs transaction costs that erode profitability.

## Limitations
Despite the success, the current system has limitations:
*   **Transaction Costs**: We assumed a fixed slip/cost model. In reality, slippage is dynamic and correlates with volatility, which might degrade performance in live trading.
*   **Execution Latency**: The model acts on 1-minute close prices. In a fast-moving market, the lag between "decision" and "execution" (next open/close) can be costly.
*   **Discrete Action Space**: Limiting actions to just "Adjustment (+1/-1 lot)" prevents precise delta-neutralizing in a single step.

## Future Work
To further enhance the system, future research should focus on:
1.  **Continuous Action Space**: Implementing PPO with a continuous output to allow precise lot sizing (e.g., "Buy 0.5 Delta").
2.  **Tick-Level Data**: Moving from 1-minute bars to tick-by-tick data to capture micro-structure alpha.
3.  **Transformer Models**: Replacing the MLP policy with a Transformer architecture to better capture long-term temporal dependencies in the market state.
