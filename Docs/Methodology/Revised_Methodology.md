# Research Methodology: Deep Hedging with Constrained Reinforcement Learning

## 1. Introduction

This research proposes a novel Deep Reinforcement Learning (DRL) framework for autonomous intraday trading and hedging of options strategies, specifically focusing on Directional Straddles. Unlike traditional Delta-hedging or rule-based algorithms, our system leverages the Proximal Policy Optimization (PPO) algorithm to learn dynamic hedging policies that balance profit maximization with risk containment. A key contribution of this work is the development of a custom OpenAI Gym environment (`IntradayOptionEnv`) that integrates rigorous financial constraints—such as ratio maintenance and liquidity logic—directly into the environment's physics, ensuring that the agent's behavior remains practically viable and risk-compliant.

## 2. Environment Architecture

The core of the methodology is the `IntradayOptionEnv`, a time-discrete simulation environment that models the microstructure of an options market for a single underlying asset (NIFTY). The environment is designed to simulate realistic intraday friction, option pricing dynamics, and margin constraints.

### 2.1. Physical Simulation & Pricing
The environment operates on 5-minute OHLCV (Open, High, Low, Close, Volume) data candles.
*   **Time Horizon**: Each episode represents one full trading day, from 09:15 AM to 03:30 PM.
*   **Pricing Engine**: While historical option prices are often noisy or incomplete, this system employs a hybridized Black-Scholes-Merton (BSM) model. The environment calculates real-time option premiums and Greeks (Delta, Gamma, Theta, Vega) using the underlying spot price and a calibrated Implied Volatility (IV) surface. This ensures that the agent observes mathematically consistent pricing relationships.
*   **Account Mechanics**: The agent manages a simulated ledger with an initial capital of INR 1,000,000. The environment tracks Mark-to-Market (MTM) PnL, realized PnL, transaction costs (0.05%), and slippage (0.02%) at every step.

### 2.2. The Challenge of Inactivity Bias
A significant challenge in financial RL is the "Inactivity Bias" or "Sit-on-Hands" problem. In high-noise environments, untrained agents often converge to a local optimum of *never trading* (Action=0), as this yields a guaranteed reward of zero, which is statistically preferable to the negative expected value of random exploration (due to spreads and costs).

To maintain the purity of the reward function while overcoming this bias, we implement a **Randomized Curriculum Learning** approach via the `reset()` mechanism:
*   **Forced Entry (50% probability)**: The agent is forced into an initial Straddle position (1 Call, 1 Put) at market open. This forces the agent to learn **Position Management**—how to mitigate losses and extract profit from an existing liability.
*   **Autonomous Entry (50% probability)**: The agent starts with zero exposure. Having learned management skills in the forced episodes, the agent is confident enough to initiate trades autonomously when it identifies profitable setups.
This "Training Wheels" mechanism bridges the gap between passive safety and active trading without artificially inflating rewards.

## 3. Markov Decision Process (MDP) Formulation

We formulate the trading problem as a discrete-time Markov Decision Process $(S, A, R, P, \gamma)$.

### 3.1. State Space ($S$)
The observation space is a 55-dimensional continuous vector representing the market state and internal portfolio status. It is composed of five feature groups:
1.  **Market Data**: Normalized returns, log-returns, and relative volume of the underlying asset.
2.  ** volatility Estimators**: Realized volatility (RV) over multiple windows (short-term vs. long-term) and ATR (Average True Range) to capture regime changes.
3.  **Technical Indicators**: RSI (Relative Strength Index), MACD, and Bollinger Band distances to provide trend and mean-reversion signals.
4.  **Option Greeks**: Delta, Gamma, Theta, and Vega of the current ATM strikes. Providing Greeks allows the agent to "see" the risk sensitivity of the market directly.
5.  **Portfolio State**: Current inventory (Call Lots, Put Lots), unrealized PnL, margin utilization, and time-to-expiry scaling factors.

### 3.2. Action Space ($A$)
To support complex structural hedging, we define the action space as a `MultiDiscrete([3, 3])` vector, allowing independent control over the Call and Put legs simultaneously.
$$ A_t = [a_{call}, a_{put}] $$
where each component $a \in \{0, 1, 2\}$:
*   **0 (Hold)**: Maintain current inventory.
*   **1 (Add/Buy)**: Buy one lot of the respective option.
*   **2 (Offload/Sell)**: Sell one lot of the respective option.

This independence allows for a rich set of strategic maneuvers:
*   **Straddle Entry**: `[1, 1]` (Buy Call, Buy Put)
*   **Delta Adjustment**: `[1, 0]` (Buy Call Only) to increase Delta.
*   **Profit Taking**: `[2, 2]` (Sell both) to close the position.

### 3.3. Constraints and Logic Layers
A raw RL agent might attempt reckless actions, such as selling naked options (unlimited risk) or pyramiding into insolvency. We embed a **Ratio Maintenance Logic** layer directly into the environment's `step()` function to enforce risk compliance:
1.  **No Naked Positions**: The agent is strictly prohibited from holding a single-leg short position. It must maintain a Straddle, Strangle, or a Ratio Spread where at least one unit of the opposing leg is held.
    *   *Rule*: A position is valid iff $(Lots_{CE} > 0 \land Lots_{PE} > 0) \lor (Lots_{CE} = 0 \land Lots_{PE} = 0)$.
    *   *Enforcement*: Any action $a_t$ that leads to a violation (e.g., selling the last Put while holding Calls) is instantly intercepted and nullified ($a_t \leftarrow 0$).
2.  **Re-Entry Symmetry**: If the agent is flat (0 lots), it can only re-enter via a paired order (Action `[1, 1]`). This enforces the "Straddle/Strangle" mandate while allowing the agent to adjust the ratio *after* entry.
3.  **Liquidity & Margin Checks**: Actions are validated against available cash and maximum lot limits ($\text{MaxLots}=10$).

### 3.4. Reward Function ($R$)
The objective is not merely profit, but risk-adjusted stability. We employ a customized reward function:
$$ R_t = \Delta \text{PnL}_t - \lambda \times \max(0, \text{Drawdown}_t) $$
*   $\Delta \text{PnL}_t$: The change in Net Asset Value (NAV) from step $t-1$ to $t$.
*   $\lambda$: A risk aversion penalty coefficient ($\lambda=0.5$).
*   $\text{Drawdown}_t$: The current drawdown from the peak equity curve.
This incentivizes the agent to smooth its equity curve and punish volatility, aligning the agent's behavior with the goals of a professional fund manager.

## 4. Training Methodology

### 4.1. Algorithm: Proximal Policy Optimization (PPO)
We utilize PPO, an on-policy gradient method, for its stability and ease of hyperparameter tuning. PPO optimizes a surrogate objective function that clips probability ratios, preventing destructive policy updates:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] $$
where $\hat{A}_t$ is the advantage function estimate.

### 4.2. Network Architecture
The Policy (Actor) and Value (Critic) networks share a feature extraction backbone consisting of:
*   **Input Layer**: 55 neurons (State dimension).
*   **Hidden Layers**: Two fully connected layers of 64 units each with Tanh activation.
*   **Output Heads**:
    *   *Actor*: Softmax layer outputting probabilities for the $3 \times 3$ action combinations.
    *   *Critic*: Linear layer outputting the state value $V(s)$.

### 4.3. Training Protocol
The model is trained on 6 years of historical NIFTY intraday data (2015-2020), with 2021 reserved for out-of-sample validation.
*   **Episodes**: 1,000,000 timesteps.
*   **Parallelization**: Multiple environment instances (`SubprocVecEnv`) run in parallel to collect experience batches, breaking temporal correlations and accelerating convergence.
*   **Curriculum Phase**:
    *   *Training*: `force_entry_prob = 0.5`.
    *   *Testing/Deployment*: `force_entry_prob = 0.0`.

## 5. Conclusion of Methodology
This methodology integrates the flexibility of Deep Reinforcement Learning with the rigid safety frameworks required in quantitative finance. By hard-coding the "rules of the game" (Ratio Maintenance) and soft-coding the "incentive to play" (Randomized Forced Entry), we create a robust environment where the agent can safely learn complex, non-linear hedging strategies without exposure to ruinous risks.
