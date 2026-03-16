# Methodology

## 1. Problem Formulation: The Markov Decision Process

We model the intraday directional straddle trading problem as a discrete-time Markov Decision Process (MDP) [1], formalized as a tuple $(S, A, P, R, \gamma)$. This framework is widely adopted in financial reinforcement learning literature to solve optimal execution and hedging problems under market friction [2, 3]. Unlike traditional delta-hedging strategies that rely on rigid assumptions (e.g., continuous trading, zero transaction costs), the MDP allows the agent to learn an adaptive policy $\pi_\theta(a_t|s_t)$, parameterized by a neural network $\theta$, that maximizes risk-adjusted returns in the presence of realistic market constraints.

### 1.1 State Space ($S$)
At each time step $t$ (representing 1 minute), the agent observes a state vector $s_t \in \mathbb{R}^{55}$. To ensure the agent captures both market regime and option pricing dynamics, we construct a feature set combining deep learning inputs with financial theory, consisting of 7 distinct feature groups:

1. **Volatility Features (12 features)**: Realized volatility estimators including Parkinson and Garman-Klass volatility over multiple rolling windows (5, 15, 30, 60 minutes) to adapt to changing volatility regimes.
2. **Price & Returns (10 features)**: Normalized close prices, distances from VWAP, and returns across multiple timeframes.
3. **Option Greeks (10 features)**: Theoretical sensitivities ($\Delta, \Gamma, \nu, \Theta$, Vanna, Volga) calculated using the Black-Scholes-Merton model to provide structured information about portfolio sensitivity.
4. **Technical Indicators (6 features)**: Momentum indicators such as RSI, MACD, and Bollinger Bands.
5. **Position Features (10 features)**: Current inventory (lots), entry prices, and strike distances to ensure the state is Markovian inclusive of inventory.
6. **Time Features (4 features)**: Time to expiry and intraday time progress.
7. **Risk Metrics (3 features)**: Peak PnL drop, max drawdown, and rolling Sharpe ratio.

### 1.2 Action Space ($A$)
To handle the complexities of intraday liquidity and execution, we discretize the action space $A = \{0, 1, 2\}$, simplifying the continuous hedging decisions typically found in theoretical literature [2]:
*   $a_t=0$ (**Hold**): No trade. Minimizes transaction costs.
*   $a_t=1$ (**Bullish Adjustment**): Increases portfolio Delta ($\Delta_{net}$) by entering/adding Call positions or exiting Put positions.
*   $a_t=2$ (**Bearish Adjustment**): Decreases portfolio Delta by entering/adding Put positions or exiting Call positions.

### 1.3 Reward Function ($R$)
Designing an appropriate reward function is crucial for preventing the agent from taking excessive risks. We employ a risk-sensitive reward function inspired by the Deep Hedging framework [2] and Sharpe Ratio maximization [7]. The peak performing runs (e.g., HPC configuration) utilize a multiplicative reward to heavily penalize deep drawdowns. The reward $r_t$ is defined essentially as:

$$r_t = \left(\frac{\text{Total PnL}_t}{\text{Denom}_t}\right) \times \left(1 - 0.001 \cdot \left(\frac{\text{Max Drawdown}_t}{\text{Denom}_t}\right)\right)$$
*(Implemented as: `reward = (total_pnl/denom) * (0.001 * (max_drawdown/denom))`)*

Where $0.001$ represents the coefficient of risk aversion (`REWARD_LAMBDA`), dynamically punishing volatility to force the agent to trade only when the expected marginal gain of an adjustment exceeds the cost of risk and execution. Transaction and turnover costs are additionally subtracted in the environment stepping calculation to account for friction [3].

## 2. Optimization Algorithm: Proximal Policy Optimization (PPO)

We utilize Proximal Policy Optimization (PPO) [8], a policy gradient method that strikes a balance between sample efficiency and stability. In financial RL, stability is paramount due to the low signal-to-noise ratio in market data. PPO prevents catastrophic policy updates by clipping the objective function:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

Here, $\hat{A}_t$ is the estimated advantage function, and $\epsilon$ is a hyperparameter (set to 0.2) that constrains the policy update. We use a Multi-Layer Perceptron (MLP) architecture with two hidden layers of 64 units each and Tanh activation functions, which has been shown to be effective for continuous state inputs [9].

## 3. Data and Training Protocol

The model is trained on high-frequency 1-minute OHLCV data for NIFTY/BANKNIFTY options.
*   **Episode definition**: One episode corresponds to a single trading day (9:30 AM to 3:15 PM).
*   **Initialization**: At the start of each episode, a straddle is initiated at the At-The-Money (ATM) strike.
*   **Termination**: Positions are forcibly squared off at 3:15 PM to avoid overnight risk.

### Training Flowchart

The following flowchart illustrates the interaction between the Agent, the MDP Environment, and the PPO Optimizer.

```mermaid
flowchart TD
    subgraph Environment
    A[Market Data Feed\n(1-min OHLCV)] --> B(Feature Engineering\nBSM Greeks + Volatility)
    B --> S[State S_t]
    P[Portfolio State] --> S
    end

    subgraph Agent_PPO
    S --> N[Policy Network\nMLP(64, 64)]
    N --> Act{Action A_t}
    Act -- 0: Hold --> E[No Cost]
    Act -- 1/2: Adjust --> T[Execute Trade]
    end
    
    T --> C[Transaction Cost\n& Slippage]
    C --> P
    T --> P
    
    subgraph Reward_Calc
    P --> R[Reward R_t\n= dPnL - Lambda*DD]
    end
    
    R --> BUF[Rollout Buffer]
    S --> BUF
    Act --> BUF
    
    BUF -- Batch --> OPT[PPO Optimizer\nMaximize L_CLIP]
    OPT -- Update Weights --> N

    style A fill:#f9f,stroke:#333
    style N fill:#bbf,stroke:#333
    style OPT fill:#bbf,stroke:#333
```

## 4. Agent Decision Process

The following diagram details the step-by-step decision loop for the PPO agent at each timestep $t$.

```mermaid
graph TD
    Market[Market Data Stream<br/>(1-Min OHLC)] --> |New Minute T| Env[Environment]
    
    subgraph Environment Processing
        Env --> |Raw Data| FeatCalc[Feature Calculator]
        FeatCalc --> |Calc Volatility<br/>Calc Returns<br/>Calc Technicals| Features[Feature Vector<br/>(55 Numbers)]
        
        Env --> |Option Pricing| BS[Black-Scholes Model]
        BS --> |Calc Greeks<br/>Calc PnL| Features
        
        Features --> |Normalize| State[State S_t]
    end
    
    State --> |Input| Agent[PPO Agent<br/>(Brain)]
    
    subgraph Decision Making
        Agent --> |Policy Network| Logits[Probabilities]
        Logits --> |Sample| Action[Action A_t]
    end
    
    Action --> |0: Hold<br/>1: Bullish<br/>2: Bearish| Exec[Execution Logic]
    
    subgraph Execution
        Exec --> |Update Position| Portfolio[Portfolio Stats]
        Portfolio --> |Calc PnL Change| Reward[Reward R_t]
    end
    
    Reward --> |Feedback| Agent
    Exec --> |Wait for Next Minute| Market
```

### Process Description

### Process Description

The agent's interaction loop operates as follows:

1.  **Observing the Market (Feature Engineering)**
    Before making any decision, the agent perceives the market state $S_t$. This is not just the raw price, but a vector of **55 engineered features**:
    *   **Volatility**: Realized volatility measures (Parkinson, Garman-Klass) over multiple time windows (5, 15, 30 min) to detect market turbulence.
    *   **Greeks**: Black-Scholes sensitivities ($\Delta, \Gamma, Vanna, Volga$). High $Vega$ or $Gamma$ alerts the agent to potential risks.
    *   **Technicals**: Momentum indicators (RSI, MACD) to gauge directional strength.
    *   **Portfolio State**: Current PnL, number of active lots, and entry prices.

2.  **Choosing an Action (The Policy Network)**
    The state $S_t$ is fed into the PPO Policy Network (a neural network).
    *   The network outputs **logits** for 3 possible actions: `[Hold, Bullish, Bearish]`.
    *   **Stochastic Sampling**: During training, the agent *samples* from this distribution (e.g., if probabilities are `[0.7, 0.2, 0.1]`, it picks 'Hold' 70% of the time). This ensures exploration.
    *   **Deterministic Execution**: During testing/deployment, the agent simply picks the action with the highest probability.

3.  **Execution and Feedback (Reward Calculation)**
    The selected action is executed at the close price of minute $t$. The environment then computes the **Reward** $r_t$ to guide learning:
    $$r_t = \Delta \text{PnL} - (\lambda \times \text{Drawdown})$$
    *   **Profit Incentive**: If the action leads to increased PnL, the agent receives a positive reward.
    *   **Risk Penalty**: If the action increases drawdown (loss from peak), a penalty is applied. This teaches the agent to profit *smoothly* rather than taking wild gambles.

## References

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.
[2] Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271-1291.
[3] Ritter, G., & Kolm, P. N. (2019). Data-driven hedging for options using deep reinforcement learning. *Available at SSRN 3514586*.
[4] Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *The Journal of Business*, 73(3), 477-501.
[5] Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of political economy*, 81(3), 637-654.
[6] Du, J., Jin, M., & Kolm, P. N. (2020). High-frequency multi-period investment: A reinforcement learning approach. *The Journal of Financial Data Science*, 2(4), 107-119.
[7] Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875-889.
[8] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
[9] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence*.
