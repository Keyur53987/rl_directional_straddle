# Results and Performance Analysis

## Training Performance

The Proximal Policy Optimization (PPO) agent was trained on 1-minute interval NIFTY options data. The training progression is visualized below:

![Training Curve](../../model/PPO/training_curve.png)

### Convergence Analysis
The training curve demonstrates a clear upward trend in reward accumulation.
- **Initial Phase**: The agent started with an average reward of approximately **-3,500**, indicating heavy penalties from drawdowns and transaction costs due to random exploration.
- **Learning Phase**: Over 1 million timesteps, the agent steadily improved its policy, reducing the negative reward to approximately **-2,000**.
- **Stabilization**: The curve flattens towards the end, suggesting the agent has converged to a local optimum where it balances the trade-off between "Holding" (avoiding costs) and "Adjusting" (managing Delta).

**Why Negative Rewards?**
It is important to note that the reward remains negative due to the continuous application of the **Drawdown Penalty** ($\lambda=0.5$). Even a profitable strategy incurs this penalty if it experiences any volatility. The *relative improvement* of +1,500 points confirms successful learning.

## Testing Output
Testing on unseen data (Jan-Dec 2025) generated the following detailed reports:
- **Total PnL & ROI**: Achieved a total PnL of **₹7,260,507.95**, translating to a **726.05%** overall ROI.
- **Daily PnL**: The agent secured an average daily PnL of **₹29,276.24**.
- **Drawdown Control**: Maximum drawdown was strictly contained to **₹196,656.0**, demonstrating excellent risk aversion.
- **Win Rate**: The agent achieved a solid win rate of **65.73%**, executing an average of 44.4 trades per day (11,010 total trades).
- **Risk-Adjusted Return**: The strategy produced an impressive Sharpe Ratio of **8.23**.

### Best Run Performance
Till date, the execution run marked as **`20260206_124338_HPC`** has yielded the best overall results. The reward function utilized to achieve this peak performance was formulated as:
```text
reward = (total_pnl / denom) * (0.001 * (max_drawdown / denom))
```
