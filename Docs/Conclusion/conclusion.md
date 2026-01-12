# Conclusion

This thesis successfully demonstrated the feasibility and efficacy of using Deep Reinforcement Learning for intraday options hedging. By integrating a sophisticated **Proximal Policy Optimization (PPO)** agent with a custom environment driven by 55 real-time market features, we established a robust framework for autonomous trading.

The primary contribution of this work is the **hybrid feature engineering approach**, which combines theoretical Black-Scholes Greeks with data-driven volatility estimators. This "guided" learning allowed the agent to converge faster and achieve superior risk-adjusted returns compared to a standard benchmark. While challenges remain in transaction cost modeling and execution speed, the results confirm that RL agents can learn complex, non-linear hedging strategies that adapt dynamically to changing market volatility regimes. This sets a strong foundation for future research into fully autonomous, algorithmic option portfolios.
