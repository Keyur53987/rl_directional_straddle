# 1. Introduction

The financial derivatives market, particularly the options market, has grown exponentially in complexity and volume over the past few decades. Options serve as critical instruments for hedging risk, speculating on market movements, and facilitating price discovery. Theoretical foundations for pricing these instruments were established by the seminal works of Black and Scholes (1973) and Merton (1973), which provided the Black-Scholes-Merton (BSM) framework. This framework assumes a frictionless market where continuous delta hedging is possible, allowing for the perfect replication of option payoffs. However, real-world financial markets are characterized by discrete trading intervals, transaction costs, market impact, and liquidity constraints, rendering the assumption of perfect replication invalid (Kolm & Ritter, 2019).

In the presence of these market frictions, the problem of option hedging and strategy execution transforms from a pure replication task into a stochastic control problem where risk cannot be entirely eliminated but must be managed. Traditional heuristic adjustments to the BSM model, such as volatility surfaces or transaction cost bands, often fail to capture the optimal trade-off between hedging error and execution costs dynamically. Consequently, there has been a paradigm shift towards data-driven approaches, specifically Reinforcement Learning (RL), to solve these complex sequential decision-making problems (Sutton & Barto, 2018).

Reinforcement Learning offers a robust framework for developing adaptive trading strategies that learn directly from interactions with the market environment. Unlike traditional models which rely on rigid assumptions about market dynamics (e.g., Geometric Brownian Motion), RL agents are model-free and can learn to exploit distinct market microstructures and anomalies. A landmark development in this domain is the concept of "Deep Hedging" introduced by Buehler et al. (2019), who demonstrated that deep neural networks could be trained to derive optimal hedging policies under convex risk measures and transaction costs, significantly outperforming traditional delta-hedging strategies.

Further research has expanded on this integration of RL and quantitative finance. Halperin (2017) proposed the "QLBS" (Q-Learner in the Black-Scholes) model, which bridges Q-learning with the BSM model, offering a discrete-time, model-free alternative for option pricing and hedging. Similarly, Cao et al. (2021) utilized Deep Reinforcement Learning (DRL) to hedge derivatives, highlighting the efficacy of these algorithms in managing large portfolios where analytical solutions are intractable. The adaptability of RL allows for the incorporation of diverse state variables, including price trends, volatility indices, and order book data, enabling the development of sophisticated directional and volatility strategies (Zhang et al., 2020). By optimizing for long-term rewards such as risk-adjusted returns (e.g., Sharpe Ratio) rather than immediate profits, RL agents effectively align with the strategic goals of institutional arbitrage and proprietary trading (Deng et al., 2017).

# 2. Problem Definition
**Development of option strategy using option pricing model and Reinforcement Learning**

The core problem addressed in this project is the limitation of static, rule-based option strategies in a dynamic and frictional market environment. Traditional strategies, such as the buying or selling of Straddles and Strangles, rely heavily on fixed entry and exit rules or simplified "Greeks" derived from the BSM model. These approaches often fail to adapt to rapidly changing intraday volatility and liquidity conditions, leading to suboptimal execution and significant drawdown during adverse market regimes. This project seeks to bridge this gap by developing an intelligent trading system that leverages Reinforcement Learning to dynamically manage option positions, utilizing option pricing models as a foundational feature set rather than a prescriptive rule.

## 1.1 Objective
The primary objective of this thesis is to design, implement, and evaluate a Deep Reinforcement Learning (DRL) agent capable of executing and managing intraday option strategies, specifically focusing on directional straddles. The agent aims to:
1.  Learn an optimal policy for entry, adjustment (e.g., delta adjustments), and exit of option positions to maximize risk-adjusted returns.
2.  Incorporate market frictions, such as transaction costs and bid-ask spreads, into the reward function to ensure the strategy's viability in a live trading environment.
3.  Utilize inputs from theoretical option pricing models (e.g., Implied Volatility, Delta, Gamma) alongside raw market data to inform decision-making.

## 1.2 Motivation
The motivation for this research stems from the increasing difficulty of generating alpha using traditional linear models in highly efficient markets. As noted by Charpentier et al. (2020), financial time series are inherently non-stationary and noisy, posing a challenge for classical econometric methods. Reinforcement Learning not only handles this non-stationarity by continuously updating its policy but also excels in optimizing complex, non-differentiable objective functions like the Sortino Ratio or Maximum Drawdown (Du et al., 2020). Furthermore, the automation of high-frequency decision-making reduces the emotional bias and cognitive load associated with manual trading, allowing for consistent execution of complex hedging maneuvers that would be impossible for a human trader to perform with the same precision and speed.

## 1.3 Limitations
While promising, the application of RL in finance is subject to several limitations:
1.  **Data Efficiency**: RL algorithms, particularly deep learning models, require vast amounts of data to converge. Financial data is limited by historical availability, often necessitating the use of calibrated simulations (sim-to-real transfer issues).
2.  **Stationarity Assumption**: Although RL adapts, extreme regime shifts (e.g., market crashes) often unseen in training data can lead to catastrophic failure if the agent has not explored such states (Zhang et al., 2020).
3.  **Interpretability**: Deep RL models are often "black boxes," making it difficult to explain specific trading decisions to stakeholders, a significant hurdle for regulatory compliance and risk management.
4.  **Computational Cost**: Training high-fidelity agents requires significant computational resources and time, as noted in large-scale studies (Lim et al., 2019).

## References

1.  Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
2.  Merton, R. C. (1973). Theory of Rational Option Pricing. *The Bell Journal of Economics and Management Science*, 4(1), 141-183.
3.  Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
4.  Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271-1291.
5.  Kolm, P. N., & Ritter, G. (2019). Dynamic replication and hedging: A reinforcement learning approach. *The Journal of Financial Data Science*, 1(1), 159-171.
6.  Halperin, I. (2017). QLBS: Q-Learner in the Black-Scholes (-Merton) Worlds. *arXiv preprint arXiv:1712.04609*.
7.  Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). Deep Hedging of Derivatives Using Reinforcement Learning. *The Journal of Financial Data Science*, 3(1), 10-27.
8.  Charpentier, A., Elie, R., & Remlinger, C. (2020). Reinforcement Learning in Economics and Finance. *Computational Economics*, 59, 1-35.
9.  Du, J., Jin, M., & Kolm, P. N. (2020). Reinforcement Learning for Option Betting? *arXiv preprint arXiv:2011.08207*.
10. Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep Reinforcement Learning for Trading. *The Journal of Financial Data Science*, 2(2), 25-40.
11. Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. *IEEE Transactions on Neural Networks and Learning Systems*, 28(3), 653-664.
12. Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing Time Series Momentum Strategies Using Deep Neural Networks. *arXiv preprint arXiv:1904.04912*.
