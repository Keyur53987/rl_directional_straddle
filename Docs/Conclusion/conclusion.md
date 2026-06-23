# Chapter 6: Conclusion

This thesis addressed the problem of autonomous intraday options hedging on the NIFTY50 index using Deep Reinforcement Learning. A MaskablePPO agent was trained to trade weekly straddles and strangles on 1-minute data with 69 engineered features. The key design innovation — a per-leg, multi-horizon reward function — resolved the credit-assignment failure that caused earlier iterations to converge to degenerate no-trade policies.

On the 2025 out-of-sample test set (48 weekly episodes), the agent achieved a total profit of **₹5,61,919**, a **93.75% win rate**, and a **profit factor of 47.66** with a maximum drawdown of ₹66,540. SHAP explainability confirmed the agent relies on time-to-expiry and volatility trend rather than price momentum, validating that it has learned principled Theta and volatility management.

While limitations remain — fixed slippage assumptions, discrete action space, and absence of bid-ask spread modeling — the system demonstrates that DRL can learn profitable, risk-aware options strategies directly from market data and execute them in a live paper-trading environment.
