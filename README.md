# Autonomous Intraday Options Trading and Hedging System using Deep RL

This repository implements a Deep Reinforcement Learning (DRL) framework designed for autonomous, intraday options trading and dynamic hedging of multi-leg strategies (Straddles and Strangles) on the **NIFTY50** index. 

By modeling option trading under market frictions as a Markov Decision Process (MDP), the system implements state-of-the-art DRL policies—principally **Maskable PPO (Proximal Policy Optimization)**—along with standard PPO and A2C baselines. The agent is trained on 1-minute historical Open-High-Low-Close (OHLC) spot index data and India VIX to learn optimal execution policies, such as asymmetric leg additions/reductions, hold times, and early square-offs.

---

## 🚀 Key Features and Enhancements

- **Discrete Action Masking**: Employs [ActionMasker](Mask_PPO_train_v3.py#L20) inside [MaskablePPO](Mask_PPO_train_v3.py#L19) to filter out invalid actions (e.g., maintaining naked positions, exceeding lot limits, or violating margin constraints) at the logit level, dramatically boosting sample efficiency.
- **69-Feature Engineered Observation Space**: Integrates high-efficiency volatility estimators (Parkinson, Garman-Klass), option Greeks (Delta, Gamma, Vega, Theta, Vanna, Volga) computed via the Black-Scholes model, technical indicators, time-based features, and portfolio state metrics.
- **Multi-Horizon Reward Shaping (v3)**: Uses a blended reward function evaluating 1-step, 30-step, and 120-step horizons, alongside separate CE/PE PnL signals, resolving hold-only or excessive overtrading pathologies.
- **Dynamic SPAN-like Margin Calculator**: Simulates exchange margin requirements (SPAN + exposure) for short strangle legs under stressed volatility scenarios (+25%) and 3.5σ price scans to ensure realistic capital constraint tracking.
- **Post-Hoc Explainability (SHAP)**: Extracts SHAP values to profile feature importance across different market volatility regimes and directional trends.
- **Interactive Dashboards**: 
  - Standard Streamlit-based backtesting dashboard.
  - Custom Plotly-based HTML explainability dashboard visualizing step-level actions, trade hold times, drawdown curves, and SHAP feature influences.

---

## 📁 Repository Structure

### Custom Environments
- [envs/intraday_option_env_maskPPO_v3.py](envs/intraday_option_env_maskPPO_v3.py): **State-of-the-art v3 environment** supporting 69 features, Discrete(8) action space, action masking, multi-horizon reward shaping, and separate Call/Put unrealized PnL inputs.
- [envs/intraday_option_env_maskPPO.py](envs/intraday_option_env_maskPPO.py): Maskable PPO environment (v1/v2) with 67 features.
- [envs/intraday_option_env.py](envs/intraday_option_env.py): Base environment with 65 features and MultiDiscrete([3,3]) action space for standard PPO/A2C.

### Model Training & Testing
- [Mask_PPO_train_v3.py](Mask_PPO_train_v3.py): Primary training entry point using 20 CPU core parallelization and weekly episodic bounds.
- [Mask_PPO_explainability_test.py](Mask_PPO_explainability_test.py): Multiprocessed testing script that evaluates models, tracks trades, and exports step-level metrics and SHAP values.
- [PPO_train.py](PPO_train.py) / [A2C_train.py](A2C_train.py): Baseline training scripts for standard PPO and A2C algorithms.
- [walk_forward_validation.py](walk_forward_validation.py): Walk-forward validation harness across different historical datasets.

### Configuration & Utilities
- [Mask_PPO_config_v3.py](Mask_PPO_config_v3.py): Settings for v3 training, risk-free rate, lot multipliers, transaction cost percentages (0.5%), and reward weights.
- [utils/black_scholes.py](utils/black_scholes.py): High-precision analytical model calculating option premiums and Greeks.
- [utils/feature_calculator.py](utils/feature_calculator.py): Incremental calculator for technical indicators (RSI, Bollinger Bands, ATR, MACD) and volatility estimators.

---

## 🛠️ Installation and Setup

### 1. Clone and Navigate
```bash
git clone <repository_url>
cd rl_directional_straddle
```

### 2. Install Dependencies
This project requires Python 3.8+ and GPU/CPU-based PyTorch. Install the prerequisites:
```bash
pip install gymnasium stable-baselines3 sb3-contrib pandas numpy scipy matplotlib plotly streamlit shimmy tqdm
```

### 3. Place Data Files
Ensure your data CSVs are in the [data/](data/) directory:
- [data/train.csv](data/train.csv): Multi-year historical NIFTY 1-minute OHLC data.
- [data/INDIA_VIX.csv](data/INDIA_VIX.csv): 1-minute INDIA VIX index data.
- [data/test.csv](data/test.csv): Out-of-sample backtesting data.

---

## 📈 Usage Guide

### 1. Model Training
To train the state-of-the-art Maskable PPO v3 model using parallel environments:
```bash
python Mask_PPO_train_v3.py
```
> [!NOTE]
> Adjust hyperparameters, number of environments (`NUM_ENVS`), and reward weights directly in [Mask_PPO_config_v3.py](Mask_PPO_config_v3.py).

### 2. Backtesting & SHAP Value Generation
To generate step-level trade files, monthly summaries, and SHAP feature importance CSVs on the out-of-sample dataset:
```bash
python Mask_PPO_explainability_test.py
```
Results will save to `model/MaskablePPO/<model_id>/results_<dataset>/`.

### 3. Launching dashboards
#### Dynamic HTML Report (Plotly)
To compile the testing results and SHAP files into a single, beautiful HTML dashboard:
```bash
python explainability_dashboard.py
```
This produces `report.html` in the results folder containing interactive performance charts, drawdown profiles, win rates, and SHAP explainability breakdowns.

#### Streamlit Live Backtest Viewer
To visually simulate trade executions and portfolio updates step-by-step:
```bash
streamlit run dashboard.py
```

---

## 📊 Evaluation Metrics (2025 Test Set)

According to thesis results (`Docs/Results/Results_v3.md`), the MaskablePPO v3 agent achieves the following out-of-sample KPIs:
- **Total Net PnL**: ₹ 561,919.30 (after 0.5% transaction cost)
- **Win Rate**: 93.75% (percentage of profitable weekly episodes)
- **Profit Factor**: 47.66
- **Max Drawdown**: ₹ 66,539.64 (strictly constrained by the reward penalty)


