# Intraday RL Trading System: Directional Straddle/Strangle

This project implements a Deep Reinforcement Learning (DRL) agent using **Stable Baselines3 (PPO)** to trade intraday options strategies (Straddles and Strangles) on NIFTY/BANKNIFTY indices. The system is designed to optimize trade management by dynamically adjusting positions (adding lots or changing hedge ratios) based on market state.

## Features

- **Custom Gym Environment**: `IntradayOptionEnv` simulates realistic intraday trading behavior using 1-minute OHLCV data. It handles order execution, PnL tracking, and position management.
- **Deep Learning Agent**: Utilizes Proximal Policy Optimization (PPO) to learn optimal policy actions such as adding legs, squaring off, or holding positions.
- **Interactive Dashboard**: A **Streamlit**-based dashboard allows for:
  - Simulating the agent's performance on specific historical dates.
  - Visualizing tick-by-tick (minute-level) PnL and underlying price updates.
  - Detailed trading logs and portfolio tracking.
- **Configurable Strategies**:
  - **Strategy Types**: Long (Buy) or Short (Sell) Straddles/Strangles.
  - **Strike Selection**: ATM (At-The-Money), Bollinger Bands, or ATR-based dynamic strikes.
  - **Risk Management**: Configurable stop-loss, transaction costs, and slippage.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd rl_directional_straddle
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages:
   ```bash
   pip install gymnasium stable-baselines3 pandas numpy matplotlib streamlit shimmy
   ```

## Usage

### 1. Training the Agent
To train the PPO agent on your historical dataset:
1. Ensure your training data is in `data/train.csv`.
2. Run the training script:
   ```bash
   python train.py
   ```
   - The script sets up a vectorized environment for faster training.
   - Models are saved to `model/PPO/`.
   - Training logs are saved to `model/PPO/logs/` and can be visualized using TensorBoard or the generated plots.

### 2. Running the Dashboard
To visualize the trained agent's performance and run simulations:
1. Ensure a trained model exists in `model/PPO/`.
2. Run the Streamlit app:
   ```bash
   streamlit run dashboard.py
   ```
3. Use the sidebar to:
   - Select the data path (e.g., `data/test.csv`).
   - Choose a specific date to simulate.
   - Adjust simulation speed.
   - View real-time PnL graphs and trade logs.

## Configuration
The system is highly configurable via `config.py`. Key settings include:

- **Account**: `INITIAL_CAPITAL`, `MAX_LOTS`, `TRANSACTION_COST_PCT`.
- **Strategy**: 
  - `STRATEGY_TYPE`: Toggle between 'LONG' and 'SHORT'.
  - `STRIKE_GAP`: Distance between strikes for strangles.
  - `START_TIME` / `END_TIME`: Trading window (e.g., 09:30 to 15:15).
- **RL Hyperparameters**: `REWARD_LAMBDA` for reward shaping.
- **Environment**: `WINDOW_SIZE` for observation lookback.

## Project Structure

- `envs/`: Contains the custom Gym environment (`intraday_option_env.py`).
- `model/`: Stores trained models and TensorBoard logs.
- `data/`: Directory for storing historical options/spot data.
- `train.py`: Main entry point for training the RL agent.
- `dashboard.py`: Interactive visualization tool.
- `config.py`: Central configuration file.
