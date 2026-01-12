# Reinforcement Learning Action Space: Multi-Discrete Logic

## Overview
This document explains the transition from a **Discrete Action Space** (choosing 1 action at a time) to a **Multi-Discrete Action Space** (choosing 2 independent actions simultaneously). This change was implemented to give the agent granular, independent control over both legs of the option strategy (Call and Put).

## The Problem with Single-Action (Discrete) Spaces
In a standard discrete action space (e.g., `Exit Call`, `Exit Put`, `Add Call`), the agent is forced to prioritize one leg over the other at every time step.

**Example Scenario**:
- **Market State**: High Volatility Crash. Calls are profitable, Puts are bleeding.
- **Agent's Goal**: Instantaneously book profits on Calls AND hedge the Puts by adding more lots.
- **Limitation**: A single-action agent must choose between "Exit Call" OR "Add Put". It cannot do both. This lag (waiting 5 minutes for the next step) can be fatal in intraday trading.

## The Solution: Multi-Discrete Action Space
We utilize a `MultiDiscrete([3, 3])` action space, meaning the agent outputs a vector of two integers at every step: `[CallAction, PutAction]`.

### Action definitions per leg:
- **0 (Hold)**: Do nothing.
- **1 (Add/Buy)**: Buy 1 more lot (Increase position size).
- **2 (Offload/Sell)**: Sell 1 lot (Realize PnL / Reduce risk).

### The Action Matrix (9 Combinations)
This creates 9 unique strategic maneuvers available at every single time step:

| Put Action $\rightarrow$ <br> Call Action $\downarrow$ | **Hold Put (0)** | **Buy More Put (1)** | **Sell Put (2)** |
| :--- | :--- | :--- | :--- |
| **Hold Call (0)** | **Hold All** `[0,0]`<br>*(Do nothing)* | **Add Put** `[0,1]`<br>*(Bearish Bias)* | **Cut Put** `[0,2]`<br>*(Cut losers/Book profit)* |
| **Buy More Call (1)** | **Add Call** `[1,0]`<br>*(Bullish Bias)* | **Double Down** `[1,1]`<br>*(Add to both sides)* | **Pivot Bullish** `[1,2]`<br>*(Buy Call, Sell Put)* |
| **Sell Call (2)** | **Cut Call** `[2,0]`<br>*(Cut losers/Book profit)* | **Pivot Bearish** `[2,1]`<br>*(Sell Call, Buy Put)* | **Reduce/Exit** `[2,2]`<br>*(Scale out both)* |

## Key Advantages
1.  **Pivot Power**: The agent can perform a "Pivot" (`[1,2]` or `[2,1]`) to flip its directional bias instantly.
2.  **Decoupling**: The decision to manage the Put leg is mathematically decoupled from the Call leg, allowing for more complex hedge ratios.
3.  **Human-Like Control**: This mimics a trader using two hands to execute orders on both legs simultaneously.
