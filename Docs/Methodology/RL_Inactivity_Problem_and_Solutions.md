# The "Inactivity Bias" Problem in RL Trading Agents

## 1. Problem Definition
In financial Reinforcement Learning (RL), agents often converge to a policy of **"Inactivity"** (never trading), also known as the "Sit-on-Hands" equilibrium.

### Why does this happen?
1.  **Exploration vs. Exploitation**: Trading is inherently risky. A random untrained agent will lose money (due to spread, transaction costs, and random price mutations) with high probability.
2.  **Local Optima**: The agent quickly learns that `Action = 0` (Do Nothing) results in `Reward = 0`.
3.  **Comparison**: Since `0` (Nothing) > `Negative` (Loss from random trading), the agent treats "Doing Nothing" as the optimal safe strategy.
4.  **The Trap**: The agent never explores enough to discover that *skilled* trading yields `Positive` rewards, because it is too afraid to enter the market in the first place.

---

## 2. Risk of Moving from Hard-Coded to Model-Decided Entry
Switching from a **Hard-Coded Entry** (where the agent *must* manage a trade) to a **Model-Decided Entry** (where the agent *chooses* when to trade) exacerbates this problem.

*   **Scenario**: The agent, previously trained to minimize loss in bad markets, may view *any* market participation as a "loss mitigation exercise" rather than a "profit opportunity."
*   **Result**: Given the choice, it will choose NOT to enter the "loss mitigation exercise" at all.

---

## 3. Mitigation Strategies

Here are the standard approaches to solve this, along with their pros and cons.

### Option A: Randomized Forced Entry ("The Training Wheels") - **Implemented Strategy**
**Mechanism**: In `reset()`, flip a coin (`force_entry_prob = 0.5`).

*   **50% of the time**: Force the agent into a trade at 9:30 (Hard Coded Entry).
    *   *Learning Objective*: **Management**. "I'm stuck in this trade, how do I minimize loss / make profit?"
*   **50% of the time**: Start with 0 lots (Model Decided).
    *   *Learning Objective*: **Timing**. "Is it worth entering right now?"

**Why this works**:
Since the agent learns "Management" from the forced episodes, it eventually realizes "Hey, I actually know how to make money here" (Positive Value Function). This confidence encourages it to start entering on its own during the "Free" episodes to seek those rewards, overcoming the inactivity bias.

| Pros | Cons |
| :--- | :--- |
| **Guaranteed Experience**: The agent *must* learn to manage trades because it is forced into them 50% of the time. | **Noise**: The agent effectively plays two different "games" (Management vs. Entry), which might destabilize the policy initially. |
| **Transfer Learning**: It learns that "Management" produces profit, which incentivizes it to "Enter" during the free episodes. | **Complexity**: Implementation requires modifying the environment reset logic. |

### Option B: Participation Reward (Reward Shaping)
**Mechanism**: Give a small positive reward for simply *having* an open position (e.g., `+0.001` per step).

| Pros | Cons |
| :--- | :--- |
| **Immediate Incentive**: Directly counters the fear of losing money. | **Gaming the System**: The agent might open huge positions just to collect the "holding bonus," regardless of market conditions. |
| **Simple**: Easy to implement in the reward function. | **False Signal**: It pollutes the true PnL objective. |

### Option C: Inactivity Penalty (The "Rent" Approach)
**Mechanism**: Give a small negative reward for *not* having a position (e.g., `-0.001` per step). "Time is money."

| Pros | Cons |
| :--- | :--- |
| **Urgency**: Forces the agent to find *something* to do. | **Forced Bad Trades**: If the market is truly terrible, the agent *should* wait. A penalty forces it to trade even in bad conditions to avoid the "rent." |

### Option D: Asymmetric Re-Entry (Top-Up Logic)
**Mechanism**: Allow the agent to re-enter, but strictly enforce that it must be a full Straddle/Strangle (paired), while exits can be partial.

| Pros | Cons |
| :--- | :--- |
| **Structural Safety**: Prevents "gambling" on naked legs to chase profit. | **Restrictive**: Limits the agent's ability to express pure directional views (though this is often desired in hedging). |

---

## 4. Recommendation for Thesis
For a robust "Deep Hedging" thesis, **Option A (Randomized Forced Entry)** is the most academically sound approach.
*   It does not artificially alter the reward function (keeping the PnL metric pure).
*   It represents a form of **Curriculum Learning**, a well-respected technique in Deep Learning.
*   It ensures the agent overcomes the "Cold Start" problem without biasing its final behavior (since you can anneal the probability to 0% over time).
