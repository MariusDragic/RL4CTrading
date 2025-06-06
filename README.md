# Deep Reinforcement Learning for Commodity Trading

## Overview

This project investigates the application of Deep Reinforcement Learning (DRL) to multi-asset trading strategies in highly volatile and non-trending commodity markets. Using a custom-designed trading environment and a Double Deep Q-Network (Double DQN) architecture enhanced with LSTM layers, we demonstrate the agent's ability to learn profitable and risk-adjusted behaviors when trading on gold, silver, and natural gas simultaneously.

A detailed report including theoretical background and extended results is available [here](https://github.com/MariusDragic/RL4CTrading/blob/main/Trading_Commodities_RL.pdf) for interested readers.

## Methodology

Commodity markets are notoriously noisy and non-bullish, rendering traditional long-only strategies ineffective. To tackle this complexity, we adopt a **value-based reinforcement learning framework** that approximates optimal Q-values via a Double DQN, supported by:

- A custom **Gym-style trading environment**
- Normalized price dynamics for simultaneous multi-asset trading
- A **discretized action space** allowing partial buys/sells
- A **reward function** encouraging net worth growth while penalizing unrealistic operations (e.g., selling unavailable assets)

We also benchmark our learned strategy against a classic **Bollinger Bands baseline**, assessing performance through advanced financial metrics like Sharpe Ratio, Calmar Ratio, and Max Drawdown.

## Reinforcement Learning Approach

We implement and train a **Double Deep Q-Network** agent that approximates the action-value function \( Q(s, a) \) via a recurrent LSTM-based architecture. To stabilize learning and reduce Q-value overestimation, the agent uses:

- A **replay buffer** to decorrelate samples
- A **target network** periodically updated
- **Epsilon-greedy exploration** with linear decay

The Bellman update for Double DQN is defined as:

```math
y_t = r_t + \gamma Q_{\theta^{-}}(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a'))
```

where $\theta$ and $\theta^{-}$ denote online and target networks respectively.

## Trading Environment

The agent interacts with a simulation environment built around real historical data (2010–2022) for gold, silver, and natural gas. Each state includes:

- OHLC prices
- Momentum and trend indicators (RSI, MACD, ADX)
- Volatility indicators (Bollinger Bands)
- CCI, volume, and moving averages

The action space is a discretized set of fractional orders:  
$−1.0, −0.75, ..., 0.75, 1.0$, where values correspond to percentage of capital to buy/sell.  

To ensure economic realism:
- Prices are normalized to [0.1, 1]
- Initial capital is capped
- A 2% transaction fee discourages overtrading

## Experimental Results

We compare the trained Double DQN agent with a rule-based Bollinger Band strategy using the following financial metrics:

| Metric                  | DQN     | Bollinger |
|-------------------------|---------|-----------|
| Expected Return E(R)    | 0.0022  | 0.0006    |
| Sharpe Ratio            | 0.045   | 0.0191    |
| Sortino Ratio           | 0.0665  | 0.025     |
| Max Drawdown (MDD)      | 0.5266  | 0.6072    |
| Calmar Ratio            | 0.0041  | 0.0009    |
| Avg. Profit / Loss      | 1.023   | 0.9301    |

**Conclusion**:  
The RL agent significantly outperforms the deterministic baseline across most metrics, especially in risk-adjusted return and capital preservation.

### Visual Analysis of Trading Performance

<p align="center">
  <img src="images/Net_worth.png" alt="Net worth evolution with Double DQN" width="80%">
</p>

*Figure 1 – Evolution of cumulative net worth (%) on the 2020-2022 test period for the Double DQN strategy.  
The agent achieves a peak of +300 % before stabilising around +150 %, illustrating both growth potential and exposure to drawdowns.*

<p align="center">
  <img src="images/NG_trading.png" alt="Natural Gas trading behaviour" width="80%">
</p>

*Figure 2 – Trading behaviour of the Double DQN agent on Natural Gas. Top : price series with buy and sell signals. Middle : action intensity (discretised order fraction). Bottom : Shares Held*

The agent progressively builds a long position during the sustained up-trend while scaling out when momentum weakens, showing its ability to adapt position sizing to changing market conditions.*


## Key Takeaways

- Double DQN is a powerful framework for capturing the stochasticity of financial markets.
- The use of LSTMs enhances the agent’s ability to model temporal dependencies.
- Despite simplifications (normalized prices, discrete actions), the trained model achieves substantial cumulative returns.

## Future Work

Possible directions include:
- Transitioning to **continuous action spaces** with TD3 or SAC
- Exploring **Actor-Critic architectures** (e.g., A2C)
- Integrating **attention mechanisms** to refine temporal feature modeling
- Extending to **real unnormalized prices** with heterogeneous transaction costs

## Authors

- **Marius Dragic** — CentraleSupélec  
- **Lancelin Poulet** — CentraleSupélec

## References

1. Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013  
2. Zhang et al., *Deep Reinforcement Learning for Trading*, 2019  
3. Yang et al., *Ensemble Strategy for Automated Stock Trading*, 2020  




