# Adaptive PID Control using Reinforcement Learning

## Overview
This project presents a lightweight reinforcement learning prototype where an RL agent learns to adapt PID controller gains to improve trajectory tracking accuracy of an Automated Guided Vehicle (AGV) in a simulated environment.

The goal is not to replace classical control strategies, but to demonstrate how RL can be used as an adaptive layer over an industrially accepted controller.

## Problem
Fixed PID gains often fail to maintain optimal performance under changes in trajectory, noise, or system asymmetry. This prototype investigates whether an RL agent can learn to adjust PID gains online to improve stability and accuracy.

## Environment
- Simulated AGV with simplified dynamics
- Straight and curved trajectories
- Controlled noise and asymmetry

### State
- Lateral tracking error
- Integral of error
- Error derivative
- Trajectory type

### Actions
Discrete actions that increment or decrement PID gains (Kp, Ki, Kd).

### Reward
Negative tracking error with stability penalty.

## Agent
- Reinforcement Learning agent (DQN-like, simplified)
- ε-greedy exploration
- Online learning

## Results
The adaptive controller shows reduced tracking error compared to fixed PID gains, demonstrating the feasibility of RL-based adaptive tuning in a computationally lightweight setup.

## Purpose
This prototype is intended as an experimental tool to test hypotheses and guide future development, not as a production-ready controller.

---

## Version 2.0 — Architecture Extension

The V1 agent (`agent/dqn_agent.py`) was a DQN-shaped placeholder: `select_action` and `update`
had no real learning behind them. V2 keeps V1 untouched and adds a second, fully implemented
agent alongside it, built around a dual network and search-based planning instead of a bare
ε-greedy stub.

### What was added
- **Dual Network** (`agent/network_v2.py`): one shared MLP torso with two heads — a **Policy
  head** producing a prior distribution over the 7 gain-adjustment actions, and a **Value
  head** estimating the expected return from a given state.
- **Policy Network**: the policy head above, used to guide tree search instead of a flat
  softmax over raw Q-values.
- **Value Network**: the value head above, used to evaluate leaf nodes during search without
  rolling out full episodes.
- **Monte Carlo Tree Search** (`agent/mcts_v2.py`): a PUCT-style search (AlphaZero-flavored)
  adapted from two-player games to a single-agent control problem — the tree searches over
  sequences of PID gain adjustments, using the environment itself as the forward model.
- **Policy+Value MCTS Agent** (`agent/pmcts_agent_v2.py`): wraps the dual network and MCTS
  into an agent trained by self-play-style updates — the network learns to imitate the MCTS
  visit distribution (policy loss) and the observed discounted return (value loss).
- **Training / evaluation scripts**: `train_v2.py` and `evaluate_v2.py`, mirroring `train.py`
  and `evaluate.py` but wired to the new agent; `main_v2.ipynb` mirrors the structure of
  `main.ipynb` (imports → init → training loop → results → V1 vs V2 benchmark).

### Why
Adding MCTS on top of a value/policy network is only meaningful once both heads produce
useful signal, so the priority was implementing a real dual-network agent first (closing the
gap left by the empty V1 stub) and only then layering search on top of it — the same
progression used in AlphaGo/AlphaZero-style systems (policy+value network first, tree search
second).

### Architecture

```
                    AGV Environment
                           |
                           v
                      State s
                           |
                           v
                 +-------------------+
                 |   Dual Network    |
                 |                  |
                 | Policy Head p(a) |
                 | Value Head  V(s) |
                 +--------+---------+
                          |
                    Policy + Value
                          |
                          v
                         MCTS
                          |
                          v
                    Best Action
                          |
                          v
                    PID Adaptation
                          |
                          v
                         AGV
```

### V1 vs V2

| | V1 | V2 |
|---|---|---|
| Agent | DQN-shaped placeholder (no real update) | Dual Network + MCTS |
| Decision | random / no-op fallback | tree search guided by learned priors |
| Value estimation | none | learned value head |
| Training signal | none (stub) | policy cross-entropy + value MSE |

Run `python train_v2.py` to train and save `agent_v2.pt`, then `python evaluate_v2.py` to
benchmark it against the V1 fixed-PID baseline. See `main_v2.ipynb` for the full walkthrough
with plots.
