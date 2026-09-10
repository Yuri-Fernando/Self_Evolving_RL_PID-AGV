"""
train_robust_v3.py

Treino de Robust RL: a mesma `DualPolicyValueNetwork` da V2, mas treinada
por REINFORCE (policy gradient com baseline da value head) contra o
`RobustAGVEnv` — o ambiente que perturba a observação de estado a cada
passo (ruído de sensor + spoofing adversarial numa fração dos passos).

Sem MCTS no loop de treino (a rede sozinha amostra a ação), então roda em
segundos. O resultado é uma política de ajuste de ganho PID que tolera
observação corrompida.

Uso:
    python train_robust_v3.py            # treina e salva agent_robust_v3.pt
    python train_robust_v3.py --nominal  # baseline: mesma rede, env nominal
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from adversarial.robust_training import RobustAGVEnv
from agent.gains_actions import ACTION_DIM, apply_action
from agent.network_v2 import DualPolicyValueNetwork
from env.agv_env import AGVEnv

INIT_GAINS = np.array([0.1, 0.01, 0.001])
GAMMA = 0.99


def _discount(rewards: list[float], gamma: float = GAMMA) -> torch.Tensor:
    out = np.zeros(len(rewards), dtype=np.float32)
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        out[t] = run
    return torch.tensor(out)


def train(robust: bool = True, episodes: int = 250, lr: float = 3e-3, seed: int = 0,
          max_steps: int = 200) -> DualPolicyValueNetwork:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = DualPolicyValueNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for ep in range(episodes):
        env = RobustAGVEnv(seed=seed + ep) if robust else AGVEnv()
        state = env.reset()
        gains = INIT_GAINS.copy()
        logps, values, rewards = [], [], []
        done = False
        t = 0
        while not done and t < max_steps:
            s = torch.tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            logits, value = net(s)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            logps.append(dist.log_prob(action))
            values.append(value.squeeze())
            gains = apply_action(gains, int(action.item()))
            state, reward, done = env.step(gains)
            rewards.append(reward)
            t += 1

        returns = _discount(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        values_t = torch.stack(values)
        advantage = returns - values_t.detach()
        policy_loss = -(torch.stack(logps) * advantage).sum()
        value_loss = F.mse_loss(values_t, returns)
        loss = policy_loss + 0.5 * value_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(f"ep {ep:03d} | return {sum(rewards):.3f} | loss {loss.item():.3f}")

    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", action="store_true", help="treina no env nominal (baseline)")
    ap.add_argument("--episodes", type=int, default=250)
    args = ap.parse_args()

    net = train(robust=not args.nominal, episodes=args.episodes)
    path = "agent_nominal_v3.pt" if args.nominal else "agent_robust_v3.pt"
    torch.save(net.state_dict(), path)
    print(f"salvo: {path}")


if __name__ == "__main__":
    main()
