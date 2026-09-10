"""Robust RL por domain randomization + treino adversarial de observação.

Envolve `AGVEnv` de forma que, durante o treino, a observação passada ao
agente é perturbada (ruído aleatório na maior parte dos passos + spoofing
adversarial numa fração deles). O agente aprende uma política de ajuste de
ganhos que tolera observação degradada.

`RobustAGVEnv` mantém a mesma API de `AGVEnv` (`reset`, `step`), então
serve de drop-in para `train_v2.py` / `PolicyValueMCTSAgent`.
"""
from __future__ import annotations

import numpy as np

from adversarial.state_attacks import gaussian_noise, spoofing
from env.agv_env import AGVEnv


class RobustAGVEnv:
    def __init__(
        self,
        sigma: float = 0.03,
        adv_prob: float = 0.25,
        adv_epsilon: float = 0.08,
        policy_fn=None,
        seed: int = 0,
    ):
        self._env = AGVEnv()
        self.dt = self._env.dt
        self.sigma = sigma
        self.adv_prob = adv_prob
        self.adv_epsilon = adv_epsilon
        self.policy_fn = policy_fn  # opcional: usado pelo spoofing adversarial
        self._rng = np.random.default_rng(seed)
        self._last_true_state = None

    def _perturb(self, state: np.ndarray, gains: np.ndarray) -> np.ndarray:
        if self.policy_fn is not None and self._rng.random() < self.adv_prob:
            return spoofing(state, gains, self.policy_fn, epsilon=self.adv_epsilon,
                            seed=int(self._rng.integers(1 << 30)))
        return gaussian_noise(state, self.sigma, rng=self._rng)

    def reset(self) -> np.ndarray:
        s = self._env.reset()
        self._last_true_state = s
        return gaussian_noise(s, self.sigma, rng=self._rng)

    def step(self, gains: np.ndarray):
        next_state, reward, done = self._env.step(gains)
        self._last_true_state = next_state
        obs = self._perturb(next_state, np.asarray(gains, float))
        return obs, reward, done


def domain_randomized_finetune(agent, episodes: int = 60, **env_kwargs):
    """Fine-tune curto de um agente que exponha `select_action(env, state,
    gains)` + `train_step(...)` (interface do PolicyValueMCTSAgent) contra o
    ambiente robusto. Importado sob demanda para não acoplar este módulo ao
    agente V2 pesado."""
    from agent.gains_actions import apply_action  # noqa: PLC0415

    try:
        from train_v2 import discount_returns  # noqa: PLC0415
    except Exception:  # fallback local
        def discount_returns(rewards, gamma=0.99):
            out = np.zeros(len(rewards), dtype=np.float32)
            run = 0.0
            for t in reversed(range(len(rewards))):
                run = rewards[t] + gamma * run
                out[t] = run
            return out

    env = RobustAGVEnv(**env_kwargs)
    for _ in range(episodes):
        state = env.reset()
        gains = np.array([0.1, 0.01, 0.001])
        done = False
        S, P, R = [], [], []
        while not done:
            action, probs = agent.select_action(env, state, gains)
            gains = apply_action(gains, action)
            state, reward, done = env.step(gains)
            S.append(state); P.append(probs); R.append(reward)
        agent.train_step(S, P, discount_returns(R))
    return agent
