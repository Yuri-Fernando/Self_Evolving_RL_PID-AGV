"""Ataques sobre a observação de estado do AGV — s_t^adv = s_t + delta.

Modelam degradações reais de sensor/comunicação num AGV:

- `gaussian_noise`  — ruído de leitura (LiDAR/encoder).
- `bias`            — offset sistemático de calibração.
- `dropout`         — perda de um componente da observação (falha de sensor).
- `latency`         — o controlador vê a observação de k passos atrás.
- `spoofing`        — perturbação ADVERSARIAL: dentro de uma bola L-inf de
                      raio epsilon, escolhe o delta que mais afasta a ação
                      da política da ação "nominal" (worst-case por busca,
                      já que a política é uma caixa-preta consultável).

Todos recebem e devolvem um vetor de estado numpy; `latency` é stateful e
por isso é uma classe.
"""
from __future__ import annotations

import numpy as np


def gaussian_noise(state: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return np.asarray(state, float) + rng.normal(0.0, sigma, size=np.shape(state))


def bias(state: np.ndarray, offset: float = 0.1, mask: np.ndarray | None = None) -> np.ndarray:
    s = np.asarray(state, float).copy()
    if mask is None:
        s = s + offset
    else:
        s[mask] = s[mask] + offset
    return s


def dropout(state: np.ndarray, index: int = 0) -> np.ndarray:
    s = np.asarray(state, float).copy()
    s[index] = 0.0
    return s


class Latency:
    """Devolve a observação de `k` passos atrás (buffer FIFO)."""

    def __init__(self, k: int = 1):
        self.k = k
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf = []

    def __call__(self, state: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(state, float).copy())
        if len(self._buf) > self.k + 1:
            self._buf.pop(0)
        return self._buf[0]


def spoofing(
    state: np.ndarray,
    gains: np.ndarray,
    policy_fn,
    epsilon: float = 0.1,
    n_dirs: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Perturbação adversarial de raio `epsilon` (L-inf) que maximiza a
    distância entre `policy_fn(s+delta, gains)` e `policy_fn(s, gains)`.

    Busca aleatória sobre `n_dirs` direções de sinal — sem gradiente,
    só consultas à política (caixa-preta)."""
    state = np.asarray(state, float)
    rng = np.random.default_rng(seed)
    base_action = np.asarray(policy_fn(state, gains), float)

    best_delta = np.zeros_like(state)
    best_dist = -1.0
    for _ in range(n_dirs):
        delta = rng.choice([-1.0, 1.0], size=state.shape) * epsilon
        cand_action = np.asarray(policy_fn(state + delta, gains), float)
        dist = float(np.linalg.norm(cand_action - base_action))
        if dist > best_dist:
            best_dist, best_delta = dist, delta
    return state + best_delta


ATTACKS = {
    "gaussian_noise": lambda s, g, p: gaussian_noise(s),
    "bias": lambda s, g, p: bias(s, 0.1),
    "dropout": lambda s, g, p: dropout(s, 0),
    "spoofing": lambda s, g, p: spoofing(s, g, p, epsilon=0.1),
}
