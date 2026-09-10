"""Adversarial / Robust RL (V3) — ataques sobre a observação de estado do
AGV e avaliação/treino de robustez, por cima da V2 (Dual Network + MCTS).

- `state_attacks`: ruído de sensor, bias, dropout, latência e spoofing
  adversarial (worst-case por busca) sobre a observação.
- `robust_eval`: PID fixo vs. RL+PID vs. Robust RL, nominal vs. ataque, com
  métricas de rastreamento (erro médio, IAE, ITAE, retorno, tolerância a
  perturbação).
- `robust_training`: `RobustAGVEnv` (drop-in de `AGVEnv` com observação
  perturbada) + fine-tune por domain randomization.
"""
from __future__ import annotations

from adversarial.robust_eval import compare, perturbation_tolerance, rollout
from adversarial.robust_training import RobustAGVEnv, domain_randomized_finetune
from adversarial.state_attacks import ATTACKS, Latency, bias, dropout, gaussian_noise, spoofing

__all__ = [
    "rollout",
    "compare",
    "perturbation_tolerance",
    "RobustAGVEnv",
    "domain_randomized_finetune",
    "ATTACKS",
    "gaussian_noise",
    "bias",
    "dropout",
    "Latency",
    "spoofing",
]
