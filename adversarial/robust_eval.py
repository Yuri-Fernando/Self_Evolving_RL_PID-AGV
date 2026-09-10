"""Avaliação de robustez do controle sob ataque de observação.

Compara controladores (PID fixo, RL+PID, Robust RL) em cenário nominal vs.
cada ataque de `state_attacks`, com as métricas clássicas de rastreamento:

- `mean_abs_error` — erro médio absoluto de trajetória;
- `IAE`  — integral do erro absoluto;
- `ITAE` — integral de (tempo · erro absoluto), penaliza erro que persiste;
- `return` — retorno acumulado (recompensa);
- `perturbation_tolerance` — maior epsilon de spoofing antes do retorno cair
  abaixo de `tol_frac` do retorno nominal.

Um controlador é `policy_fn(state, gains) -> gains'` (ajusta os ganhos a
partir da observação). PID fixo é `lambda s, g: g`.
"""
from __future__ import annotations

import numpy as np

from adversarial.state_attacks import Latency, gaussian_noise, spoofing
from env.agv_env import AGVEnv

INIT_GAINS = np.array([0.1, 0.01, 0.001])


def rollout(
    policy_fn,
    attack=None,
    episodes: int = 10,
    max_steps: int = 200,
    seed: int = 0,
) -> dict:
    """`attack(state, gains, policy_fn) -> state_obs` (ou None para nominal)."""
    metrics = {"mean_abs_error": [], "IAE": [], "ITAE": [], "return": []}

    for ep in range(episodes):
        env = AGVEnv()
        state = env.reset()
        gains = INIT_GAINS.copy()
        done = False
        t = 0
        abs_errs, ret = [], 0.0

        while not done and t < max_steps:
            obs = attack(state, gains, policy_fn) if attack is not None else state
            gains = np.asarray(policy_fn(obs, gains), float)
            state, reward, done = env.step(gains)
            err = abs(state[0])
            abs_errs.append(err)
            ret += reward
            t += 1

        abs_errs = np.array(abs_errs)
        dt = env.dt
        metrics["mean_abs_error"].append(float(abs_errs.mean()))
        metrics["IAE"].append(float(abs_errs.sum() * dt))
        metrics["ITAE"].append(float((abs_errs * np.arange(len(abs_errs)) * dt).sum() * dt))
        metrics["return"].append(float(ret))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def perturbation_tolerance(
    policy_fn,
    epsilons=(0.0, 0.05, 0.1, 0.2, 0.4),
    tol_frac: float = 0.7,
    episodes: int = 6,
) -> float:
    nominal = rollout(policy_fn, attack=None, episodes=episodes)["return"]
    threshold = nominal * tol_frac if nominal > 0 else nominal / tol_frac
    tol = 0.0
    for eps in sorted(epsilons):
        atk = lambda s, g, p, e=eps: spoofing(s, g, p, epsilon=e)
        ret = rollout(policy_fn, attack=atk, episodes=episodes)["return"]
        if ret >= threshold:
            tol = eps
        else:
            break
    return tol


def compare(policies: dict, attacks: dict, episodes: int = 8) -> dict:
    """`{nome_politica: {nome_cenario: metrics}}` para nominal + cada ataque."""
    out: dict[str, dict] = {}
    for pname, pfn in policies.items():
        out[pname] = {"nominal": rollout(pfn, attack=None, episodes=episodes)}
        for aname, afn in attacks.items():
            out[pname][aname] = rollout(pfn, attack=afn, episodes=episodes)
        out[pname]["perturbation_tolerance"] = perturbation_tolerance(pfn, episodes=max(episodes // 2, 3))
    return out
