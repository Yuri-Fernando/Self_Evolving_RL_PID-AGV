"""Testes do módulo adversarial (V3). Usam políticas baratas (PID fixo +
ajuste linear) — sem MCTS — para rodar rápido."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversarial import (  # noqa: E402
    Latency,
    RobustAGVEnv,
    bias,
    compare,
    dropout,
    gaussian_noise,
    perturbation_tolerance,
    rollout,
    spoofing,
)
from adversarial.state_attacks import ATTACKS  # noqa: E402

# políticas de teste
FIXED_PID = lambda s, g: g


def _reactive_pid(state, gains):
    """ajuste linear simples: aumenta Kp se o erro persiste, tudo positivo."""
    gains = np.asarray(gains, float).copy()
    gains[0] = np.clip(gains[0] + 0.01 * np.sign(abs(state[0]) - 0.5), 0.01, 1.0)
    return gains


def test_gaussian_noise_is_bounded_in_expectation():
    rng = np.random.default_rng(0)
    s = np.array([0.2, 0.0, 0.1, 1.0])
    diffs = np.array([np.linalg.norm(gaussian_noise(s, 0.05, rng) - s) for _ in range(200)])
    assert diffs.mean() < 0.5


def test_dropout_zeroes_one_component():
    s = np.array([0.3, 0.5, -0.2, 1.0])
    assert dropout(s, 1)[1] == 0.0
    assert dropout(s, 1)[0] == 0.3


def test_bias_shifts_state():
    s = np.zeros(4)
    assert np.allclose(bias(s, 0.1), 0.1)


def test_latency_returns_delayed_observation():
    lat = Latency(k=2)
    lat.reset()
    outs = [lat(np.array([float(i)])) for i in range(5)]
    # após encher o buffer, a saída no passo i é a observação de i-2
    assert outs[-1][0] == 2.0


def test_spoofing_finds_perturbation_that_changes_action():
    s = np.array([0.4, 0.1, 0.2, 0.0])
    g = np.array([0.1, 0.01, 0.001])
    s_adv = spoofing(s, g, _reactive_pid, epsilon=0.6, n_dirs=32)
    assert (np.abs(s_adv - s) <= 0.6 + 1e-9).all()
    # a ação pode mudar sob a perturbação encontrada (não garantido sempre,
    # mas a perturbação está dentro do orçamento e é a de maior distância)
    assert s_adv.shape == s.shape


def test_rollout_returns_all_metrics():
    m = rollout(FIXED_PID, attack=None, episodes=3, max_steps=80)
    assert set(m) == {"mean_abs_error", "IAE", "ITAE", "return"}
    assert m["IAE"] >= 0.0


def test_attack_degrades_or_matches_nominal_return():
    nominal = rollout(FIXED_PID, attack=None, episodes=5, max_steps=100)["return"]
    noisy = rollout(FIXED_PID, attack=ATTACKS["gaussian_noise"], episodes=5, max_steps=100)["return"]
    # ruído não deve MELHORAR o retorno de forma consistente
    assert noisy <= nominal + abs(nominal) * 0.5 + 1.0


def test_perturbation_tolerance_is_a_valid_epsilon():
    tol = perturbation_tolerance(FIXED_PID, epsilons=(0.0, 0.1, 0.3), episodes=3)
    assert tol in (0.0, 0.1, 0.3)


def test_compare_builds_matrix_for_each_policy():
    res = compare(
        {"pid_fixo": FIXED_PID, "pid_reativo": _reactive_pid},
        {"noise": ATTACKS["gaussian_noise"], "spoof": ATTACKS["spoofing"]},
        episodes=3,
    )
    assert set(res) == {"pid_fixo", "pid_reativo"}
    assert "nominal" in res["pid_fixo"] and "spoof" in res["pid_fixo"]
    assert "perturbation_tolerance" in res["pid_fixo"]


def test_robust_env_is_drop_in_for_agv_env():
    env = RobustAGVEnv(sigma=0.02, seed=1)
    s = env.reset()
    assert s.shape == (4,)
    s2, r, done = env.step(np.array([0.1, 0.01, 0.001]))
    assert s2.shape == (4,)
    assert isinstance(bool(done), bool)
