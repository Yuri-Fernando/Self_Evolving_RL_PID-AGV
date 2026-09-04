"""
evaluate_v2.py

Compara o controlador V1 (ganhos PID fixos) e o agente V2 (MCTS + rede dual)
sob as mesmas condições, medindo o erro médio de rastreamento (menor é
melhor). Espera um modelo já treinado por `train_v2.py` (arquivo
`agent_v2.pt`); se não encontrar, avisa e roda só o benchmark da V1.

Uso:
    python train_v2.py      # treina e salva agent_v2.pt
    python evaluate_v2.py   # compara V1 x V2
"""

import os

import numpy as np

from agent.gains_actions import apply_action
from agent.pmcts_agent_v2 import PolicyValueMCTSAgent
from env.agv_env import AGVEnv
from evaluate import evaluate as evaluate_fixed_pid

MODEL_PATH = "agent_v2.pt"


def evaluate_v2(agent, episodes=10):
    env = AGVEnv()
    errors = []

    for _ in range(episodes):
        state = env.reset()
        gains = np.array([0.1, 0.01, 0.001])
        done = False
        ep_error = []

        while not done:
            action, _ = agent.select_action(env, state, gains, greedy=True)
            gains = apply_action(gains, action)
            state, reward, done = env.step(gains)
            ep_error.append(abs(state[0]))

        errors.append(np.mean(ep_error))

    return np.mean(errors), np.std(errors)


if __name__ == "__main__":
    fixed_pid = np.array([0.1, 0.01, 0.001])
    mean_fixed, std_fixed = evaluate_fixed_pid(fixed_pid)
    print(f"V1 - PID Fixo             | Mean Error: {mean_fixed:.4f} | Std: {std_fixed:.4f}")

    if os.path.exists(MODEL_PATH):
        agent = PolicyValueMCTSAgent()
        agent.load(MODEL_PATH)
        mean_v2, std_v2 = evaluate_v2(agent)
        print(f"V2 - MCTS + Rede Dual     | Mean Error: {mean_v2:.4f} | Std: {std_v2:.4f}")
    else:
        print(
            f"Aviso: '{MODEL_PATH}' não encontrado. "
            "Rode 'python train_v2.py' antes para treinar e comparar a V2."
        )
