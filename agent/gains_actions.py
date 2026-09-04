"""
agent/gains_actions.py

Mapeamento de ações -> ajuste de ganhos PID, extraído da V1 (estava hardcoded
dentro de train.py) para ser reutilizado tanto pelo DQNAgent original quanto
pela rede dual e pelo MCTS da V2. Mantém o comportamento idêntico ao da V1:
7 ações discretas, 6 delas incrementam/decrementam Kp, Ki ou Kd e a última
funciona como no-op.
"""

import numpy as np

ACTION_DIM = 7


def apply_action(gains, action):
    """Aplica a mesma lógica de ajuste de ganho PID usada na V1 (train.py)."""
    gains = np.array(gains, dtype=np.float64, copy=True)

    if action == 0:
        gains[0] += 0.01
    elif action == 1:
        gains[0] -= 0.01
    elif action == 2:
        gains[1] += 0.001
    elif action == 3:
        gains[1] -= 0.001
    elif action == 4:
        gains[2] += 0.0001
    elif action == 5:
        gains[2] -= 0.0001
    # action == 6 -> no-op (mantém os ganhos atuais)

    return gains
