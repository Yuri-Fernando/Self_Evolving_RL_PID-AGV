"""
evaluate_robust_v3.py

Compara três controladores sob os ataques de observação de `adversarial/`:

- PID fixo
- Política treinada no ambiente NOMINAL (agent_nominal_v3.pt)
- Política treinada no RobustAGVEnv (agent_robust_v3.pt)

Métricas por cenário (nominal + cada ataque): erro médio absoluto, IAE,
ITAE, retorno, e tolerância a perturbação.

Uso:
    python train_robust_v3.py --nominal
    python train_robust_v3.py
    python evaluate_robust_v3.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from adversarial.robust_eval import compare
from adversarial.state_attacks import ATTACKS
from agent.gains_actions import apply_action
from agent.network_v2 import DualPolicyValueNetwork

FIXED_PID = lambda s, g: g


def load_policy(path: str):
    net = DualPolicyValueNetwork()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def policy_fn(state, gains):
        with torch.no_grad():
            s = torch.tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            logits, _ = net(s)
            action = int(torch.argmax(logits.squeeze(0)).item())
        return apply_action(gains, action)

    return policy_fn


def main() -> None:
    policies = {"pid_fixo": FIXED_PID}
    for name, path in (("policy_nominal", "agent_nominal_v3.pt"),
                       ("policy_robust", "agent_robust_v3.pt")):
        if os.path.exists(path):
            policies[name] = load_policy(path)
        else:
            print(f"aviso: {path} não encontrado — rode train_robust_v3.py antes")

    results = compare(policies, ATTACKS, episodes=12)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
