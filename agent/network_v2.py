"""
agent/network_v2.py

Rede Dual (Policy + Value) que substitui a Q-network placeholder da V1
(dqn_agent.py tinha `select_action`/`update` vazios). Uma única "torso" MLP
compartilhada extrai features do estado do AGV (erro lateral, integral do
erro, derivada do erro, tipo de trajetória) e alimenta duas cabeças:

- Policy head: logits sobre as 7 ações discretas de ajuste de Kp/Ki/Kd,
  usados como prior para guiar o MCTS (agent/mcts_v2.py).
- Value head: estimativa escalar V(s) do retorno esperado a partir do
  estado atual, usada para avaliar folhas do MCTS sem precisar simular até
  o fim do episódio.

Essa é a mesma ideia de "dual network" popularizada por AlphaGo/AlphaZero,
aqui adaptada de um jogo de tabuleiro para um problema de controle contínuo
de único agente.
"""

import torch
import torch.nn as nn

from agent.gains_actions import ACTION_DIM

STATE_DIM = 4  # erro, integral do erro, derivada do erro, tipo de trajetória


class DualPolicyValueNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=64):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.torso(state)
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features))
        return policy_logits, value

    @torch.no_grad()
    def predict(self, state_np):
        """Interface numpy-friendly usada pelo MCTS: recebe um estado (array
        de tamanho STATE_DIM) e devolve (probs_de_acao, valor_escalar)."""
        state_t = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs, float(value.item())
