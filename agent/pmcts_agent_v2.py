"""
agent/pmcts_agent_v2.py

Agente V2: substitui o DQNAgent placeholder da V1 (que tinha `select_action`
e `update` vazios) por uma política guiada por MCTS + rede dual
(Policy/Value), no espírito de agentes tipo AlphaZero adaptados para
controle de único agente.
"""

import numpy as np
import torch
import torch.nn.functional as F

from agent.gains_actions import ACTION_DIM
from agent.mcts_v2 import MCTS
from agent.network_v2 import DualPolicyValueNetwork


class PolicyValueMCTSAgent:
    def __init__(self, state_dim=4, hidden_dim=64, n_simulations=32, c_puct=1.5, lr=1e-3):
        self.network = DualPolicyValueNetwork(
            state_dim=state_dim, action_dim=ACTION_DIM, hidden_dim=hidden_dim
        )
        self.mcts = MCTS(self.network, c_puct=c_puct, n_simulations=n_simulations)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, env, state, gains, temperature=1.0, greedy=False):
        """Roda o MCTS a partir do estado/ganhos atuais e escolhe uma ação.

        `greedy=True` (usado na avaliação) sempre escolhe a ação mais
        visitada pela busca; caso contrário, amostra da distribuição de
        visitas ajustada por `temperature` (usado no treino, para manter
        exploração)."""
        action_probs = self.mcts.search(env, state, gains)

        if greedy or temperature == 0:
            action = int(np.argmax(action_probs))
        else:
            sharpened = action_probs ** (1.0 / temperature)
            sharpened /= sharpened.sum()
            action = int(np.random.choice(ACTION_DIM, p=sharpened))

        return action, action_probs

    def train_step(self, states, mcts_probs, returns):
        """Atualização estilo AlphaZero: a policy head aprende a imitar a
        distribuição de visitas do MCTS (cross-entropy) e a value head
        aprende a prever o retorno observado (MSE) -- assim, com o tempo, a
        rede sozinha (sem busca) já aproxima o que o MCTS descobriria."""
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        target_probs_t = torch.as_tensor(np.array(mcts_probs), dtype=torch.float32)
        returns_t = torch.as_tensor(np.array(returns), dtype=torch.float32).unsqueeze(-1)

        logits, values = self.network(states_t)
        log_probs = F.log_softmax(logits, dim=-1)

        policy_loss = -(target_probs_t * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
        self.network.eval()
