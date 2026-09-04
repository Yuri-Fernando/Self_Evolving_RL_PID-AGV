"""
agent/mcts_v2.py

Monte Carlo Tree Search guiado por rede dual (PUCT, no estilo
AlphaGo/AlphaZero), adaptado para o problema de controle de único agente do
AGV em vez de um jogo adversarial de dois jogadores.

Diferença chave em relação ao MCTS clássico de jogos: aqui não existe
oponente. A árvore representa sequências de AJUSTES de ganho PID (as mesmas
7 ações discretas da V1) e usa o próprio ambiente simulado (`AGVEnv`) como
modelo forward para prever o efeito de cada ajuste antes de aplicá-lo de
verdade no episódio. Como a dinâmica do ambiente é estocástica (ruído em
`AGVEnv.step`), cada simulação amostra uma trajetória diferente a partir do
mesmo nó — a árvore acaba refletindo uma média sobre as transições
possíveis, e não um valor determinístico único.
"""

import copy
import math

import numpy as np

from agent.gains_actions import ACTION_DIM, apply_action


class MCTSNode:
    def __init__(self, gains, state, parent=None, prior=0.0):
        self.gains = gains
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0  # recompensa observada ao entrar neste nó

    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network, c_puct=1.5, n_simulations=32):
        self.network = network
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, env, state, gains):
        """Executa `n_simulations` simulações a partir de (state, gains) e
        devolve a distribuição de probabilidade sobre as 7 ações, derivada
        do número de visitas de cada filho da raiz (quanto mais simulações
        passaram por um ramo, mais promissor ele é)."""
        root = MCTSNode(gains=gains, state=state)
        self._expand(root)

        for _ in range(self.n_simulations):
            sim_env = copy.deepcopy(env)
            node = root
            path = [node]
            done = False

            while node.children:
                action, child = self._select_child(node)
                next_state, reward, done = sim_env.step(child.gains)
                child.state = next_state
                child.reward = reward
                node = child
                path.append(node)
                if done:
                    break

            leaf_value = 0.0
            if not done:
                self._expand(node)
                _, leaf_value = self.network.predict(node.state)

            self._backup(path, leaf_value)

        return self._visit_distribution(root)

    def _expand(self, node):
        priors, _ = self.network.predict(node.state)
        for action in range(ACTION_DIM):
            child_gains = apply_action(node.gains, action)
            node.children[action] = MCTSNode(
                gains=child_gains, state=None, parent=node, prior=priors[action]
            )

    def _select_child(self, node):
        best_score, best_action, best_child = -math.inf, None, None
        for action, child in node.children.items():
            exploration = (
                self.c_puct * child.prior * math.sqrt(node.visit_count + 1) / (1 + child.visit_count)
            )
            score = child.q_value() + exploration
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _backup(self, path, leaf_value):
        cumulative = leaf_value
        for node in reversed(path):
            cumulative = node.reward + cumulative if node.parent is not None else cumulative
            node.visit_count += 1
            node.value_sum += cumulative

    def _visit_distribution(self, root):
        counts = np.array(
            [root.children[a].visit_count for a in range(ACTION_DIM)], dtype=np.float32
        )
        if counts.sum() == 0:
            return np.ones(ACTION_DIM) / ACTION_DIM
        return counts / counts.sum()
