import numpy as np

class DQNAgent:
    def __init__(self, action_dim):
        self.epsilon = 0.1
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return 0  # placeholder (policy learned depois)

    def update(self, state, action, reward, next_state):
        pass  # atualização da Q-network (simplificada)
