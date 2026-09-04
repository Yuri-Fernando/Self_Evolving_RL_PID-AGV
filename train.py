from env.agv_env import AGVEnv
from agent.dqn_agent import DQNAgent
import numpy as np

env = AGVEnv()
agent = DQNAgent(action_dim=7)

gains = np.array([0.1, 0.01, 0.001])

for episode in range(100):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)

        if action == 0: gains[0] += 0.01
        elif action == 1: gains[0] -= 0.01
        elif action == 2: gains[1] += 0.001
        elif action == 3: gains[1] -= 0.001
        elif action == 4: gains[2] += 0.0001
        elif action == 5: gains[2] -= 0.0001

        next_state, reward, done = env.step(gains)
        agent.update(state, action, reward, next_state)
        state = next_state
