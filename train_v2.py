"""
train_v2.py

Loop de treino da V2: em cada passo do episódio, o MCTS usa a rede dual para
planejar o melhor ajuste de ganho PID; o ambiente executa a ação escolhida e
os alvos (distribuição de visitas do MCTS + retorno descontado observado)
alimentam o treino supervisionado da rede -- a mesma lógica de "treino por
auto-jogo" do AlphaZero, adaptada para um único agente de controle em vez de
um jogo de dois jogadores.

Uso:
    python train_v2.py
"""

import numpy as np

from agent.gains_actions import apply_action
from agent.pmcts_agent_v2 import PolicyValueMCTSAgent
from env.agv_env import AGVEnv

EPISODES = 100
GAMMA = 0.99
MODEL_PATH = "agent_v2.pt"


def discount_returns(rewards, gamma=GAMMA):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_episode(env, agent, temperature=1.0):
    state = env.reset()
    gains = np.array([0.1, 0.01, 0.001])
    done = False

    ep_states, ep_probs, ep_rewards = [], [], []

    while not done:
        action, action_probs = agent.select_action(env, state, gains, temperature=temperature)
        gains = apply_action(gains, action)
        next_state, reward, done = env.step(gains)

        ep_states.append(state)
        ep_probs.append(action_probs)
        ep_rewards.append(reward)

        state = next_state

    return ep_states, ep_probs, ep_rewards


def main():
    env = AGVEnv()
    agent = PolicyValueMCTSAgent()
    history = []

    for episode in range(EPISODES):
        ep_states, ep_probs, ep_rewards = run_episode(env, agent)
        returns = discount_returns(ep_rewards)
        stats = agent.train_step(ep_states, ep_probs, returns)

        episode_return = float(np.sum(ep_rewards))
        history.append({"episode": episode, "return": episode_return, **stats})

        if episode % 10 == 0:
            print(
                f"Ep {episode:03d} | Return: {episode_return:.3f} "
                f"| Loss: {stats['loss']:.4f} (policy {stats['policy_loss']:.4f}, "
                f"value {stats['value_loss']:.4f})"
            )

    agent.save(MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")
    return agent, history


if __name__ == "__main__":
    main()
