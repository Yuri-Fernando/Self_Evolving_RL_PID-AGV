import numpy as np
from env.agv_env import AGVEnv

def evaluate(gains, episodes=10):
    env = AGVEnv()
    errors = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_error = []

        while not done:
            state, reward, done = env.step(gains)
            ep_error.append(abs(state[0]))

        errors.append(np.mean(ep_error))

    return np.mean(errors), np.std(errors)

if __name__ == "__main__":
    fixed_pid = np.array([0.1, 0.01, 0.001])
    mean_err, std_err = evaluate(fixed_pid)

    print(f"PID Fixed | Mean Error: {mean_err:.4f} | Std: {std_err:.4f}")
