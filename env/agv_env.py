import numpy as np

class AGVEnv:
    def __init__(self):
        self.dt = 0.1
        self.reset()

    def reset(self):
        self.error = 0.0
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.trajectory_type = np.random.choice([0, 1])  # 0=straight, 1=curve
        return self._get_state()

    def step(self, gains):
        Kp, Ki, Kd = gains

        control = (
            Kp * self.error +
            Ki * self.error_integral +
            Kd * (self.error - self.prev_error) / self.dt
        )

        self.prev_error = self.error
        self.error += np.random.randn() * 0.01 - control * 0.1
        self.error_integral += self.error * self.dt

        reward = -(
            abs(self.error) +
            0.1 * abs(self.error - self.prev_error)
        )

        done = abs(self.error) > 10
        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([
            self.error,
            self.error_integral,
            self.error - self.prev_error,
            self.trajectory_type
        ])
