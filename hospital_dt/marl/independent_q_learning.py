import numpy as np
import os
import csv

# ------------------------------------------------------------------
# Simple MARL environment (toy model for demonstration)
# ------------------------------------------------------------------
class SimpleHospitalEnv:
    def __init__(self):
        self.state = 0  # 0–9 represent occupancy level
        self.max_state = 9

    def reset(self):
        self.state = np.random.randint(0, 3)
        return self.state

    def step(self, action):
        # Actions: 0 = allocate patient, 1 = defer patient
        change = 1 if action == 0 else -1
        next_state = np.clip(self.state + change, 0, self.max_state)

        # Reward: avoid overload (state > 7)
        reward = -1 if next_state > 7 else 1

        self.state = next_state
        done = False
        return next_state, reward, done, {}


# ------------------------------------------------------------------
# Independent Q-learning for two agents (ICU + OT)
# ------------------------------------------------------------------
class IndependentQLearning:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.q_table = np.zeros((10, 2))  # states: 0–9, actions: 0/1
        self.lr = 0.1
        self.gamma = 0.9
        self.eps = 0.2

    def choose_action(self, state):
        if np.random.random() < self.eps:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)


# ------------------------------------------------------------------
# Training Loop + Logging to CSV
# ------------------------------------------------------------------
def train_marl():
    env1 = SimpleHospitalEnv()
    env2 = SimpleHospitalEnv()

    icu = IndependentQLearning(env1, "ICU_Agent")
    ot = IndependentQLearning(env2, "OT_Agent")

    # Ensure experiments folder exists
    os.makedirs("experiments", exist_ok=True)
    log_path = "experiments/qlearning_results.csv"

    # Write CSV Header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "icu_util", "ot_util", "waiting_time"])

    for episode in range(200):
        s1 = env1.reset()
        s2 = env2.reset()

        a1 = icu.choose_action(s1)
        a2 = ot.choose_action(s2)

        ns1, r1, _, _ = env1.step(a1)
        ns2, r2, _, _ = env2.step(a2)

        icu.learn(s1, a1, r1, ns1)
        ot.learn(s2, a2, r2, ns2)

        # Dummy performance metrics
        avg_reward = (r1 + r2) / 2
        icu_util = ns1
        ot_util = ns2
        waiting_time = max(0, 10 - (ns1 + ns2))

        # Append episode log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, avg_reward, icu_util, ot_util, waiting_time])

        if episode % 20 == 0:
            print(f"Episode {episode}: reward={avg_reward}")

    print(f"\nTraining completed. Logs saved to: {log_path}")


if __name__ == "__main__":
    train_marl()
