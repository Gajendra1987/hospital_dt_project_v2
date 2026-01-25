import numpy as np
import os
import csv

# ------------------------------------------------------------------
# Simple MARL environment (toy model for demonstration)
# ------------------------------------------------------------------
# MODIFIED BY AI ASSISTANT [2026-01-25]
# TASK: Align Q-Learning constraints with Baseline (ICU=4, OT=2)

class SimpleHospitalEnv:
    def __init__(self, capacity=4): # Match Baseline ICU capacity
        self.state = 0  
        self.max_state = capacity # Constraint now matches Baseline
        self.patients_served = 0 # Initialize counter here

    def reset(self):
        self.state = 0
        self.patients_served = 0 # CRITICAL: Reset to 0 every episode
        return self.state

    def step(self, action):
        # Action 0: Admit, Action 1: Defer
        reward = 0
        
        if action == 0:
            if self.state < self.max_state:
                self.state += 1
                self.patients_served += 1 # INCREMENT HERE
                reward = 10  # Reward for successful admission
            else:
                reward = -50 # MASSIVE penalty for trying to admit when full
        else:
            if self.state > 0:
                self.state -= 1
            reward = 0 # No reward for deferring
            
        return self.state, reward, False, {"served": self.patients_served}


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
        writer.writerow(["episode", "reward", "icu_util", "ot_util", "waiting_time", "patients_served"])

    for episode in range(400):
        s1 = env1.reset()
        s2 = env2.reset()
        avg_reward = 0
        for hour in range(48):
            a1 = icu.choose_action(s1)
            a2 = ot.choose_action(s2)

            ns1, r1, _, info1 = env1.step(a1)
            ns2, r2, _, info2 = env2.step(a2)

            icu.learn(s1, a1, r1, ns1)
            ot.learn(s2, a2, r2, ns2)

            # Dummy performance metrics
            avg_reward = (r1 + r2) / 2
            icu_util = ns1
            ot_util = ns2
            waiting_time = max(0, 10 - (ns1 + ns2))
            real_hourly_waiting_time = (icu_util + ot_util) * 0.15
            # Logic: Total served is the sum of admissions across both units
            total_served = info1.get("served", 0) + info2.get("served", 0)
        # Append episode log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, avg_reward, icu_util, ot_util, real_hourly_waiting_time, total_served])

        if episode % 20 == 0:
            print(f"Episode {episode}: reward={avg_reward}")

    print(f"\nTraining completed. Logs saved to: {log_path}")


if __name__ == "__main__":
    train_marl()
