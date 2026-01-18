# MODIFIED BY AI ASSISTANT [2026-01-18]
# TASK: Wrap SimpleHospitalEnv for PPO and Integrate Industry Data Patterns
import numpy as np
import pandas as pd

class HospitalPPOEnv:
    # TASK: Enable Sensitivity Analysis (What-If Scenarios)
    def __init__(self, data_path="data/industry_data.csv",surge_factor=1.0):
        self.surge_factor = surge_factor  # 1.0 = Normal, 1.5 = 50% Surge
        self.max_state = 9
        self.state = 0
        # Load Authentic MIMIC-III Data to drive simulation peaks
        self.industry_data = pd.read_csv(data_path)
        self.data_index = 0

    def reset(self):
       # Apply surge_factor to starting occupancy
        arrival_hour = self.industry_data.iloc[self.data_index % len(self.industry_data)]['arrival_hour']
        # Sensitivity Logic: Surge increases the starting pressure
        self.state = int((arrival_hour % 5) * self.surge_factor)
        self.data_index += 1
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # Action 0: Allocate (Increase Occupancy), Action 1: Defer (Decrease/Maintain)
        change = 1 if action == 0 else -1
        self.state = np.clip(self.state + change, 0, self.max_state)

        # ENHANCED REWARD LOGIC:
        # 1. Penalize wait times (Overload state > 7)
        # 2. Reward serving patients (Action 0 when state is low)
        reward = 0
        if self.state > 7:
            reward -= 5  # Strong penalty for patient waiting/overload
        elif action == 0 and self.state < 5:
            reward += 2  # Reward for efficient resource utilization
        else:
            reward += 0.5 # Small baseline reward for stability

        done = False
        return np.array([self.state], dtype=np.float32), reward, done, {}
    
    def get_baseline_action(self, state):
        """
        Simulates a standard rule-based policy:
        If occupancy < 7, always accept patients. If > 7, defer.
        """
        occupancy = state[0]
        return 0 if occupancy < 7 else 1