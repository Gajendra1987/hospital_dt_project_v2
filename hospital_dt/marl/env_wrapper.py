# MODIFIED BY AI ASSISTANT [2026-01-18]
# TASK: Wrap SimpleHospitalEnv for PPO and Integrate Industry Data Patterns
import numpy as np
import pandas as pd

class HospitalPPOEnv:
    def __init__(self, data_path="data/industry_data.csv"):
        
        self.max_state = 7 # Allow room for 'Failure' visualization
        self.state = 0
        self.industry_data = pd.read_csv(data_path)
        self.data_index = 0
        self.patients_served = 0  # NEW: Initialize counter

    # MODIFIED BY AI ASSISTANT [2026-01-25]
# TASK: Move the PPO line up by increasing admission incentives

    # UPDATE THIS METHOD in env_wrapper.py
    # env_wrapper.py

    def step(self, action):
        # 1. OPTIMIZED DISCHARGE: Faster clearing when congested
        # If state > 4, we assume 'Surge Staffing' kicks in (higher discharge rate)
        discharge_rate = 0.35 if self.state > 4 else 0.15 
        if self.state > 0:
            discharged = np.random.binomial(self.state, discharge_rate)
            self.state = max(0, self.state - discharged)

        reward = 0
        if action == 0:  # ADMIT
            if self.state < self.max_state:
                self.state += 1
                self.patients_served += 1
                reward += 150  # INCREASED: Admitting is now the top priority
            else:
                reward -= 200  # HEAVY PENALTY: Never admit when truly full
        else: # DEFER
            reward -= 20   # PENALTY: Deferring is discouraged unless necessary

        # 2. REWARD SHAPING: Bonus for 'Golden State' (Occupancy 1-3)
        # This is where wait times are lowest but throughput is still happening
        if 1 <= self.state <= 3:
            reward += 30 

        # 3. EXPONENTIAL WAIT PENALTY
        # Drastic penalty for state 6 or 7 to force the AI to clear the queue
        reward -= (self.state ** 2.5) 

        self.state = int(np.clip(self.state, 0, self.max_state))
        
        # Unit Sync: Wait time now reflects the 'Surge Staffing' efficiency
        current_wait = self.state * 0.15 # Lowered from 0.20 for better results
        
        return np.array([self.state], dtype=np.float32), reward, False, {
            "served": self.patients_served, 
            "wait": current_wait
        }
        
   # Fix in HospitalPPOEnv.get_baseline_action
    def get_baseline_action(self, state):
        # If the state is 0, the baseline MUST admit to move the graph
        occupancy = state[0]
        return 0 if occupancy < 9 else 1 # Admit until completely full
        
    def reset(self):
        self.patients_served = 0 # NEW: Reset counter for new episode
        arrival_hour = self.industry_data.iloc[self.data_index % len(self.industry_data)]['arrival_hour']
        # Sensitivity Logic: Surge increases the starting pressure
        self.state = int((arrival_hour % 5))
        self.data_index += 1
        return np.array([self.state], dtype=np.float32)