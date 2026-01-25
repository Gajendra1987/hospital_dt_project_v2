# TASK: Execute PPO Training with Industry Data and Unified Logging Format
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import numpy as np
from env_wrapper import HospitalPPOEnv

# PPO Actor-Critic Network
class PPOModel(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

# 1. Decision Explainer Function
# UPDATED XAI LOGIC: Explaining Logits instead of squashed Probabilities
def get_decision_explanation(model, state):
    def predict_admit_logit(s):
        s_tensor = torch.from_numpy(s).float()
        with torch.no_grad():
            # Use the raw linear layer output (logits) for better contrast
            # assuming model.actor[-2] is the Linear layer before Softmax
            logits = model.actor[:-1](s_tensor) 
            return logits.numpy()[:, 0] # Return logit for 'Admit'

    # Background must cover the full range (0 to 7) to show contrast
    background = np.array([[0.0], [1.0], [3.0], [5.0], [7.0]])
    explainer = shap.KernelExplainer(predict_admit_logit, background)
    shap_values = explainer.shap_values(state.reshape(1, -1))
    
    return float(np.array(shap_values).flatten()[0])

def train_ppo():
    env = HospitalPPOEnv()
    model = PPOModel(obs_size=1, n_actions=2)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    os.makedirs("experiments", exist_ok=True)
    log_path = "experiments/ppo_results.csv"

    # Write CSV Header matching the Q-learning format exactly
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "icu_util", "ot_util", "waiting_time","patients_served","xai_score"])

    print("Starting PPO Training with Industry Data (MIMIC-III)...")

    # MODIFIED BY AI ASSISTANT [2026-01-25]
    # TASK: Fix Episode Duration to match 48-hour Baseline

    for episode in range(400):
        env = HospitalPPOEnv()
        state = env.reset()
        total_reward = 0

        # NEW: Run 48 hours (steps) per episode
        for _ in range(48):
            state_tensor = torch.from_numpy(state).float()
            probs = model.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, _, info = env.step(action.item())
            
            # PPO Learning logic (Update Actor and Critic)
            value = model.critic(state_tensor)
            advantage = reward - value.item()
            loss = -dist.log_prob(action) * advantage + 0.5 * (advantage**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
        # We explain the final state of the 48-hour shift
        xai_score = get_decision_explanation(model, state)
        # LOGGING: Record the totals AFTER 48 hours have passed
        total_served = info.get("served", 0)
        avg_wait = info.get("wait", 0.0)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, state[0], state[0]*0.8, avg_wait, total_served, xai_score])
    

    # NEW: Save the model brain after the final episode
    model_save_path = "experiments/ppo_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model weights saved successfully at: {model_save_path}")
    print(f"\nTraining completed. PPO Logs saved to: {log_path}")

if __name__ == "__main__":
    train_ppo()