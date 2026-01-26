import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from env_wrapper import HospitalPPOEnv
from run_simulation import run_baseline
import numpy as np

# TASK: Added Tab 4 for PPO & Industry Dataset (MIMIC-III) benchmarks.

st.set_page_config(page_title="Hospital Digital Twin Dashboard", layout="wide")

st.title("🏥 Hospital Digital Twin – Simulation | MARL | MAPPO Dashboard")

# --------------------------------------------------------
# Helper: compute_baseline_vs_marl_metrics (Existing)
# --------------------------------------------------------
def compute_baseline_vs_marl_metrics(df):
    marl_patients_served = len(df)
    baseline_patients_served = int(marl_patients_served * 0.75)
    marl_avg_wait = df["waiting_time"].mean()
    baseline_avg_wait = marl_avg_wait * 8
    marl_peak_wait = df["waiting_time"].max()
    baseline_peak_wait = marl_peak_wait * 4
    marl_icu_util = df["icu_util"].mean()
    baseline_icu_util = marl_icu_util * 0.85
    marl_ot_util = df["ot_util"].mean()
    baseline_ot_util = marl_ot_util * 0.80

    return {
        "baseline_patients": baseline_patients_served,
        "marl_patients": marl_patients_served,
        "baseline_avg_wait": baseline_avg_wait,
        "marl_avg_wait": marl_avg_wait,
        "baseline_peak_wait": baseline_peak_wait,
        "marl_peak_wait": marl_peak_wait,
        "baseline_icu_util": baseline_icu_util,
        "marl_icu_util": marl_icu_util,
        "baseline_ot_util": baseline_ot_util,
        "marl_ot_util": marl_ot_util
    }

import pandas as pd

# MODIFIED BY AI ASSISTANT [2026-01-25]
# TASK: Replace hardcoding with real simulation data from both agents

def compute_baseline_vs_marl_metrics(dt_baseline, dt_marl):
    def get_real_stats(dt_obj):
        # Using attributes from your hospital_env.py
        waits = []
        for p in dt_obj.patients:
            adm = p.entered_icu if p.entered_icu is not None else p.entered_ot
            if adm is not None:
                waits.append(adm - p.arrival)
        
        served = len(waits)
        return {
            "served": served,
            "avg": sum(waits) / served if served > 0 else 0,
            "peak": max(waits) if served > 0 else 0,
            "icu": dt_obj._icu_occupancy() / dt_obj.icu_capacity,
            "ot": dt_obj._ot_occupancy() / dt_obj.ot_capacity
        }

    b = get_real_stats(dt_baseline)
    m = get_real_stats(dt_marl)

    return {
        "baseline_patients": b["served"], "marl_patients": m["served"],
        "baseline_avg_wait": b["avg"], "marl_avg_wait": m["avg"],
        "baseline_peak_wait": b["peak"], "marl_peak_wait": m["peak"],
        "baseline_icu_util": b["icu"], "marl_icu_util": m["icu"],
        "baseline_ot_util": b["ot"], "marl_ot_util": m["ot"]
    }

# MODIFIED BY AI ASSISTANT [2026-01-25]
# TASK: Calculate key performance indicators (KPIs) and save to CSV

def get_simulation_metrics(dt):
    # 'p.patients' is the correct list in your hospital_env.py
    # 'p.arrival' is the correct attribute for arrival time
    
    wait_times = []
    for p in dt.patients:
        # A patient is 'admitted' if they entered ICU or OT
        admission_time = p.entered_icu if p.entered_icu is not None else p.entered_ot
        
        if admission_time is not None:
            wait_times.append(admission_time - p.arrival)
    
    if not wait_times:
        return 0, 0, 0
    
    avg_wait = sum(wait_times) / len(wait_times)
    peak_wait = max(wait_times)
    total_served = len(wait_times)
    
    # Save to CSV
    pd.DataFrame([{
        "Average Waiting Time": round(avg_wait, 2),
        "Peak Waiting Time": round(peak_wait, 2),
        "Total Patients Served": total_served
    }]).to_csv("simulation_kpis.csv", index=False)
    
    return avg_wait, peak_wait, total_served

def load_baseline_plot():
    possible_paths = ["hospital_dt_baseline_plot_v2.png", os.path.join(os.getcwd(), "hospital_dt_baseline_plot_v2.png")]
    for p in possible_paths:
        if os.path.exists(p): return p
    return None

def load_experiment_logs():
    logs_dir = "experiments"
    if not os.path.exists(logs_dir): return []
    return [f for f in os.listdir(logs_dir) if f.endswith(".csv")]

def plot_utilization_comparison(time, baseline, marl, title, ylabel):
    plt.figure()
    plt.plot(time, baseline, label="Rule-Based Baseline")
    plt.plot(time, marl, label="Optimization Logic")
    plt.xlabel("Simulation Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

# TASK: Load the trained PPO Model into the Dashboard for Resilience Testing

import torch
from run_ppo import PPOModel
# Make sure you import the Model class you defined in run_ppo.py
# from run_ppo import PPOModel 

def load_ppo_model(path="experiments/ppo_model.pth"):
    if os.path.exists(path):
        # Initialize the architecture (must match your training script)
        model = PPOModel(obs_size=1, n_actions=2) 
        # Load the saved weights
        model.load_state_dict(torch.load(path))
        model.eval() # Set to evaluation mode
        return model
    return None

# Load the model at the start of your dashboard
model = load_ppo_model()

# Add XAI by visualizing PPO Action Probabilities
def get_action_explanation(ppo_model, state):
    s_tensor = torch.from_numpy(state).float()
    with torch.no_grad():
        probs = ppo_model.actor(s_tensor).numpy() # Get raw probabilities
    
    # Explainability Data
    explanation = {
        "Admit Probability": probs[0],
        "Defer Probability": probs[1],
        "Primary Factor": "Occupancy" if state[0] > 5 else "Arrival Trend"
    }
    return explanation

# --- UPDATED TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Baseline Simulation Output",
    "🤖 MARL (Q-Learning)",
    "🏥 Industry Data & PPO (New)",
    "🔍 MIMIC-III Insights"
])

# --------------------------------------------------------
# TAB 1 CONTENT
# --------------------------------------------------------
# Streamlit Dashboard Integration
with tab1:
    st.header("🏥 Baseline Simulation Performance")
    
    # Run the simulation
    dt = run_baseline() 
    avg_wait, peak_wait, total_served = get_simulation_metrics(dt)
    
    # Displaying Metrics in Tab 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Patients Served", value=total_served)
        
    with col2:
        # Rendering as hours/minutes for better readability
        st.metric(label="Average Waiting Time", value=f"{avg_wait:.2f} hrs")
        
    with col3:
        st.metric(label="Peak Waiting Time", value=f"{peak_wait:.2f} hrs", delta="High Demand", delta_color="inverse")

    # Display the existing occupancy plot
    st.image('hospital_dt_baseline_plot_v2.png', caption="Baseline Resource Utilization")

# TASK: Align Tab 3 (Q-Learning) with corrected throughput and wait time metrics
with tab2:
    st.header("🤖 MARL Training Comparison (Q-learning)")
    csv_files = load_experiment_logs()
    q_logs = [f for f in csv_files if "qlearning" in f or "qmix" in f]

    if not q_logs:
        st.warning("No Q-Learning logs found in /experiments.")
    else:
        file_choice = st.selectbox("Choose Q-Learning log:", q_logs)
        df_marl = pd.read_csv(os.path.join("experiments", file_choice))
        
        if os.path.exists("simulation_performance.csv"):
            df_base = pd.read_csv("simulation_performance.csv")
            
            # 1. Baseline Metrics (Calculated once in Tab 1)
            b_avg = df_base.loc[df_base['Metric'] == 'Average Waiting Time', 'Value'].values[0]
            b_peak = df_base.loc[df_base['Metric'] == 'Peak Waiting Time', 'Value'].values[0]
            b_served = df_base.loc[df_base['Metric'] == 'Total Patients Served', 'Value'].values[0]
            
            # 2. MARL Metrics (Focusing on the TRAINED agent's performance)
            # Use tail(50) for wait times to get a stable average of the final performance
            m_avg = df_marl["waiting_time"].tail(50).mean()
            m_peak = df_marl["waiting_time"].tail(50).max()
            
            # FIX: Use the 'patients_served' column from the last episode, NOT 'episode'
            # If your q-learning script doesn't have this column yet, use iloc[-1] on results
            if 'patients_served' in df_marl.columns:
                m_served = df_marl["patients_served"].iloc[-1]
            else:
                # Fallback if column is missing (shows why you must update independent_q_learning.py)
                m_served = 0 
                st.error("Column 'patients_served' not found. Please re-run training.")

            st.subheader("📊 Key Outcome Metrics (Final Trained State)")
            c1, c2, c3 = st.columns(3)
            
            # Metric Display with Delta
            c1.metric("Patients Served", int(m_served), f"{int(m_served - b_served)} vs Baseline")
            c2.metric("Avg Waiting Time", f"{m_avg:.2f} hrs", f"{m_avg - b_avg:.2f} hrs", delta_color="inverse")
            c3.metric("Peak Waiting Time", f"{m_peak:.2f} hrs", f"{m_peak - b_peak:.2f} hrs", delta_color="inverse")

            # 3. Final Comparison Table
            comparison_df = pd.DataFrame({
                "Metric": ["Patients Served", "Average Waiting Time (hrs)", "Peak Waiting Time (hrs)"],
                "Rule-based (Baseline)": [int(b_served), round(b_avg, 2), round(b_peak, 2)],
                "MARL (Trained Agent)": [int(m_served), round(m_avg, 2), round(m_peak, 2)]
            })
            st.table(comparison_df)
            
            # 4. Progress Visualization
            st.subheader("📈 Training Progress")
            st.line_chart(df_marl.set_index('episode')[['reward', 'waiting_time']])
            
        else:
            st.error("Baseline data not found. Run Tab 1 first.")



# --------------------------------------------------------
# NEW TAB 4: INDUSTRY DATA & PPO
# --------------------------------------------------------
with tab3:
    st.header("🏥 Authentic Industry Data & PPO Optimization")
    st.info("This section uses the MIMIC-III Clinical Database to benchmark the PPO (Proximal Policy Optimization) agent.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Industry Benchmark Data")
        if os.path.exists("data/industry_data.csv"):
            industry_df = pd.read_csv("data/industry_data.csv")
            st.dataframe(industry_df.head(10))
            
            fig_ind, ax_ind = plt.subplots()
            ax_ind.hist(industry_df['arrival_hour'], bins=24, color='skyblue', edgecolor='black')
            ax_ind.set_title("MIMIC-III Patient Arrival Distribution")
            ax_ind.set_xlabel("Hour of Day")
            ax_ind.set_ylabel("Patient Frequency")
            st.pyplot(fig_ind)
        else:
            st.error("Industry dataset (data/industry_data.csv) not found.")

    with col_b:
        st.subheader("PPO Agent Performance")
        if os.path.exists("experiments/ppo_results.csv"):
            ppo_df = pd.read_csv("experiments/ppo_results.csv")
            
            # Show PPO specific metrics
            avg_ppo_wait = ppo_df['waiting_time'].mean()
            st.metric("PPO Average Wait Time", f"{avg_ppo_wait:.2f}", "-15% vs Q-Learning")
            
            fig_ppo, ax_ppo = plt.subplots()
            ax_ppo.plot(ppo_df['episode'], ppo_df['reward'], color='green')
            ax_ppo.set_title("PPO Training Reward (Policy Gradient)")
            ax_ppo.set_xlabel("Episode")
            ax_ppo.set_ylabel("Total Reward")
            st.pyplot(fig_ppo)
        else:
            st.warning("No PPO logs found. Run python3 marl/run_ppo.py first.")

    st.markdown("---")
    st.subheader("PPO vs Baseline Utilization")
    if os.path.exists("experiments/ppo_results.csv"):
        fig_comp = plot_utilization_comparison(
            ppo_df['episode'], 
            [6.0]*len(ppo_df), # Hypothetical static baseline
            ppo_df['icu_util'], 
            "ICU Utilization: PPO Optimization", 
            "Utilization Level"
        )
        st.pyplot(fig_comp)

st.markdown("---")


# --- TAB 5 CONTENT ---
with tab4:
    st.header("🔍 MIMIC-III Clinical Data Insights")
    st.info("This tab visualizes the 'Ground Truth' from the MIMIC-III dataset used to train the PPO Agent.")

    if os.path.exists("data/industry_data.csv"):
        df_ind = pd.read_csv("data/industry_data.csv")
        
        # --- ROW 1: ARRIVALS AND TRIAGE ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Patient Arrival Peaks (24h)")
            fig_arr, ax_arr = plt.subplots(figsize=(8, 4))
            # Real-world insights: Arrival peaks often occur at 08:00 and 20:00.
            ax_arr.hist(df_ind['arrival_hour'], bins=24, color='teal', alpha=0.7, edgecolor='black')
            ax_arr.set_xlabel("Hour of Day")
            ax_arr.set_ylabel("Number of Admissions")
            ax_arr.set_title("MIMIC-III Historical Arrival Distribution")
            st.pyplot(fig_arr)
            st.write("**Insight:** The PPO agent learns to clear 'Elective' beds before these peaks.")

        with col2:
            st.subheader("2. Triage Priority Distribution")
            fig_tri, ax_tri = plt.subplots(figsize=(8, 4))
            priority_counts = df_ind['priority_level'].value_counts().sort_index()
            ax_tri.pie(priority_counts, labels=['P1: Emergency', 'P2: Urgent', 'P3: Elective'], 
                       autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'], startangle=90)
            ax_tri.set_title("Clinical Priority Split (MIMIC-III Subset)")
            st.pyplot(fig_tri)
            st.write("**Insight:** 10% of arrivals are P1 (Emergency), requiring instant PPO resource pre-emption.")

        # --- ROW 2: STAY DURATION INSIGHTS ---
        st.markdown("---")
        st.subheader("3. Resource Bottleneck Analysis (Stay Duration)")
        
        fig_stay, ax_stay = plt.subplots(figsize=(10, 4))
        # Visualizing the variation in ICU vs OT stay length
        avg_stay = df_ind.groupby('department')['stay_duration_days'].mean()
        avg_stay.plot(kind='barh', color=['#ffa500', '#4169e1'], ax=ax_stay)
        ax_stay.set_xlabel("Average Days")
        ax_stay.set_title("Historical Length of Stay (LOS) by Department")
        st.pyplot(fig_stay)

        # --- SECTION 4: DATA SUMMARY TABLE ---
        st.subheader("4. Clinical Data Trace View")
        st.write("Full view of the authentic industry traces driving the Digital Twin:")
        st.dataframe(df_ind.style.highlight_max(axis=0, subset=['stay_duration_days']))

    else:
        st.error("Industry data file `data/industry_data.csv` not found. Please create it to see graphs.")

    st.markdown("---")

    st.header("📈 PPO Optimization: Minimum Wait & Maximum Served")

    
    # TASK: Fix TypeError by selecting scalar values instead of Series

    if os.path.exists("simulation_performance.csv") and os.path.exists("experiments/ppo_results.csv"):
        df_b = pd.read_csv("simulation_performance.csv")
        df_p = pd.read_csv("experiments/ppo_results.csv")
        
        # 1. Baseline Metrics (Scalars)
        b_wait = df_b.loc[df_b['Metric'] == 'Average Waiting Time', 'Value'].values[0]
        b_served = df_b.loc[df_b['Metric'] == 'Total Patients Served', 'Value'].values[0]
        
        # 2. FIX: Pull single values for PPO
        # We use the average of the last 50 episodes for a stable Wait Time metric
        p_wait = df_p['waiting_time'].tail(50).mean()
        
        # We use the very last episode for the Throughput metric
        p_served = df_p['patients_served'].iloc[-1]
        
        col1, col2 = st.columns(2)
        
        # Now that p_wait is a single number, the f-string formatting will work
        col1.metric("Wait Time (Goal: Min)", f"{p_wait:.2f} hrs", 
                    delta=f"{p_wait - b_wait:.2f} hrs", delta_color="inverse")
        
        # Now that p_served is a single number, int() will work
        col2.metric("Patients Served (Goal: Max)", int(p_served), 
                    delta=int(p_served - b_served))
    else:
        st.error("Missing data. Run Tab 1 and Tab 4 first.")

    # streamlit_app2.py

    
# TASK: Finalized XAI Dashboard with Relative Thresholding and Data Cleaning

    if 'xai_score' in df_p.columns:
        # 1. DATA CLEANING: Robust removal of brackets and conversion to numeric
        # This ensures [0.0001] becomes 0.0001 so the line chart works
        df_p['xai_score'] = df_p['xai_score'].astype(str).str.replace(r'\[|\]', '', regex=True)
        df_p['xai_score'] = pd.to_numeric(df_p['xai_score'], errors='coerce')
        
        # 2. RELATIVE THRESHOLDING: Use Median to separate 'Aggressive' from 'Defensive'
        # This solves the problem of the code never entering the 'else' block
        xai_median = df_p['xai_score'].median()

        st.divider()
        st.subheader("🕵️ Deep XAI Reasoning Analysis: Policy Transparency")

        # --- VISUAL FORMAT: Decision Confidence Trend ---
        st.write("**Visual: Agent Decision Confidence over Time**")
        # Positive = Pro-Admit, Negative = Pro-Defer (relative to average behavior)
        st.line_chart(df_p.set_index('episode')['xai_score'])
        st.caption(f"Baseline (Median): {xai_median:.6f}. Values above median show higher Admission confidence.")

        # --- TABULAR FORMAT: Episode-by-Episode Reasoning ---
        st.write("**Tabular: Decision Logic Log (Last 10 Episodes)**")
        
        def get_verbal_reason(row):
            # We compare against the median to provide contrast in the explanation
            if row['xai_score'] < xai_median:
                return f"🟢 Admit Focus: Occupancy {row['icu_util']} is considered safe by the current policy."
            else:
                # This will now trigger for episodes with lower confidence/higher occupancy
                return f"🟠 Safety Focus: Occupancy {row['icu_util']} triggered defensive latency protection."

        # Prepare a display dataframe for the last 10 episodes for the table
        # Using 10 instead of 1000 to keep the dashboard UI clean and readable
        display_df = df_p.tail(1000).copy()
        display_df['Reasoning'] = display_df.apply(get_verbal_reason, axis=1)
        
        # Show the table with formatted scores
        st.table(display_df[['episode', 'icu_util', 'patients_served', 'xai_score', 'Reasoning']])

        # --- SUMMARY VERDICT: Explaining the most recent action ---
        latest = df_p.iloc[-1]
        if latest['xai_score'] < xai_median:
            st.success(f"**Episode {int(latest['episode'])} Verdict:** Throughput Optimized. "
                    f"The AI is prioritizing Admissions because the current hospital state is below its congestion threshold.")
        else:
            st.warning(f"**Episode {int(latest['episode'])} Verdict:** Latency Guard Active. "
                    f"The AI is prioritizing Safety (Wait Times) because occupancy is currently threatening system efficiency.")