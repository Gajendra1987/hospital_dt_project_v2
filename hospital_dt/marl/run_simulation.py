from hospital_env import HospitalDT
from rule_based_agents import icu_agent_rule, ot_agent_rule, staff_agent_rule, merge_actions
import matplotlib.pyplot as plt
import pandas as pd

def run_baseline(seed=1, duration=48):
    """
    Runs a baseline simulation of the Hospital Digital Twin (DT) environment 
    using simple rule-based agents for resource allocation.

    The simulation runs for a specified duration, collecting time-series metrics.

    :param seed: The random seed for reproducibility.
    :param duration: The total simulation time (e.g., in hours).
    :return: The final HospitalDT object containing all simulation data and metrics.
    """
    # 1. Environment Initialization
    # Instantiate the HospitalDT environment.
    # Note: These parameters (ICU=4, OT=2, Arrival Rate=0.9) define the baseline scenario.
    dt = HospitalDT(seed=seed, icu_capacity=4, ot_capacity=2, base_arrival_rate=0.9)
    
    # Reset the environment to start a new episode.
    # This initializes all resources, queues, and starts the patient generator process.
    obs = dt.reset(duration=duration)
    
    done = False
    step = 0
    
    # 2. Simulation Loop (Time Stepping)
    # The simulation continues until the duration is reached (done=True) 
    # or a safety limit of 1000 steps is hit.
    while not done and step < 1000:
        # A. Agent Decisions: Generate actions based on the current observation (obs).
        # These are simple, predefined rules (e.g., "always admit if a bed is free").
        a1 = icu_agent_rule(obs)
        a2 = ot_agent_rule(obs)
        a3 = staff_agent_rule(obs)
        
        # B. Action Merging: Combine decisions from all rule-based agents 
        # into a single action dictionary that the environment expects.
        actions = merge_actions(a1, a2, a3)
        
        # C. Environment Step: Advance the simulation by one time unit/event.
        # This executes the action, processes patient movement, updates time, 
        # and calculates the reward.
        obs, r, done, info = dt.step(actions)
        step += 1
        
    # 3. Post-Simulation Analysis (Plotting)
    try:
        # Create a time-series plot of resource usage.
        # Plot the ICU occupancy over time
        plt.plot(dt.metrics['time'], dt.metrics['icu_occupancy'], label='ICU occupancy')
        # Plot the OT occupancy over time
        plt.plot(dt.metrics['time'], dt.metrics['ot_occupancy'], label='OT occupancy')
        
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Occupancy')
        # Save the resulting plot to a file
        plt.savefig('hospital_dt_baseline_plot_v2.png')
        print('Saved plot to hospital_dt_baseline_plot_v2.png')
    except Exception as e:
        print('Plotting failed:', e)
        
    return dt

def save_simulation_results(dt, filename="simulation_performance.csv"):
    # 1. Extract raw patient data from the DT object
    patient_data = []
    
    for p in dt.patients:
        # Determine admission based on your hospital_env.py attributes
        admission_time = p.entered_icu if p.entered_icu is not None else p.entered_ot
        
        if admission_time is not None:
            # Calculate wait for THIS patient only
            current_wait = admission_time - p.arrival
            
            patient_data.append({
                'patient_id': p.id,
                'wait_time': current_wait, # Store the scalar number
                'served': True
            })
    
    if not patient_data:
        print("No patients were served during this simulation run.")
        return

    # 2. Create DataFrame and Calculate Aggregates
    df_patients = pd.DataFrame(patient_data)
    
    avg_wait = df_patients['wait_time'].mean()
    peak_wait = df_patients['wait_time'].max()
    total_served = len(df_patients)
    
    # 3. Create Summary DataFrame
    summary_data = {
        'Metric': ['Average Waiting Time', 'Peak Waiting Time', 'Total Patients Served'],
        'Value': [round(avg_wait, 2), round(peak_wait, 2), total_served]
    }
    df_summary = pd.DataFrame(summary_data)
    
    # 4. Save to CSV
    df_summary.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

if __name__ == '__main__':
    # Execute the simulation and print a confirmation message.
    dt = run_baseline()
    print('Simulation finished.')
    save_simulation_results(dt)
    print('Saved Simulation Results.')