from hospital_dt.env.hospital_env import HospitalDT
from hospital_dt.agents.rule_based_agents import icu_agent_rule, ot_agent_rule, staff_agent_rule, merge_actions
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    # Execute the simulation and print a confirmation message.
    dt = run_baseline()
    print('Simulation finished.')