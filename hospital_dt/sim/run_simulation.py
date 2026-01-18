from hospital_dt.env.hospital_env import HospitalDT
from hospital_dt.agents.rule_based_agents import icu_agent_rule, ot_agent_rule, staff_agent_rule, merge_actions
import matplotlib.pyplot as plt
def run_baseline(seed=1, duration=48):
    dt = HospitalDT(seed=seed, icu_capacity=4, ot_capacity=2, base_arrival_rate=0.9)
    obs = dt.reset(duration=duration)
    done = False; step=0
    while not done and step < 1000:
        a1 = icu_agent_rule(obs); a2 = ot_agent_rule(obs); a3 = staff_agent_rule(obs)
        actions = merge_actions(a1,a2,a3)
        obs, r, done, info = dt.step(actions); step += 1
    try:
        plt.plot(dt.metrics['time'], dt.metrics['icu_occupancy'], label='ICU occupancy')
        plt.plot(dt.metrics['time'], dt.metrics['ot_occupancy'], label='OT occupancy')
        plt.legend(); plt.xlabel('Time'); plt.ylabel('Occupancy'); plt.savefig('hospital_dt_baseline_plot_v2.png')
        print('Saved plot to hospital_dt_baseline_plot_v2.png')
    except Exception as e:
        print('Plotting failed:', e)
    return dt
if __name__ == '__main__':
    dt = run_baseline(); print('Simulation finished.')    
