"""Hospital Digital Twin (DT) simulation (SimPy) - same as earlier version."""
import simpy, random, numpy as np
from collections import deque, defaultdict

class Patient:
    """
    Represents a single patient entity within the Hospital Digital Twin simulation.

    Tracks the patient's ID, initial severity, arrival time, and key timestamps 
    for entering/exiting the ICU and OT, and final departure.
    """
    def __init__(self, pid, severity, arrival_time):
        """
        Initializes a new Patient object.

        :param pid: Unique integer identifier for the patient.
        :param severity: The patient's initial medical severity level (e.g., 1, 2, or 3).
        :param arrival_time: The time (in simulation units) the patient arrived.
        """
        self.id = pid
        self.severity = severity
        self.arrival = arrival_time
        self.entered_icu = None  # Timestamp when patient entered the ICU
        self.entered_ot = None   # Timestamp when patient entered the Operating Theater
        self.departed = None     # Timestamp when patient left the hospital

class HospitalDT:
    """
    The main simulation environment for the Hospital Digital Twin using SimPy.

    It models patient arrivals, processing through the Emergency Department (ED), 
    and resource contention for critical units like the Intensive Care Unit (ICU) 
    and Operating Theater (OT). It is designed to function as a Reinforcement 
    Learning (RL) environment (using reset and step methods).
    """
    def __init__(self, seed=42, icu_capacity=5, ot_capacity=2, base_arrival_rate=2):
        """
        Initializes the Hospital Digital Twin environment.

        Sets up random seeds, SimPy environment and resources, capacity limits, 
        and internal tracking lists/queues.

        :param seed: Random seed for reproducibility.
        :param icu_capacity: Maximum number of patients the ICU can hold simultaneously.
        :param ot_capacity: Maximum number of patients the Operating Theater can hold simultaneously.
        :param base_arrival_rate: The average number of patients arriving per time unit (lambda for exponential distribution).
        """
        random.seed(seed); np.random.seed(seed)
        self.env = simpy.Environment()
        # SimPy Resource objects for managing capacity
        self.icu = simpy.Resource(self.env, capacity=icu_capacity)
        self.ot = simpy.Resource(self.env, capacity=ot_capacity)
        self.icu_capacity = icu_capacity; self.ot_capacity = ot_capacity
        self.base_arrival_rate = base_arrival_rate
        # Data structures for tracking patients and queues
        self.patients = []
        self.waiting_ed = deque()  # Emergency Department waiting queue
        self.waiting_icu = deque() # ICU waiting queue (e.g., for critical patients)
        self.waiting_ot = deque()  # Operating Theater waiting queue
        
        self.time = 0.0
        self.pid_counter = 0
        self.metrics = defaultdict(list)
        self.running = False

    def reset(self, duration=24):
        """
        Resets the simulation environment to its initial state.

        This prepares the environment for a new simulation run or episode.
        It initializes the SimPy environment, resources, internal counters, 
        and starts the patient generation process.

        :param duration: The total simulation time limit (in time units).
        :return: The initial observation state of the environment.
        """
        self.env = simpy.Environment()
        self.icu = simpy.Resource(self.env, capacity=self.icu_capacity)
        self.ot = simpy.Resource(self.env, capacity=self.ot_capacity)
        self.patients = []; self.waiting_ed = deque(); self.waiting_icu = deque(); self.waiting_ot = deque()
        self.time = 0.0; self.pid_counter = 0; self.metrics = defaultdict(list); self.duration = duration; self.running = True
        
        # Start the continuous patient generation process
        self.env.process(self._patient_generator())
        return self._get_observation()

    def step(self, action_dict):
        """
        Advances the simulation by one time step (or until the next event).

        This is the core execution method, typically called repeatedly in an RL loop.
        It advances the SimPy environment, calculates metrics, and determines 
        the next state, reward, and termination status.

        :param action_dict: Dictionary representing actions to be taken in the environment (currently ignored/placeholder).
        :return: A tuple of (observation, reward, done, info) following the standard RL environment interface.
        """
        try:
            # Try to execute until the next scheduled event
            self.env.step()
        except RuntimeError:
            # Fallback: run for a fixed time unit if no immediate event is scheduled
            self.env.run(until=self.env.now + 1)
            
        self.time = self.env.now
        
        # Get current state, compute performance, check termination
        obs = self._get_observation()
        reward = self._compute_reward()
        done = (self.time >= self.duration)
        info = {}
        
        # Record operational metrics for analysis
        self.metrics['time'].append(self.time)
        self.metrics['icu_occupancy'].append(self._icu_occupancy())
        self.metrics['ot_occupancy'].append(self._ot_occupancy())
        self.metrics['ed_wait_len'].append(len(self.waiting_ed))
        
        return obs, reward, done, info

    def _patient_generator(self):
        """
        A SimPy process that continuously generates new patients based on an 
        exponential inter-arrival time distribution.

        Patients are created and immediately routed based on their severity.
        """
        while True:
            # Sample inter-arrival time from an exponential distribution
            inter = np.random.exponential(1.0 / max(1e-6, self.base_arrival_rate))
            yield self.env.timeout(inter)  # Wait for the inter-arrival time
            
            # Create new patient
            pid = self.pid_counter; self.pid_counter += 1
            # Assign severity: 1 (60%), 2 (30%), 3 (10%)
            sev = np.random.choice([1,2,3], p=[0.6,0.3,0.1])
            p = Patient(pid, sev, self.env.now); self.patients.append(p)
            
            # Route patient based on severity
            if sev == 3: # Critical patients go straight to ICU queue
                self.waiting_icu.append(p)
                self.env.process(self._attempt_icu_admit(p))
            else: # Lower severity patients go to Emergency Department
                self.waiting_ed.append(p)
                self.env.process(self._process_ed(p))

    def _process_ed(self, patient):
        """
        A SimPy process simulating the patient's time in the Emergency Department.

        After a processing time, the patient is either discharged or routed 
        to the Operating Theater queue.

        :param patient: The Patient object currently being processed.
        """
        # Simulate ED processing time (uniform distribution)
        yield self.env.timeout(np.random.uniform(0.5, 2.0))
        
        # Determine next destination: 20% chance to need OT, 80% chance to depart
        if random.random() < 0.2:
            self.waiting_ot.append(patient)
            self.env.process(self._attempt_ot_admit(patient))
        else:
            patient.departed = self.env.now

    def _attempt_icu_admit(self, patient):
        """
        A SimPy process for a patient attempting to gain admission to the ICU.

        The patient waits for an available ICU bed, occupies it for a duration, 
        and is then discharged.

        :param patient: The Patient object requiring ICU admission.
        """
        # Request an ICU resource unit
        with self.icu.request() as req:
            yield req  # Patient waits until a unit is available
            patient.entered_icu = self.env.now
            # Simulate ICU stay duration (uniform distribution)
            yield self.env.timeout(np.random.uniform(4, 10))
            patient.departed = self.env.now # Release the resource and patient departs

    def _attempt_ot_admit(self, patient):
        """
        A SimPy process for a patient attempting to gain admission to the Operating Theater (OT).

        The patient waits for an available OT, occupies it for a duration, 
        and is then discharged.

        :param patient: The Patient object requiring OT admission.
        """
        # Request an OT resource unit
        with self.ot.request() as req:
            yield req  # Patient waits until a unit is available
            patient.entered_ot = self.env.now
            # Simulate OT procedure duration (uniform distribution)
            yield self.env.timeout(np.random.uniform(2, 6))
            patient.departed = self.env.now # Release the resource and patient departs

    def _get_observation(self):
        """
        Gathers the current state information of the simulation environment.

        The observation includes current time, resource availability, and queue lengths.

        :return: A dictionary representing the current observation state.
        """
        obs = {
            'time': self.env.now,
            'icu_free': self.icu_capacity - self._icu_occupancy(),
            'ot_free': self.ot_capacity - self._ot_occupancy(),
            'ed_wait': len(self.waiting_ed),
            'icu_wait': len(self.waiting_icu),
            'ot_wait': len(self.waiting_ot),
            'num_patients': len(self.patients)
        }
        return obs
    
    def _icu_occupancy(self):
        """
        Calculates the current number of patients occupying an ICU bed 
        by querying the SimPy resource object directly.

        This ensures the reported occupancy is consistent with the capacity 
        enforced by the simulation framework.

        :return: The current number of occupied ICU beds (integer).
        """
        # self.icu.count returns the number of processes currently holding a resource unit.
        return self.icu.count 
        # Alternatively, self.icu.users can be used if you need a list of the Request objects.

    def _ot_occupancy(self):
        """
        Calculates the current number of patients occupying an Operating Theater (OT)
        by querying the SimPy resource object directly.

        This ensures the reported occupancy is consistent with the capacity 
        enforced by the simulation framework.

        :return: The current number of occupied OT units (integer).
        """
        # self.ot.count returns the number of processes currently holding a resource unit.
        return self.ot.count

    # The rest of the original methods (like _get_observation and step) 
    # should be updated to call these new, corrected functions.

    # def _icu_occupancy(self):
    #     """
    #     Calculates the current number of patients occupying an ICU bed.

    #     This is determined by counting patients whose ICU entry time is set 
    #     and who have not yet departed.

    #     :return: The current number of occupied ICU beds (integer).
    #     """
    #     used = 0
    #     for p in self.patients:
    #         # Occupied if entered_icu is set AND (departed is None OR departed is after entry)
    #         if p.entered_icu is not None and (p.departed is None or p.departed > p.entered_icu):
    #             used += 1
    #     return used

    # def _ot_occupancy(self):
    #     """
    #     Calculates the current number of patients occupying an Operating Theater (OT).

    #     This is determined by counting patients whose OT entry time is set 
    #     and who have not yet departed.

    #     :return: The current number of occupied OT units (integer).
    #     """
    #     used = 0
    #     for p in self.patients:
    #         # Occupied if entered_ot is set AND (departed is None OR departed is after entry)
    #         if p.entered_ot is not None and (p.departed is None or p.departed > p.entered_ot):
    #             used += 1
    #     return used

    def _compute_reward(self):
        """
        Calculates the reward signal for the current time step.

        The reward is typically a penalty based on patient waiting times/queue lengths 
        and resource over-utilization (if applicable).

        :return: The calculated reward value (float).
        """
        # Penalty based on queue length (higher penalty for ICU wait)
        r = - (len(self.waiting_ed) * 0.5 + len(self.waiting_icu) * 1.0)
        
        # Severe penalty if the number of patients in the ICU exceeds capacity 
        # (could represent a critical operational failure or error)
        if self._icu_occupancy() > self.icu_capacity:
            r -= 5.0
            
        return r