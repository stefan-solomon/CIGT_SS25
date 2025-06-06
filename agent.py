import sys
import numpy as np


from simple_simulation import Country, World
from plotting import plot_q_table

class QLearningEnv:
    def __init__(self):
        self.country = Country(
            name='Testland',
            population=1_000_000,
            area=100_000,
            budget=10_000_000_000,
            total_energy=0,
            energy_needed_per_person=0.4,
            carbon_footprint=0
        )
        self.world = World([self.country])

        self.max_plants = 10
        # Action indices → meaning
        self.action_map = {
            0: 'nothing',
            1: 'commission_solar',
            2: 'commission_nuclear',
            3: 'decommission_solar',
            4: 'decommission_nuclear',
            5: 'decommission_coal'
        }
        self.n_actions = len(self.action_map)

    def reset(self):
        self.__init__()
        return self._get_state()

    def _get_state(self):
        ns = self.country.n_solar_plants
        nn = self.country.n_nuclear_plants
        nc = self.country.n_coal_plants


        return (ns, nn, nc)

    def step(self, action):
        # Map action index → a list of env actions
        if action == 1:
            env_action = [{'action': 'commission', 'type': 'solar', 'number': 1}]
        elif action == 2:
            env_action = [{'action': 'commission', 'type': 'nuclear', 'number': 1}]
        elif action == 3:
            env_action = [{'action': 'decommission', 'type': 'solar', 'number': 1}]
        elif action == 4:
            env_action = [{'action': 'decommission', 'type': 'nuclear', 'number': 1}]
        elif action == 5:
            env_action = [{'action': 'decommission', 'type': 'coal', 'number': 1}]
        else:
            env_action = []

        # Advance one month
        self.world.step([env_action])

        # After `step`, the country’s history[-1]['carbon_footprint'] holds the updated carbon footprint.
        month_data = self.country.history[-1]
        carbon = month_data['carbon_footprint']
        energy_penalty = -100000 if month_data['total_energy'] < month_data['energy_demand'] else 0
        budget_penalty = -100000 if month_data['budget'] < 0 else 0
        reward = -carbon + energy_penalty + budget_penalty   # we want to minimize carbon
        


        next_state = self._get_state()
        done = (energy_penalty != 0) or (budget_penalty != 0)

        return next_state, reward, done

# Q-learning hyperparameters
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.2     # epsilon for ε-greedy
n_episodes = 5000
horizon = 12      # 12 months per episode

env = QLearningEnv()

# Q-table: keys = state tuples, values = array of length n_actions
Q = {}
def get_Q(state):
    if state not in Q:
        Q[state] = np.zeros(env.n_actions)
    return Q[state]

# ===== Training loop =====
for _ in range(n_episodes):
    state = env.reset()
    for _ in range(horizon):
        # ε-greedy selection
        if np.random.rand() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(get_Q(state))

        next_state, reward, done = env.step(action)
        if done:
            print(f"Episode ended at state {state} with action {env.action_map[action]}.")
            break

        # Q-learning update
        q_current = get_Q(state)[action]
        q_next_max = np.max(get_Q(next_state))
        get_Q(state)[action] = q_current + alpha * (reward + gamma * q_next_max - q_current)

        state = next_state

# ===== Display the learned Q-values =====
print("Learned Q-values (state → [Q(nothing), Q(solar), Q(nuclear), Q(decommission solar), Q(decommission nuclear), Q(decommission coal)]):")
for s, qvals in Q.items():
    print(f"  {s}: {qvals}")

# ===== Simulate one 12-month run using the greedy policy =====
state = env.reset()
print("\nSimulation using the learned greedy policy:")
for month in range(1, horizon + 1):
    qvals = get_Q(state)
    action = np.argmax(qvals)
    print(f"Month {month}, State {state} → Action = {env.action_map[action]}")
    # print budget, energy, carbon footprint
    print(f"  Budget: {env.country.budget}, Total Energy: {env.country.total_energy}, Carbon Footprint: {env.country.carbon_footprint}")
    
    # print last month data
    state, _, done = env.step(action)
    if done:
        print("Simulation ended due to budget or energy demand issues.")
        break

plot_q_table(Q, env)



