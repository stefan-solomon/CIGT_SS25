import sys
import numpy as np


from simple_simulation import Country, World
from plotting import plot_q_table

class QLearningEnv():
    def __init__(self,Country1, Country2):
        self.Country1 = Country1
        self.Country2 = Country2
        self.world = World([self.Country1, self.Country2])

        self.max_plants = 10
        # Action indices → meaning
        self.action_map = {
            0: 'nothing',
            1: 'commission_solar',
            2: 'commission_nuclear',
            3: 'decommission_solar',
            4: 'decommission_nuclear',
            5: 'decommission_coal',
            6: 'buy_energy',
            7: 'sell_energy'
        }
        self.n_actions = len(self.action_map)

    def reset(self):
        self.__init__(self.Country1, self.Country2)
        return self._get_state(self.Country1), self._get_state(self.Country2)

    def _get_state(self, country):
        ns = country.n_solar_plants
        nn = country.n_nuclear_plants
        nc = country.n_coal_plants


        return (ns, nn, nc)
    
    def _compute_reward(self, country):
        month_data = country.history[-1]
        carbon = month_data['carbon_footprint']
        energy_deficit = True if month_data['total_energy'] < month_data['energy_demand'] else False
        budget_deficit = True if month_data['budget'] < 0 else False
        dead = budget_deficit or energy_deficit
        reward = -carbon - 100000*dead

        
        return reward, dead

    def step(self, action1, action2):
        # Map action index → a list of env actions
        env_action = []
        for act in [action1, action2]:
            if act == 1:
                env_action.append([{'action': 'commission', 'type': 'solar', 'number': 1}])
            elif act == 2:
                env_action.append([{'action': 'commission', 'type': 'nuclear', 'number': 1}])
            elif act == 3:
                env_action.append([{'action': 'decommission', 'type': 'solar', 'number': 1}])
            elif act == 4:
                env_action.append([{'action': 'decommission', 'type': 'nuclear', 'number': 1}])
            elif act == 5:
                env_action.append([{'action': 'decommission', 'type': 'coal', 'number': 1}])
            elif act == 6:
                # Buy energy from the world market
                env_action.append([{'action': 'buy_energy', 'amount': 10}])
            elif act == 7:
                # Sell energy to the world market
                env_action.append([{'action': 'sell_energy', 'amount': 10}])
            else:
                env_action.append([])
        
        
        
        
        # Advance one month
        self.world.step(env_action)

        # After `step`, the country’s history[-1]['carbon_footprint'] holds the updated carbon footprint.
        next_state = []
        for country in self.world.countries:
            reward,dead = self._compute_reward(country)
        
            next_state.append(self._get_state(country))
            


        return next_state, reward, dead



Norway = Country(
            name='Norway',
            population=1_000_000,
            area=100_000,
            budget=10_000_000_000,
            total_energy=10000,
            energy_needed_per_person=0.4,
            carbon_footprint=0
        )
Indonesia = Country(
            name='Indonesia',
            population=800_000,
            area=80_000,
            budget=8_000_000_000,
            total_energy=8000,
            energy_needed_per_person=0.5,
            carbon_footprint=0
        )




# Q-learning hyperparameters
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.2     # epsilon for ε-greedy
n_episodes = 5000
horizon = 12      # 12 months per episode
n_countries = 2

env = QLearningEnv(Norway, Indonesia)

# Q-table: keys = state tuples, values = array of length n_actions
Q_tables = [dict() for _ in range(n_countries)]
def get_Q(state, Q):
    if state not in Q:
        Q[state] = np.zeros(env.n_actions)
    return Q[state]

# ===== Training loop =====
for _ in range(n_episodes):
    states = env.reset()
    for _ in range(horizon):
        # ε-greedy selection
        actions = []
        for i in range(n_countries):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(env.n_actions))
            else:
                actions.append(np.argmax(get_Q(states[i], Q_tables[i])))

        next_state, reward, dead = env.step(actions[0], actions[1])


        # Q-learning update
        for i in range(n_countries):
            action = actions[i]
            q_current = get_Q(states[i],Q_tables[i])[action]
            q_next_max = np.max(get_Q(next_state[i],Q_tables[i]))
            get_Q(states[i],Q_tables[i])[action] = q_current + alpha * (reward + gamma * q_next_max - q_current)

        state = next_state

# ===== Display the learned Q-values =====
print("Learned Q-values (state → [Q(nothing), Q(solar), Q(nuclear), Q(decommission solar), Q(decommission nuclear), Q(decommission coal)]):")
for Q in Q_tables:
    for s, qvals in Q.items():
        print(f"  {s}: {qvals}")

# ===== Simulate one 12-month run using the greedy policy =====
states = env.reset()
print("\nSimulation using the learned greedy policy:")
for month in range(1, horizon + 1):
    qvals1 = get_Q(states[0], Q_tables[0])
    qvals2 = get_Q(states[1], Q_tables[1])
    action1 = np.argmax(qvals)
    action2 = np.argmax(qvals2)
    print(f"Month {month}, State {states[0]} → Action = {env.action_map[action1]},  State {states[1]} → Action = {env.action_map[action2]}")
    # print budget, energy, carbon footprint
    for country in env.world.countries:
        print(f"  {country.name}:")
        print(f"    Budget: {country.budget}, Total Energy: {country.total_energy}, Carbon Footprint: {country.carbon_footprint}")
    
    # print last month data
    state, _, dead = env.step(action1, action2)
    # if done:
    #     print("Simulation ended due to budget or energy demand issues.")
    #     break

# plot_q_table(Q, env)



