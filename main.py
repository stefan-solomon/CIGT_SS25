from classes_and_functions import *





if __name__ == "__main__":

    Norway_params = {
                'name': 'Norway',
                'population': 6_000_000,
                'area': 100_000,
                'budget': 50_000_000_000,
                'total_energy': 0,
                'energy_needed_per_person': 0.58,
                'carbon_footprint': 0,
                'weather_data': IRRADIATION_DATA_NORWAY,
                'monthly_funds': 3_000_000_000
            }
    Indonesia_params = {
                'name':'Indonesia',
                'population': 281_000_000,
                'area': 80_000,
                'budget': 10_000_000_000,
                'total_energy': 0,
                'energy_needed_per_person':0.12,
                'carbon_footprint':0,
                'weather_data': IRRADIATION_DATA_INDONESIA,
                'monthly_funds': 4_500_000_000
    }

    


    # Q-learning hyperparameters
    alpha = 0.1       # learning rate
    gamma = 0.9       # discount factor
    epsilon = 0.2     # epsilon for ε-greedy
    n_episodes = 5000
    horizon = 12 * 4      # 12 months per episode
    n_countries = 2

    env = QLearningEnv(Norway_params, Indonesia_params)

    # Q-tables
    Q_tables = [ dict() for _ in range(n_countries) ]

    def get_Q(state, Q):
        if state not in Q:
            Q[state] = np.zeros(env.n_actions)

        return Q[state]

    # ===== Training loop =====
    for _ in range(n_episodes):
        states = env.reset()
        
        for _ in range(horizon):
            actions = []
            # Select actions for each country
            for i in range(n_countries):
                # ε-greedy action
                if np.random.rand() < epsilon:
                    action = np.random.randint(env.n_actions)
                else:
                    action = np.argmax(get_Q(states[i], Q_tables[i]))

                actions.append(action)

           
            # Step environment
            # Unpack the tuple two separate arguments
            act1, act2 = actions

            # Call the step function with two separate arguments
            next_states, rewards, dones = env.step(act1, act2)

            # Q-learning update
            for i in range(n_countries):
                buyer = (i == 0)  # Country 1 is the buyer, Country 2 is the seller
                # Infrastructure update
                action = actions[i]
                q_current = get_Q(states[i], Q_tables[i])[action]
                q_next = np.max(get_Q(next_states[i], Q_tables[i]))
                get_Q(states[i], Q_tables[i])[action] += alpha * (
                    rewards[i] + gamma * q_next - q_current
                )

            states = next_states

    # ===== Display the learned Q-values =====
    print("Learned Q-values (state → [Q(nothing), Q(solar), Q(nuclear), Q(decommission solar), Q(decommission nuclear), Q(decommission coal) ]):")
    for Q in Q_tables:
        for s, qvals in Q.items():
            print(f"  {s}: {qvals}")

    # ===== Simulate one 12-month run using the greedy policy =====
    states = env.reset()
    print("\nSimulation using the learned greedy policy:")
    for month in range(1, horizon + 1):
        # Get Q-values for both countries
        qvals1 = get_Q(states[0], Q_tables[0])
        qvals2 = get_Q(states[1], Q_tables[1])
        

        # Choose best actions
        action1 = np.argmax(qvals1)
        action2 = np.argmax(qvals2)



        # Print status
        print(f"Month {month},")
        print(f"  State {states[0]} → Action = {env.action_map[action1]}")
        print(f"  State {states[1]} → Action = {env.action_map[action2]}")

        for country in env.world.countries:
            print(f"  {country.name}:")
            print(f"    Budget: {country.budget}, Total Energy: {country.total_energy}, Carbon Footprint: {country.carbon_footprint}")

        # Take a step in the environment
        states, _, dead = env.step(action1, action2)

  
    plot_budget_history(env.world.countries)
    plot_energy_history(env.world.countries)
    plot_carbon_footprint_history(env.world.countries)
    plot_number_of_plants(country1=env.world.countries[0], country2=env.world.countries[1], color_solar1='gold', color_solar2 = 'yellow', color_nuclear1='limegreen', color_nuclear2="lime", color_coal1='dimgray', color_coal2='silver')
