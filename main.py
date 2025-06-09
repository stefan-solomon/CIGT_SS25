from classes_and_functions import *






if __name__ == "__main__":

    Norway_params = {
                'name': 'Norway',
                'population': 1_000_000,
                'area': 100_000,
                'budget': 10_000_000_000,
                'total_energy': 0,
                'energy_needed_per_person': 0.8,
                'carbon_footprint': 0,
                'weather_data': IRRADIATION_DATA_NORWAY
            }
    Indonesia_params = {
                'name':'Indonesia',
                'population': 800_000,
                'area': 80_000,
                'budget': 8_000_000_000,
                'total_energy': 0,
                'energy_needed_per_person':0.2,
                'carbon_footprint':0,
                'weather_data': IRRADIATION_DATA_INDONESIA
    }




    # Q-learning hyperparameters
    alpha = 0.1       # learning rate
    gamma = 0.9       # discount factor
    epsilon = 0.2     # epsilon for ε-greedy
    n_episodes = 5000
    horizon = 12      # 12 months per episode
    n_countries = 2

    env = QLearningEnv(Norway_params, Indonesia_params)

    # Q-tables: separate for infrastructure and trade actions
    Q_tables = [ {"infra": dict(), "trade": dict()} for _ in range(n_countries) ]

    def get_Q(state, Q, action_type):
        if state not in Q[action_type]:
            if action_type == "infra":
                Q[action_type][state] = np.zeros(env.n_infra_actions)
            else:
                Q[action_type][state] = np.zeros(env.n_trade_actions)
        return Q[action_type][state]

    # ===== Training loop =====
    for _ in range(n_episodes):
        states = env.reset()
        for _ in range(horizon):
            actions = []

            # Select actions for each country
            for i in range(n_countries):
                # ε-greedy infrastructure action
                if np.random.rand() < epsilon:
                    infra_action = np.random.randint(env.n_infra_actions)
                else:
                    infra_action = np.argmax(get_Q(states[i], Q_tables[i], "infra"))

                # ε-greedy trade action
                if np.random.rand() < epsilon:
                    trade_action = np.random.randint(env.n_trade_actions)
                else:
                    trade_action = np.argmax(get_Q(states[i], Q_tables[i], "trade"))

                actions.append((infra_action, trade_action))

            # Step environment
            # Unpack the tuples into four separate arguments
            infra1, trade1 = actions[0]
            infra2, trade2 = actions[1]

            # Call the step function with four separate arguments
            next_states, rewards, dones = env.step(infra1, infra2, trade1, trade2)

            # Q-learning update
            for i in range(n_countries):
                # Infrastructure update
                a_infra = actions[i][0]
                q_infra_current = get_Q(states[i], Q_tables[i], "infra")[a_infra]
                q_infra_next = np.max(get_Q(next_states[i], Q_tables[i], "infra"))
                get_Q(states[i], Q_tables[i], "infra")[a_infra] += alpha * (
                    rewards[i] + gamma * q_infra_next - q_infra_current
                )

                # Trade update
                a_trade = actions[i][1]
                q_trade_current = get_Q(states[i], Q_tables[i], "trade")[a_trade]
                q_trade_next = np.max(get_Q(next_states[i], Q_tables[i], "trade"))
                get_Q(states[i], Q_tables[i], "trade")[a_trade] += alpha * (
                    rewards[i] + gamma * q_trade_next - q_trade_current
                )

            states = next_states

    # ===== Display the learned Q-values =====
    print("Learned Q-values (state → [Q(nothing), Q(solar), Q(nuclear), Q(decommission solar), Q(decommission nuclear), Q(decommission coal), Q(buy energy),Q(sell energy) ]):")
    for Q in Q_tables:
        for s, qvals in Q.items():
            print(f"  {s}: {qvals}")

    # ===== Simulate one 12-month run using the greedy policy =====
    states = env.reset()
    print("\nSimulation using the learned greedy policy:")
    for month in range(1, horizon + 1):
        # Get Q-values for both countries
        qvals1_infra = get_Q(states[0], Q_tables[0], "infra")
        qvals1_trade = get_Q(states[0], Q_tables[0], "trade")
        qvals2_infra = get_Q(states[1], Q_tables[1], "infra")
        qvals2_trade = get_Q(states[1], Q_tables[1], "trade")

        # Choose best actions
        infra1 = np.argmax(qvals1_infra)
        trade1 = np.argmax(qvals1_trade)
        infra2 = np.argmax(qvals2_infra)
        trade2 = np.argmax(qvals2_trade)

        # Print status
        print(f"Month {month},")
        print(f"  State {states[0]} → Infra = {env.infra_action_map[infra1]}, Trade = {env.trade_action_map[trade1]}")
        print(f"  State {states[1]} → Infra = {env.infra_action_map[infra2]}, Trade = {env.trade_action_map[trade2]}")

        for country in env.world.countries:
            print(f"  {country.name}:")
            print(f"    Budget: {country.budget}, Total Energy: {country.total_energy}, Carbon Footprint: {country.carbon_footprint}")

        # Take a step in the environment
        states, _, dead = env.step(infra1, infra2, trade1, trade2)

        if any(dead):
            print("One of the countries failed (e.g. ran out of energy/money). Ending early.")
            break
        # if done:
        #     print("Simulation ended due to budget or energy demand issues.")
        #     break

    # plot_q_table(Q, env)