from classes_and_functions import *
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument('--independent_carbon', action='store_true', help='Run the simulation with independent carbon footprints for each country')
argparse.add_argument('--no_nuclear', action='store_true', help='Run the simulation without nuclear energy')
argparse.add_argument('--country1', type=str, default='Germany', help='First country to simulate')
argparse.add_argument('--country2', type=str, default='Indonesia', help='Second country to simulate')

def get_country_params(country1, country2):
    params1, params2 = None, None
    if country1 == 'Germany':
        params1 = Germany_params
    elif country1 == 'Norway':
        params1 = Norway_params
    elif country1 == 'Indonesia':
        params1 = Indonesia_params
    elif country1 == 'Egypt':
        params1 = Egypt_params

    if country2 == 'Germany':
        params2 = Germany_params
    elif country2 == 'Norway':
        params2 = Norway_params
    elif country2 == 'Indonesia':
        params2 = Indonesia_params
    elif country2 == 'Egypt':
        params2 = Egypt_params
    return params1, params2


if __name__ == "__main__":
    args = argparse.parse_args()
    Norway_params = {
                'name': 'Norway',
                'population': 6_000_000,
                'area': 100_000,
                'budget': 1e20,
                'total_energy': 0,
                'energy_needed_per_person': 0.58,
                'carbon_footprint': 0,
                'weather_data': IRRADIATION_DATA_NORWAY,
                'monthly_funds': 3_000_000_000,
                'nuclear_cap': 20 # Maximum number of nuclear plants allowed
            }
    Indonesia_params = {
                'name':'Indonesia',
                'population': 100_000_000,
                'area': 80_000,
                'budget': 1e20,
                'total_energy': 0,
                'energy_needed_per_person':0.12,
                'carbon_footprint':0,
                'weather_data': IRRADIATION_DATA_INDONESIA,
                'monthly_funds': 4_500_000_000,
                'nuclear_cap': 40 # Maximum number of nuclear plants allowed
    }

    Egypt_params = {
                'name':'Egypt',
                'population': 110_000_000,
                'area': 100_000,
                'budget': 1e20,
                'total_energy': 0,
                'energy_needed_per_person':0.15,
                'carbon_footprint':0,
                'weather_data': IRRADIATION_DATA_EGYPT,
                'monthly_funds': 3_000_000_000,
                'nuclear_cap': 50 # Maximum number of nuclear plants allowed
    }

    Germany_params = {
                'name': 'Germany',
                'population': 80_000_000,
                'area': 100_000,
                'budget': 1e20,
                'total_energy': 0,
                'energy_needed_per_person': 0.7,
                'carbon_footprint': 0,
                'weather_data': IRRADIATION_DATA_GERMANY,
                'monthly_funds': 15_000_000_000,
                'nuclear_cap': 60 # Maximum number of nuclear plants allowed
    }


    # Q-learning hyperparameters
    alpha = 0.1       # learning rate
    gamma = 0.9       # discount factor
    epsilon = 0.3     # epsilon for ε-greedy
    n_episodes = 5000
    horizon = 12 * 10      # 12 months per episode
    n_countries = 2
    
    params1, params2 = get_country_params(args.country1, args.country2)
    
    env = QLearningEnv(params1, params2, independent_carbon=args.independent_carbon,no_nuclear=args.no_nuclear)

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

           
            # Call the step function with two separate arguments
            next_states, rewards, dones = env.step(actions[0], actions[1])

            # Q-learning update
            for i in range(n_countries):
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
    rewards_per_country = [[], []]
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
        states, rewards, dead = env.step(action1, action2)
        rewards_per_country[0].append(rewards[0])
        rewards_per_country[1].append(rewards[1])

    name_independent = "_independent" if args.independent_carbon else ""
    name_no_nuclear = "_no_nuclear" if args.no_nuclear else ""
    figure_folder = f"figures/{env.world.countries[0].name}_vs_{env.world.countries[1].name}{name_independent}{name_no_nuclear}"
    os.makedirs(figure_folder, exist_ok=True)
  
    plot_budget_history(env.world.countries, folder = figure_folder)
    plot_energy_history(env.world.countries, folder = figure_folder)
    plot_carbon_footprint_history(env.world.countries, folder = figure_folder)
    plot_carbon_footprint_history(env.world.countries, folder = figure_folder, per_capita=True)
    plot_number_of_plants(country1=env.world.countries[0], country2=env.world.countries[1], color_solar1='gold', color_solar2 = 'yellow', color_nuclear1='limegreen', color_nuclear2="lime", color_coal1='dimgray', color_coal2='silver', folder = figure_folder)
    plot_actions_per_country(env.world.countries, folder = figure_folder)
    plot_rewards(rewards_per_country, folder = figure_folder)
    for i, country in enumerate(env.world.countries):
        countries= env.world.countries
        name = country.name
        if countries[0].name == countries[1].name:
            name += f" ({i})"
        pie_plot_energy_production_sources(country, folder = figure_folder, name=name)
    plot_energy_production_clean_dirty(env.world.countries, folder = figure_folder)