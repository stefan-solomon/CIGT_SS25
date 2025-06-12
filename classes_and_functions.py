# Import necessary libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Important Parameters
# Irradiation data for each month - transduction to solar energy
# Values in kWh/m^2
IRRADIATION_DATA_NORWAY = {
    1:  0.5,  # January
    2:  0.9,  # February
    3:  2,  # March
    4:  3,  # April
    5:  4.9,  # May
    6:  5.3,  # June
    7:  5.1,  # July
    8:  3.9,  # August
    9:  2.1,  # September
    10: 1.1,  # October
    11: 0.3,  # November
    12: 0.1   # December
}

IRRADIATION_DATA_INDONESIA = {
    1:  5.26,  # January
    2:  5.35,  # February
    3:  5.43,  # March
    4:  5.51,  # April
    5:  5.56,  # May
    6:  5.58,  # June
    7:  5.57,  # July
    8:  5.54,  # August
    9:  5.46,  # September
    10: 5.37,   # October
    11: 5.3,   # November
    12: 5.27    # December
}

IRRADIATION_DATA_EGYPT = {
    1:  4.3,  # January
    2:  5.6,  # February
    3:  6.2,  # March
    4:  7.5,  # April
    5:  7.4,  # May
    6:  7.8,  # June
    7:  7.6,  # July
    8:  7.1,  # August
    9:  6.5,  # September
    10: 5.6,   # October
    11: 4.6,   # November
    12: 4.2    # December
}

SOLAR_PLANT_SURFACE = 100_000 # Average example area in m^2 
CONVERSION_EFFICIENCY = 0.18  # Average conversion efficiency of solar panels

def calculate_solar_energy(irradiation_data):
    """
    Create a Dictionary with monthly solar energy production based on irradiation data.
    """
    months = list(irradiation_data.keys())
    irradiation_values = list(irradiation_data.values())
    dictionary ={}
    
    # Calculate monthly energy production in kWh
    monthly_energy = [irradiation * SOLAR_PLANT_SURFACE * CONVERSION_EFFICIENCY for irradiation in irradiation_values]
    
    for month, energy in zip(months, monthly_energy):
        dictionary.update({month: energy})
    
    return dictionary

# Environmental Impact of Different Solar Plants
carbon_emission_coal = 1000 # kg CO2/MWh
carbon_emission_nuclear = 8 # kg CO2/MWh
carbon_emission_solar = 25 # kg CO2/MWh


# Commissioning costs and capacities for different energy sources
commission_cost_nuclear = 8_000_000_000 # in EUR / month
commission_cost_solar = 80_000_000

# Operating costs for different energy sources
lcoe_coal = 118 # in EUR / MWh
lcoe_nuclear = 182 # in EUR / MWh
lcoe_solar = 50 # in EUR / MWh

# Monthly energy outputs for different energy sources
output_nuclear = 650_000 # in MWh / month
output_coal = 400_000 # in MWh / month

####### MAYBE ADD CAPACITY  ########

# Energy requirement per inhabitant 
energy_requirement_per_person = 0.4 # in MWh / month

# =================================================================

"""
# 1. Buyers and Sellers

Implementing the Buyer and Seller classes instead of the single-purpose Country class used until now.
Giving the countries the possibility to be either a buyer or seller allows us to take the first steps
in creating a simple market environment, good for studying Q-learning, agents, and following strategies
of the players.

This will allow us to gather data which we will analyze in the report.
"""

# === Define the Country class ===
class Country:
    def __init__(self, name, population, area, budget, total_energy, energy_needed_per_person, carbon_footprint, weather_data, monthly_funds):
        self.name = name
        self.population = population
        self.area = area
        self.irradiation_data = {}
        self.budget = budget
        self.total_energy = total_energy
        self.carbon_footprint = carbon_footprint
        self.monthly_funds = monthly_funds  # Monthly funds available for energy production

        self.weather_data = weather_data  # Dictionary with monthly irradiation data
        
        self.energy_needed_per_person = energy_needed_per_person # in MWh / month
        self.energy_demand = self.population * self.energy_needed_per_person
        self.n_solar_plants = 0
        self.n_nuclear_plants = 0
        self.n_decommissioned_coal_plants = 0

        self.history = []

        # Initially fulfill the energy needs with coal power
        needed_energy = self.energy_demand
        n_plants_init = int(np.ceil(needed_energy / output_coal))
        self.n_coal_plants = n_plants_init
    
    def commission_plant(self, plant_type):
        """Add a power plant of the specified type to the country."""
        if plant_type == 'nuclear':
            if self.budget >= commission_cost_nuclear:
                self.n_nuclear_plants += 1
                self.budget -= commission_cost_nuclear
                print(f"Commissioned a nuclear plant. Total nuclear plants: {self.n_nuclear_plants}")
            else:
                print("Insufficient budget to commission a nuclear plant.")
        elif plant_type == 'solar':
            if self.budget >= commission_cost_solar:
                self.n_solar_plants += 1
                self.budget -= commission_cost_solar
                print(f"Commissioned a solar plant. Total solar plants: {self.n_solar_plants}")
            else:
                print("Insufficient budget to commission a solar plant.")
        
        elif plant_type == 'coal':
            pass  # Coal plants are not commissioned in this simulation
        

    def decommission_plant(self, plant_type):
        """Remove a power plant of the specified type from the country."""
        if plant_type == 'nuclear' and self.n_nuclear_plants > 0:
            if self.n_nuclear_plants > 0:
                self.n_nuclear_plants -= 1
                print(f"Decommissioned a nuclear plant. Total nuclear plants: {self.n_nuclear_plants}")
            else:
                print("No nuclear plants to decommission.")
        elif plant_type == 'solar':
            if self.n_solar_plants > 0:
                self.n_solar_plants -= 1
                print(f"Decommissioned a solar plant. Total solar plants: {self.n_solar_plants}")
            else:
                print("No solar plants to decommission.")
        elif plant_type == 'coal':
            if self.n_coal_plants > 0:
                self.n_coal_plants -= 1
                self.n_decommissioned_coal_plants += 1
                print(f"Decommissioned a coal plant. Total coal plants: {self.n_coal_plants}")
            else:
                print("No coal plants to decommission.")
            
        
    
    def __repr__(self):
        return f"Buyer(name={self.name}, population={self.population}, area={self.area}, budget={self.budget}, total_energy={self.total_energy}, energy_needed_per_person={self.energy_needed_per_person})"
    
    def update(self, month_index, population_change=0, budget_change=5e8, total_energy_change=0):
        """Update the country's data for a given month."""
        month_of_year = month_index % 12 + 1
        irradiation = self.weather_data[month_of_year]
        solar_output_per_plant = irradiation * SOLAR_PLANT_SURFACE * CONVERSION_EFFICIENCY / 1000 # in MWh
        solar_output = self.n_solar_plants * solar_output_per_plant  # Total solar output in MWh
        nuclear_output = self.n_nuclear_plants * output_nuclear  # Total nuclear output in MWh
        coal_output = self.n_coal_plants * output_coal  # Total coal output in MWh

        # Operation
        total_production = solar_output + nuclear_output + coal_output
            
        solar_cost = solar_output * lcoe_solar # in EUR
        nuclear_cost = nuclear_output * lcoe_nuclear # in EUR
        coal_cost = coal_output * lcoe_coal # in EUR
        
        total_cost = solar_cost + nuclear_cost + coal_cost
        
        self.carbon_footprint += carbon_emission_coal * coal_output + carbon_emission_nuclear * nuclear_output + carbon_emission_solar * solar_output # in kg CO2

        # Update the total energy produced and budget
        self.energy_demand = self.population * self.energy_needed_per_person

        budget_change = budget_change - total_cost + self.monthly_funds  # Monthly funds added to the budget
        total_energy_change = total_energy_change + total_production - self.energy_demand

        self.population += population_change
        self.budget += budget_change
        self.total_energy += total_energy_change

        # Write the history of the country
        self.history.append({
            'month': month_index,
            'population': self.population,
            'budget': self.budget,
            'total_energy': self.total_energy,
            'energy_demand': self.energy_demand,
            'carbon_footprint': self.carbon_footprint,
            'n_coal_plants': self.n_coal_plants,
            'n_solar_plants': self.n_solar_plants,
            'n_nuclear_plants': self.n_nuclear_plants,
        })

    def action_infrastructure(self, action, type_of_plant):
        """
        Perform an action related to infrastructure, such as commissioning or decommissioning a plant.
        """
        if action == 'commission':
            self.commission_plant(type_of_plant)
        
        elif action == 'decommission':
            self.decommission_plant(type_of_plant) 
        elif action == 'nothing':
            print(f"{self.name} does nothing this month.")
        else:
            print(f"Unknown action: {action}.")

# === Define the Buyer class ===
class Buyer(Country):
    def calculate_offer_price_per_MWh(self, fraction_of_production_cost=0.9):
        """
        Calculate the offer price per MWh based on the budget and total energy.
        """
        return 1000  # Placeholder for offer price calculation

    def action_buy_energy(self, amount):
        """
        Action to buy energy from another country.
        """
        # Condition to check if the country can afford and needs the energy in the World class
        if amount > 0 and self.budget >= self.calculate_offer_price_per_MWh() * amount:
            self.budget -= self.calculate_offer_price_per_MWh() * amount
            self.total_energy += amount
            self.history[-1]['total_energy'] += amount
        else:
            return None

# === Define the Seller class ===
class Seller(Country):
    def calculate_demand_price_per_MWh(self, profit_margin=0.1):
        """
        Calculate the offer price per MWh based on the budget and total energy.
        """
        return 1000  # Placeholder for offer price calculation
    
    def action_sell_energy(self, amount):
        """
        Action to buy energy from another country.
        """
        # Condition to check if the country can afford and needs the energy in the World class
        if amount > 0 and self.total_energy >= amount:
            self.total_energy -= amount
            self.history[-1]['total_energy'] -= amount
            self.budget += self.calculate_demand_price_per_MWh() * amount
        else:
            return None
        
# =================================================================

# === Define the World class ===
class World:
    def __init__(self, countries):
        self.countries = countries  # list of Country instances
        self.month = 0
        self.history = []  # optional global history
        self.number_successful_trades = 0  # Counter for successful trades

    def compute_external_changes(self, country):
        """
        Returns a dictionary of external changes for a country.
        Can be customized per country or per month.
        """
        return {
            'population_change': country.population * 0, # e.g., 0% population growth
            'budget_change': 0,  # optionally add tax revenue, aid, etc.
            'total_energy_change': 0  # e.g., import/export, grid loss
        }
    
    def trade_energy(self, buyer_country, amount, seller_country):
        # Check if countries correspond to the right roles
        if isinstance(buyer_country, Buyer) and isinstance(seller_country, Seller):
            # Calculate prices based on the countries' budgets and energy needs
            profit_margin = 0.1
            buyer_price_per_MWh = min(buyer_country.calculate_offer_price_per_MWh()/2, buyer_country.budget / amount)  
            seller_price_per_MWh = seller_country.calculate_demand_price_per_MWh()*(1+profit_margin)  # seller's price is higher to account for profit margin
            
            if buyer_price_per_MWh >= seller_price_per_MWh:
                price = (buyer_price_per_MWh+ seller_price_per_MWh)*amount / 2
                if price >= seller_price_per_MWh * amount / (1+profit_margin):
                    # Perform the trade
                    if price > buyer_country.budget:
                        print(f"Trade failed: {buyer_country.name} cannot afford the trade price of {price} EUR.")
                        return
                    buyer_country.action_buy_energy(amount)
                    seller_country.action_sell_energy(amount)
                    
                    print(f"Trade successful: {buyer_country.name} bought {amount} MWh from {seller_country.name} at {price} EUR.")
                    self.number_successful_trades += 1
                else:
                    print(f"Trade failed: {seller_country.name} make a profit from the trade")
                    return
                
            else:
                print(f"Trade failed: {buyer_country.name} wants to pay {buyer_price_per_MWh} per MWh while {seller_country.name} wants to sell for {seller_price_per_MWh} per MWh.")
                return
            
        else:
            print("Trade failed: Invalid buyer or seller country type.")
            return

            
            
    def step(self, actions_per_country):
        # Step 1: Apply infrastructure changes
        buyer_action = []
        seller_action = []
        for country, actions in zip(self.countries, actions_per_country):
            infra_actions = [a for a in actions if a['action'] in ['commission', 'decommission']]
            for a in infra_actions:
                country.action_infrastructure(a['action'], a['type'])
            buyer_action_list = [a for a in actions if a['action'] == 'buy_energy']
            if len(buyer_action_list) >0:
                buyer_action.append(buyer_action_list[0]) 
            seller_action_list = [a for a in actions if a['action'] == 'sell_energy' or a['action'] == 'dont_sell_energy']
            if len(seller_action_list) > 0:
                seller_action.append(seller_action_list[0])
        amount = buyer_action[0]['amount'] if len(buyer_action) > 0 else 0
        if len(buyer_action) >0 and len(seller_action)>0:
            buyer_action = buyer_action[0]
            seller_action = seller_action[0]
            if buyer_action['action'] == 'buy_energy' and seller_action['action'] == 'sell_energy':
                self.trade_energy(
                    self.countries[0], amount,
                    self.countries[1]
                )
        


        # Step 3: Apply external changes and update each country's state
        for country in self.countries:
            external_changes = self.compute_external_changes(country)
            country.update(
                month_index=self.month,
                **external_changes
            )

        self.month += 1

    def run(self, months=12, action_schedule=None):
        """
        Run simulation for multiple months.
        :param months: number of months to simulate
        :param action_schedule: optional list of actions per month
               Format: list of length 'months', where each entry is a list
               of actions for each country.
        """
        for i in range(months):
            if action_schedule:
                actions = action_schedule[i]
            else:
                actions = [[] for _ in self.countries]  # no actions this month
            self.step(actions)

    def summary(self):
        """
        Print a summary of each country after the simulation.
        """
        for country in self.countries:
            print(country)

    def return_successful_trades(self):
        """
        Return the number of successful trades.
        """
        return self.number_successful_trades

# =================================================================

# === Define the QLearningEnv class ===
class QLearningEnv():
    def __init__(self,Country1_params, Country2_params):
        self.Country1 = Buyer(**Country1_params)
        self.Country2 = Seller(**Country2_params)
        self.Country1_params = Country1_params
        self.Country2_params = Country2_params
        self.world = World([self.Country1, self.Country2])

        self.max_plants = 10
        # Action indices → meaning
        self.infra_action_map = {
            0: 'nothing',
            1: 'commission_solar',
            2: 'commission_nuclear',
            3: 'decommission_solar',
            4: 'decommission_nuclear',
            5: 'decommission_coal'
        }
        self.trade_action_map_1 = {
            0: 'nothing',
            1: 'buy 1_000 MWh energy',
            2: 'buy 10_000 MWh energy',
            3: 'buy 100_000 MWh energy',
            4: 'buy 1_000_000 MWh energy',
        }
        self.trade_action_map_2 = {
            0: 'no',
            1: 'yes'
        }
        self.n_infra_actions = len(self.infra_action_map)
        self.n_trade_actions_1 = len(self.trade_action_map_1)
        self.n_trade_actions_2 = len(self.trade_action_map_2)


    def reset(self):
        self.__init__(self.Country1_params, self.Country2_params)
        return self._get_state(self.Country1), self._get_state(self.Country2)

    def _get_state(self, country):
        ns = country.n_solar_plants
        nn = country.n_nuclear_plants
        nc = country.n_coal_plants
        budget = country.budget
        energy = country.total_energy
        enough_budget_for_solar = budget >= commission_cost_solar
        enough_budget_for_nuclear = budget >= commission_cost_nuclear
        enough_energy_for_one_month = energy >= country.energy_demand
        enough_energy_for_two_months = energy >= 2 * country.energy_demand
        enough_energy_for_three_months = energy >= 3 * country.energy_demand
        return (ns,nn,nc,
                enough_budget_for_solar, enough_budget_for_nuclear,enough_energy_for_one_month,
                enough_energy_for_two_months, enough_energy_for_three_months)
    
    def _compute_reward(self, country):
        
        month_data = country.history[-1]
        carbon = month_data['carbon_footprint']
        if len(country.history) < 2:
            carbon_increase = carbon
        else:
            carbon_increase = carbon - country.history[-2]['carbon_footprint']
        energy_deficit = month_data['total_energy'] - month_data['energy_demand']
        budget_deficit = month_data['budget']
        dead = budget_deficit or (energy_deficit<0)

        # Scale the reward terms:
        carbon_increase_rew = carbon_increase * 1e-12  # Scale down carbon increase
        total_carbon_rew = carbon * 1e-12  # Scale down total carbon footprint
        energy_deficit_rew = energy_deficit * 1e-8  # Scale down energy deficit
        budget_deficit_rew = budget_deficit * 1e-11  # Scale down budget deficit
        total_energy_rew = month_data['total_energy'] * 1e-8  # Scale down total energy
        budget_rew = month_data['budget'] * 1e-11  # Scale down budget
        decommissioned_coal_rew = country.n_decommissioned_coal_plants

        reward = - 70 * carbon_increase_rew + \
                    - 2 * total_carbon_rew + \
                    - 0 * energy_deficit_rew + \
                    - 0 * budget_deficit_rew + \
                    5 * total_energy_rew + \
                    3 * budget_rew +\
                    10 * decommissioned_coal_rew + \
                    - 1000 * dead  # Large penalty for being dead
        

        
        return reward, dead

    def step(self, infra1, infra2, trade1, trade2):
        env_actions = []
        infra_action_list = []
        trade_action_list = []
        for infra in [infra1, infra2]:
            # Infra actions
            if infra == 1:
                infra_action_list.append({'action': 'commission', 'type': 'solar', 'number': 1})
            elif infra == 2:
                infra_action_list.append({'action': 'commission', 'type': 'nuclear', 'number': 1})
            elif infra == 3:
                infra_action_list.append({'action': 'decommission', 'type': 'solar', 'number': 1})
            elif infra == 4:
                infra_action_list.append({'action': 'decommission', 'type': 'nuclear', 'number': 1})
            elif infra == 5:
                infra_action_list.append({'action': 'decommission', 'type': 'coal', 'number': 1})
            else:
                infra_action_list.append({'action': 'nothing'})
            # 0 → do nothing

        # Trade actions
        if trade1 == 1:
            trade_action_list.append({'action': 'buy_energy', 'amount': 1000})
            if trade2 == 1:
                trade_action_list.append({'action': 'sell_energy'})
            else:
                trade_action_list.append({'action': 'dont_sell_energy'})
        elif trade1 == 2:
            trade_action_list.append({'action': 'buy_energy', 'amount': 10000})
            if trade2 == 1:
                trade_action_list.append({'action': 'sell_energy'})
            else:
                trade_action_list.append({'action': 'dont_sell_energy'})
        elif trade1 == 3:
            trade_action_list.append({'action': 'buy_energy', 'amount': 100000})
            if trade2 == 1:
                trade_action_list.append({'action': 'sell_energy'})
            else:
                trade_action_list.append({'action': 'dont_sell_energy'})
        elif trade1 == 4:
            trade_action_list.append({'action': 'buy_energy', 'amount': 1000000})
            if trade2 == 1:
                trade_action_list.append({'action': 'sell_energy'})
            else:
                trade_action_list.append({'action': 'dont_sell_energy'})
        else:
            trade_action_list.append({'action': 'nothing'})
            trade_action_list.append({'action': 'no'})
        # 0 → do nothing
        for i in range(len(self.world.countries)):
            action_list = []
            action_list.append(infra_action_list[i])  # Add infrastructure action
            action_list.append(trade_action_list[i])
            env_actions.append(action_list)

        

        self.world.step(env_actions)

        next_states = []
        rewards = []
        dones = []
        for country in self.world.countries:
            reward, dead = self._compute_reward(country)
            rewards.append(reward)
            dones.append(dead)
            next_states.append(self._get_state(country))

        return next_states, rewards, dones
    

def plot_energy_history(countries, title="Country History"):
    """
    Plot the history of each country.
    """
    plt.figure(figsize=(12, 8))
    for country in countries:
        df = pd.DataFrame(country.history)
        plt.plot(df['month'], df['total_energy'], label=f"{country.name} Total Energy")
    
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_carbon_footprint_history(countries, title="Carbon Footprint History"):
    """
    Plot the carbon footprint history of each country.
    """
    plt.figure(figsize=(12, 8))
    for country in countries:
        df = pd.DataFrame(country.history)
        plt.plot(df['month'], df['carbon_footprint'], label=f"{country.name} Carbon Footprint")
    
    plt.xlabel("Month")
    plt.ylabel("Carbon Footprint (kg CO2)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_budget_history(countries, title="Budget History"):
    """
    Plot the budget history of each country.
    """
    plt.figure(figsize=(12, 8))
    for country in countries:
        df = pd.DataFrame(country.history)
        plt.plot(df['month'], df['budget'], label=f"{country.name} Budget")
    
    plt.xlabel("Month")
    plt.ylabel("Budget (EUR)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_number_of_plants(country1 ,country2 , color_solar1, color_solar2, color_nuclear1, color_nuclear2, color_coal1, color_coal2, title="Number of Plants History"):
    """
    Plot the number of plants history of each country.
    """
    plt.figure(figsize=(12, 8))
    
    df1 = pd.DataFrame(country1.history)
    df2 = pd.DataFrame(country2.history)

    plt.plot(df1['month'], df1['n_solar_plants'], color = color_solar1, label=f"{country1.name} Solar Plants")
    plt.plot(df1['month'], df1['n_nuclear_plants'], color = color_nuclear1, label=f"{country1.name} Nuclear Plants")
    plt.plot(df1['month'], df1['n_coal_plants'], color = color_coal1, label=f"{country1.name} Coal Plants")
    plt.plot(df2['month'], df2['n_solar_plants'], color = color_solar2, label=f"{country2.name} Solar Plants")
    plt.plot(df2['month'], df2['n_nuclear_plants'], color = color_nuclear2, label=f"{country2.name} Nuclear Plants")
    plt.plot(df2['month'], df2['n_coal_plants'], color = color_coal2, label=f"{country2.name} Coal Plants")
    
    plt.xlabel("Month")
    plt.ylabel("Number of Plants")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
        
