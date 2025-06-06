import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
# General Parameters

# Number of months to simulate
n_months = 12

# Average sun hours per month, e.g. for a temperate country
SUN_HOURS_DISTRIBUTION = {
    1:  (50, 10),  # January: mean=50, std=10
    2:  (70, 12),
    3:  (100, 15),
    4:  (150, 15),
    5:  (200, 10),
    6:  (250, 10),
    7:  (270, 8),
    8:  (240, 10),
    9:  (180, 12),
    10: (120, 15),
    11: (80, 12),
    12: (60, 10),
}

# Commissioning costs and capacities for different energy sources
commission_cost_nuclear = 8_000_000_000 # in EUR / month
commission_cost_solar = 80_000_000

# Operating costs for different energy sources
operating_cost_nuclear = 20_000_000 # in EUR / month
operating_cost_solar = 350_000 # in EUR / month
operating_cost_coal = 15_000_000 # in EUR / month

# Monthly energy outputs for different energy sources
output_nuclear = 650_000 # in MWh / month
output_solar = 187.5 # in MWh / hour sunshine
output_coal = 500_000 # in MWh / month

####### MAYBE ADD CAPACITY  ########

# Energy requirement per inhabitant 
energy_requirement_per_person = 0.4 # in MWh / month

# Folder for saving figures
figure_folder = "./figures"


class Country:
    def __init__(self, name, population, area, budget, total_energy, energy_needed_per_person, carbon_footprint):
        self.name = name
        self.population = population
        self.area = area
        self.sun_hours = 0
        self.budget = budget
        self.total_energy = total_energy
        self.carbon_footprint = carbon_footprint
        
        self.energy_needed_per_person = energy_needed_per_person # in MWh / month
        self.energy_demand = self.population * self.energy_needed_per_person
        self.n_coal_plants = 0
        self.n_solar_plants = 0
        self.n_nuclear_plants = 0

        self.history = []

        # Initially fulfill the energy needs with coal power
        needed_energy = self.energy_demand
        n_plants_init = int(np.ceil(needed_energy / output_coal))
        self.n_coal_plants = n_plants_init
    
    def commission_plant(self, plant_type):
        """Add a power plant of the specified type to the country."""
        if plant_type == 'nuclear':
            self.n_nuclear_plants += 1
            self.budget -= commission_cost_nuclear
        elif plant_type == 'solar':
            self.n_solar_plants += 1
            self.budget -= commission_cost_solar
            
    
    def decommission_plant(self, plant_type):
        """Remove a power plant of the specified type from the country."""
        if plant_type == 'nuclear' and self.n_nuclear_plants > 0:
            self.n_nuclear_plants -= 1
        elif plant_type == 'solar' and self.n_solar_plants > 0:
            self.n_solar_plants -= 1
        elif plant_type == 'coal' and self.n_coal_plants > 0:
            self.n_coal_plants -= 1
        else:
            print(f"No {plant_type} plants to decommission in {self.name}.")

    def __repr__(self):
        return f"Country({self.name}, Population: {self.population}, Budget: {self.budget}, Total Energy: {self.total_energy}, Carbon Footprint: {self.carbon_footprint})"
    
    def update(self, month_index, population_change=0, budget_change=5e8, total_energy_change=0):
        """Update the country's data for a given month."""
        month_of_year = month_index % 12 + 1
        sun_mean, sun_std = SUN_HOURS_DISTRIBUTION[month_of_year]
        self.sun_hours =int(np.ceil(np.random.normal(sun_mean, sun_std)))
        self.sun_hours = max(0, self.sun_hours)

        # Operation
        total_production = self.n_nuclear_plants * output_nuclear + \
            self.n_solar_plants * output_solar * self.sun_hours + \
            self.n_coal_plants * output_coal
        
        total_cost = self.n_nuclear_plants * operating_cost_nuclear + \
            self.n_solar_plants * operating_cost_solar * self.sun_hours + \
            self.n_coal_plants * operating_cost_coal
        
        self.carbon_footprint += self.n_coal_plants * 0.9 + self.n_solar_plants * 0.1 + self.n_nuclear_plants * 0.05

        # Update the total energy produced and budget
        self.energy_demand = self.population * self.energy_needed_per_person
        self.budget -= total_cost

        budget_change = budget_change - total_cost
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
            'sun_hours': self.sun_hours
        })
    
    def step_action(self, action):
        """
        Actions: list of dicts like
        [{'action': 'commission', 'type': 'solar', 'number': 2}, {'action': 'decommission', 'type': 'coal', 'number': 1}]
        """
        for act in action:
            if act['action'] == 'commission':
                if act['type'] == 'coal':
                    # Commissioning coal plants is not allowed in this model
                    print(f"Cannot commission coal plants in {self.name}.")
                elif  act['type'] == 'nuclear':
                    action_cost = commission_cost_nuclear * act['number']
                elif act['type'] == 'solar':
                    action_cost = commission_cost_solar * act['number']
                if self.budget >= action_cost:
                    for _ in range(act['number']):
                        self.commission_plant(act['type'])
                    print(f"Just commissioned {act['number']} {act['type']} plants in {self.name}.")
            elif act['action'] == 'decommission':
                for _ in range(act['number']):
                    self.decommission_plant(act['type'])
                print(f"Just decommissioned {act['number']} {act['type']} plants in {self.name}.")
                

class World:
    def __init__(self, countries):
        self.countries = countries  # list of Country instances
        self.month = 0
        self.history = []  # optional global history

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

    def step(self, actions_per_country):
        """
        Steps the simulation by one month.
        :param actions_per_country: list of lists of actions (per country)
        """
        for country, actions in zip(self.countries, actions_per_country):
            country.step_action(actions)  # e.g., build or decommission plants

            external_changes = self.compute_external_changes(country)
            country.update(
                month_index=self.month,
                **external_changes
            )

        self.month += 1  # advance global month counter

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