#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import logging
import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

# Function used to read data
from utils import get_config, read_constant, read_profile

# Functions used to build the model
from utils import calculate_annuities_capex, calculate_annuities_storage_capex, update_vom_costs_scc, \
    define_month_hours, shift_range

# Functions used to process outputs
from utils import get_technical_cost, extract_curtailment, extract_hourly_balance, \
    extract_spot_price, extract_primary_gene, extract_annualized_costs_investment_new_capa, extract_CH4_to_power,\
    extract_power_to_CH4, extract_power_to_H2, extract_annualized_costs_investment_new_capa_nofOM, \
    extract_OM_cost, extract_carbon_value, extract_H2_to_power, get_carbon_content, \
    compute_costs, extract_summary, transportation_distribution_cost, extract_balance, extract_storage_losses, \
    extract_profit, calculate_lcoe_per_tech, extract_carbon_footprint

# Functions used to plot outputs
from utils import plot_load_shift_week, plot_elec_balance_week, plot_elec_residual_balance_week, \
    plot_storage_state_year, plot_installed_power, plot_gene_per_tech, compare_operable_mix

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Set,
    NonNegativeReals,
    Constraint,
    SolverFactory,
    Suffix,
    Var,
    Objective,
    value
)


# # Model definition

# In[4]:


class ModelEOLES():
    def __init__(self, name, config, output_path,
                 include_reserve=False,
                 existing_capacity=None, existing_charging_capacity=None, existing_energy_capacity=None,
                 existing_annualized_costs_elec=0,
                 existing_annualized_costs_CH4=0, existing_annualized_costs_H2=0):
        """

        :param name: str
        :param config: dict
        :param output_path: str

        The following input(s) activate(s) certain optional functionalities of the model:
        :param include_reserve: bool
            defaults as False. Dictates whether to include reserves in the optimization.

        The following inputs allow for successive iterations of the model, with capacity installed at a given date
        still being operational in the following iteration :
        :param existing_capacity: pd.Series
        :param existing_charging_capacity: pd.Series
        :param existing_energy_capacity: pd.Series
        :param existing_annualized_costs_elec: float
        :param existing_annualized_costs_CH4: float
        :param existing_annualized_costs_H2: float
        """
        self.name = name
        self.config = config
        self.output_path = output_path
        self.model = ConcreteModel()
        # Dual Variable, used to get the marginal value of an equation.
        self.model.dual = Suffix(direction=Suffix.IMPORT)

        self.include_reserve = include_reserve

        self.existing_capacity = existing_capacity # GW
        self.existing_charging_capacity = existing_charging_capacity # GW
        self.existing_energy_capacity = existing_energy_capacity # GWh
        self.existing_annualized_costs_elec = existing_annualized_costs_elec # 1e6€/GW/yr = €/kW/yr
        self.existing_annualized_costs_CH4 = existing_annualized_costs_CH4 # 1e6€/GW/yr = €/kW/yr
        self.existing_annualized_costs_H2 = existing_annualized_costs_H2 # 1e6€/GW/yr = €/kW/yr


    def load_inputs(self):

        #############################
        ### Simulation parameters ###
        #############################

        self.nb_years = self.config["nb_optimisation_years"]
        self.years_of_interest = self.config["years_of_interest"]
        # Anticipated social cost of carbon used to calculate emissions and to find optimal power mix.
        self.scc = self.config["social_cost_of_carbon"]  # €/tCO2
        self.year_costs = self.config["year_costs"]   # Year in which all the investment and construction is done, and from which all prices are taken. Not necessarily linked to meteorological data and demand years
        self.carbon_constraint = self.config["carbon_constraint"]   # bool : if True, the carbon constraint is used rather than SCC
        self.H2_demand_is_profile = self.config["H2_demand_is_profile"]
        self.CH4_demand_is_profile = self.config["CH4_demand_is_profile"]


        ###########################
        ### Exogeneous profiles ###
        ###########################

        #List of fatal production profiles to import
        vre_list = ["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                    "river", "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge"]
        vre_profile_list = [read_profile(self.config["prod_profile_"+tech], time_scale="hourly",
                                nb_years=self.nb_years, years_of_interest=self.years_of_interest) for tech in vre_list]
        for vre_profile in vre_profile_list :       # We limit how small the coefficients can be to facilitate solving
            # Installed capacities are of the order of 1 to 100 GW, so a load_factor of 1e-6 means a production of 100 to 1 kW, which is inconsequential
            # 1e6 is also the default tolerance on constraints for Gurobi which we kept, so being more precise than that is not useful
            vre_profile.loc[vre_profile < 1e-6] = 0
        self.vre_profiles = pd.concat(vre_profile_list, axis=1, keys=vre_list)

        self.elec_demand = read_profile(self.config["elec_demand"], time_scale="hourly", nb_years=self.nb_years)  # GW
        self.capacity_factor_nuclear_hourly = read_profile(self.config["capacity_factor_nuclear_hourly"],
                                                           time_scale="hourly", nb_years=self.nb_years, years_of_interest = self.years_of_interest)
        self.lake_inflows = read_profile(self.config["lake_inflows"], time_scale="hourly", nb_years=self.nb_years)  # GWh
        self.phs_inflows = read_profile(self.config["phs_inflows"], time_scale="hourly", nb_years=self.nb_years)  # GWh

        # Inflow that cannot go into storage and has to be used instantly for electricity production
        # (exogenous because the aggregation of all lakes greatly reduces precision on the lake volume constraint)
        self.lake_spill = read_profile(self.config["lake_spill"], time_scale="hourly", nb_years=self.nb_years)
        # Minimal lake volume (effective in summer) for touristic reasons in GWh
        self.lake_minimal_volume = read_profile(self.config["lake_minimal_volume"], time_scale="hourly", nb_years=self.nb_years, years_of_interest=self.years_of_interest)
        # Minimal outflow (daily constraint) for ecological and agricultural reasons in GW
        self.lake_minimal_outflow = read_profile(self.config["lake_minimal_outflow"], time_scale="daily", nb_years=self.nb_years, years_of_interest=self.years_of_interest)

        if self.H2_demand_is_profile:
            self.H2_demand = read_profile(self.config["H2_demand"], time_scale = "hourly", nb_years=self.nb_years, years_of_interest=self.years_of_interest)   # GWh
        if self.CH4_demand_is_profile:
            self.CH4_demand = read_profile(self.config["CH4_demand"], time_scale = "hourly", nb_years=self.nb_years, years_of_interest=self.years_of_interest)   # GWh

        ############################
        ### Exogeneous constants ###
        ############################

        if not self.H2_demand_is_profile:
            self.H2_demand = self.config["H2_demand"]   # GWh
        if not self.CH4_demand_is_profile:
            self.CH4_demand = self.config["CH4_demand"]   # GWh

        if self.existing_capacity is None:    # ie was not provided by an earlier instance of the model
            self.existing_capacity = read_constant(self.config["existing_capacity"]) # GW
        if self.existing_charging_capacity is None:    # ie was not provided by an earlier instance of the model
            self.existing_charging_capacity = read_constant(self.config["existing_charging_capacity"]) # GW
        if self.existing_energy_capacity is None:    # ie was not provided by an earlier instance of the model
            self.existing_energy_capacity = read_constant(self.config["existing_energy_capacity"]) # GWh

        self.maximum_capacity = read_constant(self.config["maximum_capacity"]) # GW
        self.maximum_energy_capacity = read_constant(self.config["maximum_energy_capacity"]) # GWh
        self.fixed_installed_power = read_constant(self.config["fixed_installed_power"]) # GW
        self.fixed_charging_power = read_constant(self.config["fixed_charging_power"]) # GW
        self.fixed_energy_capacity = read_constant(self.config["fixed_energy_capacity"]) # GWh
        self.fcr_requirement = self.config["fcr_requirement"] # GW
        self.frr_requirements = read_constant(self.config["frr_requirements"]) # GW of reserve per GW of each variable renewable source
        self.reserve_activation_rate = read_constant(self.config["reserve_activation_rate"])
        self.reserve_activation_time = read_constant(self.config["reserve_activation_time"])
        self.discount_rate = read_constant(self.config["discount_rate"])
        self.lifetime = read_constant(self.config["lifetime"]) # yr
        self.construction_time = read_constant(self.config["construction_time"]) # yr
        self.capex = read_constant(self.config["capex"], year_costs=self.year_costs) # 1e6€/GW = €/kW
        self.storage_capex = read_constant(self.config["storage_capex"], year_costs=self.year_costs) # 1e6€/GWh = €/kWh
        self.phs_charge_to_discharge_ratio = self.config["phs_charge_to_discharge_ratio"]

        #We know that for an energy capacity of 3 TWh, we have a maximum charging capacity of 6.4 GW.
        self.h2_saltcavern_charge_to_storage_ratio = self.config["h2_saltcavern_charge_to_storage_ratio"] # GW/GWh
        #We know that for an energy capacity of 3 TWh, we have a maximum discharging capacity of 26 GW.
        self.h2_saltcavern_discharge_to_storage_ratio = self.config["h2_saltcavern_discharge_to_storage_ratio"] # GW/GWh

        self.fOM = read_constant(self.config["fOM"], year_costs=self.year_costs) # 1e6€/GW/yr = €/kW/yr
        self.vOM = read_constant(self.config["vOM"]) # 1e6€/GW = €/kW
        self.efficiency_storage_in = read_constant(self.config["efficiency_storage_in"])
        self.efficiency_storage_out = read_constant(self.config["efficiency_storage_out"])
        self.conversion_efficiency = read_constant(self.config["conversion_efficiency"])
        self.capacity_factor_nuclear_yearly = self.config["capacity_factor_nuclear_yearly"]
        self.ramp_rate = read_constant(self.config["ramp_rate"])
        self.load_shift_maximum_power = self.config["load_shift_maximum_power"]*self.elec_demand.sum()/self.nb_years # GW
        self.load_shift_period = self.config["load_shift_period"] # hours
        self.load_uncertainty = self.config["load_uncertainty"]
        self.load_variation = self.config["load_variation"]
        self.first_month = self.config["first_month"]
        self.CO2_fraction = read_constant(self.config["CO2_fraction"])
        self.CO2_usable = read_constant(self.config["CO2_usable"])
        self.CO2_industry_prod = self.config["CO2_industry_prod"] # equivalent GWh of CH4
        self.CO2_industry_demand = self.config["CO2_industry_demand"] # equivalent GWh of CH4
        self.prediction_transport_and_distrib_annuity = read_constant(self.config["prediction_transport_and_distrib_annuity"]) # €/W/yr
        self.prediction_transport_offshore_annuity = read_constant(self.config["prediction_transport_offshore"]) # Not used
        self.biomass_potential = read_constant(self.config["biomass_potential"], year_costs=self.year_costs)  # TWh
        self.carbon_budget = self.config["carbon_budget"] # MtCO2
        self.carbon_content = read_constant(self.config["carbon_content"]) # tCO2eq/MWh
        self.carbon_footprint = read_constant(self.config["carbon_footprint"]) # tCO2eq/MWh
        fuel_prices = read_constant(self.config["fuel_prices"]["ini"])  # €/kWh, prices from 2020
        fuel_prices_growth_rate = read_constant(self.config["fuel_prices"]["ini"])
        self.fuel_prices = fuel_prices * (1+fuel_prices_growth_rate)**(self.year_costs-2020)  # 1e6€/GWh = €/kWh
        self.vOM.at["natural_gas"]= self.fuel_prices.at["natural_gas"] # €/kWh
        self.vOM.at['coal'] = self.fuel_prices.at["coal"]  # €/kWh




        # calculate annuities
        self.annuities = calculate_annuities_capex(self.discount_rate, self.capex, self.construction_time, self.lifetime)
                            # 1e6€/GW/yr = €/kW/yr
        self.storage_annuities = calculate_annuities_storage_capex(self.discount_rate, self.storage_capex, self.construction_time, self.lifetime)
                            # 1e6€/GWh/yr = €/kWh/yr

        if not self.carbon_constraint:  # on prend en compte le scc mais pas de contrainte sur le budget
            # Update natural gaz vOM based on social cost of carbon :   €/kW(h) = 1e6€/MW(h)
            self.vOM.at["natural_gas"] = update_vom_costs_scc(self.vOM.at["natural_gas"], scc=self.scc, emission_rate=self.carbon_content.at['natural_gas'])
            self.vOM.at["coal"] = update_vom_costs_scc(self.vOM.at["coal"], scc=self.scc, emission_rate=self.carbon_content.at['coal'])

        # defining needed time steps
        self.first_hour = 0
        self.last_hour = 8760*self.nb_years - 1

        self.hours_by_months = {1: 744, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 720, 10: 744,
                                11: 720, 12: 744}

        self.months_hours = {1: range(0, self.hours_by_months[self.first_month])}
        self.month_hours = define_month_hours(self.first_month, self.nb_years, self.months_hours, self.hours_by_months)


    def define_sets(self):
        # Range of hour
        self.model.h = RangeSet(self.first_hour, self.last_hour)
        # Days
        self.model.days = RangeSet(0, 365*self.nb_years - 1)
        # Months
        self.model.months = RangeSet(1, 12 * self.nb_years)
        # Years
        self.model.years = RangeSet(0, self.nb_years - 1)

        # Tuples representing the range of hours in which the load can be shifted for each hour.
        # For all information concerning load shifting modelisation, see Zerrahn et al. "On the representation of demand-side management in pwoer system models" (2015)
        def init_shift_h(model):
            for h in model.h:
                for hh in shift_range(h, self.load_shift_period, self.last_hour):
                    yield (h, hh)
        self.model.shift_h = Set(initialize=init_shift_h)

        ### Technologies ###

        self.model.all_tech = Set(initialize=["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                                                  "river", "lake", "nuclear", "methanization", "pyrogazification", "ocgt_coge",
                                                  "natural_gas", "coal", "biomass_coge", "geothermal_coge", "waste", "marine", "rsv_dummy",
                                                 "ch4_ocgt", "ch4_ccgt", "h2_ccgt", "electrolysis", "methanation",
                                                 "phs", "battery_1h", "battery_4h", "h2_saltcavern", "ch4_reservoir", "str_dummy"])

        # Primary production
        self.model.prod_tech = Set(initialize=["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                                                  "river", "lake", "nuclear", "methanization", "pyrogazification",
                                                  "natural_gas", "coal", "biomass_coge", "geothermal_coge", "waste", "marine", "rsv_dummy"])

        # Non_operable Technologies
        self.model.vre = Set(initialize=["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                                         "river", "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge"])
        self.model.solar = Set(initialize=["pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW"])

        # Technologies participating in the electricity balance
        self.model.elec_balance = \
            Set(initialize=["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                            "river", "lake", "nuclear", "phs",
                            "battery_1h", "battery_4h", "ch4_ocgt", "ch4_ccgt", "h2_ccgt", "coal",
                           "biomass_coge", "geothermal_coge", "waste", "marine", "ocgt_coge",
                           "str_dummy"])

        # Technologies for upward FRR and FCR
        self.model.reserve = Set(initialize=["lake", "phs", "ch4_ocgt", "ch4_ccgt", "nuclear", "h2_ccgt", "coal", "battery_1h", "battery_4h", "rsv_dummy"])
        self.model.reserve_ramp_limited = Set(initialize=["ch4_ocgt", "ch4_ccgt", "nuclear", "h2_ccgt", "coal"])

        # Technologies producing each of the primary energy (ie excluding conversion and storage technologies)
        self.model.elec_prod = Set(initialize=["offshore_float", "offshore_ground", "onshore", "pv_ground_S", "pv_ground_EW", "pv_roof_com_S", "pv_roof_com_EW", "pv_roof_indiv_S", "pv_roof_indiv_EW",
                                               "river", "lake", "nuclear", "coal", "biomass_coge", "geothermal_coge", "waste", "marine"])
        self.model.CH4_prod = Set(initialize=["methanization", "pyrogazification", "methanation", "natural_gas"])
        self.model.H2_prod = Set(initialize=["electrolysis"])

        # Technologies using each energy vector
        self.model.use_elec = Set(initialize=["phs", "battery_1h", "battery_4h", "electrolysis", "methanation", "str_dummy"])
        self.model.use_CH4 = Set(initialize=["ch4_reservoir", "ch4_ocgt", "ch4_ccgt", "ocgt_coge"])
        self.model.use_H2 = Set(initialize=["h2_saltcavern", "h2_ccgt"])

        # Gas technologies used for balance (both CH4 and H2)
        self.model.CH4_balance = Set(
            initialize=["methanization", "pyrogazification", "natural_gas", "methanation", "ch4_reservoir"])
        self.model.CH4_balance_biogas = Set(initialize=["methanization", "pyrogazification", "methanation"])
        self.model.H2_balance = Set(initialize=["electrolysis", "h2_saltcavern"])

        # Conversion technologies
        self.model.conversion_tech = Set(initialize=["ch4_ocgt", "ch4_ccgt", "h2_ccgt", "electrolysis", "methanation", "ocgt_coge"])
        self.model.from_elec_to_CH4 = Set(initialize=["methanation"])
        self.model.from_elec_to_H2 = Set(initialize=["electrolysis"])
        self.model.from_CH4_to_elec = Set(initialize=["ch4_ocgt", "ch4_ccgt"])
        self.model.from_H2_to_elec = Set(initialize=["h2_ccgt"])

        # Storage technologies
        self.model.str = \
            Set(initialize=["phs", "battery_1h", "battery_4h", "h2_saltcavern", "ch4_reservoir", "str_dummy"])
        # Electricity storage Technologies
        self.model.str_elec = Set(initialize=["phs", "battery_1h", "battery_4h", "str_dummy"])
        # Battery Storage
        self.model.battery = Set(initialize=["battery_1h", "battery_4h"])
        # CH4 storage
        self.model.str_CH4 = Set(initialize=["ch4_reservoir"])
        # H2 storage
        self.model.str_H2 = Set(initialize=["h2_saltcavern"])


    def define_variables(self):

        def capacity_bounds(model, i):
            if i in self.maximum_capacity.keys():  # there exists a max capacity
                return self.existing_capacity.at[i], self.maximum_capacity.at[i]  # existing capacity is always the lower bound
            else:
                return self.existing_capacity.at[i], None  # in this case, only lower bound exists

        def charging_capacity_bounds(model, i):
            # TODO: j'ai enlevé cette contrainte, car je suppose ici que la seule contrainte provient de la discharging capacity
            # if i in self.maximum_charging_capacity.keys():
            #     return self.existing_charging_capacity[i], self.maximum_capacity.at[i]
            # else:
            return self.existing_charging_capacity.at[i], None

        def energy_capacity_bounds(model, i):
            if i in self.maximum_energy_capacity.keys():
                return self.existing_energy_capacity.at[i], self.maximum_energy_capacity.at[i]
            else:
                return self.existing_energy_capacity.at[i], None

        # Hourly energy output in GW
        self.model.gene = \
            Var(((tech, h) for tech in self.model.prod_tech for h in self.model.h), within=NonNegativeReals, initialize=0)
        self.model.str_output = \
            Var(((tech, h) for tech in self.model.str for h in self.model.h), within=NonNegativeReals, initialize=0)
        self.model.conv_output = \
            Var(((tech, h) for tech in self.model.conversion_tech for h in self.model.h), within=NonNegativeReals, initialize=0)

        # Installed capacity in GW
        self.model.nominal_power = \
            Var(self.model.prod_tech, within=NonNegativeReals, bounds=capacity_bounds)
        self.model.output_power = \
            Var(self.model.conversion_tech, within=NonNegativeReals, bounds=capacity_bounds)
        self.model.discharging_power = \
            Var(self.model.str, within=NonNegativeReals, bounds=capacity_bounds)

        # Charging power capacity of each storage technology in GW
        self.model.charging_power = \
            Var(self.model.str, within=NonNegativeReals, bounds=charging_capacity_bounds)

        # Energy volume of storage technology in GWh
        self.model.energy_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=energy_capacity_bounds)

        # Hourly electricity input of storage technologies GW
        self.model.str_input = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Energy stored in each storage technology in GWh = Stage of charge
        self.model.state_of_charge = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)
        self.model.lake_stored = \
            Var((h for h in self.model.h), within=NonNegativeReals, initialize=0)

        ## Required upward frequency restoration reserve in GW
        self.model.fcr = Var(((tech, h) for tech in self.model.reserve for h in self.model.h), within=NonNegativeReals, initialize=0)

        # Required upward frequency restoration reserve in GW
        self.model.frr = Var(((tech, h) for tech in self.model.reserve for h in self.model.h), within=NonNegativeReals, initialize=0)


        # For all information concerning load shifting modelisation, see Zerrahn et al. "On the representation of demand-side management in pwoer system models" (2015)

        # Upwards load shift at hour h in GW
        self.model.dsm_up = Var(self.model.h, within=NonNegativeReals, initialize=0)

        # Downwards load shift at hour hh (second tuple term) to compensate for an upward shift at hour h (first tuple term)
        self.model.dsm_down = Var(self.model.shift_h, within=NonNegativeReals, initialize=0)







    def fix_values(self):
        for tech in self.model.prod_tech :
            if tech in self.fixed_installed_power.keys():
                self.model.nominal_power[tech].fix(self.fixed_installed_power[tech])
        for tech in self.model.conversion_tech :
            if tech in self.fixed_installed_power.keys():
                self.model.output_power[tech].fix(self.fixed_installed_power[tech])
        for tech in self.model.str :
            if tech in self.fixed_installed_power.keys():
                self.model.discharging_power[tech].fix(self.fixed_installed_power[tech])
            if tech in self.fixed_charging_power.keys():
                self.model.charging_power[tech].fix(self.fixed_charging_power[tech])
            if tech in self.fixed_energy_capacity.keys():
                self.model.energy_capacity[tech].fix(self.fixed_energy_capacity.at[tech])


    def define_constraints(self):
        """Some scaling factors are sometimes added to both sides of constraints : they should not be removed unless by someone who knows what they are doing.
        While the equation would be equivalent, these factors help reduce the range of matrix coefficients and thus improve solving time
        We aim at a [1e-3, 1e3] range for matrix coefficients, and [1e-2, 1e2] range for RHS coefficients
        Remark : No particular thought process was used to determine those objective ranges, other than the fact that further reducing the matrix range would
        require us to further truncate the smaller load factors (the limit is currently 1e6, below which all load factors are put at zero) which was deemed
        undesireable, and I do not master the barrier algorithm (= interior-point method), so feel free to adjust scaling factors if you think it could improve
        solving time"""

        ###############
        ### GENERAL ###
        ###############

        def installed_power_constraint_rule(model, h, tech):
            """Constraint on maximum power for non-VRE technologies."""
            if tech in model.prod_tech :
                return model.nominal_power[tech] >= model.gene[tech, h]   # GW
            if tech in model.conversion_tech :
                return model.output_power[tech] >= model.conv_output[tech, h]   # GW
            if tech in model.str :
                return model.discharging_power[tech] >= model.str_output[tech, h]   # GW

        self.model.installed_power_constraint = Constraint(self.model.h, self.model.all_tech, rule=installed_power_constraint_rule)


        def carbon_budget_constraint_rule(model, y):
            """Constraint on carbon budget in MtCO2."""
            # Carbon content is in tCO2/MWh
            # Power generated is in GW(h)
            # The carbon budget is in MtCO2 (per year)
            return sum(model.gene["natural_gas", h] for h in range(8760 * y, 8760 * (y + 1))) * self.carbon_content.at['natural_gas'] / 10 \
                    + sum(model.gene["coal", h]  for h in range(8760*y,8760*(y+1))) * self.carbon_content.at['coal'] / 10 \
                    + sum(model.frr["coal", h] for h in range(8760 * y, 8760 * (y + 1))) * self.reserve_activation_rate.at["frr"] * self.carbon_content.at['coal'] / 10 \
                    + sum(model.fcr["coal", h] for h in range(8760 * y, 8760 * (y + 1))) * self.reserve_activation_rate.at["fcr"] * self.carbon_content.at['coal'] / 10 \
                    <= self.carbon_budget*100   # ktCO2

        if self.carbon_constraint:  # on ajoute la contrainte carbone
            self.model.carbon_budget_constraint = Constraint(self.model.years, rule=carbon_budget_constraint_rule)

        ###################
        ### ELECTRICITY ###
        ###################

        def reserve_power_constraint_rule(model, h, tech):
            """Constraint on maximum generation including reserves"""
            if tech in model.prod_tech :
                return model.nominal_power[tech] >= model.gene[tech, h] + model.fcr[tech, h] + model.frr[tech, h]   # GW
            if tech in model.conversion_tech :
                return model.output_power[tech] >= model.conv_output[tech, h] + model.fcr[tech, h] + model.frr[tech, h]   # GW
            if tech in model.str :
                return model.discharging_power[tech] >= model.str_output[tech, h] + model.fcr[tech, h] + model.frr[tech, h]   # GW

        self.model.reserve_power_constraint = Constraint(self.model.h, self.model.reserve, rule=reserve_power_constraint_rule)


        def fcr_provision_constraint_rule(model, h):
            """Constraint on fcr total volume provided"""
            return sum(model.fcr[tech, h] for tech in model.reserve) == self.fcr_requirement*int(self.include_reserve)   # GW

        self.model.fcr_provision_constraint = Constraint(self.model.h, rule=fcr_provision_constraint_rule)


        def fcr_ramp_constraint_rule(model, tech, h):
            """Constraint on some reserve-providing technologies that have a limited ramp-rate"""
            if tech in model.prod_tech:
                return model.fcr[tech, h]*10 <= model.nominal_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["fcr"]*10
            if tech in model.conversion_tech:
                return model.fcr[tech, h]*10 <= model.output_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["fcr"]*10
            if tech in model.str:
                return model.fcr[tech, h]*10 <= model.discharging_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["fcr"]*10

        self.model.fcr_ramp_constraint = Constraint(self.model.reserve_ramp_limited, self.model.h, rule=fcr_ramp_constraint_rule)


        def frr_provision_constraint_rule(model, h):
            """Constraint on frr total volume provided"""
            res_req = sum(self.frr_requirements.at[vre] * model.nominal_power[vre] for vre in model.vre)   # GW
            load_req = self.elec_demand.iat[h] * self.load_uncertainty * (1 + self.load_variation)   #GW
            return sum(model.frr[tech, h] for tech in model.reserve)*10 == (res_req + load_req)*10*int(self.include_reserve)   # GW

        self.model.frr_provision_constraint = Constraint(self.model.h, rule=frr_provision_constraint_rule)


        def frr_ramp_constraint_rule(model, tech, h):
            """Constraint on some reserve-providing technologies that have a limited ramp-rate"""
            if tech in model.prod_tech :
                return model.frr[tech, h]*10 <= model.nominal_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["frr"]*10
            if tech in model.conversion_tech :
                return model.frr[tech, h]*10 <= model.output_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["frr"]*10
            if tech in model.str :
                return model.frr[tech, h]*10 <= model.discharging_power[tech]*self.ramp_rate.at[tech]*self.reserve_activation_time.at["frr"]*10

        self.model.frr_ramp_constraint = Constraint(self.model.reserve_ramp_limited, self.model.h, rule=frr_ramp_constraint_rule)


        def electricity_adequacy_constraint_rule(model, h):
            """Constraint for supply/demand electricity relation'"""
            storage = sum(model.str_input[str, h] for str in model.str_elec)  # GW
            gene_from_elec = model.conv_output['electrolysis', h] / self.conversion_efficiency.at['electrolysis'] \
                        + model.conv_output['methanation', h] / self.conversion_efficiency.at['methanation']  # GW
            prod_elec = sum(model.gene[tech, h] for tech in (model.prod_tech & model.elec_balance))\
                            + sum(model.conv_output[tech, h] for tech in (model.conversion_tech & model.elec_balance))\
                            + sum(model.str_output[tech, h] for tech in (model.str & model.elec_balance))      # GW
            net_load_shift_up = model.dsm_up[h] - sum(model.dsm_down[hh, h] for hh in shift_range(h, self.load_shift_period, self.last_hour))
            return prod_elec >= (self.elec_demand.iat[h] + storage + gene_from_elec + net_load_shift_up)

        self.model.electricity_adequacy_constraint = Constraint(self.model.h, rule=electricity_adequacy_constraint_rule)


        #####################
        ### LOAD SHIFTING ###
        #####################

        def shifting_adequacy_constraint_rule(model, h):
            """Constraint for adequacy between upwards and downwards load shift
            For all information concerning load shifting modelisation, see Zerrahn et al.
            "On the representation of demand-side management in power system models" (2015)"""
            return model.dsm_up[h] == sum(model.dsm_down[h, hh] for hh in shift_range(h, self.load_shift_period, self.last_hour))

        self.model.shifting_adequacy_constraint = Constraint(self.model.h, rule=shifting_adequacy_constraint_rule)


        def shifting_upwards_limit_constraint_rule(model, h):
            """Constraint for limiting upwards load shift
            For all information concerning load shifting modelisation, see Zerrahn et al.
            "On the representation of demand-side management in power system models" (2015)"""
            return model.dsm_up[h]*1000 <= self.load_shift_maximum_power*1000

        self.model.shifting_upwards_limit_constraint = Constraint(self.model.h, rule=shifting_upwards_limit_constraint_rule)


        def shifting_downwards_limit_constraint_rule(model, h):
            """Constraint for limiting downwards load shift
            For all information concerning load shifting modelisation, see Zerrahn et al.
            "On the representation of demand-side management in power system models" (2015)"""
            return sum(model.dsm_down[hh, h] for hh in shift_range(h, self.load_shift_period, self.last_hour))*1000 <= self.load_shift_maximum_power*1000

        self.model.shifting_downwards_limit_constraint = Constraint(self.model.h, rule=shifting_downwards_limit_constraint_rule)


        def shifting_simultaneous_limit_constraint_rule(model, h):
            """Constraint for limiting simultaneous upwards and downwards load shift
            For all information concerning load shifting modelisation, see Zerrahn et al.
            "On the representation of demand-side management in power system models" (2015)"""
            return model.dsm_up[h]*1000 + sum(model.dsm_down[hh, h] for hh in shift_range(h, self.load_shift_period, self.last_hour))*1000 <= self.load_shift_maximum_power*1000

        self.model.shifting_simultaneous_limit_constraint = Constraint(self.model.h, rule=shifting_simultaneous_limit_constraint_rule)


        ###########
        ### VRE ###
        ###########

        def generation_vre_constraint_rule(model, h, tech):
            """Constraint on variables renewable profiles generation."""
            return model.gene[tech, h]*1000 == model.nominal_power[tech] * self.vre_profiles.iloc[h, self.vre_profiles.columns.get_loc(tech)]*1000   # GW

        self.model.generation_vre_constraint = Constraint(self.model.h, self.model.vre, rule=generation_vre_constraint_rule)


        def pv_ground_potential_constraint_rule(model):
            """Constraint on solar maximum capacity. Not treated the same as other potentials because the model distributes the potential
            between south-facing and east-west orientations."""
            return model.nominal_power["pv_ground_S"] + model.nominal_power["pv_ground_EW"] <= self.maximum_capacity.at["pv_ground"]

        if "pv_ground" in self.maximum_capacity.index :
            self.model.pv_ground_potential_constraint = Constraint(rule=pv_ground_potential_constraint_rule)


        def pv_roof_com_potential_constraint_rule(model):
            """Constraint on solar maximum capacity. Not treated the same as other potentials because the model distributes the potential
            between south-facing and east-west orientations."""
            return model.nominal_power["pv_roof_com_S"] + model.nominal_power["pv_roof_com_EW"] <= self.maximum_capacity.at["pv_roof_com"]

        if "pv_roof_com" in self.maximum_capacity.index :
            self.model.pv_roof_com_potential_constraint = Constraint(rule=pv_roof_com_potential_constraint_rule)


        def pv_roof_indiv_potential_constraint_rule(model):
            """Constraint on solar maximum capacity. Not treated the same as other potentials because the model distributes the potential
            between south-facing and east-west orientations."""
            return model.nominal_power["pv_roof_indiv_S"] + model.nominal_power["pv_roof_indiv_EW"] <= self.maximum_capacity.at["pv_roof_indiv"]

        if "pv_roof_indiv" in self.maximum_capacity.index :
            self.model.pv_roof_indiv_potential_constraint = Constraint(rule=pv_roof_indiv_potential_constraint_rule)


        ###############
        ### STORAGE ###
        ###############

        def storage_charging_discharging_constraint_rule(model, storage_tec):
            """Constraint to limit charging capacity to be lower than discharging capacity.
            We use this to limit charging capacity, which has no cost associated to it becasue of a lack of data."""
            return model.charging_power[storage_tec] <= model.discharging_power[storage_tec]

        self.model.storage_charging_discharging_constraint = Constraint(self.model.str, rule=storage_charging_discharging_constraint_rule)


        def storing_constraint_rule(model, h, storage_tecs):
            """Constraint on energy storage consistency."""
            hPOne = h + 1 if h < (self.last_hour) else 0
            charge = model.str_input[storage_tecs, h] * self.efficiency_storage_in.at[storage_tecs]
            if storage_tecs == "phs" :
                charge += self.phs_inflows.iat[h]
            discharge = model.str_output[storage_tecs, h] / self.efficiency_storage_out.at[storage_tecs]
            if storage_tecs in model.reserve:
                discharge += model.frr[storage_tecs, h]*self.reserve_activation_rate.at["frr"] / self.efficiency_storage_out.at[storage_tecs]
                discharge += model.fcr[storage_tecs, h]*self.reserve_activation_rate.at["fcr"] / self.efficiency_storage_out.at[storage_tecs]
            flux = charge - discharge
            return model.state_of_charge[storage_tecs, hPOne]*100 == model.state_of_charge[storage_tecs, h]*100 + flux*100

        self.model.storing_constraint = Constraint(self.model.h, self.model.str, rule=storing_constraint_rule)


        def energy_capacity_constraint(model, h, storage_tecs):
            """Constraint on maximum energy that is stored in storage units"""
            # The scaling factor does not seem necessary here, but in fact some energy capacities are fixed at very large (1e5) values and are then considered RHS values
            return model.state_of_charge[storage_tecs, h]/1000 <= model.energy_capacity[storage_tecs]/1000   # GWh

        self.model.energy_capacity_constraint = Constraint(self.model.h, self.model.str, rule=energy_capacity_constraint)


        def storage_charging_power_constraint_rule(model, h, storage_tecs):
            """Constraint on the capacity with hourly charging relationship of storage. Energy entering the battery
            during one hour cannot exceed the charging capacity."""
            return model.str_input[storage_tecs, h] <= model.charging_power[storage_tecs]

        self.model.storage_charging_power_constraint = Constraint(self.model.h, self.model.str, rule=storage_charging_power_constraint_rule)


        #################
        ### BATTERIES ###
        #################

        def battery_1h_capacity_constraint_rule(model):
            """Constraint on capacity of battery 1h."""
            return model.discharging_power['battery_1h'] == model.energy_capacity['battery_1h']   # Definition of a battery that lasts 1h

        self.model.battery_1_capacity_constraint = Constraint(rule=battery_1h_capacity_constraint_rule)


        def battery_4h_capacity_constraint_rule(model):
            """Constraint on capacity of battery 4h."""
            return model.discharging_power['battery_4h'] == model.energy_capacity['battery_4h'] / 4   # Definition of a battery that lasts 4h

        self.model.battery_4_capacity_constraint = Constraint(rule=battery_4h_capacity_constraint_rule)


        def battery_capacity_constraint_rule(model, battery):
            """Constraint on battery's capacity: battery charging capacity equals battery discharging capacity."""
            return model.charging_power[battery] == model.discharging_power[battery]

        self.model.battery_capacity_constraint = Constraint(self.model.battery, rule=battery_capacity_constraint_rule)


        def battery_simultaneous_limit_constraint_rule(model, h, battery):
            """Constraint for limiting simultaneous upwards and downwards load shift.
            Without this constraint, batteries are able to charge and discharge at maximum power at the same time"""
            return model.str_input[battery, h] + model.str_output[battery, h] <= model.discharging_power[battery]

        self.model.battery_simultaneous_limit_constraint = Constraint(self.model.h, self.model.battery, rule=battery_simultaneous_limit_constraint_rule)


        ###############
        ### METHANE ###
        ###############

        def methanization_constraint_rule(model, y):
            """Constraint on methanization. The annual power production from methanization is limited to a certain amount."""
            gene_biogas = sum(model.gene['methanization', hour] for hour in range(8760*y,8760*(y+1)))
            return gene_biogas/1000 <= self.biomass_potential.at["methanization"]  # max biogas yearly energy is expressed in TWh

        self.model.biogas_constraint = Constraint(self.model.years, rule=methanization_constraint_rule)


        def pyrogazification_constraint_rule(model, y):
            """Constraint on pyrogazification. The annual power production from pyro is limited to a certain amount."""
            gene_pyro = sum(model.gene['pyrogazification', hour] for hour in range(8760*y,8760*(y+1)))   #GWh
            return gene_pyro/1000 <= self.biomass_potential.at["pyrogazification"]  # max pyro yearly energy is expressed in TWh

        self.model.pyrogazification_constraint = Constraint(self.model.years, rule=pyrogazification_constraint_rule)


        def methanation_CO2_constraint_rule(model, y):
            """Constraint on CO2 balance from methanization, summing over all hours of the year"""
            return sum(model.conv_output['methanation', h] for h in range(8760*y,8760*(y+1))) / self.conversion_efficiency.at['methanation'] / 100 <= (
                sum(model.gene['methanization', h] for h in range(8760*y,8760*(y+1))) * self.CO2_fraction.at["methanization"] / (1 - self.CO2_fraction.at["methanization"]) * self.CO2_usable.at["methanization"]
                + sum(model.gene['pyrogazification', h] for h in range(8760*y,8760*(y+1))) * self.CO2_fraction.at["pyrogazification"] / (1 - self.CO2_fraction.at["pyrogazification"]) * self.CO2_usable.at["pyrogazification"]
                + self.CO2_industry_prod*self.CO2_usable.at["industry"] - self.CO2_industry_demand)/100
        # if x is the molar fraction of CO2 then x/(1-x) is the number of moles of CO2 per mole of CH4 (considering the output gas is a mix of only those two gases)
        # the fact that with methanation, each mole of CO2 gets converted into a mole of CH4 allows us to work out the CO2 balance in terms of equivalent GWh of CH4

        self.model.methanation_constraint = Constraint(self.model.years, rule=methanation_CO2_constraint_rule)


        def methane_balance_constraint_rule(model, h):
            """Constraint on methane's balance. Methane production must satisfy ch4_ccgt, ch4_ocgt and ocgt_coge plants' CH4 demand."""

            supply_h = sum(model.gene[tech, h] for tech in (model.prod_tech & model.CH4_balance))\
                            + sum(model.conv_output[tech, h] for tech in (model.conversion_tech & model.CH4_balance))\
                            + sum(model.str_output[tech, h] for tech in (model.str & model.CH4_balance))      # GW

            usage_h = 0
            for tech in model.use_CH4:
                if tech in model.str:
                    usage_h += model.str_input[tech, h]   # GW
                if tech in model.conversion_tech:
                    usage_h += model.conv_output[tech, h]/self.conversion_efficiency.at[tech]   # GW
                if tech in model.reserve:
                    usage_h += model.frr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["frr"]
                    usage_h += model.fcr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["fcr"]
            if self.CH4_demand_is_profile:
                usage_h += self.CH4_demand.iat[h] # GWh

            return supply_h >= usage_h   # GW

        self.model.methane_balance_constraint = Constraint(self.model.h, rule=methane_balance_constraint_rule)


        def methane_annual_demand_constraint_rule(model, y):
            """Constraint on satisfying annual demand for methane"""

            supply_tot = sum(sum(model.gene[tech, h] for tech in (model.prod_tech & model.CH4_balance))\
                            + sum(model.conv_output[tech, h] for tech in (model.conversion_tech & model.CH4_balance))\
                            + sum(model.str_output[tech, h] for tech in (model.str & model.CH4_balance))
                        for h in range(8760*y,8760*(y+1)))

            usage_tot = 0
            for tech in model.use_CH4:
                if tech in model.str:
                    usage_tot += sum(model.str_input[tech, h] for h in range(8760*y,8760*(y+1)))   # GW
                if tech in model.conversion_tech:
                    usage_tot += sum(model.conv_output[tech, h]/self.conversion_efficiency.at[tech] for h in range(8760*y,8760*(y+1)))   # GW
                if tech in model.reserve:
                    usage_tot += sum(model.frr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["frr"] for h in range(8760*y,8760*(y+1)))
                    usage_tot += sum(model.fcr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["fcr"] for h in range(8760*y,8760*(y+1)))
            usage_tot += self.CH4_demand

            return supply_tot/100 >= usage_tot/100

        if not self.CH4_demand_is_profile:
            self.model.methane_annual_demand_constraint = Constraint(self.model.years, rule=methane_annual_demand_constraint_rule)


        ################
        ### Hydrogen ###
        ################

        def h2_saltcavern_discharge_constraint_rule(model):
            """Constraint on discharge capacity of h2_saltcavern. This is a bit ad hoc, based on discussions with Marie-Alix,
            and some extrapolations for the future capacity of h2_saltcavern."""
            return model.discharging_power["h2_saltcavern"]*10 <= model.energy_capacity['h2_saltcavern'] * self.h2_saltcavern_discharge_to_storage_ratio*10

        self.model.h2_saltcavern_discharge_constraint = Constraint(rule=h2_saltcavern_discharge_constraint_rule)


        def h2_saltcavern_charge_constraint_rule(model):
            """Constraint on charging capacity of h2_saltcavern. This is a bit ad hoc, based on discussions with Marie-Alix,
            and some extrapolations for the future capacity of h2_saltcavern."""
            return model.charging_power["h2_saltcavern"]*10 <= model.energy_capacity['h2_saltcavern'] * self.h2_saltcavern_charge_to_storage_ratio*10

        self.model.h2_saltcavern_charge_constraint = Constraint(rule=h2_saltcavern_charge_constraint_rule)


        def hydrogen_balance_constraint_rule(model, h):
            """Constraint on hydrogen balance. hydrogen production must satisfy ch4_ccgt-H2 plants and H2 demand."""

            supply_h = sum(model.gene[tech, h] for tech in (model.prod_tech & model.H2_balance))\
                            + sum(model.conv_output[tech, h] for tech in (model.conversion_tech & model.H2_balance))\
                            + sum(model.str_output[tech, h] for tech in (model.str & model.H2_balance))   # GW

            usage_h = 0
            for tech in model.use_H2:
                if tech in model.str:
                    usage_h += model.str_input[tech, h]   # GW
                if tech in model.conversion_tech:
                    usage_h += model.conv_output[tech, h]/self.conversion_efficiency.at[tech]   # GW
                if tech in model.reserve:
                    usage_h += model.frr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["frr"]
                    usage_h += model.fcr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["fcr"]
            if self.H2_demand_is_profile:
                usage_h += self.H2_demand.iat[h]

            return supply_h >= usage_h   # GW

        self.model.hydrogen_balance_constraint = Constraint(self.model.h, rule=hydrogen_balance_constraint_rule)


        def hydrogen_annual_demand_constraint_rule(model, y):
            """Constraint on satisfying annual demand for H2"""

            supply_tot = sum(sum(model.gene[tech, h] for tech in (model.prod_tech & model.H2_balance))\
                            + sum(model.conv_output[tech, h] for tech in (model.conversion_tech & model.H2_balance))\
                            + sum(model.str_output[tech, h] for tech in (model.str & model.H2_balance))
                        for h in range(8760*y,8760*(y+1)))

            usage_tot = 0
            for tech in model.use_H2:
                if tech in model.str:
                    usage_tot += sum(model.str_input[tech, h] for h in range(8760*y,8760*(y+1)))   # GW
                if tech in model.conversion_tech:
                    usage_tot += sum(model.conv_output[tech, h]/self.conversion_efficiency.at[tech] for h in range(8760*y,8760*(y+1)))   # GW
                if tech in model.reserve:
                    usage_tot += sum(model.frr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["frr"] for h in range(8760*y,8760*(y+1)))
                    usage_tot += sum(model.fcr[tech, h]/self.conversion_efficiency.at[tech]*self.reserve_activation_rate.at["fcr"] for h in range(8760*y,8760*(y+1)))
            usage_tot += self.H2_demand

            return supply_tot/100 >= usage_tot/100

        if not self.H2_demand_is_profile:
            self.model.hydrogen_annual_demand_constraint = Constraint(self.model.years, rule=hydrogen_annual_demand_constraint_rule)


        ###############
        ### NUCLEAR ###
        ###############

        def generation_nuclear_constraint_rule(model, y):
            """Constraint on total nuclear production which cannot be superior to nuclear capacity times a given
            capacity factor inferior to 1."""
            return sum(model.gene["nuclear", h] for h in range(8760*y,8760*(y+1))) / 10 <= self.capacity_factor_nuclear_yearly * model.nominal_power["nuclear"] * 8760 / 10   # GW

        self.model.generation_nuclear_constraint = Constraint(self.model.years, rule=generation_nuclear_constraint_rule)


        def generation_nuclear_constraint_hourly_rule(model, h):
            """Constraint on nuclear production which cannot be superior to nuclear capacity times a given capacity factor.
            This holds for all hours."""
            return model.nominal_power['nuclear'] * self.capacity_factor_nuclear_hourly.iat[h] >= model.gene['nuclear', h]   # GW

        self.model.generation_nuclear_hourly_constraint = Constraint(self.model.h, rule=generation_nuclear_constraint_hourly_rule)


        def ramping_nuclear_up_constraint_rule(model, h):
            """Constraint setting an upper ramping limit for nuclear flexibility
            Reserves at hour h-1 are taken into account because they require the reactor to be heated even if not activated,
            and reactor heating is the reason for the limiting ramp rate"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', h] - model.gene['nuclear', previous_h] \
                    + model.frr['nuclear', h] - model.frr['nuclear', previous_h] \
                    + model.fcr['nuclear', h] - model.fcr['nuclear', previous_h] \
                    <= self.ramp_rate.at["nuclear"] * model.nominal_power['nuclear']

        self.model.ramping_nuclear_up_constraint = Constraint(self.model.h, rule=ramping_nuclear_up_constraint_rule)


        def ramping_nuclear_down_constraint_rule(model, h):
            """Constraint setting a lower ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', previous_h] - model.gene['nuclear', h] \
                    + model.frr['nuclear', previous_h] - model.frr['nuclear', h] \
                    + model.fcr['nuclear', previous_h] - model.fcr['nuclear', h] \
                    <= self.ramp_rate.at["nuclear"] * model.nominal_power['nuclear']

        self.model.ramping_nuclear_down_constraint = Constraint(self.model.h, rule=ramping_nuclear_down_constraint_rule)


        #############
        ### HYDRO ###
        #############

        def phs_charging_constraint_rule(model):
            """We model a constraint on the charging capacity of PHS. Since we only have a CAPEX associated with the discharging
            capacity, there is no limit to the charging capacity. The only constraint is that the charging capacity
            should be lower than the discharging capacity. We impose here something slightly more constraining for PHS. The value
            is based on data from annex in RTE (p.898), where we calculate the ratio between the projected charging and
            discharging capacity."""
            return model.charging_power['phs'] <= model.discharging_power['phs'] * self.phs_charge_to_discharge_ratio

        self.model.phs_charging_constraint = Constraint(rule=phs_charging_constraint_rule)


        def lake_storage_constraint_rule(model, h):
            """Constraint on maximum energy that is stored in lakes"""
            return model.lake_stored[h]/100 <= self.fixed_energy_capacity.at["lake"]/100   # GWh

        self.model.lake_storage_constraint = Constraint(self.model.h, rule=lake_storage_constraint_rule)


        def lake_spillage_constraint_rule(model, h):
            """Constraint on spilled inflows. Because the model aggregates lakes and inflows, a lot of precision is lost.
            To compensate for that, a coupling between Eoles and ORCHIDEE provided a separation of inflows between
            the net inflows (just called inflows in the model) and the spilled water (ie inflow that cannot be stored).
            The energy generated is thus the sum of spilled energy and energy taken from the reservoir."""
            return model.gene['lake', h] >= self.lake_spill.iat[h]

        self.model.lake_spillage_constraint = Constraint(self.model.h, rule=lake_spillage_constraint_rule)


        def lake_storing_constraint_rule(model, h):
            """Constraint on lake storage consistency."""
            hPOne = h + 1 if h < (self.last_hour) else 0
            # Spill is a mandatory production that does not go through storage but that is counted in gene, so it needs to be substracted from it here
            outflow = (model.gene['lake', h] - self.lake_spill.iat[h])/self.efficiency_storage_out.at["lake"] \
                        + model.frr['lake', h]*self.reserve_activation_rate.at["frr"]/self.efficiency_storage_out.at["lake"] \
                        + model.fcr['lake', h]*self.reserve_activation_rate.at["fcr"]/self.efficiency_storage_out.at["lake"]
            return model.lake_stored[hPOne] == model.lake_stored[h] + self.lake_inflows.iat[h] - outflow

        self.model.lake_storing_constraint = Constraint(self.model.h, rule=lake_storing_constraint_rule)


        def lake_minimal_stored_constraint_rule(model, h):
            """Contraint on minimal energy that must be stored in lakes for recreative purposes during summer"""
            return model.lake_stored[h]/100 >= self.lake_minimal_volume.iat[h]/100

        self.model.lake_minimal_stored_constraint = Constraint(self.model.h, rule=lake_minimal_stored_constraint_rule)



        def lake_minimal_outflow_constraint_rule(model, day):
            """Constraint on minimal outflow necessary for ecological reasons and for irrigation. The constraint is a daily one
            because dams have a retention basin after the turbines allowing for the sub-daily control of the flow rate without
            impacting electricity generation, and irrigation only constrains daily to weekly outflows."""
            return sum(model.gene['lake', hour] for hour in range(24*day, 24*(day+1))) >= self.lake_minimal_outflow.iat[day]

        self.model.lake_minimal_outflow_constraint = Constraint(self.model.days, rule=lake_minimal_outflow_constraint_rule)





    def define_objective(self):
        def objective_rule(model):
            """Objective value in 1e9€"""
            return (
                # power annuities
                (sum((model.nominal_power[tech] - self.existing_capacity.at[tech]) * self.annuities.at[tech] for tech in model.prod_tech)
                    + sum((model.output_power[tech] - self.existing_capacity.at[tech]) * self.annuities.at[tech] for tech in model.conversion_tech)
                    + sum((model.discharging_power[tech] - self.existing_capacity.at[tech]) * self.annuities.at[tech] for tech in model.str)
                )*self.nb_years   # 1e6€

                # storage capacity annuities
                + sum((model.energy_capacity[storage_tech] - self.existing_energy_capacity.at[storage_tech]) *
                        self.storage_annuities.at[storage_tech] for storage_tech in model.str)*self.nb_years   # 1e6€

                # fixed O&M
                + (sum(model.nominal_power[tech] * self.fOM.at[tech] for tech in model.prod_tech)
                    + sum(model.output_power[tech] * self.fOM.at[tech] for tech in model.conversion_tech)
                    + sum(model.discharging_power[tech] * self.fOM.at[tech] for tech in model.str)
                )*self.nb_years   # 1e6€


                # variable O&M
                + sum(sum(model.gene[tech, h] * self.vOM.at[tech] for tech in model.prod_tech)
                      + sum(model.conv_output[tech, h] * self.vOM.at[tech] for tech in model.conversion_tech)
                      + sum(model.str_output[tech, h] * self.vOM.at[tech] for tech in model.str)
                    for h in model.h)   # 1e6€

                # Reserve activation variable O&M
                + sum(sum( (model.frr[tech, h]*self.reserve_activation_rate.at["frr"] + model.fcr[tech, h]*self.reserve_activation_rate.at["fcr"])
                                  *self.vOM.at[tech] for tech in model.reserve)
                       for h in model.h)   # 1e6€

                    ) / 1000

        # Creation of the objective -> Cost
        self.model.objective = Objective(rule=objective_rule)




    def build_model(self):
        self.load_inputs()
        self.define_sets()
        self.define_variables()
        self.fix_values()
        self.define_constraints()
        self.define_objective()

    def solve(self, solver_name="gurobi", infeasible_value=1000):
        """Attention au choix de la infeasible_value: c'est la valeur que l'on donne lorsque le problème n'est pas solvable."""
        self.opt = SolverFactory(solver_name)
        print(f"Solving EOLES model using {self.opt.name}")

        print("Adjusted Gurobi settings to only use the barrier (=internal point) algorithm. This greatly accelerates the solving but means that the solution may not be basic")
        self.opt.options['Method'] = 2  # Méthode de barrière seule
        self.opt.options['Crossover'] = 0  # Désactiver le crossover
        self.opt.options['NumericFocus'] = 3
        self.opt.options['BarConvTol'] = 1e-8
        self.opt.options['Presolve'] = 2

        print("Logging in " + self.output_path + "/logfile_" + self.name + ".txt")
        self.solver_results = self.opt.solve(self.model,
                                             options={'Presolve': 2, 'LogFile': self.output_path + "/logfile_" + self.name + ".txt"})

        status = self.solver_results["Solver"][0]["Status"]
        termination_condition = self.solver_results["Solver"][0]["Termination condition"]

        if status == "ok" and termination_condition == "optimal":
            print("Optimization successful")
        elif status == "warning" and termination_condition == "other":
            print("WARNING! Optimization might be sub-optimal. Writing output anyway")
        else:
            print(f"Optimisation failed with status {status} and terminal condition {termination_condition}")
            self.system_social_cost = infeasible_value
        return self.solver_results, status, termination_condition

    def extract_optimisation_results(self):
        """

        :param m: ModelEOLES
        :return:
        """
        # get value of objective function
        self.system_social_cost = self.solver_results["Problem"][0]["Upper bound"]   # 1e9€/yr

        if self.carbon_constraint :
            self.system_technical_cost, self.emissions = get_technical_cost(self.model, self.system_social_cost, scc=0, nb_years=self.nb_years, carbon_content=self.carbon_content)   # 1e9€/yr , MtCO2/yr
        else:
            self.system_technical_cost, self.emissions = get_technical_cost(self.model, self.system_social_cost, self.scc,
                                                                 self.nb_years, self.carbon_content)   # 1e9€/yr , MtCO2/yr

        self.hourly_balance = extract_hourly_balance(self.model, self.elec_demand, self.H2_demand, self.CH4_demand, self.conversion_efficiency, self.efficiency_storage_in, self.efficiency_storage_out,
                                                          self.load_shift_period, self.last_hour)   # GW
        _ = extract_curtailment(self.model, self.conversion_efficiency, self.hourly_balance)
        self.frr = pd.Series({tech : self.hourly_balance.loc[:, tech+"_frr"].sum()/1000 for tech in self.model.reserve})   # TWh
        self.fcr = pd.Series({tech : self.hourly_balance.loc[:, tech+"_fcr"].sum()/1000 for tech in self.model.reserve})   # TWh

        self.CH4_carbon_content = get_carbon_content(self.model, self.hourly_balance, self.carbon_content, self.conversion_efficiency)   # gCO2/kWh = tCO2/GWh (also updates hourly_balance to include electricity carbon content)
        self.spot_price = extract_spot_price(self.model, nb_hours=self.last_hour + 1)   # €/MW(h)
        self.carbon_value = extract_carbon_value(self.model, self.carbon_constraint, self.scc)   # €/tCO2
        self.installed_power = pd.Series([value(self.model.nominal_power[tech]) for tech in self.model.prod_tech]
                                  + [value(self.model.output_power[tech]) for tech in self.model.conversion_tech]
                                  + [value(self.model.discharging_power[tech]) for tech in self.model.str],
                            index=self.model.prod_tech | self.model.conversion_tech | self.model.str, dtype=float)   # GW
        self.energy_capacity = pd.Series([value(self.model.energy_capacity[tech]) for tech in self.model.str], index=self.model.str,dtype=float)   # GWh
        self.charging_power = pd.Series([value(self.model.charging_power[tech]) for tech in self.model.str], index=self.model.str,dtype=float)   # GW
        self.primary_generation = extract_primary_gene(self.model, self.nb_years, self.hourly_balance)   # TWh
        self.CH4_to_power_generation = extract_CH4_to_power(self.model, self.conversion_efficiency, self.nb_years, self.hourly_balance)   # TWh
        self.H2_to_power_generation = extract_H2_to_power(self.model, self.conversion_efficiency, self.nb_years, self.hourly_balance)   # TWh
        self.power_to_CH4_generation = extract_power_to_CH4(self.model, self.conversion_efficiency, self.nb_years, self.hourly_balance)   # TWh
        self.power_to_H2_generation = extract_power_to_H2(self.model, self.conversion_efficiency, self.nb_years, self.hourly_balance)   # TWh

        # 1e6€/yr
        self.new_capacity_annualized_costs, self.new_energy_capacity_annualized_costs = \
            extract_annualized_costs_investment_new_capa(self.installed_power, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                                         self.storage_annuities, self.fOM)

        self.transport_distribution_cost = transportation_distribution_cost(self.model, self.prediction_transport_and_distrib_annuity, self.installed_power) # 1e9€/yr

        # TWh
        self.elec_supply, self.elec_usage = extract_balance("elec", self.model, self.elec_demand, self.conversion_efficiency, self.hourly_balance)
        self.CH4_supply, self.CH4_usage = extract_balance("CH4", self.model, self.CH4_demand, self.conversion_efficiency, self.hourly_balance, demand_is_profile = self.CH4_demand_is_profile)
        self.H2_supply, self.H2_usage = extract_balance("H2", self.model, self.H2_demand, self.conversion_efficiency, self.hourly_balance, demand_is_profile = self.H2_demand_is_profile)

        self.str_losses = extract_storage_losses((self.elec_usage - self.elec_supply).dropna(),   # TWh
                      (self.CH4_usage - self.CH4_supply).dropna(),
                      (self.H2_usage - self.H2_supply).dropna())


        # gene in TWh
        self.summary, self.generation_per_technology = \
                            extract_summary(self.system_social_cost, self.model, self.elec_demand,
                                            self.H2_demand, self.H2_demand_is_profile,
                                            self.CH4_demand, self.CH4_demand_is_profile,
                                            self.installed_power, self.existing_capacity,
                                            self.energy_capacity, self.existing_energy_capacity, self.annuities,
                                            self.storage_annuities, self.fOM, self.vOM, self.conversion_efficiency,
                                            self.transport_distribution_cost,
                                            self.scc, self.nb_years, self.carbon_constraint, self.carbon_content,
                                            self.hourly_balance, self.spot_price)

        self.load_factor = self.generation_per_technology*1000/(self.installed_power*8760*self.nb_years)*100

        self.profits = extract_profit(self.model, self.hourly_balance, self.spot_price, self.vOM,
                                      self.new_capacity_annualized_costs.squeeze(), self.new_energy_capacity_annualized_costs.squeeze(),
                                     self.frr_requirements, self.fcr_requirement, self.reserve_activation_rate, self.conversion_efficiency,
                                     self.installed_power)

        self.lcoe_per_tech = calculate_lcoe_per_tech(self.model, self.hourly_balance, self.annuities, self.storage_annuities, self.fOM, self.vOM,
                                                     self.spot_price, self.nb_years, self.generation_per_technology, self.installed_power, self.energy_capacity,
                                                     self.existing_capacity, self.existing_energy_capacity) # €/MWh

        self.footprint = extract_carbon_footprint(self.model, self.generation_per_technology, self.carbon_footprint, self.nb_years) # MtCO2eq/yr
        self.summary.at["footprint [MtCO2eq/yr]"] = self.footprint.at["TOTAL"]

        # 1e6€/yr
        self.new_capacity_annualized_costs_nofOM, self.new_energy_capacity_annualized_costs_nofOM = \
            extract_annualized_costs_investment_new_capa_nofOM(self.installed_power, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                                         self.storage_annuities)
        self.OM_cost = extract_OM_cost(self.model, self.installed_power, self.fOM, self.vOM,
                                                           pd.Series(self.generation_per_technology) * 1000,
                                                           self.scc, self.carbon_content,
                                                           carbon_constraint=self.carbon_constraint, nb_years=self.nb_years)  # pd.Series



        self.results = pd.concat([self.installed_power.rename("Installed power [GW]"),
                                  self.energy_capacity.rename("Energy capacity [GWh]"),
                                  self.generation_per_technology.rename("Generated energy [TWh]"),
                                  self.load_factor.rename("Load factor [%]"),
                                  self.fcr.rename("FCR [TWh]"),
                                  self.frr.rename("FRR [TWh]"),
                                  self.lcoe_per_tech], axis=1).fillna('').replace(0, "")
