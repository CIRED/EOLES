import datetime
import json
import math
from pathlib import Path

# --- Unified saving wrapper for all plot_* functions ---
from typing import Optional

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _wrap_plot_savefig(func):
    """
    Decorator to add an `output_path` kwarg (as a folder path) to plot_* functions:
      - suppresses any internal plt.show()
      - saves the returned figure to a PNG file in that folder if provided
        with the filename derived from the function name.
    """

    def wrapper(*args, output_path: Optional[str] = None, **kwargs):
        import os

        # Temporarily disable plt.show() inside the wrapped plotting function
        original_show = plt.show

        def _noop_show(*_a, **_k):
            pass

        plt.show = _noop_show
        try:
            result = func(*args, **kwargs)
        finally:
            plt.show = original_show  # always restore

        # Determine the figure to save
        fig: Optional[Figure] = result if isinstance(result, Figure) else None
        if fig is None:
            try:
                fig = plt.gcf()
            except Exception:
                fig = None

        # If output_path is provided, treat it as a folder and build file name
        if output_path and fig is not None:
            os.makedirs(output_path, exist_ok=True)
            filename = f"{func.__name__}.png"
            full_path = os.path.join(output_path, filename)
            fig.savefig(full_path, dpi=300, bbox_inches="tight")

        return result

    return wrapper


import numpy as np
import pandas as pd
from pyomo.environ import value

"""[[[ Functions used to load data ]]]"""


def get_config(path) -> dict:
    with open(Path(__file__).parent / path) as file:
        return json.load(file)


def read_constant(path, year_costs=None):
    """Reads model constants. This includes the parameters of the different technologies (efficiencies, costs, potentials, ...) as well as constant constraints (potentials, carbon budget, ...).

    There are two possible formats : either the file provided has a single column which contains the inputs, or it regroups several years in which case only one will be used. In the latter case, years should be provided as different columns with the first row as the index.

    path: str
        The path to the file. The file's first column should contain the index. If several years are provided, the year index should be the first row.
    year: int (optional)
        Required if several years are provided in the same file, in which case this is the one selected.

    Returns : pd.Series"""

    # This function could be optimized by not reading data files twice, but since files read should be very small and to improve readability, it was kept this way

    df = pd.read_csv(Path(__file__).parent / path, index_col=0, header=None)

    if (
        df.shape[1] > 1
    ):  # number of columns is greater than 1, which means we have several years
        if year_costs is None:
            raise ValueError(
                f"Error reading {path} : Year of interest needs to be provided when data includes several columns"
            )
        df = pd.read_csv(
            path, index_col=0
        )  # re-read the df with the first row (which contains years) as the column names
        df = df.loc[:, str(year_costs)].squeeze()  # Select the column of interest
    else:
        df = df.squeeze(
            "columns"
        )  # Squeez columns of the dataframe to get a pandas series

    return df


def read_profile(path, time_scale, nb_years=1, years_of_interest=None):
    """Reads model inputs that are variable in time, like demand profiles or VRE production profiles.

    There are two possible formats : if the file contains one year's worth of data, then it is duplicated to match the number of years the simulation will run for ; otherwise, the file has to contain several years' worth of data, corresponding to the number of years. We consider that a year contains 365 days.

    path: str
        The path to the file. The file should contain a single profile, with date and hour in the first two columns and data in the third.
        For hourly timeseries the date format should either be %Y-%m-%d or %m-%d. For others it does not matter as the index is never refered to
        (neither for referencing nor for operation on series like adding series).
    time_scale: str
        Can be "hourly", "daily" or "monthly"
    nb_years: int. Defaults to 1.
    years_of_interest: str or int list
        Used to generate the index when the timeseries only contains one year and is expanded to several years, or when a 1-year long series with no year is provided

    Returns: pd.Series containing data for the correct number of years"""

    if time_scale == "hourly":
        yearly_nb_lines = 8760
        index_col = [0, 1]
    elif time_scale == "daily":
        yearly_nb_lines = 365
        index_col = 0
    elif time_scale == "monthly":
        yearly_nb_lines = 12
        index_col = 0
    else:
        raise ValueError(f"time scale '{time_scale}' not recognized")

    df = pd.read_csv(
        Path(__file__).parent / path, index_col=index_col, header=0
    ).squeeze("columns")

    if nb_years == 1:
        if df.shape[0] == yearly_nb_lines:
            # For now no daily or monthly series are compared, added or accessed by index, so it is only necessary to have a rigorous datetime format for hourly series.
            # This section should be changed if this changes
            if time_scale == "hourly":
                if (
                    len(df.index.get_level_values(0)[0]) > 5
                ):  # This is a really messy way to check wether the date format includes a year
                    df.index = pd.MultiIndex.from_arrays(
                        [
                            pd.to_datetime(
                                df.index.get_level_values(0), format="%Y-%m-%d"
                            )
                            .to_series()
                            .dt.date,
                            df.index.get_level_values(1),
                        ]
                    )
                else:  # In this case the date format only constains day and month
                    if years_of_interest is None:
                        raise ValueError(
                            f"Error reading {path} : Year of interest needs to be provided when the timeseries contains one year and has no specified year"
                        )
                    else:
                        df.index = pd.MultiIndex.from_arrays(
                            [
                                pd.to_datetime(
                                    df.index.get_level_values(0), format="%m-%d"
                                )
                                .to_series()
                                .apply(lambda dt: dt.replace(year=years_of_interest[0]))
                                .dt.date,
                                df.index.get_level_values(1),
                            ]
                        )

        else:
            raise ValueError(
                f"Error reading {path} : Expected file with {yearly_nb_lines} lines, got {df.shape[0]}"
            )

    else:
        if (
            df.shape[0] == yearly_nb_lines * nb_years
        ):  # Timeseries provided already contains the correct number of years
            # For now no daily or monthly series are compared, added or accessed by index, so it is only necessary to have a rigorous datetime format for hourly series
            if time_scale == "hourly":
                df.index = pd.MultiIndex.from_arrays(
                    [
                        pd.to_datetime(df.index.get_level_values(0), format="%Y-%m-%d")
                        .to_series()
                        .dt.date,
                        df.index.get_level_values(1),
                    ]
                )
        elif (
            df.shape[0] == yearly_nb_lines
        ):  # Timeseries provided contains one year. The study is on several years, so the timeseries will be repeated
            df_1y = df.copy()
            df = pd.concat([df_1y] * nb_years, ignore_index=True)

            # For now no daily or monthly series are compared, added or accessed by index, so it is only necessary to have a rigorous datetime format for hourly series.
            # This section should be changed if this changes
            # years_of_interest allows the construction of the date index
            if time_scale == "hourly":
                if years_of_interest is None:
                    raise ValueError(
                        f"Error reading {path} : Years of interest need to be provided when the timeseries only contains one year at hourly timescale and nb_years is greater than one"
                    )
                elif len(years_of_interest) != nb_years:
                    raise ValueError(
                        f"Error reading {path} : nb_years={nb_years} but years_of_interest contains {len(years_of_interest)} years."
                    )
                else:
                    date_range = pd.concat(
                        [
                            pd.date_range(
                                f"{year}/01/01", f"{year}/12/31", freq="1d"
                            ).to_series()
                            for year in years_of_interest
                        ],
                        ignore_index=True,
                    )
                    select_29_02 = (date_range.dt.day == 29) & (
                        date_range.dt.month == 2
                    )  # create boolean series indicating 29/02 days
                    date_range = date_range.loc[
                        ~select_29_02
                    ].dt.date  # remove those days

                    df.index = pd.MultiIndex.from_product(
                        [
                            date_range,
                            pd.date_range("00:00", "23:00", freq="1h")
                            .to_series()
                            .dt.hour,
                        ],
                        names=["Date", "Hour"],
                    )

        else:
            raise ValueError(
                f"Error reading {path} : Expected file with {yearly_nb_lines} or {yearly_nb_lines * nb_years} lines, got {df.shape[0]}"
            )

    return df.squeeze()


"""[[[ Functions used when defining the model ]]]"""


def calculate_annuities_capex(discount_rate, capex, construction_time, lifetime):
    """Calculate annuities for energy technologies and renovation technologies based on capex data.
    Assumes that all provided series have the same index and that it contains all relevant technologies"""
    annuities = construction_time.copy()
    for i in annuities.index:
        if lifetime.at[i] == 0 and construction_time.at[i] == 0:
            annuities.at[i] = 0
        else:
            annuities.at[i] = (
                discount_rate.at[i]
                * capex.at[i]
                * (discount_rate.at[i] * construction_time.at[i] + 1)
                / (1 - (1 + discount_rate.at[i]) ** (-lifetime.at[i]))
            )
    return annuities


def calculate_annuities_storage_capex(
    discount_rate, storage_capex, construction_time, lifetime
):
    """Calculate annuities for storage technologies based on capex data.
    Assumes that all provided series have the same index and that it contains all relevant technologies"""
    storage_annuities = storage_capex.copy()
    for i in storage_annuities.index:
        storage_annuities.at[i] = (
            discount_rate.at[i]
            * storage_capex.at[i]
            * (discount_rate.at[i] * construction_time.at[i] + 1)
            / (1 - (1 + discount_rate.at[i]) ** (-lifetime.at[i]))
        )
    return storage_annuities


def update_vom_costs_scc(vOM_init, scc, emission_rate):
    """Add emission cost related to social cost of carbon to fossil vectors vOM costs.
    :param vOM_init: float
        Initial vOM in M€/GW = €/kW
    :param scc: int
        €/tCO2
    :param emission_rate: float
        tCO2/MWh.

    Returns
    vOM in M€/GW(h)  = €/kW(h)
    """
    return vOM_init + scc * emission_rate / 1000


def define_month_hours(first_month, nb_years, months_hours, hours_by_months):
    """
    Calculates range of hours for each month
    :param first_month: int
    :param nb_years: int
    :param months_hours: dict
    :param hours_by_months: dict
    :return:
    Dict containing the range of hours for each month considered in the model
    """
    j = first_month + 1
    for i in range(2, 12 * nb_years + 1):
        hour = months_hours[i - 1][-1] + 1  # get the first hour for a given month
        months_hours[i] = range(hour, hour + hours_by_months[j])
        j += 1
        if j == 13:
            j = 1
    return months_hours


def shift_range(h, L, last_hour):
    return [(h - L + i) % (last_hour + 1) for i in range(2 * L + 1)]


"""[[[ Functions used to process outputs ]]]"""


def get_technical_cost(model, objective, scc, nb_years, carbon_content):
    """Returns technical cost (social cost without CO2 emissions-related cost)"""
    gene_ngas = sum(value(model.gene["natural_gas", hour]) for hour in model.h)  # GWh
    gene_coal = sum(value(model.gene["coal", hour]) for hour in model.h)  # GWh
    gene_import_gaz = sum(
        value(model.gene["import_gaz", hour]) for hour in model.h
    )  # GWh
    total_emissions_gas = (
        gene_ngas * carbon_content.at["natural_gas"]
        + gene_import_gaz * carbon_content.at["import_gaz"] / 1000
    )
    total_emissions_coal = gene_coal * carbon_content.at["coal"] / 1000  # MtCO2

    emissions = pd.Series(
        {
            "natural_gas": total_emissions_gas / nb_years,
            "coal": total_emissions_coal / nb_years,
        }
    )  # MtCO2/yr
    total_emissions = total_emissions_gas + total_emissions_coal
    technical_cost = objective - total_emissions * scc / 1000  # 1e9€/yr
    return technical_cost, emissions


def extract_carbon_footprint(
    model, gene_per_tech, carbon_footprint, nb_years
):  # MtCO2eq/yr
    """Calculates the yearly and per energy carbon footprint of each technology"""
    footprint = pd.Series(dtype=float)
    for tech in model.all_tech:
        if not (tech in model.CH4_balance_biogas):
            footprint.at[tech] = (
                gene_per_tech.at[tech] * carbon_footprint.at[tech] / nb_years
            )
    footprint.at["biogas"] = (
        sum(gene_per_tech.at[tech] for tech in model.CH4_balance_biogas)
        * carbon_footprint.at["biogas"]
        / nb_years
    )
    footprint.at["elec_network"] = (
        sum(gene_per_tech.at[tech] for tech in model.elec_balance)
        * carbon_footprint.at["network"]
        / nb_years
    )
    footprint.at["elec_network_primarygene"] = (
        sum(gene_per_tech.at[tech] for tech in model.elec_primary_prod)
        * carbon_footprint.at["network"]
        / nb_years
    )

    footprint.at["TOTAL"] = footprint.sum() - footprint.at["elec_network_primarygene"]

    return footprint


def extract_hourly_balance(
    model,
    elec_demand,
    H2_demand,
    CH4_demand,
    conversion_efficiency,
    conversion_efficiency_biogas,
    eta_in,
    eta_out,
    load_shift_period,
    load_shift_period_metha,
    last_hour,
):
    """Extracts hourly defined data, including demand, generation and storage
    Returns a dataframe with hourly generation for each hour.
    Using this function, you limit the number of times model output values are extracted, which is costly in computational power.
    You can then manipulate pandas DataFrames which are must faster."""

    hourly_balance = pd.DataFrame(index=elec_demand.index)
    hourly_balance.loc[:, "elec_demand"] = elec_demand
    hourly_balance.loc[:, "H2_demand"] = H2_demand
    hourly_balance.loc[:, "CH4_demand"] = CH4_demand
    hourly_balance.loc[:, "load_shift_up"] = value(model.dsm_up[:])  # GW
    hourly_balance.loc[:, "load_shift_down"] = [
        sum(
            value(model.dsm_down[hh, h])
            for hh in shift_range(h, load_shift_period, last_hour)
        )
        for h in model.h
    ]  # GW
    hourly_balance.loc[:, "load_shift_up_metha"] = value(model.dsm_up_metha[:])  # GW
    hourly_balance.loc[:, "load_shift_down_metha"] = [
        sum(
            value(model.dsm_down_metha[hh, h])
            for hh in shift_range(h, load_shift_period_metha, last_hour)
        )
        for h in model.h
    ]  # GW
    hourly_balance.loc[:, "elec_demand_w/_shift"] = (
        hourly_balance.loc[:, "elec_demand"]
        + hourly_balance.loc[:, "load_shift_up"]
        + hourly_balance.loc[:, "load_shift_up_metha"]
        - hourly_balance.loc[:, "load_shift_down"]
        - hourly_balance.loc[:, "load_shift_down_metha"]
    )
    hourly_balance.loc[:, "demand_on_hold"] = pd.Series(
        index=elec_demand.index, dtype=float
    )
    hourly_balance.iloc[0, hourly_balance.columns.get_loc("demand_on_hold")] = (
        hourly_balance.iloc[0, hourly_balance.columns.get_loc("load_shift_down")]
        + hourly_balance.iloc[0, hourly_balance.columns.get_loc("load_shift_down_metha")]
        - hourly_balance.iloc[0, hourly_balance.columns.get_loc("load_shift_up")]
        - hourly_balance.iloc[0, hourly_balance.columns.get_loc("load_shift_up_metha")]
    )
    for h in range(1, last_hour + 1):
        hourly_balance.iloc[h, hourly_balance.columns.get_loc("demand_on_hold")] = (
            hourly_balance.iloc[h - 1, hourly_balance.columns.get_loc("demand_on_hold")]
            + hourly_balance.iloc[h, hourly_balance.columns.get_loc("load_shift_down")]
            + hourly_balance.iloc[
                h, hourly_balance.columns.get_loc("load_shift_down_metha")
            ]
            - hourly_balance.iloc[h, hourly_balance.columns.get_loc("load_shift_up")]
            - hourly_balance.iloc[
                h, hourly_balance.columns.get_loc("load_shift_up_metha")
            ]
        )
    for tech in model.prod_tech:
        hourly_balance.loc[:, tech] = value(model.gene[tech, :])  # GW
    for tech in model.toute_conversion_tech:
        hourly_balance.loc[:, tech] = value(model.conv_output[tech, :])  # GW
    for tech in model.str:
        hourly_balance.loc[:, tech] = value(model.str_output[tech, :])  # GW
    for tech in model.reserve:
        hourly_balance.loc[:, tech + "_frr"] = value(model.frr[tech, :])  # GW
        hourly_balance.loc[:, tech + "_fcr"] = value(model.fcr[tech, :])  # GW
    # We add technologies which include a conversion parameter, to express their hourly generation in GWh of the input vector
    for tech in model.conversion_tech:
        hourly_balance.loc[:, tech + "_input"] = (
            value(model.conv_output[tech, :]) / conversion_efficiency.at[tech]
        )
    # We add technologies which include a double conversion parameter, to express their hourly generation in GWh of input vectors : electricity (input) and biomass (input2)
    for tech in model.double_conversion_tech:
        hourly_balance.loc[:, tech + "_input"] = (
            value(model.conv_output[tech, :]) / conversion_efficiency.at[tech]
        )
        hourly_balance.loc[:, tech + "_input2"] = (
            value(model.conv_output[tech, :]) / conversion_efficiency_biogas.at[tech]
        )
    for tech in model.str:
        hourly_balance.loc[:, tech + "_input"] = value(model.str_input[tech, :])  # GW
        hourly_balance.loc[:, tech + "_state_charge"] = value(
            model.state_of_charge[tech, :]
        )  # GW
    hourly_balance.loc[:, "lake_state_charge"] = value(model.lake_stored[:])  # GW
    hourly_balance.loc[:, "storage_input_losses"] = sum(
        hourly_balance.loc[:, tech + "_input"] * (1 - eta_in.at[tech])
        for tech in model.str
    )
    hourly_balance.loc[:, "storage_output_losses"] = sum(
        hourly_balance.loc[:, tech] * (1 / eta_out.at[tech] - 1) for tech in model.str
    )

    return hourly_balance  # GW


def extract_hourly_demand(vector, model, conversion_efficiency, hourly_balance):
    """Only useful when annual demand is provided instead of a demand profile.
    Extracts how much of the demand is satisfied at each hour.
    :param vector: string
        Should be one of : 'methane', 'CH4', 'hydrogen', 'H2'
    :param model: pyomo model
    :param conversion_efficiency: pd.Series
    :return: pd.Series"""

    if vector == "methane" or vector == "CH4":
        supply_list = model.CH4_balance
        usage_list = model.use_CH4
    elif vector == "hydrogen" or vector == "H2":
        supply_list = model.H2_balance
        usage_list = model.use_H2

    supply = pd.Series(index=hourly_balance.index, dtype=float)
    for h in model.h:
        supply.iat[h] = sum(
            hourly_balance.iloc[h, hourly_balance.columns.get_loc(tech)]
            for tech in supply_list
        )  # GW

    usage = pd.Series(index=model.hourly_balance.index, dtype=float)
    for h in model.h:
        usage.iat[h] = sum(
            hourly_balance.iloc[h, hourly_balance.columns.get_loc(tech + "_input")]
            for tech in usage_list
        )

    return supply - usage


def extract_curtailment(model, conversion_efficiency, hourly_balance):
    """Calculates and returns hourly curtailed electricity.
    Also adds it to hourly_balance"""

    curtailment = pd.Series(index=hourly_balance.index, dtype=float)
    for h in model.h:
        elec_supply_h = sum(
            hourly_balance.iloc[h, hourly_balance.columns.get_loc(tech)]
            for tech in model.elec_balance
        )
        elec_usage_h = sum(
            hourly_balance.iloc[h, hourly_balance.columns.get_loc(tech + "_input")]
            for tech in model.use_elec
        )
        curtailment.iat[h] = (
            elec_supply_h
            - elec_usage_h
            - hourly_balance.iloc[
                h, hourly_balance.columns.get_loc("elec_demand_w/_shift")
            ]
        )
        if math.isclose(curtailment.iat[h], 0, abs_tol=1e-09):
            curtailment.iat[h] = 0

    hourly_balance.loc[:, "curtailment"] = curtailment
    return curtailment


def get_carbon_content(model, hourly_balance, carbon_content, conversion_efficiency):
    """Estimates the carbon content of gas, based on methodology by ADEME and RTE (méthode moyenne horaire).
    Returns the result in gCO2/kWh = tCO2/GWh
    Also adds a column in hourly_balance that indicates the carbon content of electricity in gCO2/kWh = tCO2/GWh"""

    # Estimate carbon content of gas
    CH4_carbon_content = (
        (
            hourly_balance.loc[:, "natural_gas"].sum()
            + hourly_balance.loc[:, "import_gaz"].sum()
        )
        * carbon_content.at["natural_gas"]
    ) / (
        hourly_balance.loc[:, "natural_gas"]
        + hourly_balance.loc[:, "import_gaz"].sum()
        + hourly_balance.loc[:, "methanization_injection"]
        + hourly_balance.loc[:, "pyrogazification"]
    ).sum()

    hourly_balance.loc[:, "CH4_carbon_content"] = hourly_balance.apply(
        lambda row: 1000
        * (
            row.at["ch4_ocgt"]
            / conversion_efficiency.at["ch4_ocgt"]
            * CH4_carbon_content
            + row.at["ch4_ccgt"]
            / conversion_efficiency.at["ch4_ccgt"]
            * CH4_carbon_content
            + row.at["coal"] * carbon_content.at["coal"]
        )
        / sum(row[tech] for tech in model.elec_balance),
        axis=1,
    )

    return CH4_carbon_content * 1e3


def extract_peak_load(hourly_balance: pd.DataFrame, conversion_efficiency):
    """Deprecated
    Returns the value of peak load for electricity in GW. Includes electricity demand, as well as demand for electrolysis and methanation."""

    peak_load = hourly_balance.copy()[
        [
            "elec_demand",
            "electrolysis",
            "methanation",
            "metha_coge",
            "metha_injection",
            "methanization_injection",
            "methanization_coge",
        ]
    ]

    peak_load["peak_electricity_load"] = (
        peak_load["elec_demand"]
        + peak_load["electrolysis"] / conversion_efficiency["electrolysis"]
        + peak_load["methanation"] / conversion_efficiency["methanation"]
        + peak_load["methanization_coge"] / conversion_efficiency["methanization_coge"]
        + peak_load["methanization_injection"]
        / conversion_efficiency["methanization_injection"]
        + peak_load["metha_coge"] / conversion_efficiency["metha_coge"]
        + peak_load["metha_injection"] / conversion_efficiency["metha_injection"]
    )
    ind = peak_load["peak_electricity_load"].idxmax()
    peak_load_info = peak_load.loc[ind]
    peak_load_info.name = "Peak Load"
    peak_load_info["hour"] = str(ind)
    peak_load_info["year"] = str(ind // 8760)

    return peak_load_info  # GW


def extract_spot_price(model, nb_hours):
    """Extracts spot prices in 1e3€/GW(h) = €/MW(h). Dual is in 1e9€/GW(h) = €/W(h) because the objective function is in 1e9€ and constraints are in GW(h)"""
    spot_price = pd.DataFrame(
        {
            "elec": [
                -1e6 * model.dual[model.electricity_adequacy_constraint[h]]
                for h in model.h
            ],
            "CH4": [
                -1e6 * model.dual[model.methane_balance_constraint[h]] for h in model.h
            ],
            "H2": [
                -1e6 * model.dual[model.hydrogen_balance_constraint[h]] for h in model.h
            ],
            "fcr": [
                1e6 * model.dual[model.fcr_provision_constraint[h]] for h in model.h
            ],
            "frr": [
                1e6 * model.dual[model.frr_provision_constraint[h]] for h in model.h
            ],
            "shifting_upwards_metha": [
                -1e6 * model.dual[model.shifting_upwards_limit_constraint_metha[h]]
                for h in model.h
            ],
            "shifting_downwards_metha": [
                -1e6 * model.dual[model.shifting_downwards_limit_constraint_metha[h]]
                for h in model.h
            ],
            "shifting_simultaneous_metha": [
                -1e6 * model.dual[model.shifting_simultaneous_limit_constraint_metha[h]]
                for h in model.h
            ],
            "shifting_upwards": [
                -1e6 * model.dual[model.shifting_upwards_limit_constraint[h]]
                for h in model.h
            ],
            "shifting_downwards": [
                -1e6 * model.dual[model.shifting_downwards_limit_constraint[h]]
                for h in model.h
            ],
            "shifting_simultaneous": [
                -1e6 * model.dual[model.shifting_simultaneous_limit_constraint[h]]
                for h in model.h
            ],
            "storage_charging_discharging": [
                -1e6
                * model.dual[
                    model.storage_charging_discharging_constraint["injection_reservoir"]
                ]
                for h in model.h
            ],
            "storing": [
                -1e6 * model.dual[model.storing_constraint[h, "injection_reservoir"]]
                for h in model.h
            ],
            "energy_capacity": [
                -1e6
                * model.dual[model.energy_capacity_constraint[h, "injection_reservoir"]]
                for h in model.h
            ],
            "storage_injection_reservoir": [
                -1e6 * model.dual[model.min_biogas_injection_storage] for h in model.h
            ],
    
            "biogas_coge_balance": [
                -1e6 * model.dual[model.biogas_coge_balance_constraint[h]]
                for h in model.h
            ],
            "biogas_inj_balance": [
                -1e6 * model.dual[model.biogas_injection_balance_constraint[h]]
                for h in model.h
            ],
        }
    )
    return spot_price


def extract_carbon_value(model, carbon_constraint, scc):
    """Extracts the social value of carbon in the considered model. Corresponds to the given SCC or to the shadow price of the carbon constraint if the latter is used. Carbon constraint is in ktCO2 and the objective function is in 1e9€, so the dual is in 1e6€/tCO2"""
    if carbon_constraint:
        # TODO: here we only consider the carbon value for one of the given years !! to modify in future
        carbon_value = -1e6 * model.dual[model.carbon_budget_constraint[0]]  # €/tCO2
    else:
        carbon_value = scc
    return carbon_value


def extract_primary_gene(model, nb_years, hourly_balance):
    """Extracts yearly primary energy generation per source of energy in TWh"""
    primary_generation = pd.Series(dtype=float)

    for tech in model.prod_tech:
        primary_generation.at[tech] = (
            hourly_balance.loc[:, tech].sum() / 1000 / nb_years
        )  # TWh
    return primary_generation


def extract_CH4_to_power(model, conversion_efficiency, nb_years, hourly_balance):
    """Extracts CH4 used to produce electricity in TWh"""
    gas_to_power_input = pd.Series(dtype=float)

    for tech in model.from_CH4_to_elec:
        gas_to_power_input.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000 / nb_years
        )  # TWh
    return gas_to_power_input


def extract_H2_to_power(model, conversion_efficiency, nb_years, hourly_balance):
    """Extracts H2 used to produce electricity in TWh"""
    H2_to_power_input = pd.Series(dtype=float)

    for tech in model.from_H2_to_elec:
        H2_to_power_input.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000 / nb_years
        )  # TWh
    return H2_to_power_input


def extract_power_to_CH4(model, conversion_efficiency, nb_years, hourly_balance):
    """Extracts electricity generation necessary to produce CH4 in TWh"""
    power_to_CH4_input = pd.Series(dtype=float)

    for tech in model.from_elec_to_CH4 | model.from_biogas_to_CH4:
        power_to_CH4_input[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000 / nb_years
        )  # TWh

    return power_to_CH4_input


def extract_power_to_H2(model, conversion_efficiency, nb_years, hourly_balance):
    """Extracts electricity used to produce H2 in TWh"""
    power_to_H2_input = pd.Series(dtype=float)

    for tech in model.from_elec_to_H2:
        power_to_H2_input.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000 / nb_years
        )  # TWh
    return power_to_H2_input


def extract_balance(
    vector, model, demand, conversion_efficiency, hourly_balance, demand_is_profile=True
):
    """Extracts total supply and usage (including demand) of the given vector (elec, CH4 or H2) in TWh
    :param vector: string
        Should be one of : 'elec', 'electricity', 'methane', 'CH4', 'hydrogen', 'H2'
    :param model: pyomo model
    :param demand: pd.Series or int
        Type should be consistent with 'demand_is_profile'
    :param conversion_efficiency: pd.Series
    :param demand_is_profile: string
    :return: pd.Series tuple
    """

    if vector == "elec" or vector == "electricity":
        supply_list = model.elec_balance
        usage_list = model.use_elec
    elif vector == "methane" or vector == "CH4":
        supply_list = model.CH4_balance
        usage_list = model.use_CH4
    elif vector == "hydrogen" or vector == "H2":
        supply_list = model.H2_balance
        usage_list = model.use_H2
    else:
        raise ValueError(f"{vector} is not a recognized vector")

    supply = pd.Series(dtype=float)
    for tech in supply_list:
        supply.at[tech] = hourly_balance.loc[:, tech].sum() / 1000  # TWh

    usage = pd.Series(dtype=float)
    for tech in usage_list:
        usage.at[tech] = hourly_balance.loc[:, tech + "_input"].sum() / 1000  # TWh

    if demand_is_profile:
        usage.at["demand"] = demand.sum() / 1000  # TWh
    else:
        usage.at["demand"] = demand / 1000  # TWh

    if vector == "elec" or vector == "electricity":
        usage.at["curtailment"] = (
            hourly_balance.loc[:, "curtailment"].sum() / 1000
        )  # TWh

    return supply, usage


def extract_storage_losses(elec_str_losses, CH4_str_losses, H2_str_losses):
    str_losses = pd.concat([elec_str_losses, CH4_str_losses, H2_str_losses])
    str_losses.loc[np.isclose(str_losses, 0)] = 0
    str_losses.at["TOTAL"] = str_losses.sum()
    return str_losses


def extract_annualized_costs_investment_new_capa(
    capacities,
    energy_capacities,
    existing_capacities,
    existing_energy_capacities,
    annuities,
    storage_annuities,
    fOM,
):
    """
    Returns the annualized costs coming from newly invested capacities and energy capacities. This includes annualized CAPEX + fOM.
    Unit: 1e6€/yr
    :param model: pyomo model
    :param existing_capacities: pd.Series
    :return:
    """
    new_capacity = capacities - existing_capacities  # pd.Series
    costs_new_capacity = pd.concat(
        [new_capacity, annuities, fOM], axis=1, ignore_index=True
    ).rename(columns={0: "new_capacity", 1: "annuities", 2: "fOM"})
    costs_new_capacity.loc[:, "annualized_costs"] = costs_new_capacity.loc[
        :, "new_capacity"
    ] * (
        costs_new_capacity.loc[:, "annuities"] + costs_new_capacity.loc[:, "fOM"]
    )  # includes both annuity and fOM ! not to be counted twice in the LCOE

    new_storage_capacity = energy_capacities - existing_energy_capacities
    costs_new_energy_capacity = pd.concat(
        [new_storage_capacity, storage_annuities], axis=1, ignore_index=True
    ).rename(columns={0: "new_capacity", 1: "storage_annuities"})
    costs_new_energy_capacity.loc[:, "annualized_costs"] = (
        costs_new_energy_capacity.loc[:, "new_capacity"]
        * costs_new_energy_capacity.loc[:, "storage_annuities"]
    )
    return costs_new_capacity.loc[:, "annualized_costs"], costs_new_energy_capacity.loc[
        :, "annualized_costs"
    ]


def extract_annualized_costs_investment_new_capa_nofOM(
    capacities,
    energy_capacities,
    existing_capacities,
    existing_energy_capacities,
    annuities,
    storage_annuities,
):
    """
    Returns the annualized investment coming from newly invested capacities and energy capacities, without fOM. Unit: 1e6€/yr
    :param model: pyomo model
    :param existing_capacities: pd.Series
    :return:
    """
    new_capacity = capacities - existing_capacities  # pd.Series
    costs_new_capacity = pd.concat(
        [new_capacity, annuities], axis=1, ignore_index=True
    ).rename(columns={0: "new_capacity", 1: "annuities"})
    costs_new_capacity = costs_new_capacity.dropna()
    costs_new_capacity.loc[:, "annualized_costs"] = (
        costs_new_capacity.loc[:, "new_capacity"]
        * costs_new_capacity.loc[:, "annuities"]
    )

    new_storage_capacity = energy_capacities - existing_energy_capacities
    costs_new_energy_capacity = pd.concat(
        [new_storage_capacity, storage_annuities], axis=1, ignore_index=True
    ).rename(columns={0: "new_capacity", 1: "storage_annuities"})
    costs_new_energy_capacity = costs_new_energy_capacity.dropna()
    costs_new_energy_capacity.loc[:, "annualized_costs"] = (
        costs_new_energy_capacity.loc[:, "new_capacity"]
        * costs_new_energy_capacity.loc[:, "storage_annuities"]
    )
    return costs_new_capacity.loc[:, "annualized_costs"], costs_new_energy_capacity.loc[
        :, "annualized_costs"
    ]


def extract_OM_cost(
    model,
    capacities,
    fOM,
    vOM,
    generation,
    scc,
    carbon_content,
    carbon_constraint=True,
    nb_years=1,
):
    """Returns operation and maintenance costs, which corresponds to fOM and vOM, including SCC when it is used. vOM for gas include the SCC. Unit: 1e6€/yr
    IMPORTANT REMARK: we divide generation by number of total years to get the average yearly generation
    :param scc: int
        Social cost of carbon used to estimate optimal power mix.
    """

    if not carbon_constraint:  # ie optimization with a given social cost of carbon
        # we remove the SCC in this vOM
        vOM_no_scc = vOM.copy()
        vOM_no_scc.at["natural_gas"] = update_vom_costs_scc(
            vOM_no_scc.at["natural_gas"],
            scc=(-scc),
            emission_rate=carbon_content.at["natural_gas"],
        )  # €/kWh
        vOM_no_scc.at["import_gaz"] = update_vom_costs_scc(
            vOM_no_scc.at["import_gaz"],
            scc=(-scc),
            emission_rate=carbon_content.at["import_gaz"],
        )  # €/kWh
        vOM_no_scc.at["coal"] = update_vom_costs_scc(
            vOM_no_scc.at["coal"], scc=(-scc), emission_rate=carbon_content.at["coal"]
        )  # €/kWh

        # variable cost only due to actual scc, not anticipated scc
        vOM_SCC_only = vOM - vOM_no_scc

        system_fOM_vOM = pd.concat(
            [capacities, fOM, vOM_no_scc, vOM_SCC_only, generation / nb_years],
            axis=1,
            ignore_index=True,
        ).rename(
            columns={
                0: "capacity",
                1: "fOM",
                2: "vOM_no_scc",
                3: "vOM_SCC_only",
                4: "generation",
            }
        )
        system_fOM_vOM = system_fOM_vOM.dropna()
        system_fOM_vOM.loc[:, "OM_cost_noSCC"] = (
            system_fOM_vOM.loc[:, "capacity"] * system_fOM_vOM.loc[:, "fOM"]
            + system_fOM_vOM.loc[:, "generation"] * system_fOM_vOM.loc[:, "vOM_no_scc"]
        )
        system_fOM_vOM.loc[:, "OM_cost_SCC_only"] = (
            system_fOM_vOM.loc[:, "generation"] * system_fOM_vOM.loc[:, "vOM_SCC_only"]
        )

        carbon_cost = system_fOM_vOM.loc[:, "OM_cost_SCC_only"].sum()
        system_fOM_vOM = system_fOM_vOM.loc[:, "OM_cost_noSCC"]
        system_fOM_vOM.rename(columns={"OM_cost_noSCC": "OM_cost"}, inplace=True)
        system_vOM_fOM.loc["carbon_cost", "OM_cost"] = carbon_cost

    else:
        system_fOM_vOM = pd.concat(
            [capacities, fOM, vOM, generation / nb_years], axis=1, ignore_index=True
        ).rename(columns={0: "capacity", 1: "fOM", 2: "vOM", 3: "generation"})
        system_fOM_vOM = system_fOM_vOM.dropna()
        system_fOM_vOM.loc[:, "OM_cost"] = (
            system_fOM_vOM.loc[:, "capacity"] * system_fOM_vOM.loc[:, "fOM"]
            + system_fOM_vOM.loc[:, "generation"] * system_fOM_vOM.loc[:, "vOM"]
        )
        system_fOM_vOM = system_fOM_vOM.loc[:, "OM_cost"]

    return system_fOM_vOM


def calculate_lcoe_per_tech(
    model,
    hourly_balance,
    annuities,
    storage_annuities,
    fOM,
    vOM,
    spot_price,
    nb_years,
    gene_per_tech,
    capacity,
    energy_capacity,
    existing_capacity,
    existing_energy_capacity,
    capex,
):
    """This function calculates the levelized cost of energy produced by the newly installed plants (ie not present in 2020 and not coming from an earlier
    instance of the model) for each technology. Generation is attributed proportionnately to capacity to newly constructed plants and old ones, as in the
    model all plants are aggregated."""

    lcoe = pd.DataFrame(
        columns=["lcoe [€/MWh]", "lcoe without input [€/MWh]"], dtype=object
    )
    for tech in (
        model.prod_tech
        | model.conversion_tech
        | model.double_conversion_tech
        | model.str
    ):
        if capacity.at[tech] != 0:
            gene_new_installation = (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )

            # /!\ gene is in TWh
            cost_without_input = (capacity.at[tech] - existing_capacity.at[tech]) * (
                annuities.at[tech] + fOM.at[tech]
            ) * nb_years + gene_new_installation * 1000 * vOM.at[tech]  # 1e6€

            if tech in model.str:
                cost_without_input += (
                    (energy_capacity.at[tech] - existing_energy_capacity.at[tech])
                    * storage_annuities.at[tech]
                    * nb_years
                )

            if (
                tech in model.conversion_tech
                or tech in model.str
                or tech in model.double_conversion_tech
            ):
                # /!\ spot_prices are in 1e3€/GWh
                if tech in model.use_elec:
                    cost = (
                        cost_without_input
                        + (
                            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                            * spot_price.loc[:, "elec"]
                        ).sum()
                        / 1e3
                    )  # 1e6€
                if tech in model.use_CH4:
                    cost = (
                        cost_without_input
                        + (
                            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                            * spot_price.loc[:, "CH4"]
                        ).sum()
                        / 1e3
                    )  # 1e6€
                if tech in model.use_H2:
                    cost = (
                        cost_without_input
                        + (
                            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                            * spot_price.loc[:, "H2"]
                        ).sum()
                        / 1e3
                    )  # 1e6€

                if gene_new_installation != 0:
                    lcoe.at[tech, "lcoe [€/MWh]"] = cost / gene_new_installation
                    lcoe.at[tech, "lcoe without input [€/MWh]"] = (
                        cost_without_input / gene_new_installation
                    )
                else:
                    lcoe.at[tech, "lcoe [€/MWh]"] = "No new plants"

            else:  # primary production
                if gene_new_installation != 0:
                    lcoe.at[tech, "lcoe [€/MWh]"] = (
                        cost_without_input / gene_new_installation
                    )
                else:
                    lcoe.at[tech, "lcoe [€/MWh]"] = "No new plants"
    """
    for tech in model.prod_tech:
        if nominal_power.at[tech] != 0:
            gene_new_installation = gene_per_tech.at[tech]*(capacity.at[tech] - existing_capacity.at[tech])/nominal_power.at[tech]
            # /!\ gene is in TWh
            cost_without_input = (nominal_power.at[tech] - existing_capacity.at[tech])*(annuities.at[tech] + fOM.at[tech])*nb_years + gene_new_installation*1000*vOM.at[tech] # 1e6€
            if gene_new_installation != 0:
                lcoe.at[tech, "lcoe [€/MWh]"] = cost_without_input / gene_new_installation
            else:
                lcoe.at[tech, "lcoe [€/MWh]"] = "No new plants"

    for tech in model.conversion_tech:
        if output_power.at[tech] != 0:
            gene_new_installation = gene_per_tech.at[tech]*(capacity.at[tech] - existing_capacity.at[tech])/output_power.at[tech]
            # /!\ gene is in TWh
            cost_without_input = (output_power.at[tech] - existing_capacity.at[tech])*(annuities.at[tech] + fOM.at[tech])*nb_years + gene_new_installation*1000*vOM.at[tech] # 1e6€
            if tech in model.use_elec:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "elec"]).sum()/1e3   # 1e6€
            if tech in model.use_CH4:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "CH4"]).sum()/1e3   # 1e6€
            if tech in model.use_H2:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "H2"]).sum()/1e3   # 1e6€
            if gene_new_installation != 0:
                lcoe.at[tech, "lcoe [€/MWh]"] = cost / gene_new_installation
                lcoe.at[tech, "lcoe without input [€/MWh]"] = cost_without_input / gene_new_installation
            else:
                lcoe.at[tech, "lcoe [€/MWh]"] = "No new plants"

    for tech in model.str:
        if output_power.at[tech] != 0:
            gene_new_installation = gene_per_tech.at[tech]*(capacity.at[tech] - existing_capacity.at[tech])/output_power.at[tech]
            # /!\ gene is in TWh
            cost_without_input = (output_power.at[tech] - existing_capacity.at[tech])*(annuities.at[tech] + fOM.at[tech])*nb_years + gene_new_installation*1000*vOM.at[tech] # 1e6€
            cost_without_input += (energy_capacity.at[tech] - existing_energy_capacity.at[tech])*storage_annuities.at[tech]*nb_years
            if tech in model.use_elec:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "elec"]).sum()/1e3   # 1e6€
            if tech in model.use_CH4:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "CH4"]).sum()/1e3   # 1e6€
            if tech in model.use_H2:
                cost = cost_without_input + (hourly_balance.loc[:, tech+"_input"].reset_index(drop=True)*spot_price.loc[:, "H2"]).sum()/1e3   # 1e6€
            if gene_new_installation != 0:
                lcoe.at[tech, "lcoe [€/MWh]"] = cost / gene_new_installation
                lcoe.at[tech, "lcoe without input [€/MWh]"] = cost_without_input / gene_new_installation
            else:
                lcoe.at[tech, "lcoe [€/MWh]"] = "No new plants"
    """
    return lcoe


def compute_costs(
    model,
    annuities,
    fOM,
    vOM,
    storage_annuities,
    gene_per_tech,
    capacity,
    existing_capacity,
    energy_capacity,
    existing_energy_capacity,
    nb_years,
    hourly_balance,
    capex,
):
    costs_elec = 0  # 1e6€
    for tech in model.elec_balance:
        if capacity.at[tech] != 0:
            gene_new_installation = (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )
            # /!\ gene is in TWh
            costs_elec += (capacity.at[tech] - existing_capacity.at[tech]) * (
                annuities.at[tech] + fOM.at[tech]
            ) * nb_years + gene_new_installation * 1000 * vOM.at[tech]
            if tech in model.str:
                costs_elec += (
                    (energy_capacity.at[tech] - existing_energy_capacity.at[tech])
                    * storage_annuities.at[tech]
                    * nb_years
                )
    costs_elec += (
        (
            energy_capacity.at["coge_reservoir"]
            - existing_energy_capacity.at["coge_reservoir"]
        )
        * (storage_annuities.at["coge_reservoir"] + fOM.at["coge_reservoir"])
        * nb_years
        + gene_per_tech.at["coge_reservoir"] * 1000 * vOM.at["coge_reservoir"]
        + (
            energy_capacity.at["extra_coge_reservoir"]
            - existing_energy_capacity.at["extra_coge_reservoir"]
        )
        * (storage_annuities.at["extra_coge_reservoir"] + fOM.at["extra_coge_reservoir"])
        * nb_years
        + gene_per_tech.at["extra_coge_reservoir"]
        * 1000
        * vOM.at["extra_coge_reservoir"]
        + (capacity.at["metha_coge"] - existing_capacity.at["metha_coge"])
        * (annuities.at["metha_coge"] + fOM.at["metha_coge"])
        * nb_years
        + gene_per_tech.at["metha_coge"] * 1000 * vOM.at["metha_coge"]
    )

    costs_CH4 = 0  # 1e6€
    for tech in model.CH4_balance:
        if capacity.at[tech] != 0:
            gene_new_installation = (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )
            # /!\ gene is in TWh
            costs_CH4 += (capacity.at[tech] - existing_capacity.at[tech]) * (
                annuities.at[tech] + fOM.at[tech]
            ) * nb_years + gene_new_installation * 1000 * vOM.at[tech]
            if tech in model.str:
                costs_CH4 += (
                    (energy_capacity.at[tech] - existing_energy_capacity.at[tech])
                    * storage_annuities.at[tech]
                    * nb_years
                )
    costs_CH4 += (
        (
            energy_capacity.at["injection_reservoir"]
            - existing_energy_capacity.at["injection_reservoir"]
        )
        * (storage_annuities.at["injection_reservoir"] + fOM.at["injection_reservoir"])
        * nb_years
        + gene_per_tech.at["injection_reservoir"] * 1000 * vOM.at["injection_reservoir"]
        + (
            energy_capacity.at["extra_injection_reservoir"]
            - existing_energy_capacity.at["extra_injection_reservoir"]
        )
        * (
            storage_annuities.at["extra_injection_reservoir"]
            + fOM.at["extra_injection_reservoir"]
        )
        * nb_years
        + gene_per_tech.at["extra_injection_reservoir"]
        * 1000
        * vOM.at["extra_injection_reservoir"]
        + (capacity.at["metha_injection"] - existing_capacity.at["metha_injection"])
        * (annuities.at["metha_injection"] + fOM.at["metha_injection"])
        * nb_years
        + gene_per_tech.at["metha_injection"] * 1000 * vOM.at["metha_injection"]
    )

    costs_H2 = 0  # 1e6€
    for tech in model.H2_balance:
        if capacity.at[tech] != 0:
            gene_new_installation = (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )
            # /!\ gene is in TWh
            costs_H2 += (capacity.at[tech] - existing_capacity.at[tech]) * (
                annuities.at[tech] + fOM.at[tech]
            ) * nb_years + gene_new_installation * 1000 * vOM.at[tech]
            if tech in model.str:
                costs_H2 += (
                    (energy_capacity.at[tech] - existing_energy_capacity.at[tech])
                    * storage_annuities.at[tech]
                    * nb_years
                )

    costs_biogas = 0  # 1e6€
    for tech in model.biogas_balance:
        if capacity.at[tech] != 0:
            gene_new_installation = (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )
            # /!\ gene is in TWh
            costs_biogas += (capacity.at[tech] - existing_capacity.at[tech]) * (
                annuities.at[tech] + fOM.at[tech]
            ) * nb_years + gene_new_installation * 1000 * vOM.at[tech]
            if tech in model.str:
                costs_biogas += (
                    (energy_capacity.at[tech] - existing_energy_capacity.at[tech])
                    * storage_annuities.at[tech]
                    * nb_years
                )

    return costs_elec, costs_CH4, costs_H2, costs_biogas


def compute_lcoe(
    costs_elec,
    costs_CH4,
    costs_H2,
    G2P_bought,
    P2G_CH4_bought,
    P2G_H2_bought,
    sumgene_elec,
    sumgene_CH4,
    sumgene_H2,
):
    """Compute LCOE by using the costs of buying electricity / CH4 / H2. Parameters sumgene_elec, sumgene_CH4 and
    sumgene_H2 refer to the total production from each system (which can be used either to satisfy final demand, or for
     vector coupling)."""
    if sumgene_elec != 0:
        lcoe_elec = (costs_elec + G2P_bought) / sumgene_elec  # €/MWh
    else:
        lcoe_elec = "No electricity production"
    if sumgene_CH4 != 0:
        lcoe_CH4 = (costs_CH4 + P2G_CH4_bought) / sumgene_CH4  # €/MWh
    else:
        lcoe_CH4 = "No CH4 production"
    if sumgene_H2 != 0:
        lcoe_H2 = (costs_H2 + P2G_H2_bought) / sumgene_H2  # €/MWh
    else:
        lcoe_H2 = "No H2 production"
    return lcoe_elec, lcoe_CH4, lcoe_H2


def compute_lcoe_volumetric(
    model,
    gene_per_tech,
    conversion_efficiency,
    conversion_efficiency_biogas,
    costs_elec,
    costs_CH4,
    costs_H2,
    elec_demand_tot,
    CH4_demand_tot,
    H2_demand_tot,
    hourly_balance,
):
    """Computes a volumetric LCOE, where costs of each system (respectively, electricity, methane and hydrogen) are distributed across the different systems based on volumes (eg, volume of demand versus volume of gas used for the electricity system)."""
    gene_from_CH4_to_elec = sum(
        gene_per_tech.at[tech] / conversion_efficiency.at[tech]
        for tech in model.from_CH4_to_elec
    )
    gene_from_elec_to_elec = (
        gene_per_tech.at["methanization_coge"]
        / conversion_efficiency.at["methanization_coge"]
    )
    gene_from_H2_to_elec = sum(
        gene_per_tech.at[tech] / conversion_efficiency.at[tech]
        for tech in model.from_H2_to_elec
    )  # TWh
    gene_from_elec_to_CH4 = (
        sum(
            gene_per_tech.at[tech] / conversion_efficiency.at[tech]
            for tech in model.from_elec_to_CH4
        )
        + gene_per_tech.at["methanization_injection"]
        / conversion_efficiency.at["methanization_injection"]
    )
    gene_from_elec_to_H2 = sum(
        gene_per_tech.at[tech] / conversion_efficiency.at[tech]
        for tech in model.from_elec_to_H2
    )  # TWh

    costs_CH4_to_demand = (
        costs_CH4 * CH4_demand_tot / (CH4_demand_tot + gene_from_CH4_to_elec)
    )  # 1e6 €
    costs_CH4_to_elec = (
        costs_CH4 * gene_from_CH4_to_elec / (CH4_demand_tot + gene_from_CH4_to_elec)
    )
    costs_H2_to_demand = (
        costs_H2 * H2_demand_tot / (H2_demand_tot + gene_from_H2_to_elec)
    )
    costs_H2_to_elec = (
        costs_H2 * gene_from_H2_to_elec / (H2_demand_tot + gene_from_H2_to_elec)
    )
    costs_elec_to_demand = (
        costs_elec
        * elec_demand_tot
        / (
            elec_demand_tot
            + gene_from_elec_to_H2
            + gene_from_elec_to_CH4
            + gene_from_elec_to_elec
        )
    )
    costs_elec_to_CH4 = (
        costs_elec
        * gene_from_elec_to_CH4
        / (
            elec_demand_tot
            + gene_from_elec_to_H2
            + gene_from_elec_to_CH4
            + gene_from_elec_to_elec
        )
    )
    costs_elec_to_H2 = (
        costs_elec
        * gene_from_elec_to_H2
        / (
            elec_demand_tot
            + gene_from_elec_to_H2
            + gene_from_elec_to_CH4
            + gene_from_elec_to_elec
        )
    )
    costs_elec_to_elec = (
        costs_elec
        * gene_from_elec_to_elec
        / (
            elec_demand_tot
            + gene_from_elec_to_H2
            + gene_from_elec_to_CH4
            + gene_from_elec_to_elec
        )
    )

    if elec_demand_tot != 0:
        lcoe_elec_volume = (
            costs_CH4_to_elec
            + costs_H2_to_elec
            + costs_elec_to_demand
            + costs_elec_to_elec
        ) / elec_demand_tot  # € / MWh
    else:
        lcoe_elec_volume = "No exogenous demand for electricity"
    if CH4_demand_tot != 0:
        lcoe_CH4_volume = (
            costs_elec_to_CH4 + costs_CH4_to_demand
        ) / CH4_demand_tot  # € / MWh
    else:
        lcoe_CH4_volume = "No exogenous demand for CH4"
    if H2_demand_tot != 0:
        lcoe_H2_volume = (
            costs_elec_to_H2 + costs_H2_to_demand
        ) / H2_demand_tot  # € / MWh
    else:
        lcoe_H2_volume = "No exogenous demand for H2"
    return lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume


def compute_lcoe_value(
    model,
    hourly_balance,
    costs_elec,
    costs_CH4,
    costs_H2,
    elec_demand_tot,
    CH4_demand_tot,
    H2_demand_tot,
    elec_demand,
    CH4_demand,
    H2_demand,
    spot_price,
):
    gene_from_CH4_to_elec_value = sum(
        (
            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
            * spot_price.loc[:, "CH4"]
        ).sum()
        for tech in model.from_CH4_to_elec
    )
    gene_from_H2_to_elec_value = sum(
        (
            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
            * spot_price.loc[:, "H2"]
        ).sum()
        for tech in model.from_H2_to_elec
    )
    gene_from_elec_to_CH4_value = (
        sum(
            (
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "elec"]
            ).sum()
            for tech in model.from_elec_to_CH4
        )
        + (
            hourly_balance.loc[:, "methanization_injection_input"].reset_index(drop=True)
            * spot_price.loc[:, "elec"]
        ).sum()
    )
    gene_from_elec_to_H2_value = sum(
        (
            hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
            * spot_price.loc[:, "CH4"]
        ).sum()
        for tech in model.from_elec_to_H2
    )
    gene_from_elec_to_elec_value = (
        hourly_balance.loc[:, "methanization_coge_input"].reset_index(drop=True)
        * spot_price.loc[:, "elec"]
    ).sum()

    elec_demand_tot_value = (
        elec_demand.reset_index(drop=True) * spot_price.loc[:, "elec"]
    ).sum()
    CH4_demand_tot_value = (
        CH4_demand.reset_index(drop=True) * spot_price.loc[:, "CH4"]
    ).sum()
    H2_demand_tot_value = (
        H2_demand.reset_index(drop=True) * spot_price.loc[:, "H2"]
    ).sum()

    #   1e6 €
    costs_CH4_to_demand_value = (
        costs_CH4
        * CH4_demand_tot_value
        / (CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    )
    costs_CH4_to_elec_value = (
        costs_CH4
        * gene_from_CH4_to_elec_value
        / (CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    )
    costs_H2_to_demand_value = (
        costs_H2
        * H2_demand_tot_value
        / (H2_demand_tot_value + gene_from_H2_to_elec_value)
    )
    costs_H2_to_elec_value = (
        costs_H2
        * gene_from_H2_to_elec_value
        / (H2_demand_tot_value + gene_from_H2_to_elec_value)
    )
    costs_elec_to_demand_value = (
        costs_elec
        * elec_demand_tot_value
        / (
            elec_demand_tot_value
            + gene_from_elec_to_H2_value
            + gene_from_elec_to_CH4_value
            + gene_from_elec_to_elec_value
        )
    )
    costs_elec_to_CH4_value = (
        costs_elec
        * gene_from_elec_to_CH4_value
        / (
            elec_demand_tot_value
            + gene_from_elec_to_H2_value
            + gene_from_elec_to_CH4_value
            + gene_from_elec_to_elec_value
        )
    )
    costs_elec_to_H2_value = (
        costs_elec
        * gene_from_elec_to_H2_value
        / (
            elec_demand_tot_value
            + gene_from_elec_to_H2_value
            + gene_from_elec_to_CH4_value
            + gene_from_elec_to_elec_value
        )
    )
    costs_elec_to_elec_value = (
        costs_elec
        * gene_from_elec_to_elec_value
        / (
            elec_demand_tot_value
            + gene_from_elec_to_H2_value
            + gene_from_elec_to_CH4_value
            + gene_from_elec_to_elec_value
        )
    )

    if elec_demand_tot != 0:
        lcoe_elec_value = (
            costs_CH4_to_elec_value
            + costs_H2_to_elec_value
            + costs_elec_to_demand_value
            + costs_elec_to_elec_value
        ) / elec_demand_tot  # € / MWh
    else:
        lcoe_elec_value = "No exogenous demand for electricity"
    if CH4_demand_tot != 0:
        lcoe_CH4_value = (
            costs_elec_to_CH4_value + costs_CH4_to_demand_value
        ) / CH4_demand_tot  # € / MWh
    else:
        lcoe_CH4_value = "No exogenous demand for CH4"
    if H2_demand_tot != 0:
        lcoe_H2_value = (
            costs_elec_to_H2_value + costs_H2_to_demand_value
        ) / H2_demand_tot  # € / MWh
    else:
        lcoe_H2_value = "No exogenous demand for H2"
    return lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value


def transportation_distribution_cost(
    model, prediction_transport_and_distrib_annuity, capacity
):
    """Estimation of annualized transport and distribution cost, based on solar and onshore wind capacities."""
    solar_capacity = sum(capacity.at[tech] for tech in model.solar)
    onshore_capacity = capacity.at["onshore"]
    transport_and_distrib_annuity = (
        prediction_transport_and_distrib_annuity.at["intercept"]
        + prediction_transport_and_distrib_annuity.at["solar"] * solar_capacity
        + prediction_transport_and_distrib_annuity.at["onshore"] * onshore_capacity
    )  # 1e9 €/yr
    return transport_and_distrib_annuity


def extract_profit(
    model,
    hourly_balance,
    spot_price,
    vOM,
    new_annuities,
    new_str_annuities,
    frr_requirements,
    fcr_requirement,
    reserve_activation_rate,
    conversion_efficiency,
    capacity,
):
    """Extracts profit collected by each tech. This profit should be null except for rare goods (ie for techs with a limiting potential), so this output
    is used to identify issues"""
    profits = pd.Series(dtype=float)

    for tech in model.elec_balance:
        profits.at[tech] = (
            hourly_balance.loc[:, tech].reset_index(drop=True)
            * (spot_price.loc[:, "elec"] / 1000 - vOM.at[tech])
        ).sum() - new_annuities.at[tech]  # 1e6€/yr
        if tech in model.str:
            profits.at[tech] += (
                -(
                    hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                    * spot_price.loc[:, "elec"]
                    / 1000
                ).sum()
                - new_str_annuities.at[tech]
            )  # 1e6€/yr
        if tech in model.from_CH4_to_elec:
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "CH4"]
                / 1000
            ).sum()  # 1e6€/yr
        if tech in model.from_H2_to_elec:
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "H2"]
                / 1000
            ).sum()  # 1e6€/yr
        if tech in model.reserve:
            profits.at[tech] += (
                hourly_balance.loc[:, tech + "_frr"].reset_index(drop=True)
                * spot_price.loc[:, "frr"]
                / 1000
            ).sum()  # 1e6€/yr
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_frr"].reset_index(drop=True)
                * reserve_activation_rate.at["frr"]
                * vOM.at[tech]
            ).sum()  # 1e6€/yr
            profits.at[tech] += (
                hourly_balance.loc[:, tech + "_fcr"].reset_index(drop=True)
                * spot_price.loc[:, "fcr"]
                / 1000
            ).sum()  # 1e6€/yr
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_fcr"]
                * reserve_activation_rate.at["fcr"]
                * vOM.at[tech]
            ).sum()  # 1e6€/yr
            if tech in model.from_CH4_to_elec:
                profits.at[tech] += -(
                    hourly_balance.loc[:, tech + "_frr"].reset_index(drop=True)
                    / conversion_efficiency.at[tech]
                    * reserve_activation_rate.at["frr"]
                    * spot_price.loc[:, "CH4"]
                    / 1000
                ).sum()  # 1e6€/yr
                profits.at[tech] += -(
                    hourly_balance.loc[:, tech + "_fcr"].reset_index(drop=True)
                    / conversion_efficiency.at[tech]
                    * reserve_activation_rate.at["fcr"]
                    * spot_price.loc[:, "CH4"]
                    / 1000
                ).sum()  # 1e6€/yr
            if tech in model.from_H2_to_elec:
                profits.at[tech] += -(
                    hourly_balance.loc[:, tech + "_frr"].reset_index(drop=True)
                    / conversion_efficiency.at[tech]
                    * reserve_activation_rate.at["frr"]
                    * spot_price.loc[:, "H2"]
                    / 1000
                ).sum()  # 1e6€/yr
                profits.at[tech] += -(
                    hourly_balance.loc[:, tech + "_fcr"].reset_index(drop=True)
                    / conversion_efficiency.at[tech]
                    * reserve_activation_rate.at["fcr"]
                    * spot_price.loc[:, "H2"]
                    / 1000
                ).sum()  # 1e6€/yr
        if tech in model.vre:
            profits.at[tech] += (
                -frr_requirements.at[tech]
                * capacity.at[tech]
                * spot_price.loc[:, "frr"].sum()
                / 1000
            )  # 1e6€/yr

    for tech in model.CH4_balance:
        profits.at[tech] = (
            hourly_balance.loc[:, tech].reset_index(drop=True)
            * (spot_price.loc[:, "CH4"] / 1000 - vOM.at[tech])
        ).sum() - new_annuities.at[tech]  # 1e6€/yr
        if tech in model.str:
            profits.at[tech] += (
                -(
                    hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                    * spot_price.loc[:, "CH4"]
                    / 1000
                ).sum()
                - new_str_annuities.at[tech]
            )  # 1e6€/yr
        if tech in model.from_elec_to_CH4:
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "elec"]
                / 1000
            ).sum()  # 1e6€/yr
        if tech in model.from_biogas_to_CH4:
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "elec"]
                / 1000
            ).sum()  # 1e6€/yr

    for tech in model.H2_balance:
        profits.at[tech] = (
            hourly_balance.loc[:, tech].reset_index(drop=True)
            * (spot_price.loc[:, "H2"] / 1000 - vOM[tech])
        ).sum() - new_annuities.at[tech]  # 1e6€/yr
        if tech in model.str:
            profits.at[tech] += (
                -(
                    hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                    * spot_price.loc[:, "H2"]
                    / 1000
                ).sum()
                - new_str_annuities.at[tech]
            )  # 1e6€/yr
        if tech in model.from_elec_to_H2:
            profits.at[tech] += -(
                hourly_balance.loc[:, tech + "_input"].reset_index(drop=True)
                * spot_price.loc[:, "elec"]
                / 1000
            ).sum()  # 1e6€/yr

    profits.loc[np.isclose(profits, 0)] = 0
    return profits


def extract_summary(
    objective,
    model,
    elec_demand,
    H2_demand,
    H2_demand_is_profile,
    CH4_demand,
    CH4_demand_is_profile,
    capacity,
    existing_capacity,
    energy_capacity,
    existing_energy_capacity,
    annuities,
    storage_annuities,
    fOM,
    vOM,
    conversion_efficiency,
    conversion_efficiency_biogas,
    transportation_distribution_cost,
    scc,
    nb_years,
    carbon_constraint,
    carbon_content,
    hourly_balance,
    spot_price,
    capex,
):
    """This function compiles different general statistics of the electricity mix, including in particular LCOE."""
    summary = pd.Series(dtype=float)  # final dictionary for output

    summary.at["total system cost [1e9€]"] = objective

    # Total demands
    elec_demand_tot = elec_demand.sum() / 1000  # electricity demand in TWh
    summary.at["elec_demand_tot [TWh]"] = elec_demand_tot

    if H2_demand_is_profile:
        H2_demand_tot = H2_demand.sum() / 1000  # H2 demand in TWh
    else:
        H2_demand_tot = H2_demand / 1000  # H2 demand in TWh
        H2_demand = extract_hourly_demand(
            "H2", model, conversion_efficiency, hourly_balance
        )
    summary.at["H2_demand_tot [TWh]"] = H2_demand_tot

    if CH4_demand_is_profile:
        CH4_demand_tot = CH4_demand.sum() / 1000  # CH4 demand in TWh
    else:
        CH4_demand_tot = CH4_demand / 1000  # CH4 demand in TWh
        CH4_demand = extract_hourly_demand(
            "CH4", model, conversion_efficiency, hourly_balance
        )
    summary.at["CH4_demand_tot [TWh]"] = CH4_demand_tot

    # Prices weighted by hourly demand (ie average price paid by consumers)

    if elec_demand_tot != 0:
        weighted_elec_price_demand = (
            spot_price.loc[:, "elec"] * elec_demand.reset_index(drop=True)
        ).sum() / (elec_demand_tot * 1e3)  # €/MWh
    else:
        weighted_elec_price_demand = "No exogenous demand for electricity"
    summary.at["elec_price_weighted_by_demand [€/MWh]"] = weighted_elec_price_demand
    if CH4_demand_tot != 0:
        weighted_CH4_price_demand = (
            spot_price.loc[:, "CH4"] * CH4_demand.reset_index(drop=True)
        ).sum() / (CH4_demand_tot * 1e3)  # €/MWh
    else:
        weighted_CH4_price_demand = "No exogenous demand for CH4"
    summary.at["CH4_price_weighted_by_demand [€/MWh]"] = weighted_CH4_price_demand
    if H2_demand_tot != 0:
        weighted_H2_price_demand = (
            spot_price.loc[:, "H2"] * H2_demand.reset_index(drop=True)
        ).sum() / (H2_demand_tot * 1e3)  # €/MWh
    else:
        weighted_H2_price_demand = "No exogenous demand for H2"
    summary.at["H2_price_weighted_by_demand [€/MWh]"] = weighted_H2_price_demand

    summary.at["load_shifted [TWh]"] = (
        hourly_balance.loc[:, "load_shift_up"]
        + hourly_balance.loc[:, "load_shift_up_metha"]
    ).sum() / 1000
    summary.at["load_shifted_metha [TWh]"] = (
        hourly_balance.loc[:, "load_shift_up_metha"].sum() / 1000
    )
    summary.at["load_shifted [%]"] = (
        summary.at["load_shifted [TWh]"] / elec_demand_tot * 100
    )

    # Overall energy generated by the technology in TWh over total considered years
    gene_per_tech = pd.Series(dtype=float)
    for tech in model.all_tech:
        gene_per_tech.at[tech] = hourly_balance.loc[:, tech].sum() / 1000  # TWh
        if math.isclose(gene_per_tech.at[tech], 0, abs_tol=5e-04):
            gene_per_tech.at[tech] = 0

    input_per_tech = pd.Series(dtype=float)
    input2_per_tech = pd.Series(dtype=float)
    for tech in model.conversion_tech:
        input_per_tech.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000
        )  # TWh
        if math.isclose(input_per_tech.at[tech], 0, abs_tol=5e-04):
            input_per_tech.at[tech] = 0
    for tech in model.double_conversion_tech:
        input_per_tech.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000
        )  # TWh
        if math.isclose(input_per_tech.at[tech], 0, abs_tol=5e-04):
            input_per_tech.at[tech] = 0
    for tech in model.double_conversion_tech:
        input2_per_tech.at[tech] = (
            hourly_balance.loc[:, tech + "_input2"].sum() / 1000
        )  # TWh
        if math.isclose(input2_per_tech.at[tech], 0, abs_tol=5e-04):
            input2_per_tech.at[tech] = 0
    for tech in model.str:
        input_per_tech.at[tech] = (
            hourly_balance.loc[:, tech + "_input"].sum() / 1000
        )  # TWh
        if math.isclose(input_per_tech.at[tech], 0, abs_tol=5e-04):
            input_per_tech.at[tech] = 0

    primary_gene_elec = sum(gene_per_tech.at[tech] for tech in model.elec_primary_prod)
    summary.at["primary_gene_elec [TWh]"] = primary_gene_elec
    gene_elec = sum(gene_per_tech.at[tech] for tech in model.elec_prod)
    summary.at["gene_elec [TWh]"] = gene_elec
    gene_elec_new_installation = 0
    for tech in model.elec_balance:
        if capacity.at[tech] != 0:
            gene_elec_new_installation += (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )

    summary.at["gene_curtailed [TWh]"] = (
        hourly_balance.loc[:, "curtailment"].sum() / 1000
    )
    summary.at["gene_curtailed [%]"] = (
        summary.at["gene_curtailed [TWh]"] / summary.at["primary_gene_elec [TWh]"] * 100
    )

    primary_gene_CH4 = sum(gene_per_tech.at[tech] for tech in model.CH4_primary_prod)
    summary.at["primary_gene_CH4 [TWh]"] = primary_gene_CH4
    gene_CH4 = sum(gene_per_tech.at[tech] for tech in model.CH4_prod)
    summary.at["gene_CH4 [TWh]"] = gene_CH4
    gene_CH4_new_installation = 0
    for tech in model.CH4_balance:
        if capacity.at[tech] != 0:
            gene_CH4_new_installation += (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )

    gene_H2 = sum(gene_per_tech.at[tech] for tech in model.H2_balance)
    summary.at["gene_H2 [TWh]"] = gene_H2
    gene_H2_new_installation = 0
    for tech in model.H2_balance:
        if capacity.at[tech] != 0:
            gene_H2_new_installation += (
                gene_per_tech.at[tech]
                * (capacity.at[tech] - existing_capacity.at[tech])
                / capacity.at[tech]
            )

    # Monetary values of the energy converted between gas and electricity in 1e6€

    methanization_coge_cost = (
        spot_price.loc[:, "elec"]
        * hourly_balance.loc[:, "methanization_coge_input"].reset_index(drop=True)
    ).sum() / 1e3
    G2P_CH4_bought = (
        spot_price.loc[:, "CH4"]
        * sum(
            hourly_balance.loc[:, tech + "_input"] for tech in model.from_CH4_to_elec
        ).reset_index(drop=True)
    ).sum() / 1e3
    G2P_H2_bought = (
        spot_price.loc[:, "H2"]
        * sum(
            hourly_balance.loc[:, tech + "_input"] for tech in model.from_H2_to_elec
        ).reset_index(drop=True)
    ).sum() / 1e3
    G2P_bought = G2P_CH4_bought + G2P_H2_bought + methanization_coge_cost

    P2G_CH4_elec_cost = (
        spot_price.loc[:, "elec"]
        * sum(
            hourly_balance.loc[:, tech + "_input"]
            for tech in model.from_elec_to_CH4 | model.from_biogas_to_CH4
        ).reset_index(drop=True)
    ).sum() / 1e3
    P2G_CH4_bought = P2G_CH4_elec_cost

    P2G_H2_bought = (
        spot_price.loc[:, "elec"]
        * sum(
            hourly_balance.loc[:, tech + "_input"] for tech in model.from_elec_to_H2
        ).reset_index(drop=True)
    ).sum() / 1e3

    # We calculate the costs associated to functioning of each system (elec, CH4, gas)
    costs_elec, costs_CH4, costs_H2, costs_biogas = compute_costs(
        model,
        annuities,
        fOM,
        vOM,
        storage_annuities,
        gene_per_tech,
        capacity,
        existing_capacity,
        energy_capacity,
        existing_energy_capacity,
        nb_years,
        hourly_balance,
        capex,
    )  # 1e6 €

    # We first calculate LCOE by using total costs.
    lcoe_elec, lcoe_CH4, lcoe_H2 = compute_lcoe(
        costs_elec,
        costs_CH4,
        costs_H2,
        G2P_bought,
        P2G_CH4_bought,
        P2G_H2_bought,
        gene_elec_new_installation,
        gene_CH4_new_installation,
        gene_H2_new_installation,
    )
    summary.at["lcoe_elec [€/MWh]"] = lcoe_elec
    summary.at["lcoe_CH4 [€/MWh]"] = lcoe_CH4
    summary.at["lcoe_H2 [€/MWh]"] = lcoe_H2
    summary.at["Biogas costs [1e6€]"] = costs_biogas

    # We now redistribute the costs depending on which vector's demand it helps fulfill.
    # Option 1: based only on total volumes converted
    lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume = compute_lcoe_volumetric(
        model,
        gene_per_tech,
        conversion_efficiency,
        conversion_efficiency_biogas,
        costs_elec,
        costs_CH4,
        costs_H2,
        elec_demand_tot,
        CH4_demand_tot,
        H2_demand_tot,
        hourly_balance,
    )
    (
        summary.at["lcoe_elec_volume [€/MWh]"],
        summary.at["lcoe_CH4_volume [€/MWh]"],
        summary.at["lcoe_H2_volume [€/MWh]"],
    ) = lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume

    # Option 2: based on volumes converted at each hour and the associated spot prices

    lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value = compute_lcoe_value(
        model,
        hourly_balance,
        costs_elec,
        costs_CH4,
        costs_H2,
        elec_demand_tot,
        CH4_demand_tot,
        H2_demand_tot,
        elec_demand,
        CH4_demand,
        H2_demand,
        spot_price,
    )  # €/MWh
    (
        summary.at["lcoe_elec_value [€/MWh]"],
        summary.at["lcoe_CH4_value [€/MWh]"],
        summary.at["lcoe_H2_value [€/MWh]"],
    ) = lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value

    # We compute CH4 LCOE without SCC. This is needed for the calibration and estimation of gas prices.
    if not carbon_constraint:
        vOM_noSCC = vOM.copy()  # we remove the SCC in this vOM
        vOM_noSCC.at["natural_gas"] = update_vom_costs_scc(
            vOM_noSCC.at["natural_gas"],
            scc=(-scc),
            emission_rate=carbon_content.at["natural_gas"],
        )  # €/kWh
        vOM_noSCC.at["import_gaz"] = update_vom_costs_scc(
            vOM_noSCC.at["import_gaz"],
            scc=(-scc),
            emission_rate=carbon_content.at["import_gaz"],
        )  # €/kWh
        vOM_noSCC.at["coal"] = update_vom_costs_scc(
            vOM_noSCC.at["coal"], scc=(-scc), emission_rate=carbon_content.at["coal"]
        )  # €/kWh

        costs_elec_noSCC, costs_CH4_noSCC, costs_H2_noSCC, costs_biogas_noSCC = (
            compute_costs(
                model,
                annuities,
                fOM,
                vOM_noSCC,
                storage_annuities,
                gene_per_tech,
                capacity,
                existing_capacity,
                energy_capacity,
                existing_energy_capacity,
                nb_years,
                capex,
            )
        )  # 1e6 €

        lcoe_elec_noSCC, lcoe_CH4_noSCC, lcoe_H2_noSCC = compute_lcoe(
            costs_elec_noSCC,
            costs_CH4_noSCC,
            costs_H2_noSCC,
            G2P_bought,
            P2G_CH4_bought,
            P2G_H2_bought,
            gene_elec_new_installation,
            gene_CH4_new_installation,
            gene_H2_new_installation,
        )

        lcoe_elec_volume_noSCC, lcoe_CH4_volume_noSCC, lcoe_H2_volume_noSCC = (
            compute_lcoe_volumetric(
                model,
                gene_per_tech,
                conversion_efficiency,
                conversion_efficiency_biogas,
                costs_elec_noSCC,
                costs_CH4_noSCC,
                costs_H2_noSCC,
                elec_demand_tot,
                CH4_demand_tot,
                H2_demand_tot,
                hourly_balance,
            )
        )

        summary.at["lcoe_CH4_noSCC [€/MWh]"] = lcoe_CH4_noSCC
        summary.at["lcoe_CH4_volume_noSCC [€/MWh]"] = lcoe_CH4_volume_noSCC

    # Estimation of transportation and distribution costs
    transport_and_distrib_lcoe = (
        transportation_distribution_cost * 1000 * nb_years
    ) / elec_demand_tot  # € / yr / MWh

    summary.at["transport_and_distrib_lcoe [€/yr/MWh]"] = transport_and_distrib_lcoe

    return summary, gene_per_tech, input_per_tech, input2_per_tech


"""[[[ Functions used to plot outputs ]]]"""


def plot_load_shift_week(model, hourly_balance, hour, lang="EN"):
    fig, ax = plt.subplots(figsize=(16, 10))

    net_shift = (
        hourly_balance.loc[:, "load_shift_up"] - hourly_balance.loc[:, "load_shift_down"]
    )
    net_shift_metha = (
        hourly_balance.loc[:, "load_shift_up_metha"]
        - hourly_balance.loc[:, "load_shift_down_metha"]
    )
    positive_net_shift = (
        net_shift.mask(net_shift < 0, 0)
        .rename("positive_net_shift")
        .iloc[hour : (hour + 7 * 24)]
    )
    negative_net_shift = (
        -net_shift.mask(net_shift > 0, 0)
        .rename("negative_net_shift")
        .iloc[hour : (hour + 7 * 24)]
    )
    positive_net_shift_metha = (
        net_shift_metha.mask(net_shift_metha < 0, 0)
        .rename("positive_net_shift_metha")
        .iloc[hour : (hour + 7 * 24)]
    )
    negative_net_shift_metha = (
        -net_shift_metha.mask(net_shift_metha > 0, 0)
        .rename("negative_net_shift_metha")
        .iloc[hour : (hour + 7 * 24)]
    )
    elec_demand = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand")
    ]
    elec_demand_w_shift = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand_w/_shift")
    ]
    nonoperable_prod = (
        sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.solar
        )
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("biomass_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("geothermal_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("waste")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("ocgt_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("river")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("marine")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("onshore")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_ground")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_float")
        ]
    )

    time = pd.to_datetime(
        elec_demand.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_demand.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    if lang == "EN":
        ax.stackplot(
            time,
            nonoperable_prod,
            lw=0,
            color="#d66b0d60",
            labels=["Non-operable production"],
        )

        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift,
            positive_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "orange", "gold"],
            labels=[
                None,
                "Positive net shift",
                "Positive anaerobic digestion process net shift",
            ],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift,
            negative_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "steelblue", "cornflowerblue"],
            labels=[
                None,
                "Negative net shift",
                "Negative anaerobic digestion process net shift",
            ],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demand before load shift")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demand after load shift",
        )

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Electricity demand load shift over one week",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Electricity demand [GW]", fontsize=13)

    if lang == "FR":
        ax.stackplot(
            time,
            nonoperable_prod,
            lw=0,
            color="#d66b0d60",
            labels=["Production non pilotable"],
        )

        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "orange"],
            labels=[None, "Décalage net positif"],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "steelblue"],
            labels=[None, "Décalage net négatif"],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demande avant pilotage")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demande après pilotage",
        )

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Pilotage de la demande sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Demande d'électricité [GW]", fontsize=13)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.grid(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_load_shift_metha_week(model, hourly_balance, hour, lang="EN"):
    fig, ax = plt.subplots(figsize=(16, 10))

    net_shift = (
        hourly_balance.loc[:, "load_shift_up"] - hourly_balance.loc[:, "load_shift_down"]
    )
    net_shift_metha = (
        hourly_balance.loc[:, "load_shift_up_metha"]
        - hourly_balance.loc[:, "load_shift_down_metha"]
    )
    positive_net_shift_metha = (
        net_shift_metha.mask(net_shift_metha < 0, 0)
        .rename("positive_net_shift_metha")
        .iloc[hour : (hour + 7 * 24)]
    )
    negative_net_shift_metha = (
        -net_shift_metha.mask(net_shift_metha > 0, 0)
        .rename("negative_net_shift_metha")
        .iloc[hour : (hour + 7 * 24)]
    )
    elec_demand = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand")
    ]
    elec_demand_w_shift = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand_w/_shift")
    ]
    nonoperable_prod = (
        sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.solar
        )
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("biomass_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("geothermal_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("waste")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("ocgt_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("river")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("marine")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("onshore")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_ground")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_float")
        ]
    )

    time = pd.to_datetime(
        elec_demand.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_demand.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    if lang == "EN":
        ax.stackplot(
            time,
            nonoperable_prod,
            lw=0,
            color="#d66b0d60",
            labels=["Non-operable production"],
        )

        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "gold"],
            labels=[None, "Positive anaerobic digestion process net shift"],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "cornflowerblue"],
            labels=[None, "Negative anaerobic digestion process net shift"],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demand before load shift")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demand after load shift",
        )

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Electricity demand load shift over one week",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Electricity demand [GW]", fontsize=13)

    if lang == "FR":
        ax.stackplot(
            time,
            nonoperable_prod,
            lw=0,
            color="#d66b0d60",
            labels=["Production non pilotable"],
        )

        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "orange"],
            labels=[None, "Décalage net positif"],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "steelblue"],
            labels=[None, "Décalage net négatif"],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demande avant pilotage")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demande après pilotage",
        )

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Pilotage de la demande sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Demande d'électricité [GW]", fontsize=13)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.grid(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_just_load_shift_metha_week(model, hourly_balance, hour, lang="EN"):
    fig, ax = plt.subplots(figsize=(16, 10))

    net_shift = (
        hourly_balance.loc[:, "load_shift_up"] - hourly_balance.loc[:, "load_shift_down"]
    )
    net_shift_metha = (
        hourly_balance.loc[:, "load_shift_up_metha"]
        - hourly_balance.loc[:, "load_shift_down_metha"]
    )
    positive_net_shift_metha = (
        net_shift_metha.mask(net_shift_metha < 0, 0)
        .rename("positive_net_shift_metha")
        .iloc[hour : (hour + 4 * 24)]
    )
    negative_net_shift_metha = (
        -net_shift_metha.mask(net_shift_metha > 0, 0)
        .rename("negative_net_shift_metha")
        .iloc[hour : (hour + 4 * 24)]
    )
    elec_demand = hourly_balance.iloc[
        hour : (hour + 4 * 24), hourly_balance.columns.get_loc("elec_demand")
    ]
    elec_demand_w_shift = hourly_balance.iloc[
        hour : (hour + 4 * 24), hourly_balance.columns.get_loc("elec_demand_w/_shift")
    ]

    time = pd.to_datetime(
        elec_demand.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_demand.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    if lang == "EN":
        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "gold"],
            labels=[None, "Positive anaerobic digestion process net shift"],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift_metha,
            lw=0,
            colors=[(0, 0, 0, 0), "cornflowerblue"],
            labels=[None, "Negative anaerobic digestion process net shift"],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demand before load shift")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demand after load shift",
        )
        ax.set_ylim(40, 100)

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Electricity demand load shift over one week",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Electricity demand [GW]", fontsize=13)

    if lang == "FR":
        ax.stackplot(
            time,
            elec_demand,
            positive_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "orange"],
            labels=[None, "Décalage net positif"],
        )
        ax.stackplot(
            time,
            elec_demand_w_shift,
            negative_net_shift,
            lw=0,
            colors=[(0, 0, 0, 0), "steelblue"],
            labels=[None, "Décalage net négatif"],
        )

        ax.plot(time, elec_demand, lw=1.8, color="red", label="Demande avant pilotage")
        ax.plot(
            time,
            elec_demand_w_shift,
            lw=1.8,
            color="black",
            label="Demande après pilotage",
        )
        ax.set_ylim(40, 100)

        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.38, +1.06),
            columnspacing=0.5,
        )
        ax.text(
            x=0.113,
            y=0.93,
            s="Pilotage de la demande sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        ax.set_ylabel("Demande d'électricité [GW]", fontsize=13)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax.yaxis.grid(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_elec_balance_week(
    model, hourly_balance, installed_power, hour, include_str=False, lang="EN"
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_demand = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand_w/_shift")
    ]
    time = pd.to_datetime(
        elec_demand.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_demand.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    elec_prod = pd.DataFrame(dtype=float)

    if lang == "EN":
        elec_prod["Cogeneration"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("biomass_coge")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("geothermal_coge")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("waste")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("ocgt_coge")
            ]
        )
        elec_prod["Cogeneration by methanization"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanization_coge")
        ]
        elec_prod["Hydropower - Other"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("river")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("marine")
            ]
        )
        elec_prod["Wind - onshore"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("onshore")
        ]
        elec_prod["Wind - offshore"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_ground")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_float")
            ]
        )
        elec_prod["Hydropower - Dams"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("lake")
        ]
        elec_prod["Solar"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.solar
        )
        if installed_power.at["nuclear"] > 0:
            elec_prod["Nuclear"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("nuclear")
            ]
        if installed_power.at["coal"] > 0:
            elec_prod["Coal"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("coal")
            ]
        elec_prod["PHS"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs")
        ]
        elec_prod["Batteries"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.battery
        )
        elec_prod["CH4 turbines"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_CH4_to_elec
        )
        elec_prod["H2 turbines"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_H2_to_elec
        )
        elec_prod["Missing storage"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy")
        ]
    if lang == "FR":
        elec_prod["Cogénération"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("biomass_coge")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("geothermal_coge")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("waste")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("ocgt_coge")
            ]
        )
        elec_prod["Cogénération par méthanisation"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanization_coge")
        ]
        elec_prod["Hydraulique - Autres"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("river")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("marine")
            ]
        )
        elec_prod["Eolien - Terrestre"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("onshore")
        ]
        elec_prod["Eolien - En mer"] = (
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_ground")
            ]
            + hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_float")
            ]
        )
        elec_prod["Hydraulique - Barrages"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("lake")
        ]
        elec_prod["Photovoltaïque"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.solar
        )
        if installed_power.at["nuclear"] > 0:
            elec_prod["Nucléaire"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("nuclear")
            ]
        if installed_power.at["coal"] > 0:
            elec_prod["Charbon"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("coal")
            ]
        elec_prod["STEP"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs")
        ]
        elec_prod["Batteries"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.battery
        )
        elec_prod["Turbines CH4"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_CH4_to_elec
        )
        elec_prod["Turbines H2"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_H2_to_elec
        )
        elec_prod["Stockage manquant"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy")
        ]

    colors_prod = ["#156956", "#2672b0", "#74cb2e", "#72cbb7", "#2672b0", "#d66b0d"]
    if installed_power.at["nuclear"] > 0:
        colors_prod.append("#e4a701")
    if installed_power.at["coal"] > 0:
        colors_prod.append("#a68832")
    colors_prod.extend(
        ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575", "#2ca25f", "#ffbf00"]
    )

    handles_prod = ax.stackplot(
        time, elec_prod.T, labels=elec_prod.columns, colors=colors_prod
    )

    elec_str = pd.DataFrame(dtype=float)

    if include_str:
        if lang == "EN":
            elec_str.loc[:, "PHS"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs_input")
            ]
            elec_str.loc[:, "Batteries"] = (
                -hourly_balance.iloc[
                    hour : (hour + 7 * 24),
                    hourly_balance.columns.get_loc("battery_1h_input"),
                ]
                + -hourly_balance.iloc[
                    hour : (hour + 7 * 24),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )
            elec_str.loc[:, "Methanation"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanation_input"),
            ]
            elec_str.loc[:, "Electrolysis"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("electrolysis_input"),
            ]
            elec_str.loc[:, "Missing storage"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy_input")
            ]
            elec_str.loc[
                :, "Cogeneration input by methanization"
            ] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanization_coge_input"),
            ]
            elec_str.loc[:, "Injection input by methanization"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanization_injection_input"),
            ]

        if lang == "FR":
            elec_str.loc[:, "STEP"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs_input")
            ]
            elec_str.loc[:, "Batteries"] = (
                -hourly_balance.iloc[
                    hour : (hour + 7 * 24),
                    hourly_balance.columns.get_loc("battery_1h_input"),
                ]
                + -hourly_balance.iloc[
                    hour : (hour + 7 * 24),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )
            elec_str.loc[:, "Méthanation"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanation_input"),
            ]
            elec_str.loc[:, "Electrolyse"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("electrolysis_input"),
            ]
            elec_str.loc[:, "Stockage manquant"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy_input")
            ]
            elec_str.loc[:, "Cogénération par méthanisation"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanization_coge_input"),
            ]
            elec_str.loc[:, "Injection par méthanisation"] = -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("methanization_injection_input"),
            ]

        colors_str = ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575", "#e4a701"]
        colors_str.extend(
            ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575", "#2ca25f", "#ffbf00"]
        )

        handles_str = ax.stackplot(
            time, elec_str.T, labels=elec_str.columns, colors=colors_str
        )
    else:
        handles_str = []

    if lang == "EN":
        handle_demand = ax.plot(
            time, elec_demand, lw=2, color="black", label="Electricity demand"
        )
        ax.set_ylabel("Electricity Production and Usage [GW]", fontsize=12, labelpad=10)
        ax.text(
            x=0.05,
            y=1.02,
            s="Electricity Balance Over One Week",
            transform=fig.transFigure,
            ha="left",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            x=0.05,
            y=1.0,
            s="Electricity production by source, including storage output, and demand over one week",
            transform=fig.transFigure,
            ha="left",
            fontsize=12,
            alpha=0.8,
        )
        leg_str = ax.legend(
            handles=handles_str + handle_demand,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            title="Usage",
            title_fontsize="large",
            bbox_to_anchor=(0.21, +1.08),
            frameon=False,
            columnspacing=0.5,
        )
        leg_prod = ax.legend(
            handles=handles_prod,
            loc="upper center",
            ncol=len(elec_prod.columns),
            title="Production",
            title_fontsize="large",
            bbox_to_anchor=(0.5, +1.14),
            frameon=False,
            columnspacing=0.5,
        )

    if lang == "FR":
        handle_demand = ax.plot(
            time, elec_demand, lw=2, color="black", label="Demande d'électricité"
        )
        ax.set_ylabel(
            "Production et utilisation de l'électricité [GW]", fontsize=12, labelpad=10
        )
        ax.text(
            x=0.05,
            y=1.02,
            s="Equilibre du réseau électrique sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            x=0.05,
            y=1.0,
            s="Demande et production électrique par source, déstockage inclu, sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=12,
            alpha=0.8,
        )
        leg_str = ax.legend(
            handles=handles_str + handle_demand,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            title="Usage",
            title_fontsize="large",
            bbox_to_anchor=(0.241, +1.08),
            frameon=False,
            columnspacing=0.5,
        )
        leg_prod = ax.legend(
            handles=handles_prod,
            loc="upper center",
            ncol=len(elec_prod.columns) - 2,
            title="Production",
            title_fontsize="large",
            bbox_to_anchor=(0.45, +1.165),
            frameon=False,
            columnspacing=0.5,
        )

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_elec_residual_balance_week(
    model, hourly_balance, installed_power, hour, lang="EN"
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_demand = hourly_balance.iloc[
        hour : (hour + 7 * 24), hourly_balance.columns.get_loc("elec_demand_w/_shift")
    ]
    time = pd.to_datetime(
        elec_demand.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_demand.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    nonoperable_prod = (
        sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.solar
        )
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("biomass_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("geothermal_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("waste")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("ocgt_coge")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("river")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("marine")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("onshore")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_ground")
        ]
        + hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("offshore_float")
        ]
    )
    residual_elec_demand = elec_demand - nonoperable_prod

    operable_prod = pd.DataFrame(dtype=float)
    if lang == "EN":
        operable_prod.loc[:, "Hydropower - Dams"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("lake")
        ]
        if installed_power.at["nuclear"] > 0:
            operable_prod.loc[:, "Nuclear"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("nuclear")
            ]
        if installed_power.at["coal"] > 0:
            operable_prod.loc[:, "Coal"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("coal")
            ]
        operable_prod.loc[:, "PHS"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs")
        ]
        operable_prod.loc[:, "Batteries"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.battery
        )
        operable_prod.loc[:, "CH4 turbines"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_CH4_to_elec
        )
        operable_prod.loc[:, "H2 turbines"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_H2_to_elec
        )
        operable_prod.loc[:, "Missing storage"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy")
        ]
        operable_prod.loc[:, "Cogeneration by methanization"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanization_coge")
        ]
    if lang == "FR":
        operable_prod.loc[:, "Hydraulique - Barrages"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("lake")
        ]
        if installed_power.at["nuclear"] > 0:
            operable_prod.loc[:, "Nucléaire"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("nuclear")
            ]
        if installed_power.at["coal"] > 0:
            operable_prod.loc[:, "Charbon"] = hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc("coal")
            ]
        operable_prod.loc[:, "STEP"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs")
        ]
        operable_prod.loc[:, "Batteries"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.battery
        )
        operable_prod.loc[:, "Turbines CH4"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_CH4_to_elec
        )
        operable_prod.loc[:, "Turbines H2"] = sum(
            hourly_balance.iloc[
                hour : (hour + 7 * 24), hourly_balance.columns.get_loc(tech)
            ]
            for tech in model.from_H2_to_elec
        )
        operable_prod.loc[:, "Stockage manquant"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy")
        ]
        operable_prod.loc[:, "Cogénération par méthanisation"] = hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanization_coge")
        ]

    colors_prod = ["#2672b0"]
    if installed_power.at["nuclear"] > 0:
        colors_prod.append("#e4a701")
    if installed_power.at["coal"] > 0:
        colors_prod.append("#a68832")
    colors_prod.extend(
        ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575", "#2ca25f", "#ffbf00"]
    )

    handles_prod = ax.stackplot(
        time, operable_prod.T, labels=operable_prod.columns, colors=colors_prod
    )

    elec_str = pd.DataFrame(dtype=float)
    if lang == "EN":
        elec_str.loc[:, "PHS"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs_input")
        ]
        elec_str.loc[:, "Batteries"] = (
            -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("battery_1h_input"),
            ]
            + -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("battery_4h_input"),
            ]
        )
        elec_str.loc[:, "Methanation"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanation_input")
        ]
        elec_str.loc[:, "Electrolysis"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("electrolysis_input")
        ]
        elec_str.loc[:, "Missing storage"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy_input")
        ]
        elec_str.loc[:, "Cogeneration by methanization"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24),
            hourly_balance.columns.get_loc("methanization_coge_input"),
        ]
        elec_str.loc[:, "Injection by methanization"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24),
            hourly_balance.columns.get_loc("methanization_injection_input"),
        ]

    if lang == "FR":
        elec_str.loc[:, "STEP"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("phs_input")
        ]
        elec_str.loc[:, "Batteries"] = (
            -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("battery_1h_input"),
            ]
            + -hourly_balance.iloc[
                hour : (hour + 7 * 24),
                hourly_balance.columns.get_loc("battery_4h_input"),
            ]
        )
        elec_str.loc[:, "Méthanation"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("methanation_input")
        ]
        elec_str.loc[:, "Electrolyse"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("electrolysis_input")
        ]
        elec_str.loc[:, "Stockage manquant"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24), hourly_balance.columns.get_loc("str_dummy_input")
        ]
        elec_str.loc[:, "Cogénération par méthanisation"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24),
            hourly_balance.columns.get_loc("methanization_coge_input"),
        ]
        elec_str.loc[:, "Injection par méthanisation"] = -hourly_balance.iloc[
            hour : (hour + 7 * 24),
            hourly_balance.columns.get_loc("methanization_injection_input"),
        ]

    colors_str = ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575"]
    colors_str.extend(
        ["#0e4269", "#80549f", "#f20809", "#f252c0", "#757575", "#2ca25f", "#ffbf00"]
    )

    handles_str = ax.stackplot(
        time, elec_str.T, labels=elec_str.columns, colors=colors_str
    )

    if lang == "EN":
        handle_demand = ax.plot(
            time,
            residual_elec_demand,
            lw=2,
            color="black",
            label="Residual electricity demand",
        )
        ax.set_ylabel("Electricity Production and Usage [GW]", fontsize=12, labelpad=10)
        ax.text(
            x=0.05,
            y=1.02,
            s="Residual Electricity Balance Over One Week",
            transform=fig.transFigure,
            ha="left",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            x=0.05,
            y=1.0,
            s="Operable electricity production by source and residual demand over one week",
            transform=fig.transFigure,
            ha="left",
            fontsize=12,
            alpha=0.8,
        )
        leg_str = ax.legend(
            handles=handles_str + handle_demand,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            title="Usage",
            title_fontsize="large",
            bbox_to_anchor=(0.24, +1.08),
            frameon=False,
            columnspacing=0.5,
        )
        leg_prod = ax.legend(
            handles=handles_prod,
            loc="upper center",
            ncol=len(operable_prod.columns),
            title="Operable production",
            title_fontsize="large",
            bbox_to_anchor=(0.22, +1.14),
            frameon=False,
            columnspacing=0.5,
        )
    if lang == "FR":
        handle_demand = ax.plot(
            time,
            residual_elec_demand,
            lw=2,
            color="black",
            label="Demande d'électricité résiduelle",
        )
        ax.set_ylabel(
            "Production et utilisation de l'électricité [GW]", fontsize=12, labelpad=10
        )
        ax.text(
            x=0.05,
            y=1.02,
            s="Equilibre résiduel du réseau électrique sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            x=0.05,
            y=1.0,
            s="Demande résiduelle et production électrique pilotable par source sur une semaine",
            transform=fig.transFigure,
            ha="left",
            fontsize=12,
            alpha=0.8,
        )
        leg_str = ax.legend(
            handles=handles_str + handle_demand,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            title="Usage",
            title_fontsize="large",
            bbox_to_anchor=(0.272, +1.08),
            frameon=False,
            columnspacing=0.5,
        )
        leg_prod = ax.legend(
            handles=handles_prod,
            loc="upper center",
            ncol=len(operable_prod.columns),
            title="Operable production",
            title_fontsize="large",
            bbox_to_anchor=(0.25, +1.14),
            frameon=False,
            columnspacing=0.5,
        )

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_storage_state_year(
    hourly_balance,
    hour,
    nb_years=1,
    select_tech=[
        "ch4_reservoir",
        "h2_saltcavern",
        "lake",
        "phs",
        "battery",
        "str_dummy",
        "coge_reservoir",
        "injection_reservoir",
    ],
    lang="EN",
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_str = pd.DataFrame(dtype=float)
    if lang == "EN":
        if "ch4_reservoir" in select_tech:
            elec_str.loc[:, "Methane"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("ch4_reservoir_state_charge"),
            ]
        if "h2_saltcavern" in select_tech:
            elec_str.loc[:, "Hydrogen"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("h2_saltcavern_state_charge"),
            ]
        if "lake" in select_tech:
            elec_str.loc[:, "Hydropower"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("lake_state_charge"),
            ]
        if "phs" in select_tech:
            elec_str.loc[:, "PHS"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("phs_state_charge"),
            ]
        if "battery" in select_tech:
            elec_str.loc[:, "Batteries"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_1h_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )
        if "str_dummy" in select_tech:
            elec_str.loc[:, "Missing storage"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("str_dummy_state_charge"),
            ]
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogas for cogeneration"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogas for injection"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    if lang == "FR":
        if "ch4_reservoir" in select_tech:
            elec_str.loc[:, "Méthane"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("ch4_reservoir_state_charge"),
            ]
        if "h2_saltcavern" in select_tech:
            elec_str.loc[:, "Hydrogène"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("h2_saltcavern_state_charge"),
            ]
        if "lake" in select_tech:
            elec_str.loc[:, "Hydraulique"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("lake_state_charge"),
            ]
        if "phs" in select_tech:
            elec_str.loc[:, "STEP"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("phs_state_charge"),
            ]
        if "battery" in select_tech:
            elec_str.loc[:, "Batteries"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_1h_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )
        if "str_dummy" in select_tech:
            elec_str.loc[:, "Stockage manquant"] = hourly_balance.iloc[
                hour : (hour + 8760 * nb_years),
                hourly_balance.columns.get_loc("str_dummy_state_charge"),
            ]
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour cogénération"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour injection"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    time = pd.to_datetime(
        elec_str.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_str.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    colors_str = []
    if "ch4_reservoir" in select_tech:
        colors_str.append("#f20809")
    if "h2_saltcavern" in select_tech:
        colors_str.append("#f252c0")
    if "lake" in select_tech:
        colors_str.append("#2672b0")
    if "phs" in select_tech:
        colors_str.append("#0e4269")
    if "battery" in select_tech:
        colors_str.append("#80549f")
    if "str_dummy" in select_tech:
        colors_str.append("#757575")
    if "coge_reservoir" in select_tech:
        colors_str.append("#f2b705")
    if "injection_reservoir" in select_tech:
        colors_str.append("#1abc9c")

    handles_str = ax.stackplot(
        time, elec_str.T, labels=elec_str.columns, colors=colors_str
    )

    if lang == "EN":
        ax.set_ylabel("State of charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s=f" Storage State-of-Charge Over the Year",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.19, +1.06),
            frameon=False,
            columnspacing=0.5,
        )
    if lang == "FR":
        ax.set_ylabel("Niveau de charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s="Niveau de charge sur un an",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.22, +1.06),
            frameon=False,
            columnspacing=0.5,
        )

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.set_ylim([elec_str.sum(axis=1).min(), None])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_storage_biogas_state_year(
    hourly_balance,
    spot_price,
    energy_capacity,
    hour,
    nb_years=1,
    select_tech=["coge_reservoir", "injection_reservoir"],
    lang="EN",
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_str = pd.DataFrame(dtype=float)
    if lang == "EN":
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogas for cogeneration"] = (
                hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogas for injection"] = (
                hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    if lang == "FR":
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour cogénération"] = (
                hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour injection"] = (
                hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 365 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    time = pd.to_datetime(
        elec_str.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_str.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    colors_str = []
    if "coge_reservoir" in select_tech:
        colors_str.append("#f2b705")
    if "injection_reservoir" in select_tech:
        colors_str.append("#1abc9c")

    handles_str = ax.stackplot(
        time, elec_str.T, labels=elec_str.columns, colors=colors_str
    )

    if lang == "EN":
        ax.set_ylabel("State of charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s=f" Storage State-of-Charge Over the Year",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.19, +1.06),
            frameon=False,
            columnspacing=0.5,
        )
    if lang == "FR":
        ax.set_ylabel("Niveau de charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s="Niveau de charge sur un an",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.22, +1.06),
            frameon=False,
            columnspacing=0.5,
        )

    max_storage = energy_capacity.at["injection_reservoir"]
    meth_output = hourly_balance.iloc[
        hour : (hour + 365 * 24), hourly_balance.columns.get_loc("metha_injection")
    ].reset_index(drop=True)
    ax.axhline(
        y=max_storage,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Maximum biogas injection storage capacity",
    )
    color3 = "#2ca02c"
    ax.plot(
        time,
        meth_output,
        color=color3,
        linewidth=1.5,
        linestyle="--",
        label="Methanization total output",
    )

    prix_elec = spot_price.loc[:, "elec"]
    # Crée un deuxième axe Y pour le prix de l'électricité
    ax2 = ax.twinx()
    ax2.plot(
        time,
        prix_elec.iloc[hour : (hour + 365 * 24)],
        color="black",
        linewidth=1.5,
        label="Electricity price",
    )

    ax2.set_ylabel("Electricity price [€/MWh]", fontsize=12, labelpad=10)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 250)

    # Ajouter la légende du prix (à droite)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.set_ylim([elec_str.sum(axis=1).min(), None])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_storage_biogas_state_year_monthly(
    hourly_balance,
    spot_price,
    energy_capacity,
    hour,
    nb_years=1,
    select_tech=["coge_reservoir", "injection_reservoir"],
    lang="EN",
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_str = pd.DataFrame(dtype=float)
    if lang == "EN":
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogas for cogeneration"] = (
                hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogas storage for injection state of charge"] = (
                hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    if lang == "FR":
        if "coge_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour cogénération"] = (
                hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("coge_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("extra_coge_reservoir_state_charge"),
                ]
            )
        if "injection_reservoir" in select_tech:
            elec_str.loc[:, "Biogaz pour injection"] = (
                hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc("injection_reservoir_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 30 * 24),
                    hourly_balance.columns.get_loc(
                        "extra_injection_reservoir_state_charge"
                    ),
                ]
            )

    time = pd.to_datetime(
        elec_str.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_str.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    colors_str = []
    if "coge_reservoir" in select_tech:
        colors_str.append("#f2b705")
    if "injection_reservoir" in select_tech:
        colors_str.append("#1abc9c")

    handles_str = ax.stackplot(
        time, elec_str.T, labels=elec_str.columns, colors=colors_str
    )

    if lang == "EN":
        ax.set_ylabel("State of charge [GWh]", fontsize=12, labelpad=10)
        #ax.text(
          #  x=0.06,
          #  y=0.93,
           # s=f" Storage State-of-Charge Over the Year",
          #  transform=fig.transFigure,
          #  ha="left",
          #  fontsize=16,
           # weight="bold",
       # )
        #leg_str = ax.legend(
         #   handles=handles_str,
          #  loc="upper center",
           # ncol=len(elec_str.columns) + 1,
            #bbox_to_anchor=(0.19, +1.06),
            #frameon=False,
            #columnspacing=0.5,
        #)
    if lang == "FR":
        ax.set_ylabel("Niveau de charge [GWh]", fontsize=12, labelpad=10)
        #ax.text(
          #  x=0.06,
           # y=0.93,
            #s="Niveau de charge sur un an",
            #transform=fig.transFigure,
            #ha="left",
            #fontsize=16,
            #weight="bold",
        #)
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.22, +1.06),
            frameon=False,
            columnspacing=0.5,
        )

    max_storage = energy_capacity.at["injection_reservoir"]
    meth_inj_output = hourly_balance.iloc[
        hour : (hour + 30 * 24),
        hourly_balance.columns.get_loc("methanization_injection"),
    ].reset_index(drop=True)
    color1 = "#1f77b4"
    ax.plot(
        time,
        meth_inj_output,
        color=color1,
        linewidth=2,
        label="Injection from anaerobic digestion output",
    )
    meth_output = hourly_balance.iloc[
        hour : (hour + 30 * 24), hourly_balance.columns.get_loc("metha_injection")
    ].reset_index(drop=True)
    ax.axhline(
        y=max_storage,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Maximum biogas storage for injection capacity",
    )
    color3 = "#2ca02c"
    ax.plot(
        time,
        meth_output,
        color=color3,
        linewidth=1.5,
        linestyle="--",
        label="Biogas production",
    )

    prix_elec = spot_price.loc[:, "elec"]
    # Crée un deuxième axe Y pour le prix de l'électricité
    ax2 = ax.twinx()
    ax2.plot(
        time,
        prix_elec.iloc[hour : (hour + 30 * 24)],
        color="black",
        linewidth=1.5,
        label="Electricity price",
    )

    # -------------------------------
    # LÉGENDE UNIQUE (PROPRE)
    # -------------------------------
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
     bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    ax2.set_ylabel("Electricity price [€/MWh]", fontsize=12, labelpad=10)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 250)

    # Ajouter la légende du prix (à droite)
    #lines_1, labels_1 = ax.get_legend_handles_labels()
    #lines_2, labels_2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)

    #ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_spot_and_meth_output(hourly_balance, spot_price, hour=0, lang="EN"):
    # Créer un axe temporel de 8760 heures = 1 an
    time = pd.date_range(start="2050-01-01 00:00:00", periods=365 * 24, freq="H")

    # Extraire les signaux
    biogas_injection_output = hourly_balance.iloc[
        hour : (hour + 365 * 24), hourly_balance.columns.get_loc("metha_injection")
    ].reset_index(drop=True)
    biogas_coge_output = hourly_balance.iloc[
        hour : (hour + 365 * 24), hourly_balance.columns.get_loc("metha_coge")
    ].reset_index(drop=True)
    meth_inj_output = hourly_balance.iloc[
        hour : (hour + 365 * 24),
        hourly_balance.columns.get_loc("methanization_injection"),
    ].reset_index(drop=True)
    meth_coge_output = hourly_balance.iloc[
        hour : (hour + 365 * 24),
        hourly_balance.columns.get_loc("methanization_coge"),
    ].reset_index(drop=True)
    prix_elec = (
        spot_price.loc[:, "elec"].iloc[hour : (hour + 365 * 24)].reset_index(drop=True)
    )

    # Créer la figure
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Courbe 1 : méthanisation (gauche)
    color1 = "#1f77b4"
    ax1.plot(
        time,
        meth_inj_output,
        color=color1,
        linewidth=2,
        label="Methanization injection output",
    )
    color2 = "#1a086e"
    ax1.plot(
        time,
        meth_coge_output,
        color=color2,
        linewidth=2,
        label="Methanization cogeneration output",
    )

    color3 = "#2ca02c"
    ax1.plot(
        time,
        biogas_injection_output,
        color=color3,
        linewidth=1.5,
        linestyle="--",
        label="Biogas for injection output",
    )

    color4 = "#012b01"
    ax1.plot(
        time,
        biogas_coge_output,
        color=color4,
        linewidth=1.5,
        linestyle="--",
        label="Biogas for cogeneration output",
    )

    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Methanization output [GWh]", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Courbe 2 : prix élec (droite)
    ax2 = ax1.twinx()
    color2 = "black"
    ax2.plot(time, prix_elec, color=color2, linewidth=1.5, label="Electricity price")
    ax2.set_ylabel("Electricity price [€/MWh]", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 250)

    # Légende combinée
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    # Titre
    title = "Methanization Output and Electricity Price"
    if lang == "FR":
        title = "Production de méthanisation et prix de l'électricité"
    plt.title(title, fontsize=14, weight="bold")

    plt.show()

    return fig


def plot_storage_batteries_state_year(
    hourly_balance, energy_capacity, hour, nb_years=1, select_tech=["battery"], lang="EN"
):
    fig, ax = plt.subplots(figsize=(16, 10))

    elec_str = pd.DataFrame(dtype=float)
    if lang == "EN":
        if "battery" in select_tech:
            elec_str.loc[:, "Batteries"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_1h_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )

    if lang == "FR":
        if "battery" in select_tech:
            elec_str.loc[:, "Batteries"] = (
                hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_1h_state_charge"),
                ]
                + hourly_balance.iloc[
                    hour : (hour + 8760 * nb_years),
                    hourly_balance.columns.get_loc("battery_4h_input"),
                ]
            )

    time = pd.to_datetime(
        elec_str.index.get_level_values(0)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        + " "
        + elec_str.index.get_level_values(1)
        .to_series()
        .reset_index(drop=True)
        .astype("str")
        .str.zfill(2),
        format="%Y-%m-%d %H",
    )

    colors_str = []
    if "battery" in select_tech:
        colors_str.append("#f2b705")

    handles_str = ax.stackplot(
        time, elec_str.T, labels=elec_str.columns, colors=colors_str
    )

    if lang == "EN":
        ax.set_ylabel("State of charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s=f" Storage State-of-Charge Over the Year",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.19, +1.06),
            frameon=False,
            columnspacing=0.5,
        )
    if lang == "FR":
        ax.set_ylabel("Niveau de charge [GWh]", fontsize=12, labelpad=10)
        ax.text(
            x=0.06,
            y=0.93,
            s="Niveau de charge sur un an",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        leg_str = ax.legend(
            handles=handles_str,
            loc="upper center",
            ncol=len(elec_str.columns) + 1,
            bbox_to_anchor=(0.22, +1.06),
            frameon=False,
            columnspacing=0.5,
        )

    ax.add_artist(leg_str)

    ax.yaxis.set_tick_params(pad=2, bottom=True, labelsize=12)
    ax.set_ylim([elec_str.sum(axis=1).min(), None])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_installed_power(lang="EN", *args):
    fig, ax = plt.subplots(figsize=(16, 10))
    nb_models = len(args)
    colors = plt.cm.tab20(
        np.linspace(0, 1, nb_models)
    )  # ou une autre palette avec 20+ couleurs
    ax.set_prop_cycle(color=colors)

    for i, (model, name, installed_power) in enumerate(args):
        if lang == "EN":
            installed_power_to_plot = pd.Series(dtype=float)
            installed_power_to_plot.at["Cogeneration"] = (
                installed_power.at["biomass_coge"]
                + installed_power.at["geothermal_coge"]
                + installed_power.at["waste"]
                + installed_power.at["ocgt_coge"]
            )
            installed_power_to_plot.at["Cogeneration by methanization"] = (
                installed_power.at["methanization_coge"]
            )
            installed_power_to_plot.at["Hydropower - Other"] = (
                installed_power.at["river"] + installed_power.at["marine"]
            )
            installed_power_to_plot.at["Wind - onshore"] = installed_power.at["onshore"]
            installed_power_to_plot.at["Wind - offshore"] = (
                installed_power.at["offshore_ground"]
                + installed_power.at["offshore_float"]
            )
            installed_power_to_plot.at["Hydropower - Dams"] = installed_power.at["lake"]
            installed_power_to_plot.at["Solar"] = sum(
                installed_power.at[tech] for tech in model.solar
            )
            installed_power_to_plot.at["Nuclear"] = installed_power.at["nuclear"]
            installed_power_to_plot.at["Coal"] = installed_power.at["coal"]
            installed_power_to_plot.at["PHS"] = installed_power.at["phs"]
            installed_power_to_plot.at["Batteries"] = sum(
                installed_power.at[tech] for tech in model.battery
            )
            installed_power_to_plot.at["CH4 turbines"] = sum(
                installed_power.at[tech] for tech in model.from_CH4_to_elec
            )
            installed_power_to_plot.at["H2 turbines"] = sum(
                installed_power.at[tech] for tech in model.from_H2_to_elec
            )

        if lang == "FR":
            installed_power_to_plot.at["Cogénération"] = (
                installed_power.at["biomass_coge"]
                + installed_power.at["geothermal_coge"]
                + installed_power.at["waste"]
                + installed_power.at["ocgt_coge"]
            )
            installed_power_to_plot.at["Cogénération par méthanisation"] = (
                installed_power.at["methanization_coge"]
            )
            installed_power_to_plot.at["Hydraulique - Autres"] = (
                installed_power.at["river"] + installed_power.at["marine"]
            )
            installed_power_to_plot.at["Hydraulique - Barrages"] = installed_power.at[
                "lake"
            ]
            installed_power_to_plot.at["Hydraulique - STEP"] = installed_power.at["phs"]
            installed_power_to_plot.at["Eolien - Terrestre"] = installed_power.at[
                "onshore"
            ]
            installed_power_to_plot.at["Eolien - En mer"] = (
                installed_power.at["offshore_ground"]
                + installed_power.at["offshore_float"]
            )
            installed_power_to_plot.at["Photovoltaïque"] = sum(
                installed_power.at[tech] for tech in model.solar
            )
            installed_power_to_plot.at["Nucléaire"] = installed_power.at["nuclear"]
            installed_power_to_plot.at["Charbon"] = installed_power.at["coal"]
            installed_power_to_plot.at["Batteries"] = sum(
                installed_power.at[tech] for tech in model.battery
            )
            installed_power_to_plot.at["Turbines CH4"] = sum(
                installed_power.at[tech] for tech in model.from_CH4_to_elec
            )
            installed_power_to_plot.at["Turbines H2"] = sum(
                installed_power.at[tech] for tech in model.from_H2_to_elec
            )

    barWidth = 1 / (len(args) + 1)
    bar_pos = [base + i * barWidth for base in range(len(installed_power_to_plot))]

    ax.barh(bar_pos, installed_power_to_plot, height=barWidth, label=name)

    if lang == "EN":
        ax.legend(
            loc="upper left",
            ncol=len(args) + 1,
            bbox_to_anchor=(0, +1.06),
            frameon=False,
            columnspacing=0.5,
        )
        ax.text(
            x=0.132,
            y=0.93,
            s="Installed Power of each technology",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        ax.set_xlabel("Installed power [GW]", fontsize=12)
    if lang == "FR":
        ax.legend(
            loc="upper left",
            ncol=len(args) + 1,
            bbox_to_anchor=(0, +1.06),
            frameon=False,
            columnspacing=0.5,
        )
        ax.text(
            x=0.132,
            y=0.93,
            s="Puissance installée de chaque technologie",
            transform=fig.transFigure,
            ha="left",
            fontsize=16,
            weight="bold",
        )
        ax.set_xlabel("Puissance installée [GW]", fontsize=12)

    ax.set_yticks(
        [
            base + (len(args) // 2) * barWidth
            for base in range(len(installed_power_to_plot))
        ],
        installed_power_to_plot.index,
    )
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def plot_gene_per_tech(*args):
    fig, ax = plt.subplots(figsize=(16, 10))
    nb_models = len(args)
    colors = plt.cm.tab20(
        np.linspace(0, 1, nb_models)
    )  # ou une autre palette avec 20+ couleurs
    ax.set_prop_cycle(color=colors)
    barWidth = 1 / (len(args) + 1)

    for i, (model, name, prod_per_tech) in enumerate(args):
        prod_per_tech_to_plot = pd.Series(dtype=float)
        prod_per_tech_to_plot.at["Cogeneration"] = (
            prod_per_tech.at["biomass_coge"]
            + prod_per_tech.at["geothermal_coge"]
            + prod_per_tech.at["waste"]
            + prod_per_tech.at["ocgt_coge"]
        )
        prod_per_tech_to_plot.at["Cogeneration by methanization"] = prod_per_tech.at[
            "methanization_coge"
        ]
        prod_per_tech_to_plot.at["Hydropower - Other"] = (
            prod_per_tech.at["river"] + prod_per_tech.at["marine"]
        )
        prod_per_tech_to_plot.at["Wind - onshore"] = prod_per_tech.at["onshore"]
        prod_per_tech_to_plot.at["Wind - offshore"] = (
            prod_per_tech.at["offshore_ground"] + prod_per_tech.at["offshore_float"]
        )
        prod_per_tech_to_plot.at["Hydropower - Dams"] = prod_per_tech.at["lake"]
        prod_per_tech_to_plot.at["Solar"] = sum(
            prod_per_tech.at[tech] for tech in model.solar
        )
        prod_per_tech_to_plot.at["Nuclear"] = prod_per_tech.at["nuclear"]
        # prod_per_tech_to_plot.at["Coal"] = prod_per_tech.at["coal"]
        prod_per_tech_to_plot.at["PHS"] = prod_per_tech.at["phs"]
        prod_per_tech_to_plot.at["Batteries"] = sum(
            prod_per_tech.at[tech] for tech in model.battery
        )
        prod_per_tech_to_plot.at["CH4 turbines"] = sum(
            prod_per_tech.at[tech] for tech in model.from_CH4_to_elec
        )
        prod_per_tech_to_plot.at["H2 turbines"] = sum(
            prod_per_tech.at[tech] for tech in model.from_H2_to_elec
        )

    bar_pos = [base + i * barWidth for base in range(len(prod_per_tech_to_plot))]

    ax.barh(bar_pos, prod_per_tech_to_plot, height=barWidth, label=name)

    ax.legend(
        loc="upper left",
        ncol=len(args) + 1,
        bbox_to_anchor=(0, +1.06),
        frameon=False,
        columnspacing=0.5,
    )
    ax.text(
        x=0.132,
        y=0.93,
        s="Energy Generation per Technology",
        transform=fig.transFigure,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    ax.set_yticks(
        [
            base + (len(args) // 2) * barWidth
            for base in range(len(prod_per_tech_to_plot))
        ],
        prod_per_tech_to_plot.index,
    )
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.grid(True)
    ax.set_xlabel("Total energy generated [TWh]", fontsize=12)
    ax.set_axisbelow(True)

    plt.show()

    return fig


def compare_operable_mix(names, results, lang="EN"):
    results = [df.apply(pd.to_numeric, errors="coerce") for df in results]

    fig, ax = plt.subplots(figsize=(16, 10))
    bottom = np.zeros(len(names))

    if lang == "EN":
        ax.bar(
            names,
            [result.at["lake", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Hydropower - Dams",
            bottom=bottom,
            color="#2672b0",
        )
        bottom += [result.at["lake", "Installed power [GW]"] for result in results]
        if any(
            nuc > 0
            for nuc in [
                result.at["nuclear", "Installed power [GW]"] for result in results
            ]
        ):
            ax.bar(
                names,
                [result.at["nuclear", "Installed power [GW]"] for result in results],
                width=0.5,
                label="Nuclear",
                bottom=bottom,
                color="#e4a701",
            )
            bottom += [
                result.at["nuclear", "Installed power [GW]"] for result in results
            ]
        if any(
            coal > 0
            for coal in [result.at["coal", "Installed power [GW]"] for result in results]
        ):
            ax.bar(
                names,
                [result.at["coal", "Installed power [GW]"] for result in results],
                width=0.5,
                label="Coal",
                bottom=bottom,
                color="#a68832",
            )
            bottom += [result.at["coal", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["phs", "Installed power [GW]"] for result in results],
            width=0.5,
            label="PHS",
            bottom=bottom,
            color="#0e4269",
        )
        bottom += [result.at["phs", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [
                (
                    result.at["battery_1h", "Installed power [GW]"]
                    + result.at["battery_4h", "Installed power [GW]"]
                )
                for result in results
            ],
            width=0.5,
            label="Batteries",
            bottom=bottom,
            color="#80549f",
        )
        bottom += [
            (
                result.at["battery_1h", "Installed power [GW]"]
                + result.at["battery_4h", "Installed power [GW]"]
            )
            for result in results
        ]
        ax.bar(
            names,
            [result.at["ch4_ocgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Open-cycle gas turbines",
            bottom=bottom,
            color="#f20809",
        )
        bottom += [result.at["ch4_ocgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["ch4_ccgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Combined-cycle gas turbines",
            bottom=bottom,
            color="#a90506",
        )
        bottom += [result.at["ch4_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["h2_ccgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Hydrogen gas turbines",
            bottom=bottom,
            color="#f252c0",
        )
        bottom += [result.at["h2_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["str_dummy", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Missing storage",
            bottom=bottom,
            color="#757575",
        )
        bottom += [result.at["str_dummy", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["rsv_dummy", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Missing reserves",
            bottom=bottom,
            color="#3d3d3d",
        )
        bottom += [result.at["rsv_dummy", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [
                result.at["methanization_coge", "Installed power [GW]"]
                for result in results
            ],
            width=0.5,
            label="Cogeneration by methanization",
            bottom=bottom,
            color="#e4a701",
        )
        bottom += [
            result.at["methanization_coge", "Installed power [GW]"] for result in results
        ]
        ax.bar(names, bottom / 5, bottom=bottom, alpha=0)  # margin

        ax.set_ylabel("Installed power [GW]", fontsize=12)
        ax.set_title("Operable part of the energy mix for different scenarios")

    if lang == "FR":
        ax.bar(
            names,
            [result.at["lake", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Hydraulique - Barrages",
            bottom=bottom,
            color="#2672b0",
        )
        bottom += [result.at["lake", "Installed power [GW]"] for result in results]
        if any(
            nuc > 0
            for nuc in [
                result.at["nuclear", "Installed power [GW]"] for result in results
            ]
        ):
            ax.bar(
                names,
                [result.at["nuclear", "Installed power [GW]"] for result in results],
                width=0.5,
                label="Nucléaire",
                bottom=bottom,
                color="#e4a701",
            )
            bottom += [
                result.at["nuclear", "Installed power [GW]"] for result in results
            ]
        if any(
            coal > 0
            for coal in [result.at["coal", "Installed power [GW]"] for result in results]
        ):
            ax.bar(
                names,
                [result.at["coal", "Installed power [GW]"] for result in results],
                width=0.5,
                label="Charbon",
                bottom=bottom,
                color="#a68832",
            )
            bottom += [result.at["coal", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["phs", "Installed power [GW]"] for result in results],
            width=0.5,
            label="STEP",
            bottom=bottom,
            color="#0e4269",
        )
        bottom += [result.at["phs", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [
                (
                    result.at["battery_1h", "Installed power [GW]"]
                    + result.at["battery_4h", "Installed power [GW]"]
                )
                for result in results
            ],
            width=0.5,
            label="Batteries",
            bottom=bottom,
            color="#80549f",
        )
        bottom += [
            (
                result.at["battery_1h", "Installed power [GW]"]
                + result.at["battery_4h", "Installed power [GW]"]
            )
            for result in results
        ]
        ax.bar(
            names,
            [result.at["ch4_ocgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Turbines CH4 cycle ouvert",
            bottom=bottom,
            color="#f20809",
        )
        bottom += [result.at["ch4_ocgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["ch4_ccgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Turbines CH4 cycle combiné",
            bottom=bottom,
            color="#a90506",
        )
        bottom += [result.at["ch4_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["h2_ccgt", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Turbines H2",
            bottom=bottom,
            color="#f252c0",
        )
        bottom += [result.at["h2_ccgt", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["str_dummy", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Stockage manquant",
            bottom=bottom,
            color="#757575",
        )
        bottom += [result.at["str_dummy", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [result.at["rsv_dummy", "Installed power [GW]"] for result in results],
            width=0.5,
            label="Réserves manquantes",
            bottom=bottom,
            color="#3d3d3d",
        )
        bottom += [result.at["rsv_dummy", "Installed power [GW]"] for result in results]
        ax.bar(
            names,
            [
                result.at["methanization_coge", "Installed power [GW]"]
                for result in results
            ],
            width=0.5,
            label="Cogénération par méthanisation",
            bottom=bottom,
            color="#e4a701",
        )
        bottom += [
            result.at["methanization_coge", "Installed power [GW]"] for result in results
        ]
        ax.bar(names, bottom / 5, bottom=bottom, alpha=0)  # margin

        ax.set_ylabel("Puissance installée [GW]", fontsize=12)
        ax.set_title("Partie pilotable du mix énergétique pour différents scénarios")

    ax.legend(loc="upper left", ncol=2, frameon=False)
    if output_path is not None:
        import os
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(f"{output_path}/compare_operable_mix.png", dpi=300, bbox_inches="tight")

    plt.show()

    return fig


for _name, _obj in list(globals().items()):
    if callable(_obj) and _name.startswith("plot_"):
        globals()[_name] = _wrap_plot_savefig(_obj)
