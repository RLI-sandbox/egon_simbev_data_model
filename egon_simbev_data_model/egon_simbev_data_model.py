import json
import multiprocessing
import re
import sys

from collections import Counter
from pathlib import Path, PurePath
from typing import Union

import numpy as np
import pandas as pd

from config.config import settings
from loguru import logger
from numpy.random import Generator

logger.remove()
logger.add(sys.stderr, level=settings.debugging["debug_level"])


def reduce_mem_usage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Function to automatically check if columns of a pandas DataFrame can
    be reduced to a smaller data type. Source:
    https://www.mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas/

    :param df: DataFrame to reduce memory usage on
    :type df: pandas.DataFrame

    :return: DataFrame with memory usage decreased
    :rtype: pd.DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype("int16")
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype("int32")
                else:
                    df[col] = df[col].astype("int64")
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype("float32")
                else:
                    df[col] = df[col].astype("float64")

        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2

    logger.debug(
        "Reduced memory usage of DataFrame by "
        f"{(1 - end_mem/start_mem) * 100:.2f} %."
    )

    return df


def egon_scenario(
    rng: Generator,
    num_evs: int = 50,
) -> pd.DataFrame:
    """
    TODO: @Jonathan hier deine SQL Abfrage

    :return: DataFrame containing the scenario data
    :rtype: pandas.DataFrame
    """
    ev_ids = rng.integers(0, 100, size=num_evs)

    mv_grid_id = rng.integers(1000, 1002, size=num_evs)

    data = list(zip(mv_grid_id, ev_ids))

    dummy_df = pd.DataFrame(data=data, columns=["mv_grid_id", "ev_id"])

    logger.debug("Loaded egon scenario data.")

    if settings.memory_management["reduce_memory"]:
        return reduce_mem_usage(dummy_df)
    return dummy_df


def simbev_config() -> dict:
    """
    TODO: Add this function as soon as SimBEV provides config data as output.
     Until then add your config settings by hand (settings.toml)

    :return: Dict containing simbev config data
    :rtype: dict
    """
    simbev_config_dict = dict(settings.simbev_config)

    logger.debug("Loaded simbev config data.")

    return simbev_config_dict


def simbev_data(
    simbev_data_dir: PurePath,
) -> pd.DataFrame:
    """Read in SimBEV data within a given directory.

    :param simbev_data_dir: Path to directory containing SimBEV data
    :type simbev_data_dir: pathlib.PurePath

    :return: pandas DataFrame containing simbev data
    :rtype: pandas.DataFrame
    """
    scenario_files = []

    # iterate over all regions
    for region_path in simbev_data_dir.iterdir():  # type: ignore
        assert region_path.is_dir()

        scenario_files.extend(list(region_path.iterdir()))

    dummy_df = pd.DataFrame()

    for count, f in enumerate(scenario_files):
        df = pd.read_csv(f, index_col=0)

        # only keep rows with a charging event
        df = df.loc[df.chargingdemand_kWh > 0]  # pylint: disable=E1101

        bat_cap = int(re.findall(r"(\d+(?:\.\d+)?)kWh", f.parts[-1])[0])

        df = df.assign(ev_id=count, bat_cap=bat_cap)

        dummy_df = dummy_df.append(df, ignore_index=True)

    logger.debug("Loaded simbev EV data.")

    if settings.memory_management["reduce_memory"]:
        return reduce_mem_usage(dummy_df)
    return dummy_df


def generate_dsm_profile(
    restriction_time: int,
    min_soc: float,
    timestep_count: int,
) -> pd.DataFrame:
    dsm_profile_week = np.zeros((24 * 7,))
    dsm_profile_week[(np.arange(0, 7, 1) * 24 + restriction_time)] = min_soc
    weeks, rest = divmod(timestep_count, len(dsm_profile_week))
    dsm_profile = np.concatenate(
        (np.tile(dsm_profile_week, weeks), dsm_profile_week[0:rest])
    )

    return pd.DataFrame({"min_soc": dsm_profile})


def static_params(
    ev_data_df: pd.DataFrame,
    load_time_series_df: pd.DataFrame,
    region: Union[int, str],
) -> dict:
    """Calculate static parameters from SimBEV data.

    :param ev_data_df: DataFrame containing all SimBEV data.
    :type ev_data_df: pandas.DataFrame
    :param load_time_series_df: pandas DataFrame containing time series of the
    load and the flex potential
    :type load_time_series_df: pandas.DataFrame
    :param region: Region key
    :type region: int or str

    :return: Dictionary containing static parameters.
    :rtype: dict
    """
    max_df = (
        ev_data_df[["ev_id", "bat_cap", "grid_charging_capacity_MW"]]
        .groupby("ev_id")
        .max()
    )

    static_params_dict = {
        "store_ev_battery.e_nom_MWh": float(max_df.bat_cap.sum() / 10 ** 3),
        "link_bev_charger.p_nom_MW": float(
            load_time_series_df.simultaneous_plugged_in_charging_capacity.max()
        ),
        "store_ev_battery.e_max_pu": 1,
    }

    logger.debug(f"Calculated static parameters for region {region}.")

    return static_params_dict


def load_time_series(
    ev_data_df: pd.DataFrame,
    simbev_cfg_dict: dict,
    region: Union[int, str],
) -> pd.DataFrame:
    """Calculate the load time series from the given SimBEV data. A dumb
    charging strategy is assumed where each EV starts charging immediately
    after plugging it in. Simultaneously the flexible charging capacity is
    calculated.

    :param ev_data_df: DataFrame containing all SimBEV data.
    :type ev_data_df: pandas.DataFrame
    :param simbev_cfg_dict: Dict containing SimBEV config data
    :type simbev_cfg_dict: dict
    :param region: Region key
    :type region: int or str

    :return: pandas DataFrame containing time series of the load and the flex
    potential
    :rtype: pandas.DataFrame
    """
    # instantiate timeindex
    timeindex = pd.date_range(
        simbev_cfg_dict["start_date"],
        periods=ev_data_df.last_timestep.max() + 1,
        freq=simbev_cfg_dict["timestep"],
    )

    load_time_series_df = pd.DataFrame(
        data=0.0,
        index=timeindex,
        columns=["load_time_series", "flex_time_series"],
    )

    load_time_series_array = np.zeros(len(load_time_series_df))
    flex_time_series_array = load_time_series_array.copy()
    simultaneous_plugged_in_charging_capacity = load_time_series_array.copy()

    columns = [
        "park_start_timesteps",
        "charge_end",
        "grid_charging_capacity_MW",
        "last_timestep",
        "last_timestep_grid_charging_capacity_MW",
        "flex_grid_charging_capacity_MW",
        "flex_last_timestep_grid_charging_capacity_MW",
        "park_end_timesteps",
    ]

    # iterate over charging events
    for (
        _,
        start,
        end,
        cap,
        last_ts,
        last_ts_cap,
        flex_cap,
        flex_last_ts_cap,
        park_end,
    ) in ev_data_df[columns].itertuples():
        load_time_series_array[start:end] += cap
        load_time_series_array[last_ts] += last_ts_cap

        flex_time_series_array[start:end] += flex_cap
        flex_time_series_array[last_ts] += flex_last_ts_cap

        simultaneous_plugged_in_charging_capacity[start:park_end] += cap

    load_time_series_df = load_time_series_df.assign(
        load_time_series=load_time_series_array,
        flex_time_series=flex_time_series_array,
        simultaneous_plugged_in_charging_capacity=(
            simultaneous_plugged_in_charging_capacity
        ),
    )

    np.testing.assert_almost_equal(
        load_time_series_df.load_time_series.sum() / 4,
        ev_data_df.chargingdemand_kWh.sum()
        / 1000
        / (
            ev_data_df.nominal_charging_capacity_kW
            / ev_data_df.grid_charging_capacity_kW
        ).mean(),
        decimal=4,
    )

    logger.debug(f"Calculated load time series for region {region}.")

    if settings.memory_management["reduce_memory"]:
        return reduce_mem_usage(load_time_series_df)
    return load_time_series_df


def data_preprocessing(
    scenario_data: pd.DataFrame,
    ev_data_df: pd.DataFrame,
    simbev_cfg_dict: dict,
    region: Union[int, str],
) -> pd.DataFrame:
    """Filters SimBEV data to match region requirements. Duplicates profiles
    if necessary. Pre-calculates necessary parameters for the load time series.

    :param scenario_data: DataFrame containing eGo^n scenario data.
    :type scenario_data: pandas.DataFrame
    :param ev_data_df: DataFrame containing all SimBEV data.
    :type ev_data_df: pandas.DataFrame
    :param simbev_cfg_dict: Dict containing SimBEV config data
    :type simbev_cfg_dict: dict
    :param region: Region key
    :type region: int or str

    :return: DataFrame containing SimBEV data of the region
    :rtype: pandas.DataFrame
    """
    # get scenario data for region
    scenario_data = scenario_data.loc[scenario_data.mv_grid_id == region]

    # count profiles to respect profiles which are used multiple times
    count_profiles = Counter(scenario_data.ev_id)  # type: dict

    max_duplicates = max(count_profiles.values())

    # get ev data for given profiles
    ev_data_df = ev_data_df.loc[ev_data_df.ev_id.isin(scenario_data.ev_id.unique())]

    # drop faulty data
    ev_data_df = ev_data_df.loc[
        ev_data_df.park_start_timesteps < ev_data_df.park_end_timesteps
    ]

    if max_duplicates >= 2:
        # duplicate profiles if necessary
        temp = ev_data_df.copy()

        for count in range(2, max_duplicates + 1):
            duplicates = [key for key, val in count_profiles.items() if val >= count]

            duplicates_df = temp.loc[temp.ev_id.isin(duplicates)]

            duplicates_df = duplicates_df.assign(
                ev_id=duplicates_df.ev_id.astype(str) + f"_{count}"
            )

            ev_data_df = ev_data_df.append(duplicates_df)

    # calculate time necessary to fulfill the charging demand and brutto
    # charging capacity in mva
    ev_data_df = ev_data_df.assign(
        grid_charging_capacity_MW=(ev_data_df.grid_charging_capacity_kW / 10 ** 3),
        minimum_charging_time=(
            ev_data_df.chargingdemand_kWh / ev_data_df.nominal_charging_capacity_kW * 4
        ),
        location=ev_data_df.location.str.replace("/", "_"),
    )

    # calculate charging capacity for last timestep
    (
        full_timesteps,
        last_timestep_share,
    ) = ev_data_df.minimum_charging_time.divmod(1)

    full_timesteps = full_timesteps.astype(int)

    ev_data_df = ev_data_df.assign(
        full_timesteps=full_timesteps,
        last_timestep_share=last_timestep_share,
        last_timestep_grid_charging_capacity_MW=(
            last_timestep_share * ev_data_df.grid_charging_capacity_MW
        ),
        charge_end=ev_data_df.park_start_timesteps + full_timesteps,
        last_timestep=ev_data_df.park_start_timesteps + full_timesteps + 1,
    )

    # calculate flexible charging capacity
    # TODO: change this when SimBEV issue #33 is solved
    #  https://github.com/rl-institut/simbev/issues/33
    flex_dict = dict(settings.flex_share)

    flex_share = ev_data_df.location.map(flex_dict)

    ev_data_df = ev_data_df.assign(
        flex_grid_charging_capacity_MW=(
            ev_data_df.grid_charging_capacity_MW * flex_share
        ),
        flex_last_timestep_grid_charging_capacity_MW=(
            ev_data_df.last_timestep_grid_charging_capacity_MW * flex_share
        ),
    )

    logger.debug(f"Preprocessed data for region {region}.")

    if settings.memory_management["reduce_memory"]:
        return reduce_mem_usage(ev_data_df)
    return ev_data_df


def export_results(
    static_params_dict: dict,
    load_time_series_df: pd.DataFrame,
    dsm_profile_df: pd.DataFrame,
    region: Union[int, str],
) -> None:
    """Export all results as CSVs and add Metadata JSON.

    :param static_params_dict: Dictionary containing static parameters.
    :type static_params_dict: dict
    :param load_time_series_df: pandas DataFrame containing time series of the
    load and the flex potential
    :type load_time_series_df: pandas.DataFrame
    :param region: Region key
    :type region: int or str
    """
    load_time_series_df = load_time_series_df.assign(
        ev_availability=(
            load_time_series_df.flex_time_series
            / static_params_dict["link_bev_charger.p_nom_MW"]
        )
    )

    results_dir = Path(settings.directories["results_dir"]) / str(region)

    results_dir.mkdir(exist_ok=True, parents=True)

    hourly_load_time_series_df = load_time_series_df.resample("1H").mean()

    if len(hourly_load_time_series_df) >= len(dsm_profile_df):
        hourly_load_time_series_df = hourly_load_time_series_df.iloc[
            : len(dsm_profile_df)
        ]
    else:
        dsm_profile_df = dsm_profile_df.iloc[: len(hourly_load_time_series_df)]

    dsm_profile_df.index = hourly_load_time_series_df.index

    hourly_load_time_series_df[["load_time_series"]].to_csv(
        results_dir / "ev_load_time_series.csv"
    )

    hourly_load_time_series_df[["ev_availability"]].to_csv(
        results_dir / "ev_availability.csv"
    )

    dsm_profile_df.to_csv(results_dir / "ev_dsm_profile.csv")

    static_params_dict["load_land_transport_ev.p_set_MW"] = "ev_load_time_series.csv"
    static_params_dict["link_bev_charger.p_max_pu"] = "ev_availability.csv"
    static_params_dict["store_ev_battery.e_min_pu"] = "ev_dsm_profile.csv"

    file = results_dir / "ev_static_params.json"

    with open(file, "w") as f:
        json.dump(static_params_dict, f, indent=4)


def calculate_scenario(
    scenario_data: pd.DataFrame,
    ev_data_df: pd.DataFrame,
    simbev_cfg_dict: dict,
    region: Union[int, str],
) -> None:
    """Calculates static parameters and load time series for a given scenario
    and region.

    :param scenario_data: DataFrame containing eGo^n scenario data.
    :type scenario_data: pandas.DataFrame
    :param ev_data_df: DataFrame containing all SimBEV data.
    :type ev_data_df: pandas.DataFrame
    :param simbev_cfg_dict: Dict containing SimBEV config data
    :type simbev_cfg_dict: dict
    :param region: Region key
    :type region: int or str
    """
    ev_data_df = data_preprocessing(scenario_data, ev_data_df, simbev_cfg_dict, region)

    load_time_series_df = load_time_series(ev_data_df, simbev_cfg_dict, region)

    static_params_dict = static_params(ev_data_df, load_time_series_df, region)

    dsm_profile_df = generate_dsm_profile(
        restriction_time=settings.dsm_profile["restriction_time"],
        min_soc=settings.dsm_profile["min_soc"],
        timestep_count=settings.dsm_profile["timestep_count"],
    )

    export_results(static_params_dict, load_time_series_df, dsm_profile_df, region)


def run_egon_simbev_data_model() -> None:
    """Main function to run all sub functions"""
    # setup random generator
    seed = settings.random_state["seed"]
    rng = np.random.default_rng(seed)

    # get egon scenario
    scenario_data = egon_scenario(rng)

    # get corresponding simbev data
    simbev_data_dir = Path(settings.directories["simbev_data_dir"])
    ev_data_df = simbev_data(simbev_data_dir)

    simbev_cfg_dict = simbev_config()

    # run regions
    num_threads = settings.multiprocessing["num_threads"]
    regions = scenario_data.mv_grid_id.unique()

    if num_threads <= 1:
        for region in regions:
            calculate_scenario(scenario_data, ev_data_df, simbev_cfg_dict, region)
    else:
        region_tuples = [
            (scenario_data, ev_data_df, simbev_cfg_dict, region) for region in regions
        ]

        with multiprocessing.Pool(num_threads) as pool:
            pool.starmap(calculate_scenario, region_tuples)


if __name__ == "__main__":
    run_egon_simbev_data_model()
