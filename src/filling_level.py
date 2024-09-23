import numpy as np
import pandas as pd


def estimate_filling_level(
    df: pd.DataFrame,
    net_prod_name: str,
    efficiency_name: str = None,
    max_filling_level: float = -1,
) -> pd.Series:
    """
    Determine the filling level of a storage system based on the net
    production dataframe, the efficiency of the storage system and the maximum
    filling level. The filling level is calculated by cumulating the net
    production and applying the efficiency to the pumping periods. The storage
    level is then iterated over the dataframe to determine the filling level
    at each point in time. The filling level is limited by the maximum storage
    level.

    :param df: The net production dataframe
    :param net_prod_name: The name of the net production column
    :param inflow_name: The name of the inflow column
    :param efficiency: The efficiency of the storage system
    :param max_storage_capacity: The maximum storage capacity of the storage system
    :return: A dataframe with the filling level
    """
    assert df is not None, "net_prod_df must not be None"
    assert isinstance(df, pd.DataFrame), "net_prod_df must be a DataFrame object"
    assert net_prod_name is not None, "net_prod_name must not be None"
    assert isinstance(net_prod_name, str), "net_prod_name must be a string"
    assert efficiency_name is not None, "efficiency_name must not be None"
    assert isinstance(efficiency_name, str), "efficiency_name must be a string"
    # apply efficiency to pumping periods
    df["net_prod_eff"] = df[net_prod_name]
    df.loc[df[net_prod_name] < 0, "net_prod_eff"] = (
        df[net_prod_name] * df[efficiency_name]
    )
    # calculate net pumping
    df["net_pump_eff"] = df["net_prod_eff"] * -1.0
    # initialize filling level
    df["filling_level_initial"] = df["net_pump_eff"].cumsum()
    # look for the smallest filling level
    min_filling_level = df["filling_level_initial"].min()
    print(f"Smallest filling level: {min_filling_level}")
    # adjust the filling level
    df["filling_level"] = df["filling_level_initial"] - min_filling_level
    # look for all filling levels below zero
    neg_filling_level = df[df["filling_level"] < 0]
    # print warning with indexes of negative filling levels
    if len(neg_filling_level) > 0:
        print(
            f"Warning: Found {len(neg_filling_level)} negative filling levels at indexes {neg_filling_level.index}"
        )
    # look for all filling levels above the maximum filling level
    max_filling_level = df[df["filling_level"] > max_filling_level]
    # print warning with indexes of filling levels above the maximum filling level
    if len(max_filling_level) > 0:
        print(
            f"Warning: Found {len(max_filling_level)} filling levels above the maximum filling level at indexes {max_filling_level.index}"
        )
    # return timeseries?
    return df[["filling_level"]]
