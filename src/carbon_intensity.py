import importlib
import sys

import numpy as np
import pandas as pd

from src.flow_tracing import trace_flows

# carbon intensity per production type in kgCO2eq/MWh
# values come from IPCC 2014 if not stated otherwise
# https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7
_carbon_intensities: dict[str, float] = {
    "Biomass Power": 230,
    "Coal Power": 820,
    "Gas Power": 490,
    "Geothermal Power": 38,
    "Hydro": 24,
    "Nuclear": 12,
    # https://www.ipcc.ch/report/renewable-energy-sources-and-climate-change-mitigation/
    "Oil Power": 840,
    # only used in EE
    # https://circabc.europa.eu/sd/a/e3685bb5-cdbc-4a94-a390-64b6241513ea/15%2004%2013%20Oil%20Shale.pdf
    "Oil Shale Power": 420,
    # only in IE and NIE
    # https://www.mdpi.com/2071-1050/7/6/6376
    "Peat Power": 1120,
    "Solar Power": 48,
    "Wind Power": 11,
    "Wind Onshore Power": 11,
    "Wind Offshore Power": 12,
    "Unknown": 700,
}

_placeholders: dict[str, str] = {
    "Bioenergy Power": "Biomass Power",
    "Biogas Power": "Biomass Power",
    "CHP Power": "Biomass Power",
    "Waste Power": "Biomass Power",
    "Hard Coal Power": "Coal Power",
    "Lignite Power": "Coal Power",
    "Derived Gas Power": "Gas Power",
    "Natural Gas Power": "Gas Power",
    "Hydro Pumped-storage": "Hydro",
    "Hydro Reservoir": "Hydro",
    "Hydro Run-of-river": "Hydro",
    "Other Power": "Unknown",
}

_prod_types: list[str] = [
    "Bioenergy Power",
    "Biogas Power",
    "Biomass Power",
    "CHP Power",
    "Derived Gas Power",
    "Geothermal Power",
    "Hard Coal Power",
    "Hydro Pumped-storage",
    "Hydro Reservoir",
    "Hydro Run-of-river",
    "Lignite Power",
    "Natural Gas Power",
    "Nuclear",
    "Oil Power",
    "Oil Shale Power",
    "Other Power",
    "Peat Power",
    "Solar Photovoltaic",
    "Waste Power",
    "Wind Power",
]


def _get_carbon_intensity(prod_type: str) -> float:
    assert prod_type, "Production type must be provided."

    if prod_type in _placeholders:
        prod_type = _placeholders[prod_type]
    try:
        return _carbon_intensities[prod_type]
    except KeyError:
        return _carbon_intensities["Unknown"]


def calculate_ci_prod(
    df: pd.DataFrame, area: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prefix: str = f"{area} " if area else ""
    # copy the dataframe to avoid modifying the original
    df_copy: pd.DataFrame = df.copy()
    # remove old columns
    df_copy = df_copy.drop(
        columns=[
            f"{prefix}Total Production",
            f"{prefix}Carbon Emissions Production",
            f"{prefix}Carbon Intensity Production",
        ],
        errors="ignore",
    )
    # initialize total production and total carbon emissions
    df_copy[f"{prefix}Total Production"] = 0.0
    df_copy[f"{prefix}Carbon Emissions Production"] = 0.0
    # iterate over all production types
    for prod_type in _prod_types:
        if f"{prefix}{prod_type}" not in df_copy.columns:
            continue
        # replace nan values with 0
        df_copy[f"{prefix}{prod_type}"] = df_copy[f"{prefix}{prod_type}"].fillna(0)
        # get carbon intensity for production type
        prod_ci = _get_carbon_intensity(prod_type)
        # calculate carbon emissions for production type
        df_copy[f"{prefix}{prod_type} Carbon Emissions"] = (
            prod_ci * df_copy[f"{prefix}{prod_type}"]
        )
        # calculate total production
        df_copy[f"{prefix}Total Production"] += df_copy[f"{prefix}{prod_type}"]
        # calculate total carbon emissions
        df_copy[f"{prefix}Carbon Emissions Production"] += df_copy[
            f"{prefix}{prod_type} Carbon Emissions"
        ]
    # calculate the carbon intensity of the production mix
    df_copy[f"{prefix}Total Production"] = df_copy[f"{prefix}Total Production"].replace(
        0.0, np.nan
    )
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Emissions Production"]
        / df_copy[f"{prefix}Total Production"]
    )
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Intensity Production"]
        .replace([float("inf"), float("-inf")], np.nan)
        .clip(0.0, 1600.0)
    )
    # # replace infinite values with 0
    # # limit the carbon intensity to 0 and 1600 kgCO2eq/MWh
    # df_copy[f"{prefix}Carbon Intensity Production"] = (
    #     df_copy[f"{prefix}Carbon Intensity Production"]
    #     .replace([float("inf"), float("-inf")], 0)
    #     .clip(0, 1600)
    # )

    return (
        df_copy[f"{prefix}Carbon Intensity Production"],
        df_copy[f"{prefix}Carbon Emissions Production"],
        df_copy[f"{prefix}Total Production"],
    )


def calculate_ci_prod_series(
    row: pd.Series, area: str = ""
) -> tuple[float, float, float]:
    prefix: str = f"{area} " if area else ""
    # remove old columns
    row = row.drop(
        labels=[
            f"{prefix}Total Production",
            f"{prefix}Carbon Emissions Production",
            f"{prefix}Carbon Intensity Production",
        ],
        errors="ignore",
    )
    # initialize total production and total carbon emissions
    row[f"{prefix}Total Production"] = 0.0
    row[f"{prefix}Carbon Emissions Production"] = 0.0
    # fill all nans with zero
    # row = row.fillna(0)
    # iterate over all production types
    for prod_type in _prod_types:
        if f"{prefix}{prod_type}" not in row.index:
            continue
        # replace nan values with 0
        row[f"{prefix}{prod_type}"] = (
            0.0 if pd.isna(row[f"{prefix}{prod_type}"]) else row[f"{prefix}{prod_type}"]
        )
        # get carbon intensity for production type
        prod_ci = _get_carbon_intensity(prod_type)
        # calculate carbon emissions for production type
        row[f"{prefix}{prod_type} Carbon Emissions"] = (
            prod_ci * row[f"{prefix}{prod_type}"]
        )
        # calculate total production
        row[f"{prefix}Total Production"] += row[f"{prefix}{prod_type}"]
        # calculate total carbon emissions
        row[f"{prefix}Carbon Emissions Production"] += row[
            f"{prefix}{prod_type} Carbon Emissions"
        ]
    # calculate the carbon intensity of the production mix
    if row[f"{prefix}Total Production"] <= 0.0:
        row[f"{prefix}Total Production"] = np.nan
    row[f"{prefix}Carbon Intensity Production"] = (
        row[f"{prefix}Carbon Emissions Production"] / row[f"{prefix}Total Production"]
    )
    if row[f"{prefix}Carbon Intensity Production"] == float("inf") or row[
        f"{prefix}Carbon Intensity Production"
    ] == float("-inf"):
        row[f"{prefix}Carbon Intensity Production"] = np.nan
    elif (
        not pd.isna(row[f"{prefix}Carbon Intensity Production"])
        and row[f"{prefix}Carbon Intensity Production"] < 0.0
    ):
        row[f"{prefix}Carbon Intensity Production"] = 0.0
    elif (
        not pd.isna(row[f"{prefix}Carbon Intensity Production"])
        and row[f"{prefix}Carbon Intensity Production"] > 1600.0
    ):
        row[f"{prefix}Carbon Intensity Production"] = 1600.0

    # replace infinite values with 0
    # limit the carbon intensity to 0 and 1600 kgCO2eq/MWh
    # if (
    #     np.isinf(row[f"{prefix}Carbon Intensity Production"])
    #     or row[f"{prefix}Carbon Intensity Production"] < 0
    # ):
    #     row[f"{prefix}Carbon Intensity Production"] = np.nan
    # elif row[f"{prefix}Carbon Intensity Production"] > 1600:
    #     row[f"{prefix}Carbon Intensity Production"] = 1600
    # row[f"{prefix}Carbon Intensity Production"] = row[
    #     f"{prefix}Carbon Intensity Production"
    # ].clip(min=0, max=1600)

    return (
        row[f"{prefix}Carbon Intensity Production"],
        row[f"{prefix}Carbon Emissions Production"],
        row[f"{prefix}Total Production"],
    )


def add_average_ci_to_phs(
    df: pd.DataFrame, area: str, plant: str, suffix: str = ""
) -> tuple[pd.Series, pd.Series]:
    prefix: str = f"{area} " if area else ""
    plant_prefix = f"{prefix}@{plant} " if plant else ""

    # copy the dataframe to avoid modifying the original
    df_copy: pd.DataFrame = df.copy()
    # remove old columns
    df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
    # calculate yearly moving average of CI
    df_copy[f"{plant_prefix}CI Storage"] = (
        df_copy[f"{prefix}Carbon Intensity Consumption{suffix}"]
        .ffill()
        .rolling(window=24 * 30)
        .mean()
    )
    # calculate carbon emissions for plant
    df_copy[f"{plant_prefix}Carbon Emissions"] = df_copy[
        f"{plant_prefix}CI Storage"
    ] * df_copy[f"{plant_prefix}Hydro Pumped-storage Production"].fillna(0.0).clip(
        lower=0.0
    )
    # calculate total carbon emissions
    df_copy[f"{prefix}Carbon Emissions Production"] = (
        df_copy[f"{plant_prefix}Carbon Emissions"]
        + df_copy[f"{prefix}Carbon Emissions Production{suffix}"]
    )
    # calculate the carbon intensity of the production mix
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Emissions Production"]
        / df_copy[f"{prefix}Total Production"]
    )
    # replace infinite values with 0
    # limit the carbon intensity to 0 and 1600 kgCO2eq/MWh
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Intensity Production"]
        .clip(0.0, 1600.0)
        .replace([0.0, 1600.0], np.nan)
    )

    return (
        df_copy[f"{prefix}Carbon Intensity Production"],
        df_copy[f"{prefix}Carbon Emissions Production"],
    )


def add_temporal_matching_ci_to_phs(
    df: pd.DataFrame, area: str, plant: str, suffix: str = ""
) -> tuple[pd.Series, pd.Series]:
    prefix: str = f"{area} " if area else ""
    plant_prefix = f"{prefix}@{plant} " if plant else ""
    # copy the dataframe to avoid modifying the original
    df_copy: pd.DataFrame = df.copy()
    # remove old columns
    df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
    # calculate carbon emissions for plant
    df_copy[f"{plant_prefix}Carbon Emissions"] = df_copy[
        f"{prefix}Carbon Intensity Production{suffix}"
    ] * df_copy[f"{plant_prefix}Hydro Pumped-storage Production"].fillna(0).clip(
        lower=0
    )
    # calculate total carbon emissions
    df_copy[f"{prefix}Carbon Emissions Production"] += df_copy[
        f"{plant_prefix}Carbon Emissions"
    ]
    # calculate the carbon intensity of the production mix
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Emissions Production"]
        / df_copy[f"{prefix}Total Production"]
    )
    # replace infinite values with 0
    # limit the carbon intensity to 0 and 1600 kgCO2eq/MWh
    df_copy[f"{prefix}Carbon Intensity Production"] = (
        df_copy[f"{prefix}Carbon Intensity Production"]
        .replace([float("inf"), float("-inf")], 0)
        .clip(0, 1600)
    )

    return (
        df_copy[f"{prefix}Carbon Intensity Production"],
        df_copy[f"{prefix}Carbon Emissions Production"],
    )


def calculate_ci_teta(
    df: pd.DataFrame,
    areas: list[str],
    plants: list[tuple[str, str, float]] = [],
    prev_row: pd.Series = None,
) -> pd.DataFrame:
    print(
        f"Calculate CI Teta for df starting at {df.index[0]} and ending at {df.index[-1]} with {len(df)} rows"
    )
    # copy the dataframe to avoid modifying the original
    df_copy: pd.DataFrame = df.copy()
    df_index = df_copy.index
    first_index = df_index[0]
    # pre fill plants
    for area, plant, default_val in plants:
        print(f"Prepare {area} @{plant}")
        if prev_row is None:
            df_copy.loc[first_index, f"{area} @{plant} Carbon Intensity"] = default_val
            df_copy.loc[first_index, f"{area} @{plant} Carbon Emissions"] = (
                df_copy.loc[first_index, f"{area} @{plant} Carbon Intensity"]
                * df_copy.loc[first_index, f"{area} @{plant} Filling Level"]
            )
        # setup plant production
        df_copy[f"{area} @{plant} Production"] = (
            df_copy[f"{area} @{plant} Hydro Pumped-storage Production"]
            .fillna(0.0)
            .clip(lower=0.0)
        )
        df_copy[f"{area} @{plant} Pumping"] = (
            df_copy[f"{area} @{plant} Hydro Pumped-storage Production"]
            .fillna(0.0)
            .clip(upper=0.0)
        )
    # init results
    results: list[pd.Series] = []
    # iterate over index
    # for i in df_index:
    idx = 0
    length = len(df_copy)
    for ts, row in df_copy.iterrows():
        idx += 1
        print(f"Index {ts} // {idx} of {length} // {idx/length*100:.2f}%")
        # skip first row
        if prev_row is None:
            results.append(pd.Series(dtype="float64"))
            prev_row = row
            print(f"Skip first index {ts}")
            continue
        # initialize results for timestamp
        ts_results: pd.Series = pd.Series(dtype="float64")
        plant_results: pd.Series = pd.Series(dtype="float64")
        # calculate production-based carbon intensity for each area
        for area in areas:
            ci_prod, ce_prod, total_prod = calculate_ci_prod_series(
                df_copy.loc[ts].copy(), area
            )
            if np.isinf(ci_prod) or np.isnan(ci_prod):
                ci_prod = 0.0
            elif ci_prod < 0.0:
                ci_prod = 0.0
            elif ci_prod > 1600.0:
                ci_prod = 1600.0
            ts_results[f"{area} Carbon Intensity Production"] = ci_prod
            ts_results[f"{area} Carbon Emissions Production"] = ce_prod
            ts_results[f"{area} Total Production"] = total_prod
        # add emissions from each plant production
        for area, plant, _ in plants:
            ce_stor = (
                prev_row[f"{area} @{plant} Carbon Intensity"]
                * row[f"{area} @{plant} Production"]
            )
            # reduce emissions in storage by the produced emissions
            plant_results[f"{area} @{plant} Carbon Emissions"] = (
                prev_row[f"{area} @{plant} Carbon Emissions"] - ce_stor
            )
            # if no emissions in storage, skip the adjustment of areas emissions
            if ce_stor == 0:
                continue
            # add emissions to area emissions
            ce_prod = ts_results[f"{area} Carbon Emissions Production"] + ce_stor
            ci_prod = ce_prod / ts_results[f"{area} Total Production"]
            if np.isinf(ci_prod) or np.isnan(ci_prod):
                ci_prod = 0
            elif ci_prod < 0:
                ci_prod = 0
            elif ci_prod > 1600:
                ci_prod = 1600
            ts_results[f"{area} Carbon Intensity Production"] = ci_prod
            ts_results[f"{area} Carbon Emissions Production"] = ce_prod
        # add additional necessary columns for CI cons calculation
        for area in areas:
            ts_results[f"{area} Consumption"] = row[f"{area} Consumption"]
            ts_results[f"{area} Net Export"] = row[f"{area} Net Export"]
            if f"{area} Hydro Pumped-storage Pumping" in row:
                ts_results[f"{area} Hydro Pumped-storage Pumping"] = row[
                    f"{area} Hydro Pumped-storage Pumping"
                ]
            for o_area in areas:
                if area == o_area or f"{o_area}>{area} Exchange" not in row:
                    continue
                ts_results[f"{o_area}>{area} Exchange"] = row[
                    f"{o_area}>{area} Exchange"
                ]
        # calculate consumption-based carbon intensity for each area
        ci_cons_series: pd.Series = calculate_ci_cons_series(ts_results, areas)
        for area in areas:
            ci_cons = ci_cons_series[f"{area} Carbon Intensity Consumption"]
            # if np.isinf(ci_cons) or np.isnan(ci_cons) or ci_cons < 0:
            #     ci_cons = 0
            # elif ci_cons > 1600:
            #     ci_cons = 1600
            ts_results[f"{area} Carbon Intensity Consumption"] = ci_cons
            ts_results[f"{area} Carbon Emissions Consumption"] = ci_cons_series[
                f"{area} Carbon Emissions Consumption"
            ]
            ts_results[f"{area} Total Load"] = ci_cons_series[f"{area} Total Load"]
        # add emissions to each plant pumping
        for area, plant, _ in plants:
            # calculate emissions from pumping
            pump_emissions = (
                ts_results[f"{area} Carbon Intensity Consumption"]
                * row[f"{area} @{plant} Pumping"]
            )
            # if no pumping, plant has been producing and CI stays the same
            if pump_emissions == 0:
                plant_results[f"{area} @{plant} Carbon Intensity"] = prev_row[
                    f"{area} @{plant} Carbon Intensity"
                ]
            else:
                # plant was pumping, CI needs to be adjusted
                ce_stor = prev_row[f"{area} @{plant} Carbon Emissions"] - pump_emissions
                ci_stor = ce_stor / row[f"{area} @{plant} Filling Level"]
                if np.isinf(ci_stor) or np.isnan(ci_stor):
                    ci_stor = 0
                elif ci_stor < 0:
                    ci_stor = 0
                elif ci_stor > 1600:
                    ci_stor = 1600
                plant_results[f"{area} @{plant} Carbon Emissions"] = ce_stor
                plant_results[f"{area} @{plant} Carbon Intensity"] = ci_stor
            # add results to plant results
            ts_results[f"{area} @{plant} Carbon Intensity"] = plant_results[
                f"{area} @{plant} Carbon Intensity"
            ]
            ts_results[f"{area} @{plant} Carbon Emissions"] = plant_results[
                f"{area} @{plant} Carbon Emissions"
            ]
            ts_results[f"{area} @{plant} Filling Level"] = row[
                f"{area} @{plant} Filling Level"
            ]
        # add results to list
        results.append(ts_results)
        # update previous row
        prev_row = ts_results

    # return a df with the results and the index
    return pd.DataFrame(results, index=df_index)


def calculate_ci_cons(
    df: pd.DataFrame, areas: list[str], suffix: str = ""
) -> pd.DataFrame:
    # copy the dataframe to avoid modifying the original
    df_copy: pd.DataFrame = df.copy()
    # iterate over areas
    for area in areas:
        # remove old columns
        df_copy = df_copy.drop(
            columns=[
                f"{area} Carbon Intensity Consumption",
                f"{area} Carbon Emissions Consumption",
                f"{area} Total Load",
            ],
            errors="ignore",
        )
        # initialize columns
        # df_copy[f"{area} Carbon Intensity Consumption"] = 0.0
        # df_copy[f"{area} Carbon Emissions Consumption"] = 0.0
        # df_copy[f"{area} Total Load"] = 0.0
        # replace nan values with 0
        df_copy[
            [
                # f"{area} Carbon Intensity Production{suffix}",
                # f"{area} Carbon Emissions Production{suffix}",
                f"{area} Total Production",
                f"{area} Consumption",
                # f"{area} Hydro Pumped-storage Pumping",
                f"{area} Net Export",
            ]
        ].fillna(0.0, inplace=True)
        # replace nan values with 0 for pumped-storage pumping
        df_copy[f"{area} Hydro Pumped-storage Pumping"] = (
            df_copy[f"{area} Hydro Pumped-storage Pumping"].fillna(0.0)
            if f"{area} Hydro Pumped-storage Pumping" in df_copy.columns
            else 0.0
        )
        # iterate over other areas
        for o_area in areas:
            # skip if area is the same as o_area
            if area == o_area:
                continue
            # replace nan values with 0 for exchange columns
            exchange_columns = df_copy.filter(regex="Exchange$").columns
            df_copy[exchange_columns] = df_copy[exchange_columns].fillna(0.0)

    # trace the flows through the network
    flows_df = df_copy.apply(lambda x: trace_flows(x, areas), axis=1)

    # only keep upstream flows (in one direction)
    flows_df.clip(lower=0.0, inplace=True)  # TODO

    # add flows to the dataframe
    df_copy = pd.concat([df_copy, flows_df], axis=1)
    # remove old columns
    df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]

    # iterate over areas
    for area in areas:
        # calculate the total load for the area
        df_copy[f"{area} Total Load"] = (
            df_copy[f"{area} Consumption"]
            - df_copy[f"{area} Hydro Pumped-storage Pumping"]  # pumping is negative
            + df_copy[f"{area} Net Export"]
        ).replace(0.0, np.nan)
        # initialize carbon emissions for consumption
        df_copy[f"{area} Carbon Emissions Consumption"] = df_copy[
            f"{area} Carbon Emissions Production{suffix}"
        ]

        # iterate over other areas
        for o_area in areas:
            # skip if area is the same as o_area
            if area == o_area:
                continue

            # add carbon emissions from exchange to consumption
            df_copy[f"{area} Carbon Emissions Consumption"] += np.where(
                df_copy[f"{o_area}>{area} Flow"] > 0,
                # add carbon emission from o_area to area
                df_copy[f"{o_area}>{area} Flow"]
                * df_copy[f"{o_area} Carbon Intensity Production{suffix}"],
                # remove carbon emission from area to o_area
                df_copy[f"{o_area}>{area} Flow"]
                * df_copy[f"{area} Carbon Intensity Production{suffix}"],
            )

        # replace negative values with 0
        df_copy[f"{area} Carbon Emissions Consumption"] = df_copy[
            f"{area} Carbon Emissions Consumption"
        ].clip(lower=0)

        # calculate the carbon intensity of the consumption
        df_copy[f"{area} Carbon Intensity Consumption"] = (
            np.where(
                df_copy[f"{area} Net Export"] >= 0.0,
                # if net export is positive, the CI is the same as the production CI
                df_copy[f"{area} Carbon Intensity Production{suffix}"],
                # if net export is negative, the CI must be computed
                df_copy[f"{area} Carbon Emissions Consumption"]
                / df_copy[f"{area} Total Load"],
            )
        ).clip(0.0, 1600.0)

    # return all columns with carbon intensity, carbon emissions, and total load
    return df_copy.filter(
        regex="(Carbon Intensity Consumption|Carbon Emissions Consumption|Total Load)$"
    )


def calculate_ci_cons_series(
    series: pd.Series, areas: list[str], suffix: str = ""
) -> pd.Series:
    # iterate over areas
    for area in areas:
        # series[f"{area} Carbon Intensity Consumption"] = 0.0
        # series[f"{area} Carbon Emissions Consumption"] = 0.0
        # series[f"{area} Total Load"] = 0.0
        # replace nan values with 0
        # series[
        #     [
        #         f"{area} Carbon Intensity Production{suffix}",
        #         f"{area} Carbon Emissions Production{suffix}",
        #         f"{area} Total Production",
        #         f"{area} Consumption",
        #         # f"{area} Hydro Pumped-storage Pumping",
        #         f"{area} Net Export",
        #     ]
        # ] =
        # series[
        #     [
        #         f"{area} Carbon Intensity Production{suffix}",
        #         f"{area} Carbon Emissions Production{suffix}",
        #         f"{area} Total Production",
        #         f"{area} Consumption",
        #         # f"{area} Hydro Pumped-storage Pumping",
        #         f"{area} Net Export",
        #     ]
        # ].fillna(
        #     0.0, inplace=True
        # )
        # replace nan values with 0 for pumped-storage pumping
        if f"{area} Hydro Pumped-storage Pumping" not in series.index:
            series[f"{area} Hydro Pumped-storage Pumping"] = 0.0
        # iterate over other areas
        for o_area in areas:
            # skip if area is the same as o_area
            if area == o_area:
                continue
            # replace nan values with 0 for exchange columns
            exchange_columns = series.filter(regex="Exchange$").index
            series[exchange_columns] = series[exchange_columns].fillna(0.0)
    # fill all nans with zero
    series = series.fillna(0.0)
    # trace the flows through the network
    flows: pd.Series = trace_flows(series, areas)
    # only keep upstream flows (in one direction)
    flows.clip(lower=0, inplace=True)  # TODO
    # add flows to the dataframe
    series = pd.concat([series, flows])
    # iterate over areas
    for area in areas:
        # calculate the total load for the area
        series[f"{area} Total Load"] = (
            series[f"{area} Consumption"]
            - series[f"{area} Hydro Pumped-storage Pumping"]  # pumping is negative
            + series[f"{area} Net Export"]
        )
        # initialize carbon emissions for consumption
        series[f"{area} Carbon Emissions Consumption"] = series[
            f"{area} Carbon Emissions Production{suffix}"
        ]
        # iterate over other areas
        for o_area in areas:
            # skip if area is the same as o_area
            if area == o_area:
                continue
            # add carbon emissions from exchange to consumption
            series[f"{area} Carbon Emissions Consumption"] += (
                series[f"{o_area}>{area} Flow"]
                * series[f"{o_area} Carbon Intensity Production{suffix}"]
            )
            # np.where(
            #     series[f"{o_area}>{area} Flow"] > 0.0,
            #     # add carbon emission from o_area to area
            #     series[f"{o_area}>{area} Flow"]
            #     * series[f"{o_area} Carbon Intensity Production{suffix}"],
            #     # remove carbon emission from area to o_area
            #     series[f"{o_area}>{area} Flow"]
            #     * series[f"{area} Carbon Intensity Production{suffix}"],
            # )
        # replace negative values with 0
        series[f"{area} Carbon Emissions Consumption"] = series[
            f"{area} Carbon Emissions Consumption"
        ].clip(min=0)
        # calculate the carbon intensity of the consumption
        series[f"{area} Carbon Intensity Consumption"] = np.where(
            series[f"{area} Net Export"] >= 0.0,
            # if net export is positive, the CI is the same as the production CI
            series[f"{area} Carbon Intensity Production{suffix}"],
            # if net export is negative, the CI must be computed
            series[f"{area} Carbon Emissions Consumption"]
            / series[f"{area} Total Load"],
        ).clip(0.0, 1600.0)
        # if series[f"{area} Net Export"] >= 0:
        #     # if net export is positive, the CI is the same as the production CI
        #     series[f"{area} Carbon Intensity Consumption"] = series[
        #         f"{area} Carbon Intensity Production{suffix}"
        #     ]
        # else:
        #     # if net export is negative, the CI must be computed
        #     series[f"{area} Carbon Intensity Consumption"] = (
        #         series[f"{area} Carbon Emissions Consumption"]
        #         / series[f"{area} Total Load"]
        #     )
        # limit the carbon intensity to 0 and 1600 kgCO2eq/MWh
        # series[f"{area} Carbon Intensity Consumption"] = series[
        #     f"{area} Carbon Intensity Consumption"
        # ].clip(0, 1600)

    # return all columns with carbon intensity, carbon emissions, and total load
    return series.filter(
        regex="(Carbon Intensity Consumption|Carbon Emissions Consumption|Total Load)$"
    )
