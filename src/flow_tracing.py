import numpy as np
import pandas as pd


# The flow tracing algorithm is based on: J. Bialek, “Tracing the flow of electricity,” IEE Proceedings - Generation,
# Transmission and Distribution, vol. 143, no. 4, pp. 313–320, Jul. 1996, [Online]. Available:
# https://www.semanticscholar.org/paper/Tracing-the-flow-of-electricity-Bialek/748e3b963bc427d2b9e444150e637194e64c3eaa


def trace_flows(row: pd.Series, areas: list[str]) -> pd.Series:
    n = len(areas)
    P_g = np.zeros(n)
    P_l = np.zeros(n)
    flows = np.zeros((n, n))
    # iterate over areas
    for i, area in enumerate(areas):
        net_export = row[f"{area} Net Export"]
        # if net export is positive, the node is a source adding power to the network
        if net_export > 0:
            P_g[i] = net_export
        # if net export is negative, the node is a sink removing power from the network
        else:
            P_l[i] = -net_export
        # iterate over other areas
        for j in range(n):
            if i == j:
                continue
            o_area = areas[j]  # if the exchange column exists, use it
            if f"{area}>{o_area} Exchange" in row.index and not pd.isna(
                row[f"{area}>{o_area} Exchange"]
            ):
                # exchange between area and o_area
                flows[i][j] = row[f"{area}>{o_area} Exchange"]
    # calculate the distribution of power
    dist = flow_tracing(P_g, P_l, flows)
    # add flows to df
    columns = np.empty(n * n - n, dtype="U64")
    flows = np.zeros(n * n - n)
    k = 0
    for i in range(n):
        area = areas[i]
        for j in range(n):
            if i == j:
                continue
            o_area = areas[j]
            columns[k] = f"{o_area}>{area} Flow"
            flows[k] = dist[i][j]
            k += 1

    return pd.Series(flows, index=columns)


def flow_tracing(P_g: np.ndarray, P_l: np.ndarray, flows: np.ndarray) -> np.ndarray:
    """
    Calculate the distribution of power using upstream-looking algorithm.

    Args:
        P_g (np.ndarray): The vector of nodal generation.
        P_l (np.ndarray): The vector of nodal demands.
        flows (np.ndarray): The matrix of direct nodal flows. Exports must be positive values, imports must be negative
        values.


    Returns:
        np.ndarray: The distribution of power using upstream-looking algorithm. It gives the amount of power supplied
        by a particular generator to a particular load.
    """
    assert P_g.ndim == 1, "P_g must be a 1D array"
    assert P_g.shape == P_l.shape, "P_g and P_l must have the same shape"
    assert flows.ndim == 2, "flows must be a 2D array"
    assert (
        flows.shape[0] == P_g.shape[0]
    ), "flows must have the same number of rows as P_g"
    assert flows.shape[0] == flows.shape[1], "flows must be a square matrix"

    n = P_g.shape[0]
    # P is the vector of nodal through-flows
    # for upstream that is the sum of inflows (generation plus the imports)
    P = np.zeros(n)
    for i in range(n):
        P[i] = P_g[i] + abs(np.sum(flows[i, flows[i] < 0]))
    # A is the NxN upstream distribution matrix
    # an element is equal to
    # 1 if i = j
    # -1 * (|flow from i to j| / P[j]) if there is a link between i and j
    # 0 if i != j and there is no link between i and j
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = 1
            elif flows[i][j] < 0 and P[j] != 0:
                A[i][j] = -1 * abs(flows[i][j]) / P[j]
    try:
        # A_inv is the inverse of A
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        print(
            f"The flow tracing matrix is probably singular and could not be inverted at index {i}. The matrix A can be found in the following log. The LinaAlgError: {e}"
        )
        df: pd.DataFrame = pd.DataFrame(A)
        print(f"Matrix A CSV:\n{df.to_csv()}")
        print(f"Matrix A JSON:\n{df.to_json()}")
        return np.zeros((n, n))
    # dist is the NxN matrix of the distribution of power
    # it gives the amount of power supplied by a particular generator to a particular load
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or P_g[j] == 0 or P_g[i] > 0 or P[i] == 0:
                continue
            dist[i][j] = (P_l[i] / P[i]) * A_inv[i][j] * P_g[j]
    return dist
