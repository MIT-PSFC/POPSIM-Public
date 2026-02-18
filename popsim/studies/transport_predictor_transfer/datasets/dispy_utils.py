import numpy as np
from disruption_py.machine.tokamak import Tokamak
from disruption_py.workflow import get_database
from loguru import logger


def summary(
    summary_table: str,
    ipmax: float,
    pulse_length: float,
    min_shot: int,
    max_shot: int,
    shots: list[int] | bool = False,
) -> np.ndarray:
    """Perform a SELECT query on the `summary` table to find shots with high enough current and long enough pulse length.

    Optionally select shots from a given list.
    Snagged from https://github.com/MIT-PSFC/disruption-efit/blob/main/disruption_efit/sql.py

    Args:
        summary_table (str): name of the summary table to query.
        ipmax (float): threshold that maximum plasma current must exceed [A].
        pulse_length (float): threshold that pulse length must exceed [s].
        min_shot (int): disregard shots below this number.
        max_shot (int): disregard shots above this number.
        shots (list[int] | bool): list of shots to be queried. Defaults to False.

    Returns:
        np.ndarray: Nx2 array of [shot_id, pulse_length] for the N shots which exceed the thresholds.
    """

    # database
    db = get_database(tokamak=Tokamak.CMOD)

    # query
    query = [
        f"select distinct(shot), pulse_length from {summary_table} ",
        f"where ipmax > {ipmax} and pulse_length > {pulse_length}",
    ]
    if min_shot > 0:
        query += [f"and shot >= {min_shot}"]
    if max_shot > 0:
        query += [f"and shot <= {max_shot}"]
    if hasattr(shots, "__iter__"):
        query += [f"and shot in ({', '.join(str(s) for s in shots)})"]
    query += ["order by shot"]
    logger.trace("> {query}", query=" ".join(query))

    # results
    data = db.query(" ".join(query), use_pandas=True).values

    logger.trace("= {shape}", shape=data.shape)
    return data


def get_shotlist_from_sql(
    summary_table: str, ipmax: float, pulse_length: float, min_shot: int, max_shot: int, num_shots: int | None = None
) -> list[int]:
    """Get a list of shots from the SQL database that meet the given criteria."""
    data = summary(
        summary_table=summary_table,
        ipmax=ipmax,
        pulse_length=pulse_length,
        min_shot=min_shot,
        max_shot=max_shot,
        shots=False,
    )
    shotlist = data[:, 0].astype(int).tolist()
    if num_shots is not None:
        shotlist = shotlist[:num_shots]
    return shotlist
