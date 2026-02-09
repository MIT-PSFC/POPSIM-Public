import xarray as xr

from popsim.math_utils import eigen_decompose


def compute_eigs(ds: xr.Dataset):
    eigval, modes = xr.apply_ufunc(
        eigen_decompose,
        ds["A"],
        input_core_dims=[["state_out", "state"]],
        output_core_dims=[["eigenmode"], ["state", "eigenmode"]],
        vectorize=True,
    )

    ds["Aeigvals"] = eigval.real
    ds["Aeigenmodes"] = modes.real
    return ds


def strip_name_prefix(inp: dict | str, sep: str = ".") -> dict | str:
    if isinstance(inp, str):
        return inp.split(sep)[-1]
    elif isinstance(inp, dict):
        return {k.split(sep)[-1]: v for k, v in inp.items()}
    else:
        raise ValueError(f"Unexpected type: {type(inp)}")
