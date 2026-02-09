import xarray as xr
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter(order, cutoff, fs=fs, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


def lowpass_xr(ds, time_var, cutoff, order: int = 4):
    dts = ds[time_var].diff(time_var)
    fs = 1 / dts.mean().item()

    return xr.apply_ufunc(
        lambda data: butter_lowpass_filter(data, cutoff, fs, order),
        ds,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
