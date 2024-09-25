import dataclasses

import chex
import jax
import jax.numpy as jnp

from popsim import ModuleBase, discrete_time_field
from popsim.logic_utils import select_w_tuples

FFT_SAMPLES = 2048
SAMPLING_FREQUENCY = 60e3 # Hz
REPORT_FREQUENCY = 1e3 # Hz

"""
Mirror of rtnewspec, used to calculate n1rms (and n2rms, n3rms, etc.) in real-time on DIII-D.
"""

@chex.dataclass
class RTNewSpecMirror(ModuleBase):
    @chex.dataclass
    class Config:
        # Number of probe samples to use for the FFT
        nsamples: int = dataclasses.field(default_factory=lambda: FFT_SAMPLES)
        # TODO(ZanderKeith): Some factor needs to be included to get [T] out of the FFT.
        alpha: float = dataclasses.field(default_factory=lambda: 1.0)
        # Toroidal angle between probes [deg]
        d_theta: float
        # Sample rate of the magnetic probes [Hz]
        f_probe: float = dataclasses.field(default_factory=lambda: SAMPLING_FREQUENCY)
        # Number of samples in frequency space to smooth over
        nsmth: int = dataclasses.field(default_factory=lambda: 3)
        # Maximum number of modes to look at.
        max_modes: int = dataclasses.field(default_factory=lambda: 12)
        # The frequency of updating the calculated RMS values [Hz]
        f_report: float = dataclasses.field(default_factory=lambda: REPORT_FREQUENCY)

    @chex.dataclass
    class State:
        # The last nsamples of magnetic probe data
        probe1_data: jnp.ndarray = discrete_time_field(default=jnp.zeros(FFT_SAMPLES))
        probe2_data: jnp.ndarray = discrete_time_field(default=jnp.zeros(FFT_SAMPLES))
        previous_rms: float = discrete_time_field(default=0.0)
        time_until_next_report: float = discrete_time_field(default=0.0)
        uninitialized: bool = discrete_time_field(default=True)

    @chex.dataclass
    class Output:
        n1rms: float

    @chex.dataclass
    class Params:
        # The currently stored probe data
        probe1_signal: jnp.ndarray
        probe2_signal: jnp.ndarray

    config: Config

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        probe1_shifted_window = jnp.roll(state.probe1_data, shift=1)
        probe2_shifted_window = jnp.roll(state.probe2_data, shift=1)
        probe1_data = probe1_shifted_window.at[0].set(params.probe1_signal)
        probe2_data = probe2_shifted_window.at[0].set(params.probe2_signal)

        # TODO(ZanderKeith): It would be nice to have some way to turn off the expensive
        # FFT calculations when the module is not reporting and just waiting for data to come in.
        # But that isn't really jax-like...
        # Maybe with jax.lax.cond?

        returned_rms = jax.lax.cond(
            state.time_until_next_report <= 0.0,
            lambda _: self.calculate_rms(probe1_data, probe2_data),
            lambda _: state.previous_rms,
            operand=None,
        )

        # Update the time until the next report
        time_until_next_report = state.time_until_next_report - 1 / self.config.f_report
        # Return stuff, woohaa!

    def calculate_rms(self, probe1_data, probe2_data):
        # Calculate the RMS of the signal
        n1rms = self.calculate_rms_for_mode(probe1_data)
        n2rms = self.calculate_rms_for_mode(probe2_data)

        return n1rms, n2rms




