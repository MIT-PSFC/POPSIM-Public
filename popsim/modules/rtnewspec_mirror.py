import dataclasses

import chex
import jax
import jax.numpy as jnp

from popsim import ModuleBase, discrete_time_field
from popsim.modules.magnetic_diagnostics import BFieldPoloidalProbes

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
        # Toroidal angle between probes [deg]
        d_theta: float
        # Probe 1 identifier in the BFieldPoloidalProbes module
        probe1_id: str
        # Probe 2 identifier in the BFieldPoloidalProbes module
        probe2_id: str
        # Number of probe samples to use for the FFT
        nsamples: int = dataclasses.field(default_factory=lambda: FFT_SAMPLES)
        # TODO(ZanderKeith): Some factor needs to be included to get [T] out of the FFT.
        alpha: float = dataclasses.field(default_factory=lambda: 1.0)
        # Sample rate of the magnetic probes [Hz]
        f_probe: float = dataclasses.field(default_factory=lambda: SAMPLING_FREQUENCY)
        # Number of samples in frequency space to smooth over
        nsmth: int = dataclasses.field(default_factory=lambda: 3)
        # Maximum number of modes to look at.
        max_modes: int = dataclasses.field(default_factory=lambda: 3)
        # The frequency of updating the calculated RMS values [Hz]
        f_report: float = dataclasses.field(default_factory=lambda: REPORT_FREQUENCY)

    @chex.dataclass
    class State:
        # The last nsamples of magnetic probe data
        probe1_data: jnp.ndarray = discrete_time_field(default=jnp.zeros(FFT_SAMPLES))
        probe2_data: jnp.ndarray = discrete_time_field(default=jnp.zeros(FFT_SAMPLES))
        #previous_rms: float = discrete_time_field(default=0.0)
        #time_until_next_report: float = discrete_time_field(default=0.0)
        uninitialized: bool = discrete_time_field(default=True)

    @chex.dataclass
    class Output:
        rms: dict[int, float] # Mode number to RMS value

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

        # returned_rms = jax.lax.cond(
        #     state.time_until_next_report <= 0.0,
        #     lambda _: self.calculate_rms(probe1_data, probe2_data),
        #     lambda _: state.previous_rms,
        #     operand=None,
        # )
        rms_values = self.calculate_rms(probe1_data, probe2_data)

        state = RTNewSpecMirror.State(
            probe1_data=probe1_data,
            probe2_data=probe2_data,
            uninitialized=False
        )
        output = RTNewSpecMirror.Output(
            rms=rms_values
        )

        return state, output

    def calculate_rms(self, probe1_data, probe2_data):
        # Calculate the RMS of the signal in a similar way to rtnewspec
        # 1. Take FFT of both signals
        # 2. Get auto spectrup of probe 1, smooth based on nsmth with a boxcar average
        # 3. Calculate cross spectrum of probe 1 and probe 2
        # 4. Find coherence of probe 1 and probe 2 signals
        # 5. Filter cross spectrum to only have data where the coherence is above a threshold, and the phase matches an n-th mode

        probe1_fft = jnp.fft.fft(probe1_data)
        probe2_fft = jnp.fft.fft(probe2_data)

        # Calculate the auto spectrum of probe 1
        auto_spectrum = jnp.abs(probe1_fft) ** 2

        # Smooth the auto spectrum
        boxcar = jnp.ones(self.config.nsmth) / self.config.nsmth
        smoothed_auto_spectrum = jnp.convolve(auto_spectrum, boxcar, mode='same')

        # Calculate the cross spectrum of probe 1 and probe 2
        complex_cross_spectrum = probe1_fft * jnp.conj(probe2_fft)
        cross_phase = jnp.angle(complex_cross_spectrum)

        # Calculate coherence and set up 95% confidence interval
        coherence = jnp.abs(complex_cross_spectrum) ** 2 / (auto_spectrum * jnp.abs(probe2_fft) ** 2)
        c95 = 1.96/jnp.sqrt(2.0*self.config.nsmth - 2.0)
        c95 = (jnp.exp(c95)-jnp.exp(-c95))/(jnp.exp(c95)+jnp.exp(-c95))
        c95 = c95*c95

        rms_values = {}

        # For each mode, filter the auto spectrum based on coherence and phase
        for mode in range(1, self.config.max_modes + 1):
            filtered_spectrum = jnp.where(
                (coherence > c95) & (jnp.cos(cross_phase - mode * self.config.d_theta) > 0),
                smoothed_auto_spectrum,
                0.0
            )
            # Get the highest value in the filtered spectrum
            rms = jnp.max(filtered_spectrum)*self.config.alpha
            rms_values[mode] = rms

        return rms_values