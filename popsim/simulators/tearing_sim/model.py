import chex

from popsim import ModuleBase
from popsim.modules.magnetic_diagnostics import BFieldPoloidalProbes, LowNArray
from popsim.modules.rtnewspec_mirror import RTNewSpecMirror
from popsim.modules.tearing import Tearing

"""
Simulation for tearing modes and all the diagnostics which can measure them.
"""


class TearingSim(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        tearing_module: Tearing
        b_field_poloidal_probes_module: BFieldPoloidalProbes
        lown_array_module: LowNArray
        rtnewspec_mirror_module: RTNewSpecMirror

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        tearing_state: Tearing.State
        rtnewspec_mirror_state: RTNewSpecMirror.State

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        lown_array_out: LowNArray.Output
        b_field_poloidal_probes_out: BFieldPoloidalProbes.Output
        rtnewspec_mirror_out: RTNewSpecMirror.Output

    @chex.dataclass
    class Inputs:
        # Define the, possibly time dependent, inputs that will be passed to the module.
        tearing_inputs: Tearing.Inputs

    config: Config

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        # Compute the results of the Tearing module.
        tearing_state_dot, tearing_out = self.config.tearing_module(state.tearing_state, inputs.tearing_inputs)

        # Build the inputs for the diagnostic modules.
        lown_array_inputs = LowNArray.Inputs(tearing_out=tearing_out, modes=self.config.tearing_module.config.modes)
        b_field_poloidal_probes_inputs = BFieldPoloidalProbes.Inputs(tearing_out=tearing_out, modes=self.config.tearing_module.config.modes)

        lown_array_out = self.config.lown_array_module(None, lown_array_inputs)
        b_field_poloidal_probes_out = self.config.b_field_poloidal_probes_module(None, b_field_poloidal_probes_inputs)

        rtnewspec_inputs = RTNewSpecMirror.Inputs(
            probe1_signal=b_field_poloidal_probes_out.Bp[self.config.rtnewspec_mirror_module.config.probe1_id],
            probe2_signal=b_field_poloidal_probes_out.Bp[self.config.rtnewspec_mirror_module.config.probe2_id],
        )
        rtnewspec_mirror_state, rtnewspec_mirror_out = self.config.rtnewspec_mirror_module(state.rtnewspec_mirror_state, rtnewspec_inputs)

        # Make a state_dot.
        state = TearingSim.State(tearing_state=tearing_state_dot, rtnewspec_mirror_state=rtnewspec_mirror_state)
        out = TearingSim.Output(
            lown_array_out=lown_array_out,
            b_field_poloidal_probes_out=b_field_poloidal_probes_out,
            rtnewspec_mirror_out=rtnewspec_mirror_out,
        )
        return state, out

    def add_to_ods(self, ods, sim_xarray):
        """Given an ods object and the simulation's xarray, fill the ods with the simulation's data.
        This is entirely hard-coded for now. Should decide on a more general structure after we've done a few of these.
        TODO(ZanderKeith) this breaks index ordering. For instance, if a probe was originally at index 100 in the device description
        it might now be at index 5 or something. This is because you can't put an element in an ods array at index x + 1 if there's nothing at index x.
        Since we're only simulating a subset of probes, the indices are all smaller.
        This shouldn't be too much of a problem since the names of the probes are still there,
        we just need to ensure that indices are not used as identifiers elsewhere.

        Args:
        ----
        ods: ODS
            The ods object which will simulation data added to it.
        sim_xarray: xarray.Dataset
            The output of the TearingSim after running the simulation.
        """

        # The only data implemented so far is from the BFieldPoloidalProbes module
        probe_details = self.config.b_field_poloidal_probes_module.config.probe_details

        for i, probe in enumerate(probe_details):
            # Fields should already have a unit associated with it [T]
            ods[f"magnetics.b_field_pol_probe[{i}].area"] = probe["area"]

            ods["magnetics"]["b_field_pol_probe"][i]["field"]["data"] = sim_xarray[f"output.b_field_poloidal_probes_out.Bp.{probe['name']}"]
            ods["magnetics"]["b_field_pol_probe"][i]["field"]["time"] = sim_xarray.time

            ods["magnetics"]["b_field_pol_probe"][i]["identifier"] = probe["identifier"]
            ods["magnetics"]["b_field_pol_probe"][i]["name"] = probe["name"]
            ods["magnetics"]["b_field_pol_probe"][i]["position"]["r"] = probe["position"]["r"]
            ods["magnetics"]["b_field_pol_probe"][i]["position"]["phi"] = probe["position"]["phi"]
            ods["magnetics"]["b_field_pol_probe"][i]["position"]["z"] = probe["position"]["z"]

            ods["magnetics"]["b_field_pol_probe"][i]["turns"] = 1  # Not in the device description?
            ods["magnetics"]["b_field_pol_probe"][i]["type"]["index"] = probe["type"]["index"]
