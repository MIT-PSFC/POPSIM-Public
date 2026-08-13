import diffrax
import jax.numpy as jnp

from popsim.modules.hmode_dynamics import HmodeDynamics
from popsim.simulate import SimInput, StepperType, simulate


def test_sim_and_clip():
    state = HmodeDynamics.State(hmode=0.0)
    lh_threshold_MW = 10.0
    hl_threshold_MW = 7.0

    #
    # Four phases:
    #   1) Conducted power well above LH threshold.
    #   2) Conducted power above HL threshold but below LH threshold.
    #   3) Conducted power well below HL threshold.
    #   4) Conducted power above LH threshold, but input power below LH threshold.
    #
    times = jnp.array([0.0, 0.5, 0.6, 1.0, 1.01, 1.11, 1.3])
    conducted_powers = jnp.array([15.0, 15.0, 9.0, 9.0, 7.0, 5.0, 15.0])
    conducted_powers_traj = diffrax.LinearInterpolation(ts=times, ys=conducted_powers)
    input_powers = jnp.array([15.0, 15.0, 15.0, 15.0, 15.0, 0.0, 0.0])
    input_powers_traj = diffrax.LinearInterpolation(ts=times, ys=input_powers)

    inputs = HmodeDynamics.Inputs(transition_characteristic_time=0.1, P_tau_MW=conducted_powers_traj, P_input_MW=input_powers_traj, hl_threshold_MW=hl_threshold_MW, lh_threshold_MW=lh_threshold_MW)

    hmode_module = HmodeDynamics(config=HmodeDynamics.Config())

    sol = simulate(hmode_module, SimInput(time=times, initial_state=state, inputs=inputs), return_xarray=False, stepper_type=StepperType.DIFFRAX_TSIT5)

    in_hmodes = sol.ys['state'].in_hmode
    
    assert in_hmodes[0] == False
    assert in_hmodes[1] == True
    assert in_hmodes[2] == True
    assert in_hmodes[3] == True
    assert in_hmodes[4] == True
    assert in_hmodes[5] == False
    assert in_hmodes[6] == False