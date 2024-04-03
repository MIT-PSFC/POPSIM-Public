import diffrax
import jax.numpy as jnp
import popsim.physics.hmode_dynamics as hmode
from popsim.interp import resolve_paths

def test_state():
    state = hmode.State(hmode=0.5)
    assert state.hmode == 0.5
    assert state.in_hmode == True

    state = hmode.State(hmode=1.5)
    assert state.hmode == 1.0

    state_dot = hmode.State(hmode=10.0, is_derivative=True)
    assert state_dot.hmode == 10.0


def test_sim_and_clip():
    state = hmode.State(hmode=0.0)
    lh_threshold_MW = 10.0
    hl_threshold_MW = 7.0

    #
    # Three phases:
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

    params = hmode.Params(transition_characteristic_time=0.1, P_tau_MW=conducted_powers_traj, P_input_MW=input_powers_traj, hl_threshold_MW=hl_threshold_MW, lh_threshold_MW=lh_threshold_MW)

    def fun(t, y, args):
        params_t = resolve_paths(params, t)
        return hmode.dynamics(y, params_t)


    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(fun),
        solver=diffrax.Tsit5(),
        t0=times[0],
        t1=times[-1],
        dt0=jnp.min(jnp.diff(times)),
        y0=state,
        args=params,
        saveat=diffrax.SaveAt(ts=times),
    )

    in_hmodes = sol.ys.in_hmode
    assert in_hmodes[0] == False
    assert in_hmodes[1] == True
    assert in_hmodes[2] == True
    assert in_hmodes[3] == True
    assert in_hmodes[4] == True
    assert in_hmodes[5] == False
    assert in_hmodes[6] == False