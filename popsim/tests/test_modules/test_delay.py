from popsim.modules import delay
import jax.numpy as jnp
from popsim.simulate import simulate, SimInput

def test_delay_buffer():
    ts = 0.01 * jnp.arange(100)
    module = delay.DelayBuffer()
    n_buffer = 5
    initial_state = delay.DelayBuffer.State(buffer=jnp.zeros(n_buffer))
    inputs = delay.DelayBuffer.Inputs(inp=jnp.array(1.0))

    out = simulate(module, SimInput(time=ts, initial_state=initial_state, inputs=inputs), return_xarray=True)

    delayed_vals = out['output.delayed'].values

    assert jnp.all(delayed_vals[:n_buffer] == 0)
    assert jnp.all(delayed_vals[n_buffer:] == 1)