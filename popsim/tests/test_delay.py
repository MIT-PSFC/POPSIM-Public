from popsim.modules import delay
import jax.numpy as jnp
from popsim.simulate import simulate

def test_delay_buffer():
    ts = 0.01 * jnp.arange(100)
    module = delay.DelayBuffer()
    n_buffer = 5
    initial_state = delay.DelayBuffer.State(buffer=jnp.zeros(n_buffer))
    params = delay.DelayBuffer.Params(inp=jnp.array(1.0))

    out = simulate(module, ts, initial_state, params, return_xarray=True)

    delayed_vals = out['aux.delayed'].values

    assert jnp.all(delayed_vals[:n_buffer] == 0)
    assert jnp.all(delayed_vals[n_buffer:] == 1)