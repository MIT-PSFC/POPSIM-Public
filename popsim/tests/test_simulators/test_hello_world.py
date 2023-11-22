from popsim.simulators.simple_power_balance.build_default import build_default


def test_hello_world():
    model, state, params = build_default()
    dW_dt, debug = model(state, params, debug_info=True)
