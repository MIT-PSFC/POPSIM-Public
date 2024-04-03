import popsim.algorithms.hmode_dynamics as hmode

def test_state():
    state = hmode.State(hmode=0.5)
    assert state.hmode == 0.5
    assert state.CRITICAL_THRESHOLD == 0.5
    assert state.in_hmode == True

    state = hmode.State(hmode=1.5)
    assert state.hmode == 1.0