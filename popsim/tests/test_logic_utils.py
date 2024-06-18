from popsim.logic_utils import select_w_tuples


def test_select_tuples():
    # Case 1: Test that the first True condition is selected.
    conditions_and_choices = [
        (False, 1.0),
        (True, 2.0),
        (True, 3.0),
    ]
    assert select_w_tuples(conditions_and_choices) == 2.0

    # Case 2: Test that the default is selected when all conditions are False.
    conditions_and_choices = [(False, 1.0), (False, 2.0), (False, 3.0)]
    assert select_w_tuples(conditions_and_choices, default=4.0) == 4.0