import chex

from popsim import tree_util
from popsim.interfaces import equil_data


def test_builders():
    # MATLAB files are not loading in CI unless we disable the integrity check.
    # https://stackoverflow.com/questions/42607271/unable-to-read-mat-file-with-scipy
    model = equil_data.load(verify_compressed_data_integrity=False)

    ts_test = [0.0, 0.5, 1.0]
    for t in ts_test:
        equil_slice = model.equil_ff.evaluate(t)

        # Check we can build a state tree and reconstruct the original array.
        state_tree = model.tree_builder.build_state_tree(equil_slice.x0)
        x0_reconstructed = tree_util.leaves_as_array(state_tree)
        chex.assert_trees_all_close(equil_slice.x0, x0_reconstructed)

        # Check we can build an output tree and reconstruct the original array.
        output_tree = model.tree_builder.build_output_tree(equil_slice.y0)
        y0_reconstructed = tree_util.leaves_as_array(output_tree)
        chex.assert_trees_all_close(equil_slice.y0, y0_reconstructed)
