import popsim.interfaces.equil_data as equil_data
import popsim.tree_util as tree_util
import chex

def test_builders():
    equil, tree_builder = equil_data.load()

    ts_test = [0.0, 0.5, 1.0]
    for t in ts_test:
        equil_slice = equil.evaluate(t)

        # Check we can build a state tree and reconstruct the original array.
        state_tree = tree_builder.build_state_tree(equil_slice.x0)
        x0_reconstructed = tree_util.leaves_as_array(state_tree)
        chex.assert_trees_all_close(equil_slice.x0, x0_reconstructed)

        # Check we can build an output tree and reconstruct the original array.
        output_tree = tree_builder.build_output_tree(equil_slice.y0)
        y0_reconstructed = tree_util.leaves_as_array(output_tree)
        chex.assert_trees_all_close(equil_slice.y0, y0_reconstructed)