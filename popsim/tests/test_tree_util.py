import jax.numpy as jnp

from popsim import tree_util


def test_tree_transpose():
    def make_tree(x):
        return {"foo": {"a": x + 1.0, "b": x + 2.0}, "bar": x + 3.0}

    seq_of_pytrees = [make_tree(x) for x in range(3)]
    pytree_of_arrays = tree_util.tree_transpose(seq_of_pytrees)

    assert jnp.all(pytree_of_arrays["foo"]["a"] == jnp.array([1.0, 2.0, 3.0]))
    assert jnp.all(pytree_of_arrays["foo"]["b"] == jnp.array([2.0, 3.0, 4.0]))
    assert jnp.all(pytree_of_arrays["bar"] == jnp.array([3.0, 4.0, 5.0]))
