from typing import Callable, Literal, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from popsim.ml.rtd_activation import Activation


class RtdMLP(eqx.Module, strict=True):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Activation = eqx.field(static=True)
    final_activation: Activation = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = Activation.RELU,
        final_activation: Callable = Activation.IDENTITY,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32 if dtype is None else dtype
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0]))
        else:
            layers.append(eqx.nn.Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[0]))
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width_size, width_size, use_bias, dtype=dtype, key=keys[i + 1]))  # noqa: PERF401
            layers.append(eqx.nn.Linear(width_size, out_size, use_final_bias, dtype=dtype, key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None, return_jacobian: bool = False) -> Array:
        if return_jacobian:
            jac = jnp.eye(x.size)  # Initialize Jacobian as identity

        for layer in self.layers[:-1]:
            x = layer(x)

            if return_jacobian:
                activation_jac = self.activation.get_jac_fn()(x)
                jac = activation_jac @ layer.weight @ jac
            x = self.activation.get_fn()(x)

        x = self.layers[-1](x)

        if return_jacobian:
            activation_jac = self.final_activation.get_jac_fn()(x)
            jac = activation_jac @ self.layers[-1].weight @ jac

        x = self.final_activation.get_fn()(x)

        if return_jacobian:
            return x, jac
        else:
            return x
