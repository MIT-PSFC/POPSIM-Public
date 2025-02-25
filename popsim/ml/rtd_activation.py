from enum import Enum

import jax
import jax.numpy as jnp


class Activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"

    def get_fn(self):
        if self == Activation.RELU:
            return jax.nn.relu
        elif self == Activation.SIGMOID:
            return jax.nn.sigmoid
        elif self == Activation.SOFTMAX:
            return jax.nn.softmax
        elif self == Activation.SOFTPLUS:
            return jax.nn.softplus
        elif self == Activation.IDENTITY:
            return lambda x: x
        else:
            raise ValueError(f"Activation {self} not recognized")

    def get_jac_fn(self):
        if self == Activation.RELU:
            return lambda x: jnp.diag(jnp.where(x > 0, 1.0, 0.0))

        elif self == Activation.SIGMOID:

            def sigmoid_jac(x):
                sig_x = jax.nn.sigmoid(x)
                return jnp.diag(sig_x * (1 - sig_x))

            return sigmoid_jac

        elif self == Activation.SOFTMAX:

            def softmax_jac(x):
                sm_x = jax.nn.softmax(x)
                return jnp.diag(sm_x) - jnp.outer(sm_x, sm_x)

            return softmax_jac

        elif self == Activation.SOFTPLUS:
            return lambda x: jnp.diag(jax.nn.sigmoid(x))  # Jacobian is diag(sigmoid(x))

        elif self == Activation.IDENTITY:
            return lambda x: jnp.eye(x.shape[0])
        else:
            raise ValueError(f"Activation {self} not recognized")
