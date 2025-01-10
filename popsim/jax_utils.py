from contextlib import contextmanager

import jax


@contextmanager
def jax_cpu():
    """Convenience context manager for running JAX code on the CPU."""
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        yield


@contextmanager
def jax_gpu():
    """Convenience context manager for running JAX code on the GPU."""
    gpu_device = jax.devices("gpu")[0]
    with jax.default_device(gpu_device):
        yield
