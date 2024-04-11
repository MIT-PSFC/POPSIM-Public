# POPSIM: Plasma OPerational SIMulation
POPSIM is a control-oriented time-dependent tokamak plasma simulator build in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable.

It's role is to enable:
1) Sensitivity analysis via massively parallelized Monte Carlo simulation on GPU and automatic differentiation
2) Uncertainty-aware optimization of scenarios, trajectories, and controllers via techniques from optimal control + reinforcement learning
3) Learning unknown and computationally expensive dynamics from data

Is POPSIM a machine learning (ML) model or a physics-based simulator? It's both! For a primer on this paradigm, consider checking out this [tutorial on differentiable physics](https://physicsbaseddeeplearning.org/diffphys.html) and this [textbook on scientific machine learning](https://book.sciml.ai/).



## Installation
1) Run a `git submodule update --init --recursive` to make sure you have all the necessary submodules.
2) If you don't have it, [install poetry](https://python-poetry.org/docs/#installation) and make sure it is version >= 1.6.1.
3) In the root of this repo, run a `poetry install`
4) CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
5) Be sure to use either `poetry shell` or prefix all of your commands with `poetry run`. If you are confused, check out the [poetry docs](https://python-poetry.org/docs/basic-usage/)

## Simulators
POPSIM is designed to be a collection of Jax-based simulators that can interop (at least somewhat) with each other. As of March 27th 2024, there are two "simulators":

1) CometMirror (in `popsim/simulators/comet_mirror`)
2) Torax (POPSIM code that is using torax is in `popsim/simulators/torax`)