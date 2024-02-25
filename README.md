# POPSIM: Plasma OPerational SIMulation
POPSIM is a control-oriented time-dependent tokamak plasma simulator build in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable.

It's role is to enable:
1) Sensitivity analysis via massively parallelized Monte Carlo simulation on GPU and automatic differentiation
2) Optimization of scenarios, trajectories, and controllers via techniques from optimal control + reinforcement learning
3) Learning unknown dynamics from data

Is POPSIM a machine learning (ML) model or a physics-based simulator? It's both! For a primer on this paradigm, consider checking out this [tutorial on differentiable physics](https://physicsbaseddeeplearning.org/diffphys.html) and this [textbook on scientific machine learning](https://book.sciml.ai/).

POPSIM is the successor to the work done in the paper [Active Disruption Avoidance and Trajectory Design for Tokamak Ramp-downs with Neural Differential Equations and Reinforcement Learning](https://arxiv.org/pdf/2402.09387.pdf).

## Installation
1) Run a `git submodule update --init --recursive` to make sure cfspopcon is available.
2) If you don't have it, [install poetry](https://python-poetry.org/docs/#installation) and make sure it is version >= 1.6.1.
3) In the root of this repo, run a `poetry install`
4) CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
5) Be sure to use either `poetry shell` or prefix all of your commands with `poetry run`. If you are confused, check out the [poetry docs](https://python-poetry.org/docs/basic-usage/)
6) Ping Allen W. if things don't work.
