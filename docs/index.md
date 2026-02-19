# Welcome to Plasma OPerational SIMulation (POPSIM)!

!!! warning

    Note that Jupyter Notebooks are currently not being executed when the documentation is built, pending resolution of https://github.com/cfs-energy-internal/POPSIM/issues/91


POPSIM is an experiment at building a control-oriented modelling toolbox built in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable. While it is designed with tokamaks in mind, the core library can be applied to simulating any controlled dynamical system.

While it started off as a project to build a "time-dependent POPCON", POPSIM evolved into a set of tools to allow you to define simulation "modules". Each module is, itself, a stand-alone simulator that can be simulated using a common `simulate` function. More complex simulators can be built by chaining modules together. Data and machine learning pipelines are built in. Some of the tools were used for experiments at TCV [1,2].

## Installation
### Automatic Install
In your terminal (Linux and Mac, sorry Windows users!).

```
bash install.sh
```

### Manual Installation (in case the script doesn't work for you)
   1. Run a `git submodule update --init --recursive` to make sure you have all the necessary submodules.
   2. If you don't have it, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and make sure it is version >= 0.4.0.
   3. In the root of this repo, run a `uv sync`

      a. To install with GPU support, run `uv sync --group gpu`

      b. To install with development dependencies, run `uv sync --group dev`

      c. To install with data dependencies, run `uv sync --group data`

      d. To install with all optional dependencies, run `uv sync --all-groups`

   4. CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
   5. Install git lfs and do a `git lfs install` and a `git lfs pull` to get the necessary data files.
   6. Be sure to use either `uv run` to execute commands in the virtual environment. If you are confused, check out the [uv docs](https://docs.astral.sh/uv/)


## Getting Started
The easiest way to get oriented to POPSIM and many of its core capabilities is by exploring the [TdPopcon Tutorial](./notebooks/td_popcon.ipynb).

## Development Status
### Core Modules Infrastructure
- [x] Continuous time (e.g. diff eq), discrete time (e.g. state machine), hybrid time modules [see "Introduction to Modules"](./notebooks/intro_to_modules.ipynb)
- [x] All modules are simulateable with a common `simulate` function [see "Introduction to Modules"](./notebooks/intro_to_modules.ipynb)
- [x] All modules can have time dependent trajectories as `Inputs` (e.g. controls, disturbances, boundary conditions, inputs from other modules) [see "Using TdPopcon"](./notebooks/td_popcon.ipynb)
- [x] All modules output a common dataset format (`xarray.Dataset`) [see "Using TdPopcon"](./notebooks/td_popcon.ipynb)
    * [x] Support for multi-dimensional output variables 
- [x] Hierarchical module example [see the Controller Plus Sim Tutorial](./notebooks/controller_plus_sim.ipynb)
    * [x] Tools for auto-generating hierarchical module (2026 update: LLMs can probably do this now ;D )
- [ ] Multi-rate simulation
- [x] API for hooking up non-Jax modules (`popsim.simulate.single_step`)
  
### Monte Carlo Simulation Tools
- [x] `simulate` function supports a list of simulation cases, each with a different initial state and `Inputs` struct [see "Using TdPopcon"](./notebooks/td_popcon.ipynb)
    * [x] `simulate` function is both CPU and GPU capable (modulo the setup and post-processing steps)
- [x] Tools to generate multiple simulation cases, including combinatorially [see "Using TdPopcon"](./notebooks/td_popcon.ipynb)
- [x] Tools to generate time-dependent random walks [see "Randomness and Stochasticity](./notebooks/random_and_stochastic.ipynb)
- [x] State-dependent noise with `PRNGModule` [see "Randomness and Stochasticity](./notebooks/random_and_stochastic.ipynb)

### Module Training and Automated Evaluation Tools
- [x] `xarray.Dataset` => `Dataloader` pipeline [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [x] Running a simple module on C-Mod data [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [x] Basic example of automated evaluation of modules [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [x] Support for vectorized evaluation of time-independent modules
- [x] Machine learning tools and pipeline `popsim.ml`

## References
[1] Wang, Allen, et al. "Magnetic Control with an Inverse Grad-Shafranov Neural Network." DPP 2025. 2025.
[2] Wang, Allen M., et al. "Technical Aspects of Plasma Operational Simulation (POPSIM): A Framework for Data-Driven Simulation and Control." arXiv preprint arXiv:2509.10244 (2025).