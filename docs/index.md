# Welcome to Plasma OPerational SIMulation (POPSIM)!

!!! warning

    The documentation (and this package overall) is still in an early stage of development. If you have any questions and/or problems, please reach out to the developers.

    Note that Jupyter Notebooks are currently not being executed when the documentation is built, pending resolution of https://github.com/cfs-energy-internal/POPSIM/issues/91


POPSIM is a control-oriented simulation toolbox built in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable. While it is designed with tokamaks in mind, it's core library can be applied to simulating any controlled dynamical system.

It's role is to enable:

1. Sensitivity analysis and off-normal simulation via massively parallelized Monte Carlo simulation on GPU and automatic differentiation
2. Uncertainty-aware optimization of scenarios, trajectories, and controllers via techniques from optimal control + reinforcement learning
3. Parameter estimation and learning unknown and computationally expensive dynamics from data

POPSIM is not a single simulator, but rather a set of tools to allow you to define simulation "modules". Each module is, itself, a stand-alone simulator that can be simulated using a common `simulate` function. More complex simulators can be built by chaining modules together.

## Installation
### Automatic Install
In your terminal (Linux and Mac, sorry Windows users!).

```
bash install.sh
```

### Manual Installation (in case the script doesn't work for you)
   1. Run a `git submodule update --init --recursive` to make sure you have all the necessary submodules.
   2. If you don't have it, [install poetry](https://python-poetry.org/docs/#installation) and make sure it is version >= 1.6.1.
   3. In the root of this repo, run a `poetry install`
   
      a. To install with GPU support, run `poetry install --with gpu`

      b. To install with development dependencies, run `poetry install --with dev`

   4. CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
   5. Install git lfs and do a `git lfs install` and a `git lfs pull` to get the necessary data files.
   6. Be sure to use either `poetry shell` or prefix all of your commands with `poetry run`. If you are confused, check out the [poetry docs](https://python-poetry.org/docs/basic-usage/)


## Getting Started
The easiest way to get oriented to POPSIM and many of its core capabilities is by exploring the [CometMirror Tutorial](./notebooks/comet_mirror.ipynb).

## Development Status
### Core Modules Infrastructure
- [x] Continuous time (e.g. diff eq), discrete time (e.g. state machine), hybrid time modules [see "Introduction to Modules"](./notebooks/intro_to_modules.ipynb)
- [x] All modules are simulateable with a common `simulate` function [see "Introduction to Modules"](./notebooks/intro_to_modules.ipynb)
- [x] All modules can have time dependent trajectories as `Inputs` (e.g. controls, disturbances, boundary conditions, inputs from other modules) [see "Using CometMirror"](./notebooks/comet_mirror.ipynb)
- [x] All modules output a common dataset format (`xarray.Dataset`) [see "Using CometMirror"](./notebooks/comet_mirror.ipynb)
    * [ ] Support for multi-dimensional output variables 
- [x] Hierarchical module example [see the Controller Plus Sim Tutorial](./notebooks/controller_plus_sim.ipynb)
    * [ ] Tools for auto-generating hierarchical modules
- [ ] Multi-rate simulation
- [ ] API for inter-process communication (IPC) with other programs
- [ ] API for hooking up non-Jax modules
  
### Monte Carlo Simulation Tools
- [x] `simulate` function supports a list of simulation cases, each with a different initial state and `Inputs` struct [see "Using CometMirror"](./notebooks/comet_mirror.ipynb)
    * [x] `simulate` function is both CPU and GPU capable (modulo the setup and post-processing steps)
- [x] Tools to generate multiple simulation cases, including combinatorially [see "Using CometMirror"](./notebooks/comet_mirror.ipynb)
- [x] Tools to generate time-dependent random walks [see "Randomness and Stochasticity](./notebooks/random_and_stochastic.ipynb)
- [x] State-dependent noise with `PRNGModule` [see "Randomness and Stochasticity](./notebooks/random_and_stochastic.ipynb)
- [ ] Tools to specify probability distributions in the initial `State` and `Inputs` structs

### Module Training and Automated Evaluation Tools
- [x] `xarray.Dataset` => `Dataloader` pipeline [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [x] Running a simple module on C-Mod data [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [x] Basic example of automated evaluation of modules [see "Dataloading and Xarray-Based Pre-processing
"](./notebooks/preprocess_and_dataloader.ipynb)
- [ ] Support for vectorized evaluation of time-independent modules
- [ ] Model `Trainer`
    * [ ] Gradient-based optimization API
    * [ ] Gradient-free optimization API
    * [ ] Logging and experiment tracking system
    * [ ] Model checkpointing system
    * [ ] `wandb` sweep and hyper-parameter optimization
