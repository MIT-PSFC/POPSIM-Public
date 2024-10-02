# POPSIM: Plasma OPerational SIMulation
POPSIM is a control-oriented simulation toolbox built in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable. While it is designed with tokamaks in mind, it's core library can be applied to simulating any controlled dynamical system.

It's role is to enable:
1) Sensitivity analysis and off-normal simulation via massively parallelized Monte Carlo simulation on GPU and automatic differentiation
2) Uncertainty-aware optimization of scenarios, trajectories, and controllers via techniques from optimal control + reinforcement learning
3) Parameter estimation and learning unknown and computationally expensive dynamics from data

POPSIM is not a single simulator, but rather a set of tools to allow you to define simulation "modules". Each module is, itself, a stand-alone simulator that can be simulated using a common `simulate` function. More complex simulators can be built by chaining modules together.

## Automatic Install
In your terminal (Linux and Mac, sorry Windows users!).

```
chmod +x install.sh
sudo ./install.sh
```

## Manual Installation
   1. Run a `git submodule update --init --recursive` to make sure you have all the necessary submodules.
   2. If you don't have it, [install poetry](https://python-poetry.org/docs/#installation) and make sure it is version >= 1.6.1.
   3. In the root of this repo, run a `poetry install`
   
      a. To install with GPU support, run `poetry install --with gpu`

      b. To install with development dependencies, run `poetry install --with dev`

   4. CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
   5. Install git lfs and do a `git lfs install` and a `git lfs pull` to get the necessary data files.
   6. Be sure to use either `poetry shell` or prefix all of your commands with `poetry run`. If you are confused, check out the [poetry docs](https://python-poetry.org/docs/basic-usage/)
