# POPSIM: Plasma OPerational SIMulation
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

      c. To install with both GPU and dev dependencies, run `uv sync --group gpu --group dev`

   4. CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
   5. Install git lfs and do a `git lfs install` and a `git lfs pull` to get the necessary data files.
   6. Be sure to use either `uv run` to execute commands in the virtual environment. If you are confused, check out the [uv docs](https://docs.astral.sh/uv/)

## Citing POPSIM
To cite this repository:
```
@software{Wang_POPSIM_Plasma_OPerational,
    author = {Wang, Allen and Keith, Zander and Boyer, M. Dan and Nelson, A. Oak},
    license = {MIT},
    title = {{POPSIM: Plasma OPerational SIMulation}},
    url = {https://github.com/MIT-PSFC/POPSIM-Public},
    version = {0.1.0},
}
```

## References
[1] Wang, Allen, et al. "Magnetic Control with an Inverse Grad-Shafranov Neural Network." DPP 2025. 2025.

[2] Wang, Allen M., et al. "Technical Aspects of Plasma Operational Simulation (POPSIM): A Framework for Data-Driven Simulation and Control." arXiv preprint arXiv:2509.10244 (2025).