# POPSIM: Plasma OPerational SIMulation
POPSIM is a control-oriented tokamak plasma simulation toolbox built in the machine-learning framework [JAX](https://github.com/google/jax). Thus, it is JIT-compilable, massively parallelizable on GPU, and auto-differentiable.

It's role is to enable:
1) Sensitivity analysis and off-normal simulation via massively parallelized Monte Carlo simulation on GPU and automatic differentiation
2) Uncertainty-aware optimization of scenarios, trajectories, and controllers via techniques from optimal control + reinforcement learning
3) Parameter estimation and learning unknown and computationally expensive dynamics from data

POPSIM is not a single simulator, but rather a set of tools to allow you to define simulation "modules". Each module is, itself, a stand-alone simulator that can be simulated using a common `simulate` function. More complex simulators can be built by chaining modules together. 


## Installation
1. Run a `git submodule update --init --recursive` to make sure you have all the necessary submodules.
2. If you don't have it, [install poetry](https://python-poetry.org/docs/#installation) and make sure it is version >= 1.6.1.
3. In the root of this repo, run a `poetry install`
   
   a. To install with GPU support, run `poetry install --with gpu`

   b. To install with development dependencies, run `poetry install --with dev`
4. CFSPOPCON currently requires a bit of manual work to get working. Follow the instructions [here](https://cfspopcon.readthedocs.io/en/latest/doc_sources/Usage.html)
5. Install git lfs and do a `git lfs init` and a `git lfs pull` to get the necessary data files.
6. Be sure to use either `poetry shell` or prefix all of your commands with `poetry run`. If you are confused, check out the [poetry docs](https://python-poetry.org/docs/basic-usage/)

## Development Workflow

### Dev Install

- Install development related packages with `poetry install --with dev`
- Run `poetry run pre-commit install` to install pre-commit hooks

### Branching
Create a branch off `main` to commit your changes to. For those new to git workflows, [this tutorial](https://webtuu.com/blog/04/git-basics-branching-merging-push-to-github) may be helpful.


The recommended pattern for the branch name is `name/description` (e.g. `allenw/dev_workflow`)

### Pre-Commits
The `dev` install introduces `pre-commit` hooks that need to pass before you can commit. For example, the auto-formatter will run on every commit, and if it detects improperly formatted code you will get an error like that shown below.
![pre-commit-error-example](img/pre-commit-error-example.png)

In many cases, like the one above, the `pre-commit` will automatically fix the problem for you, and you just need to `git add` the change. In the above example, we can see the formatting fix the pre-commit made with a `git diff`:

![pre-commit-diff-example](img/pre-commit-format-diff.png)

Once all pre-commits pass, the commit should go through.

⚠️ **Bypassing pre-commits:** sometimes you may want to bypass the pre-commits. If this is necessary, you can use `git commit --no-verify` to bypass the pre-commits. Note that Github Actions will run the pre-commits on your pull request (PR), so if the issue isn't fixed by that point, CI will fail.

### VSCode Autoformatting + Linting
VSCode users can install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extensions to have `ruff` autoformat on save by introducing the following into your `settings.json`:
    
```json
{   
    # Other stuff
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "never" # At the time of writing, there are issues with imports organizing.
        },
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    # Other stuff
}
```
The ruff extension also helpfully explains to you the issues it finds in the code:
![ruff-warning-message](img/ruff-warning-message.png)

### Automated Tests
POPSIM uses the `pytest` framework for automated testing. For those new to Pytest, the documentation provides a [nice tutorial](https://docs.pytest.org/en/8.2.x/getting-started.html) for getting started. 

Ideally, every function and module would have automated test coverage. Tests are located in the `popsim/tests` directory.



**Running Automated Tests**

To run all of the automated tests, run `poetry run pytest popsim/`.

You can run the tests for just a specific file, for example: `poetry run pytest popsim/tests/test_tree_util.py`

You can also run just one specific test, for example: `poetry run pytest popsim/tests/test_tree_util.py::test_tree_transpose`


### Pull Requests
When you are ready to merge your code back into `main`, you should create a pull request (PR) on Github. For those new to PRs, follow [this tutorial](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

If you would like to indicate that you are not quite ready to merge your code into `main`, but you would like to share the code with others for the purposes of generating discussion and/or soliciting feedback, you should create a "Draft PR" which can be easily done by following the instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request#creating-the-pull-request).