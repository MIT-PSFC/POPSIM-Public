from pathlib import Path

import yaml


# Define a custom constructor for Python tuples
def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


# Add the constructor to the SafeLoader
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python.tuple", construct_python_tuple)

# Load the YAML file with the custom loader
with open(Path(__file__).parent / "dim_map.yaml") as stream:
    DIM_TO_VARS_MAP = yaml.load(stream, Loader=yaml.SafeLoader)

# Inverting the dictionary
VARS_TO_DIM_MAP = {var: key for key, vars_list in DIM_TO_VARS_MAP.items() for var in vars_list}
