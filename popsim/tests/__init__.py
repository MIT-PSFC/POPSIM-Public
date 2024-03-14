import os

import pytest

import cfspopcon
from popsim import PACKAGE_ROOT
from popsim.enums import Impurity


def load_sparc_prd_data_f():
    sparc_prd_path = os.path.join(PACKAGE_ROOT, "../cfspopcon/example_cases/SPARC_PRD")
    input_parameters, algorithm, points = cfspopcon.read_case(sparc_prd_path)
    algorithm.validate_inputs(input_parameters)


    impurity_types = [Impurity(impurity.value) for impurity in input_parameters["impurities"].dim_species.data]

    impurity_concentrations = dict(
        zip(
            impurity_types,
            input_parameters["impurities"].values,
        )
    )

    return input_parameters, algorithm, points, impurity_types, impurity_concentrations

def load_sparc_q1l_data_f():
    sparc_path = os.path.join(PACKAGE_ROOT, "../cfspopcon/example_cases/SPARC_Q1L")
    input_parameters, algorithm, points = cfspopcon.read_case(sparc_path)
    algorithm.validate_inputs(input_parameters)


    impurity_types = [Impurity(impurity.value) for impurity in input_parameters["impurities"].dim_species.data]

    impurity_concentrations = dict(
        zip(
            impurity_types,
            input_parameters["impurities"].values,
        )
    )

    return input_parameters, algorithm, points, impurity_types, impurity_concentrations

@pytest.fixture
def load_sparc_prd_data():
    return load_sparc_prd_data_f()
