"""Tests for the formulas module."""
import os

import pytest

import cfspopcon
from popsim import PACKAGE_ROOT


@pytest.fixture
def load_data():
    sparc_prd_path = os.path.join(PACKAGE_ROOT, "../cfspopcon/example_cases/SPARC_PRD")
    input_parameters, algorithm, points = cfspopcon.read_case(sparc_prd_path)
    algorithm.validate_inputs(input_parameters)
    return input_parameters, algorithm, points


def test_profiles(load_data):
    input_parameters, algorithm, points = load_data
    assert True  # Temporary placeholder
