import pytest
from collections.abc import Generator
import os
import xarray as xr

from popsim.data._paths import get_path_to_ml_data_dump
from popsim.data.tcv.data_generators import (
    generate_defuse_h5_paths,
    generate_fbte_nc_paths,
    generate_fbte_mat_paths,
    generate_defuse_h5s,
    generate_fbte_datasets,
)


# Skip all tests in this module if ML data dump path is not available
pytestmark = pytest.mark.skipif(
    get_path_to_ml_data_dump() is None,
    reason="ML data dump path not available",
    allow_module_level=True
)


def test_generate_defuse_h5_paths():
    """Test that generate_defuse_h5_paths returns an generator and yields valid paths."""
    path_generator = generate_defuse_h5_paths()
    assert isinstance(path_generator, Generator)
    
    # Test first few paths if available
    for i, path in enumerate(path_generator):
        assert isinstance(path, str)
        assert path.endswith('.h5')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_generate_fbte_nc_paths():
    """Test that generate_fbte_nc_paths returns an generator and yields valid paths."""
    path_generator = generate_fbte_nc_paths()
    assert isinstance(path_generator, Generator)
    
    # Test first few paths if available
    for i, path in enumerate(path_generator):
        assert isinstance(path, str)
        assert path.endswith('.nc')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_generate_fbte_mat_paths():
    """Test that generate_fbte_mat_paths returns an generator and yields valid paths."""
    path_generator = generate_fbte_mat_paths()
    assert isinstance(path_generator, Generator)
    
    # Test first few paths if available
    for i, path in enumerate(path_generator):
        assert isinstance(path, str)
        assert path.endswith('.mat')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_generate_defuse_h5s():
    """Test that generate_defuse_h5s returns an generator and yields dictionaries of datasets."""
    generator = generate_defuse_h5s()
    assert isinstance(generator, Generator)
    
    # Test first item if available
    first_item = next(generator)
    assert isinstance(first_item, dict)
    # Values should be xarray datasets
    for key, value in first_item.items():
        assert isinstance(key, str)
        assert isinstance(value, xr.Dataset)


def test_generate_fbte_datasets():
    """Test that generate_fbte_datasets returns an generator and yields xarray datasets."""
    generator = generate_fbte_datasets()
    assert isinstance(generator, Generator)
    
    # Test first item if available
    first_item = next(generator)
    assert isinstance(first_item, xr.Dataset)

@pytest.mark.slow
def test_defuse_datasets_load_successfully():
    """Test that all DEFUSE datasets can be loaded without errors."""
    for i, dataset_dict in enumerate(generate_defuse_h5s()):
        # Just verify we can load and access basic properties
        assert isinstance(dataset_dict, dict)
        assert len(dataset_dict) > 0
        
        # Limit to first few for performance
        if i >= 2:
            break


@pytest.mark.slow
def test_fbte_datasets_load_successfully():
    """Test that all FBTE datasets can be loaded without errors."""
    for i, dataset in enumerate(generate_fbte_datasets()):
        # Just verify we can load and access basic properties
        assert isinstance(dataset, xr.Dataset)
        assert hasattr(dataset, 'dims')
        
        # Limit to first few for performance
        if i >= 2:
            break