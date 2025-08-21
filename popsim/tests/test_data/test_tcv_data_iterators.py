import pytest
from collections.abc import Iterator
import os
import xarray as xr

from popsim.data._paths import get_path_to_ml_data_dump
from popsim.data.tcv.data_iterators import (
    iterate_defuse_h5_paths,
    iterate_fbte_nc_paths,
    iterate_fbte_mat_paths,
    iterate_defuse_h5s,
    iterate_fbte_datasets,
)


# Skip all tests in this module if ML data dump path is not available
pytestmark = pytest.mark.skipif(
    get_path_to_ml_data_dump() is None,
    reason="ML data dump path not available",
    allow_module_level=True
)


def test_iterate_defuse_h5_paths():
    """Test that iterate_defuse_h5_paths returns an iterator and yields valid paths."""
    path_iterator = iterate_defuse_h5_paths()
    assert isinstance(path_iterator, Iterator)
    
    # Test first few paths if available
    for i, path in enumerate(path_iterator):
        assert isinstance(path, str)
        assert path.endswith('.h5')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_iterate_fbte_nc_paths():
    """Test that iterate_fbte_nc_paths returns an iterator and yields valid paths."""
    path_iterator = iterate_fbte_nc_paths()
    assert isinstance(path_iterator, Iterator)
    
    # Test first few paths if available
    for i, path in enumerate(path_iterator):
        assert isinstance(path, str)
        assert path.endswith('.nc')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_iterate_fbte_mat_paths():
    """Test that iterate_fbte_mat_paths returns an iterator and yields valid paths."""
    path_iterator = iterate_fbte_mat_paths()
    assert isinstance(path_iterator, Iterator)
    
    # Test first few paths if available
    for i, path in enumerate(path_iterator):
        assert isinstance(path, str)
        assert path.endswith('.mat')
        assert os.path.isfile(path)
        
        # Limit to first few for performance
        if i >= 2:
            break


def test_iterate_defuse_h5s():
    """Test that iterate_defuse_h5s returns an iterator and yields dictionaries of datasets."""
    iterator = iterate_defuse_h5s()
    assert isinstance(iterator, Iterator)
    
    # Test first item if available
    first_item = next(iterator)
    assert isinstance(first_item, dict)
    # Values should be xarray datasets
    for key, value in first_item.items():
        assert isinstance(key, str)
        assert isinstance(value, xr.Dataset)


def test_iterate_fbte_datasets():
    """Test that iterate_fbte_datasets returns an iterator and yields xarray datasets."""
    iterator = iterate_fbte_datasets()
    assert isinstance(iterator, Iterator)
    
    # Test first item if available
    first_item = next(iterator)
    assert isinstance(first_item, xr.Dataset)

@pytest.mark.slow
def test_defuse_datasets_load_successfully():
    """Test that all DEFUSE datasets can be loaded without errors."""
    for i, dataset_dict in enumerate(iterate_defuse_h5s()):
        # Just verify we can load and access basic properties
        assert isinstance(dataset_dict, dict)
        assert len(dataset_dict) > 0
        
        # Limit to first few for performance
        if i >= 2:
            break


@pytest.mark.slow
def test_fbte_datasets_load_successfully():
    """Test that all FBTE datasets can be loaded without errors."""
    for i, dataset in enumerate(iterate_fbte_datasets()):
        # Just verify we can load and access basic properties
        assert isinstance(dataset, xr.Dataset)
        assert hasattr(dataset, 'dims')
        
        # Limit to first few for performance
        if i >= 2:
            break