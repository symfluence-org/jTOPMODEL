"""Tests for jTOPMODEL plugin registration."""

import pytest


def test_register_function_exists():
    """jtopmodel module should have a register() function."""
    import jtopmodel
    assert hasattr(jtopmodel, 'register')
    assert callable(jtopmodel.register)


def test_entry_point_discoverable():
    """The topmodel entry point should be discoverable."""
    from importlib.metadata import entry_points
    eps = entry_points(group='symfluence.plugins')
    names = [ep.name for ep in eps]
    assert 'topmodel' in names


def test_register_creates_config_adapter():
    """Calling register() should register TOPMODEL config adapter."""
    import jtopmodel
    jtopmodel.register()

    from symfluence.core.registries import R
    assert 'TOPMODEL' in R.config_adapters
