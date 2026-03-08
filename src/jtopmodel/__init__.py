# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TOPMODEL (Beven & Kirkby 1979) -- Standalone Plugin Package.

A native Python/JAX implementation of TOPMODEL, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- DDS and evolutionary calibration integration

Algorithms:
    - Degree-day snow module
    - Exponential transmissivity baseflow (Beven & Kirkby 1979)
    - Saturation-excess overland flow with topographic index distribution
    - Linear reservoir channel routing

Components:
    - TopmodelPreProcessor: Prepares forcing data (P, T, PET)
    - TopmodelRunner: Executes model simulations
    - TopmodelPostprocessor: Extracts streamflow results
    - TopmodelWorker: Handles calibration

References:
    Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable
    contributing area model of basin hydrology. Hydrological Sciences
    Bulletin, 24(1), 43-69.
"""

from typing import TYPE_CHECKING


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # Configuration
    'TOPMODELConfig': ('.config', 'TOPMODELConfig'),
    'TopmodelConfigAdapter': ('.config', 'TopmodelConfigAdapter'),

    # Main components
    'TopmodelPreProcessor': ('.preprocessor', 'TopmodelPreProcessor'),
    'TopmodelRunner': ('.runner', 'TopmodelRunner'),
    'TopmodelPostprocessor': ('.postprocessor', 'TopmodelPostprocessor'),
    'TopmodelResultExtractor': ('.extractor', 'TopmodelResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'TopmodelParameters': ('.parameters', 'TopmodelParameters'),
    'TopmodelState': ('.parameters', 'TopmodelState'),
    'create_params_from_dict': ('.parameters', 'create_params_from_dict'),
    'create_initial_state': ('.parameters', 'create_initial_state'),
    'generate_ti_distribution': ('.parameters', 'generate_ti_distribution'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'snow_step': ('.model', 'snow_step'),
    'topmodel_step': ('.model', 'topmodel_step'),
    'route_step': ('.model', 'route_step'),
    'step': ('.model', 'step'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Loss functions (for gradient-based calibration)
    'nse_loss': ('.losses', 'nse_loss'),
    'kge_loss': ('.losses', 'kge_loss'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),

    # Calibration
    'TopmodelWorker': ('.calibration.worker', 'TopmodelWorker'),
    'TopmodelParameterManager': ('.calibration.parameter_manager', 'TopmodelParameterManager'),
    'get_topmodel_calibration_bounds': ('.calibration.parameter_manager', 'get_topmodel_calibration_bounds'),
}


def __getattr__(name: str):
    """Lazy import handler for TOPMODEL module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return available attributes for tab completion."""
    return list(_LAZY_IMPORTS.keys()) + ['register']


def register() -> None:
    """Register TOPMODEL components with symfluence plugin registry."""
    from symfluence.core.registry import model_manifest
    from .calibration.optimizer import TopmodelModelOptimizer
    from .calibration.parameter_manager import TopmodelParameterManager
    from .calibration.worker import TopmodelWorker
    from .config import TopmodelConfigAdapter
    from .extractor import TopmodelResultExtractor
    from .postprocessor import TopmodelPostprocessor
    from .preprocessor import TopmodelPreProcessor
    from .runner import TopmodelRunner

    model_manifest(
        "TOPMODEL",
        preprocessor=TopmodelPreProcessor,
        runner=TopmodelRunner,
        runner_method='run_topmodel',
        postprocessor=TopmodelPostprocessor,
        config_adapter=TopmodelConfigAdapter,
        result_extractor=TopmodelResultExtractor,
        optimizer=TopmodelModelOptimizer,
        worker=TopmodelWorker,
        parameter_manager=TopmodelParameterManager,
    )


# Type hints for IDE support
if TYPE_CHECKING:
    from .calibration.parameter_manager import TopmodelParameterManager, get_topmodel_calibration_bounds
    from .calibration.worker import TopmodelWorker
    from .config import TOPMODELConfig, TopmodelConfigAdapter
    from .extractor import TopmodelResultExtractor
    from .losses import (
        get_kge_gradient_fn,
        get_nse_gradient_fn,
        kge_loss,
        nse_loss,
    )
    from .model import (
        HAS_JAX,
        route_step,
        simulate,
        simulate_jax,
        simulate_numpy,
        snow_step,
        step,
        topmodel_step,
    )
    from .parameters import (
        DEFAULT_PARAMS,
        PARAM_BOUNDS,
        TopmodelParameters,
        TopmodelState,
        create_initial_state,
        create_params_from_dict,
        generate_ti_distribution,
    )
    from .postprocessor import TopmodelPostprocessor
    from .preprocessor import TopmodelPreProcessor
    from .runner import TopmodelRunner


__all__ = [
    # Main components
    'TopmodelPreProcessor',
    'TopmodelRunner',
    'TopmodelPostprocessor',
    'TopmodelResultExtractor',

    # Configuration
    'TOPMODELConfig',
    'TopmodelConfigAdapter',

    # Parameters
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'TopmodelParameters',
    'TopmodelState',
    'create_params_from_dict',
    'create_initial_state',
    'generate_ti_distribution',

    # Core model
    'simulate',
    'simulate_jax',
    'simulate_numpy',
    'snow_step',
    'topmodel_step',
    'route_step',
    'step',
    'HAS_JAX',

    # Loss functions
    'nse_loss',
    'kge_loss',
    'get_nse_gradient_fn',
    'get_kge_gradient_fn',

    # Calibration
    'TopmodelWorker',
    'TopmodelParameterManager',
    'get_topmodel_calibration_bounds',

    # Plugin registration
    'register',
]
