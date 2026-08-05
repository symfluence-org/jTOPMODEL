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
    - TopmodelPostProcessor: Extracts streamflow results
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
    'TopmodelPostProcessor': ('.postprocessor', 'TopmodelPostProcessor'),
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
    from .postprocessor import TopmodelPostProcessor
    from .preprocessor import TopmodelPreProcessor
    from .runner import TopmodelRunner

    model_manifest(
        "TOPMODEL",
        preprocessor=TopmodelPreProcessor,
        runner=TopmodelRunner,
        runner_method='run_topmodel',
        postprocessor=TopmodelPostProcessor,
        config_adapter=TopmodelConfigAdapter,
        result_extractor=TopmodelResultExtractor,
        optimizer=TopmodelModelOptimizer,
        worker=TopmodelWorker,
        parameter_manager=TopmodelParameterManager,
    )

    # Contribute TOPMODEL's calibration bounds to symfluence's catalogue.
    #
    # TOPMODEL predates the register_model_bounds seam, so symfluence carried
    # a get_topmodel_bounds() entry as a compatibility shim -- meaning a change
    # to TOPMODEL's bounds needed a FRAMEWORK release. Registering here makes
    # this package the owner: get_model_bounds('TOPMODEL') resolves what we
    # register, ahead of the built-in entry, so this works against current
    # symfluence and lets a later release drop the compat entry entirely.
    #
    # Values come from jtopmodel.parameters.PARAM_BOUNDS, already the single
    # source for every other bounds consumer in this package, and verified
    # identical to what the framework served (11 names, zero differences) --
    # so adopting the seam changes no calibration result.
    #
    # The catalogue entries are namespaced 'topmodel_*' and served with the
    # prefix stripped, because bare 'm' is a DIFFERENT parameter there (RHESSys
    # decay, 0.5-5.0). Registering unprefixed would collide with it and the
    # central definition would win, silently widening TOPMODEL's 'm'. Keep the
    # prefix + strip_prefix, matching get_topmodel_bounds().
    from symfluence.core.calibration.parameters import ParameterInfo, register_model_bounds

    from .parameters import PARAM_BOUNDS

    register_model_bounds(
        "TOPMODEL",
        params={
            f"topmodel_{name}": ParameterInfo(
                float(lo), float(hi), description=f"TOPMODEL {name}"
            )
            for name, (lo, hi) in PARAM_BOUNDS.items()
        },
        names=[f"topmodel_{name}" for name in PARAM_BOUNDS],
        strip_prefix="topmodel_",
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
    from .postprocessor import TopmodelPostProcessor
    from .preprocessor import TopmodelPreProcessor
    from .runner import TopmodelRunner


__all__ = [
    # Main components
    'TopmodelPreProcessor',
    'TopmodelRunner',
    'TopmodelPostProcessor',
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
