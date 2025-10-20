"""Unified package initialisation for :mod:`OSC_Reader`.

This module exposes the core OSC file processing helpers alongside the
newly integrated diffraction analysis utilities that originated from the
``DVB_pack`` project.  All heavy scientific dependencies (``pyFAI``,
``lmfit``, ``pandas`` …) remain optional – the corresponding submodules are
loaded lazily so that importing :mod:`OSC_Reader` continues to work even in
lightweight environments.  Attempting to access an optional function or
submodule will raise a descriptive :class:`ImportError` if the underlying
requirements are missing.
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from .OSC_Reader import ShapeError, convert_to_asc, osc2jpg, read_osc
from .OSC_Viewer import visualize_osc_data

__all__ = [
    "ShapeError",
    "convert_to_asc",
    "osc2jpg",
    "read_osc",
    "visualize_osc_data",
]

# Mapping of attribute name -> (submodule name, attribute name inside module).
_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Tools module exports
    "parse_poni_file": ("tools", "parse_poni_file"),
    "display": ("tools", "display"),
    "setup_azimuthal_integrator": ("tools", "setup_azimuthal_integrator"),
    "xrd_peaks": ("tools", "xrd_peaks"),
    "rotate_image": ("tools", "rotate_image"),
    "get_radial": ("tools", "get_radial"),
    "load_data": ("tools", "load_data"),
    "image": ("tools", "image"),
    "integrate_spec": ("tools", "integrate_spec"),
    "plot_qz_vs_qr": ("tools", "plot_qz_vs_qr"),
    "plot_q_3d": ("tools", "plot_q_3d"),
    "q": ("tools", "q"),
    "plot_q": ("tools", "plot_q"),
    "plot_qr": ("tools", "plot_qr"),
    "integrate_qr_vs_qz": ("tools", "integrate_qr_vs_qz"),
    "integrate_qz_vs_qr": ("tools", "integrate_qz_vs_qr"),
    "clean_data": ("tools", "clean_data"),
    "background_spec": ("tools", "background_spec"),
    "plot_interactive_2d": ("viewer", "plot_interactive_2d"),
    # Peak analysis exports
    "estimate_initial_parameters": ("peak_analysis", "estimate_initial_parameters"),
    "compute_fwhm": ("peak_analysis", "compute_fwhm"),
    "derivative_wrt_sigma": ("peak_analysis", "derivative_wrt_sigma"),
    "derivative_wrt_fraction": ("peak_analysis", "derivative_wrt_fraction"),
    "compute_fwhm_and_error_from_cov": ("peak_analysis", "compute_fwhm_and_error_from_cov"),
    "perform_fit": ("peak_analysis", "perform_fit"),
    "plot_fit_with_components": ("peak_analysis", "plot_fit_with_components"),
    "fit_pvoigt_peaks": ("peak_analysis", "fit_pvoigt_peaks"),
    "process_data": ("peak_analysis", "process_data"),
}

# Submodules that can be imported on demand via attribute access.
_OPTIONAL_SUBMODULES = {
    name for name, _ in _OPTIONAL_EXPORTS.values()
}
_OPTIONAL_SUBMODULES.update({"tools", "peak_analysis", "viewer"})

# Import cache and error registry for optional modules.
_module_cache: Dict[str, object] = {}
_module_errors: Dict[str, Exception] = {}


def _load_optional_module(module_name: str):
    """Import *module_name* relative to :mod:`OSC_Reader` on demand."""
    if module_name in _module_cache:
        return _module_cache[module_name]
    try:
        module = import_module(f".{module_name}", __name__)
    except Exception as exc:  # pragma: no cover - exercised only when deps missing
        _module_errors[module_name] = exc
        raise
    else:
        _module_cache[module_name] = module
        globals()[module_name] = module
        return module


def __getattr__(name: str):
    """Lazily expose optional modules and symbols."""
    if name in _OPTIONAL_SUBMODULES:
        try:
            return _load_optional_module(name)
        except Exception as exc:  # pragma: no cover - optional dependency failure
            raise ImportError(
                f"Optional module '{name}' could not be imported. "
                "Install the required dependencies to enable this feature."
            ) from exc
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        try:
            module = _load_optional_module(module_name)
        except Exception as exc:  # pragma: no cover - optional dependency failure
            raise ImportError(
                f"Optional functionality '{name}' is unavailable because the "
                f"'{module_name}' module failed to import."
            ) from exc
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():  # pragma: no cover - trivial helper
    optional_names = set(_OPTIONAL_SUBMODULES) | set(_OPTIONAL_EXPORTS)
    return sorted(set(globals()) | set(__all__) | optional_names)


# Advertise optional names in ``__all__`` while keeping ordering stable.
__all__ = list(dict.fromkeys(__all__ + sorted(_OPTIONAL_SUBMODULES) + sorted(_OPTIONAL_EXPORTS)))
