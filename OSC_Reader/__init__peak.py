"""Public API for the dvb package.

This module also performs a series of convenience imports so that simply
executing ``import dvb`` provides a ready-to-use analysis environment.
The behaviour mirrors the manual imports typically used in notebooks.
"""

###############################################################################
# Convenience imports executed on ``import dvb``
###############################################################################

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

import pyFAI
import fabio

# Additional pyFAI modules. These imports require optional Qt bindings, so they
# are guarded to prevent ImportError when the GUI stack is not available.
try:  # pragma: no cover - optional GUI dependencies
    import pyFAI.test.utilstest  # noqa: F401 (imported for side effects)
    from pyFAI.gui import jupyter  # noqa: F401
    from pyFAI.gui.jupyter.calib import Calibration  # noqa: F401
except Exception:
    Calibration = None
    jupyter = None

# Re-import modules so users can access them via ``dvb.pa`` and ``dvb.viewer``
import importlib as _importlib
pa = _importlib.import_module('.peak_analysis', __package__)
viewer = _importlib.import_module('.viewer', __package__)

###############################################################################
# DVB package imports
###############################################################################

from .tools import (
    xrd_peaks,
    rotate_image,
    get_radial,
    load_data,
    image,
    integrate_spec,
    integrate_qr_vs_qz,
    integrate_qz_vs_qr,
    plot_qz_vs_qr,
    plot_q_3d,
    q,
    plot_q,
    plot_qr,
    clean_data,
    background_spec,
    parse_poni_file,
    setup_azimuthal_integrator,
    display,
)
from .viewer import plot_interactive_2d
from .peak_analysis import perform_fit, process_data, fit_pvoigt_peaks

__all__ = [
    "xrd_peaks",
    "rotate_image",
    "get_radial",
    "load_data",
    "image",
    "integrate_spec",
    "integrate_qr_vs_qz",
    "integrate_qz_vs_qr",
    "plot_qz_vs_qr",
    "plot_q_3d",
    "q",
    "plot_q",
    "plot_qr",
    "clean_data",
    "background_spec",
    "plot_interactive_2d",
    "perform_fit",
    "fit_pvoigt_peaks",
    "process_data",
    "parse_poni_file",
    "setup_azimuthal_integrator",
    "display",
    "np",
    "pd",
    "plt",
    "subplots",
    "pyFAI",
    "fabio",
    "jupyter",
    "Calibration",
    "pa",
    "viewer",
]

# Attempt to enable IPython's autoreload extension if running inside a
# Jupyter environment. Fail silently if IPython is not available.
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("load_ext", "autoreload")
except Exception:
    pass

# Display pyFAI version information on import for quick reference
print("pyFAI version:", pyFAI.version)
