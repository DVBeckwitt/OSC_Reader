# OSC Reader and Diffraction Toolkit

![OSC Reader viewer preview](docs/viewer_preview.png)

`OSC_Reader` is a Python toolkit for working with Rigaku RAXIS `.osc` detector
images. It combines a low-level file reader, export helpers, a high-FPS desktop
viewer, an exact detector-to-angle-space conversion pipeline, and optional
diffraction-analysis utilities that were folded in from the earlier
`DVB_pack` work.

The project is primarily a local scientific desktop tool and Python library. It
is not a web service and does not require a deployment stack to be useful.

## Key Features

- Read Rigaku RAXIS `.osc` files into NumPy arrays.
- Export detector frames to `.asc` text grids and `.jpg` images.
- Inspect data in a PySide6/pyqtgraph GUI with crosshairs, linked line
  profiles, histogram/level controls, and zooming.
- Pick a beam center interactively on the detector image, with snapping to the
  nearest local peak top and manual pixel-coordinate override via spin boxes.
- Convert detector images to `φ`/`2θ` space using explicit geometry
  parameters.
- Display the full angle-view `φ` range by default from `-180` to `180`
  degrees.
- Re-orient the GUI `φ=0` direction to `Up`, `Left`, `Down`, or `Right`.
- Use the exact splitter conversion path with a numba-backed parallel engine
  when available.
- Access optional diffraction-analysis helpers for integration, reciprocal
  space plotting, and peak fitting.

## Tech Stack

| Area | Technology |
|----------|-------------|
| Language | Python 3.8+ |
| Core numeric layer | NumPy |
| Fast angle conversion | numba |
| GUI | PySide6 + pyqtgraph |
| Image export helper | OpenCV |
| Packaging | setuptools |
| Optional plotting/analysis | matplotlib, pandas, SciPy, pyFAI, fabio, lmfit, datashader |
| Platform helpers | Windows batch launchers for local desktop use |

## Repository Layout

```text
OSC_Reader/
|-- OSC_Reader/
|   |-- __init__.py          # Unified public package entry point with lazy exports
|   |-- OSC_Reader.py        # Core .osc parser plus .asc/.jpg conversion helpers
|   |-- OSC_Viewer.py        # Main Qt GUI / high-FPS desktop viewer
|   |-- angle_space.py       # Exact detector-to-angle-space conversion pipeline
|   |-- tools.py             # Optional diffraction utilities (pyFAI, plotting, etc.)
|   |-- peak_analysis.py     # Optional peak-fitting helpers built around lmfit
|   `-- viewer.py            # Legacy matplotlib-based interactive viewer
|-- Validation/
|   |-- Methodology.md       # Validation notes for ASCII export comparisons
|   |-- Validator.ipynb      # Notebook for validation experiments
|   |-- generated_file_*.asc # Example generated outputs
|-- docs/
|   `-- viewer_preview.png   # README preview image
|-- Run_OSC_Viewer.bat       # Windows launcher for the Qt viewer
|-- Register_OSC_Default_App.bat
|                           # Windows file association helper for .osc files
|-- setup.py                 # Package metadata and dependencies
|-- CHANGELOG.md
|-- LICENSE
`-- README.md
```

## Prerequisites

### Required for the core package

- Python 3.8 or newer
- `pip`
- A desktop session if you plan to use the GUI viewer

### Required for optional scientific-analysis workflows

Install these only if you need the optional modules in `OSC_Reader.tools`,
`OSC_Reader.peak_analysis`, or `OSC_Reader.viewer`:

- `matplotlib`
- `pandas`
- `scipy`
- `pyFAI`
- `fabio`
- `lmfit`
- `datashader`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DVBeckwitt/OSC_Reader.git
cd OSC_Reader
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

Editable install for development:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Standard install:

```bash
python -m pip install --upgrade pip
pip install .
```

The package metadata in [setup.py](setup.py)
installs these core dependencies automatically:

- `numpy`
- `numba`
- `tifffile`
- `docopt`
- `logbook`
- `opencv-python`
- `pyqtgraph`
- `pyside6`

### 4. Install optional analysis dependencies

If you need the optional diffraction-analysis modules, install their scientific
stack separately:

```bash
pip install matplotlib pandas scipy pyfai fabio lmfit datashader
```

### 5. No environment variables are required

This project does not currently rely on an `.env` file or runtime secrets for
normal local usage.

## Getting Started

### Read a detector image

```python
from OSC_Reader import read_osc

image = read_osc("example.osc")
print(image.shape, image.dtype)
```

### Convert an `.osc` file to `.asc`

```python
from OSC_Reader import convert_to_asc

convert_to_asc("example.osc")
```

### Convert an `.osc` file to `.jpg`

```python
from OSC_Reader import osc2jpg

osc2jpg("example.osc")
```

### Launch the main GUI

Open a specific file:

```bash
python -m OSC_Reader.OSC_Viewer path/to/example.osc
```

Open the viewer and choose a file from the dialog:

```bash
python -m OSC_Reader.OSC_Viewer
```

Launch through Python directly:

```python
from OSC_Reader import visualize_osc_data

visualize_osc_data("example.osc")
```

## Main GUI Guide

The main desktop GUI lives in
[OSC_Reader/OSC_Viewer.py](OSC_Reader/OSC_Viewer.py)
and is the primary way to inspect detector frames interactively.

### Viewer controls

| Control | What it does |
|----------|-------------|
| Mouse move over image | Updates the crosshair and the linked bottom/side intensity profiles |
| Left-click + drag | Zooms to a rectangular region |
| `Reset Zoom` | Restores the full current view |
| `Save Image` | Exports the rendered image plot |
| `Show Detector` | Returns from angle view to the raw detector image |
| `Pick Beam Center` | Lets you click near the direct-beam location; the picker snaps to the nearest local peak top |
| `Center X` / `Center Y` | Manual beam-center override in detector pixels |
| `Distance` | Sample-to-detector distance in millimetres |
| `Pixel Size` | Detector pixel size in millimetres |
| `2θ Bins` / `φ Bins` | Output resolution for the angle-space image |
| `φ=0 At` | Chooses which on-screen direction should correspond to `φ=0` in the GUI display |
| `Convert to φ/2θ` | Runs the detector-to-angle conversion on the currently loaded detector frame |
| `Bottom Log Y` | Enables logarithmic intensity scaling on the bottom profile plot |
| `Side Log X` | Enables logarithmic intensity scaling on the side profile plot |

### Recommended detector-to-angle workflow

1. Open an `.osc` detector image.
2. Click `Pick Beam Center`.
3. Click near the direct beam or reference peak on the detector image.
4. Let the viewer snap to the nearest local peak top automatically.
5. If needed, fine-tune the result with `Center X` and `Center Y`.
6. Enter `Distance` and `Pixel Size`.
7. Adjust `2θ Bins` and `φ Bins` to control output resolution.
8. Optionally change `φ=0 At` if you want a different on-screen orientation.
9. Click `Convert to φ/2θ`.
10. Use `Show Detector` to switch back to detector space without reloading.

### Beam center behavior

- Detector clicks do not set the beam center blindly.
- The GUI searches a local window around the clicked point and snaps to the
  nearest local maximum.
- A small subpixel refinement step is then applied to estimate the peak top
  more precisely.
- Manual spin-box values remain the authoritative override when you want exact
  coordinates.

### Detector view vs angle view

- Detector view keeps the image aspect locked so detector pixels remain square.
- Angle view unlocks the image aspect ratio so the full `φ` range remains
  visible instead of being cropped by square-pixel constraints.
- The default GUI angle range is `-180` to `180` degrees.

### Performance notes

- Cursor/profile updates are rate-limited for responsive interaction.
- The viewer targets roughly `60 FPS` for cursor/profile rendering work.
- A background warm-up thread primes the numba angle-conversion path so the
  first interactive conversion is less disruptive.
- The default angle conversion worker setting is `8`, capped by available CPU
  count.

### Windows helper scripts

Start the GUI using the included launcher:

```bat
Run_OSC_Viewer.bat
```

You can also drag and drop an `.osc` file onto `Run_OSC_Viewer.bat`.

Register the viewer as the default app for `.osc` files for the current user:

```bat
Register_OSC_Default_App.bat
```

This writes the relevant `HKCU\Software\Classes` registry keys and points the
file association at the batch launcher.

## Python API Overview

### Core parser and export helpers

The low-level parser lives in
[OSC_Reader/OSC_Reader.py](OSC_Reader/OSC_Reader.py).

Public helpers:

- `read_osc(path)`
- `convert_to_asc(path, force=False)`
- `osc2jpg(path)`
- `ShapeError`

### Detector-to-angle conversion API

```python
from OSC_Reader import (
    convert_image_to_phi_2theta_space,
    prepare_gui_phi_display,
    read_osc,
)

image = read_osc("example.osc")

result = convert_image_to_phi_2theta_space(
    image,
    distance_mm=75.0,
    pixel_size_mm=0.1,
    center_row_px=1500.0,
    center_col_px=1500.0,
    radial_bins=1000,
    azimuth_bins=720,
)

cake_image, two_theta_deg, phi_deg = prepare_gui_phi_display(
    result,
    zero_direction="up",
)

print(cake_image.shape)
print(two_theta_deg.min(), two_theta_deg.max())
print(phi_deg.min(), phi_deg.max())
```

`convert_image_to_phi_2theta_space(...)` returns a `DetectorCakeResult` with:

- `radial_deg`
- `azimuthal_deg`
- `intensity`
- `sum_signal`
- `sum_normalization`
- `count`

### Angle conversion arguments

| Argument | Meaning | Default |
|----------|-------------|---------|
| `distance_mm` | Sample-to-detector distance in millimetres | required |
| `pixel_size_mm` | Detector pixel size in millimetres | required |
| `center_row_px` | Beam-center row in detector pixels | required |
| `center_col_px` | Beam-center column in detector pixels | required |
| `radial_bins` | Number of `2θ` bins | detector width |
| `azimuth_bins` | Number of azimuth bins | detector height |
| `two_theta_min_deg` | Lower output `2θ` limit | `0.0` |
| `two_theta_max_deg` | Upper output `2θ` limit | `90.0` |
| `phi_min_deg` | Lower raw azimuth limit | `-180.0` |
| `phi_max_deg` | Upper raw azimuth limit | `180.0` |
| `correct_solid_angle` | Enable flat-detector solid-angle correction | `False` |
| `engine` | Backend: `"numba"` or `"python"` | `"numba"` |
| `workers` | Worker count, `"auto"`, or `None` | `8` |

`prepare_gui_phi_display(...)` is a separate presentation helper. It reorders
and wraps the raw azimuth axis for the GUI view and accepts:

- `phi_min_deg`
- `phi_max_deg`
- `zero_direction`

### Optional diffraction-analysis helpers

The package root lazily exposes optional functionality from
[OSC_Reader/__init__.py](OSC_Reader/__init__.py).
These imports will fail with a clear `ImportError` if their optional
dependencies are missing.

Examples:

```python
from OSC_Reader import (
    setup_azimuthal_integrator,
    display,
    integrate_spec,
    plot_qz_vs_qr,
    fit_pvoigt_peaks,
    process_data,
)
```

Typical usage:

```python
from OSC_Reader import setup_azimuthal_integrator, display, integrate_spec

ai = setup_azimuthal_integrator("calibration.poni")
image = display("sample.osc", ai, show=False)
spec = integrate_spec(
    image,
    d=0.3,
    c=(1500, 1500),
    th_range=(5, 50),
    phi_range=(-90, 90),
)
```

## Architecture

### 1. Core `.osc` parser

The binary reader in
[OSC_Reader/OSC_Reader.py](OSC_Reader/OSC_Reader.py)
does the following:

- Validates that the file starts with the `RAXIS` signature.
- Detects byte order from the version field.
- Reads detector width and height from the file header.
- Extracts pixel data starting at byte offset `6000`.
- Converts the raw detector payload into a 2D integer array.

This module also implements the `.asc` and `.jpg` export helpers.

### 2. Unified package exports

[OSC_Reader/__init__.py](OSC_Reader/__init__.py)
acts as the public package facade:

- Core functions are imported eagerly.
- Heavy scientific utilities are exposed lazily.
- Optional submodules raise explicit import errors only when accessed.

That design keeps `import OSC_Reader` usable even when the full scientific
stack is not installed.

### 3. Qt desktop viewer

[OSC_Reader/OSC_Viewer.py](OSC_Reader/OSC_Viewer.py)
contains the main GUI:

- File loading happens through a worker object when Qt signal support is
  available.
- Angle conversion also runs in a worker thread so the UI stays responsive.
- The central image plot is linked to bottom and side profile plots.
- Histogram LUT controls manage the image intensity window.
- Beam center is displayed with a dedicated marker.
- Detector mode and angle mode share the same display framework but use
  different coordinate systems and aspect-lock rules.

### 4. Angle-space conversion engine

[OSC_Reader/angle_space.py](OSC_Reader/angle_space.py)
implements the detector-to-`φ`/`2θ` conversion path:

- Geometry is represented with `DetectorCakeGeometry`.
- Output data is returned as `DetectorCakeResult`.
- Input and output axes are validated for monotonicity and uniform spacing.
- The conversion can run with a pure-Python backend or a numba backend.
- Work is chunked across threads when the numba backend is active.
- A GUI helper wraps and reorders the azimuth axis to match the requested
  `φ=0` direction.

### 5. Optional analysis modules

- [OSC_Reader/tools.py](OSC_Reader/tools.py)
  contains pyFAI-driven integration helpers, image display utilities, and
  reciprocal-space plotting routines.
- [OSC_Reader/peak_analysis.py](OSC_Reader/peak_analysis.py)
  contains peak-fitting helpers oriented around Gaussian, Lorentzian, and
  pseudo-Voigt models.
- [OSC_Reader/viewer.py](OSC_Reader/viewer.py)
  contains an older matplotlib-based interactive viewer. The Qt viewer is the
  main GUI path.

## Validation and Quality Checks

The repository includes a `Validation/` directory with supporting material for
checking exported `.asc` data and documenting methodology:

- [Validation/Methodology.md](Validation/Methodology.md)
- [Validation/Validator.ipynb](Validation/Validator.ipynb)

At the time of writing, the repository does not include a dedicated automated
test suite. If you modify the GUI or conversion code, a reasonable minimum
sanity check is:

```bash
python -m compileall OSC_Reader
```

Then verify:

- an `.osc` file loads correctly,
- beam-center picking lands on the intended peak,
- `Convert to φ/2θ` produces a full-range angle view,
- `Show Detector` returns to detector coordinates cleanly.

## Packaging and Distribution

This project is best thought of as a library plus local desktop application.
There is no service deployment story because nothing here is designed to run as
a long-lived server.

If you want to distribute it internally:

1. Build a wheel or source distribution.
2. Install it into a managed Python environment.
3. On Windows, ship `Run_OSC_Viewer.bat` alongside the package if you want a
   double-click launcher experience.

Example build flow:

```bash
pip install build
python -m build
```

## Troubleshooting

### The GUI does not start

Check that `pyside6` and `pyqtgraph` are installed in the active environment:

```bash
pip show pyside6 pyqtgraph
```

### An optional function raises `ImportError`

You are likely calling something from `tools.py`, `peak_analysis.py`, or the
legacy matplotlib viewer without the required scientific stack installed.
Install the missing package and try again.

### The first angle conversion feels slower than later ones

That is expected. The viewer performs a background warm-up of the numba-backed
angle-space engine.

### Beam-center snapping picked the wrong bright spot

Zoom in further and click closer to the intended peak, or set `Center X` and
`Center Y` manually.

### Windows still opens `.osc` files in another application

Run `Register_OSC_Default_App.bat`, then sign out/in or restart Explorer if the
previous association remains cached.

## Contributing

Pull requests and issue reports are welcome. Before making large changes:

1. Confirm whether the change affects the core parser, GUI viewer, or optional
   scientific modules.
2. Keep the README and usage examples in sync with actual defaults.
3. Validate GUI behavior manually if you touch `OSC_Viewer.py` or
   `angle_space.py`.

## License

Released under the GPL-3.0 License. See [LICENSE](LICENSE).

## Contact

David Beckwitt - david.beckwitt@gmail.com
