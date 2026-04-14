# OSC Reader and Diffraction Toolkit

![OSC Reader viewer preview](docs/viewer_preview.png)

`OSC_Reader` is a Python package and local desktop viewer for Rigaku R-AXIS
`.osc` detector images and other common 2D diffraction image formats. It
combines a validated native `.osc` parser, a FabIO-backed compatibility layer
for other detector formats, a PySide6/pyqtgraph GUI for interactive
inspection, an exact detector-to-angle-space conversion pipeline, and optional
diffraction-analysis helpers inherited from earlier `DVB_pack` work.

This repository is primarily a local scientific tool, not a web service. The
main use cases are:

- opening detector frames interactively,
- exporting detector data to `.asc` or preview `.jpg`,
- converting detector images into `φ/2θ` and `q-space` views,
- selecting one or more ROIs in angle space and comparing integrated profiles,
- scripting the same workflows from Python.

## Key Features

- Native Rigaku `.osc` parsing with a dedicated binary reader.
- Shared detector import for `.osc`, `.cbf`, `.edf`, `.img`, `.mccd`, `.sfrm`,
  `.gfrm`, `.tif`, `.tiff`, `.h5`, `.hdf5`, and `.nxs`.
- High-FPS Qt viewer with detector, `φ/2θ`, and `q-space` views.
- Interactive beam-center picking with local peak snapping.
- Exact detector-to-angle-space rebinning with a numba-backed fast path.
- Multi-ROI angle-space comparison with peak-centered profile overlays.
- Optional diffraction-analysis helpers for pyFAI integration, plotting, and
  peak fitting.

## Table of Contents

- [Tech Stack](#tech-stack)
- [Repository Layout](#repository-layout)
- [Supported Formats](#supported-formats)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Desktop Viewer Guide](#desktop-viewer-guide)
- [Python API Overview](#python-api-overview)
- [Architecture](#architecture)
- [Core Math and Domain Conventions](#core-math-and-domain-conventions)
- [Available Commands](#available-commands)
- [Testing and Validation](#testing-and-validation)
- [Packaging and Distribution](#packaging-and-distribution)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.8+ |
| Numeric core | NumPy |
| Fast conversion backend | numba |
| Detector format compatibility | FabIO |
| GUI | PySide6 + pyqtgraph |
| Preview image export | OpenCV |
| Packaging | setuptools |
| Optional analysis stack | matplotlib, pandas, SciPy, pyFAI, lmfit, datashader |
| Windows desktop helpers | Batch launchers and file-association scripts |

## Repository Layout

```text
OSC_Reader/
|-- OSC_Reader/
|   |-- __init__.py          # Public package facade with lazy optional exports
|   |-- OSC_Reader.py        # Native .osc parser plus .asc/.jpg conversion helpers
|   |-- image_import.py      # Unified detector-image loader (.osc + FabIO formats)
|   |-- OSC_Viewer.py        # Main PySide6/pyqtgraph desktop viewer
|   |-- angle_space.py       # Exact detector -> φ/2θ and q-space helpers
|   |-- tools.py             # Optional diffraction and plotting utilities
|   |-- peak_analysis.py     # Optional lmfit-based peak-fitting helpers
|   `-- viewer.py            # Legacy matplotlib-based interactive viewer
|-- Validation/
|   |-- Methodology.md       # Validation notes for exported data and comparisons
|   `-- Validator.ipynb      # Notebook for validation experiments
|-- docs/
|   `-- viewer_preview.png   # Preview image used in this README
|-- Run_OSC_Viewer.bat       # Windows launcher for the Qt viewer
|-- Register_OSC_Default_App.bat
|                           # Windows file-association helper for .osc files
|-- setup.py                 # Package metadata and install requirements
|-- CHANGELOG.md
|-- LICENSE
`-- README.md
```

## Supported Formats

The shared detector loader in `OSC_Reader/image_import.py` exposes a curated
format list used by the GUI file picker and the public API.

### Core detector image extensions

- `.osc`
- `.cbf`
- `.edf`
- `.img`
- `.mccd`
- `.sfrm`
- `.gfrm`
- `.tif`
- `.tiff`
- `.h5`
- `.hdf5`
- `.nxs`

### Loader behavior

- `.osc` files use the in-repo native Rigaku parser by default.
- All other supported formats use FabIO.
- Multi-frame formats can be loaded through FabIO with `frame_index`.
- Native `.osc` loading currently supports `frame_index=0` only.

## Prerequisites

### Required for the core package

- Python 3.8 or newer
- `pip`
- A desktop environment if you want to run the GUI viewer

### Optional dependencies for extended analysis features

Install these only if you need functions from `OSC_Reader.tools`,
`OSC_Reader.peak_analysis`, or the legacy matplotlib viewer:

- `matplotlib`
- `pandas`
- `scipy`
- `pyFAI`
- `lmfit`
- `datashader`

## Getting Started

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

macOS or Linux:

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

The package metadata in `setup.py` installs these core runtime dependencies:

- `numpy>=1.18.0`
- `numba>=0.59.0`
- `tifffile>=2020.9.3`
- `fabio`
- `docopt>=0.6.2`
- `logbook>=1.5.3`
- `opencv-python>=4.0.0`
- `pyqtgraph>=0.14.0`
- `pyside6>=6.5.0`

### 4. Install optional scientific extras if needed

```bash
pip install matplotlib pandas scipy pyfai lmfit datashader
```

### 5. No environment variables are required

The project does not currently depend on an `.env` file, API keys, or runtime
secrets for normal local usage.

### 6. Sanity-check the installation

```bash
python -m compileall OSC_Reader
```

## Quick Start

### Open a detector image from Python

```python
from OSC_Reader import read_detector_image

image = read_detector_image("example.tif")
print(image.shape, image.dtype)
```

### Load a detector image with metadata

```python
from OSC_Reader import load_detector_image

result = load_detector_image("example.cbf")
print(result.format_name)
print(result.loader_name)
print(result.frame_index, result.frame_count)
print(result.data.shape)
```

### Read a native Rigaku `.osc` image

```python
from OSC_Reader import read_osc

image = read_osc("example.osc")
print(image.shape, image.dtype)
```

### Export a detector image to ASCII

```python
from OSC_Reader import convert_to_asc

convert_to_asc("example.osc")
```

### Export a detector image to a JPEG preview

```python
from OSC_Reader import detector2jpg

detector2jpg("example.tif")
```

### Launch the desktop viewer

Open a specific file:

```bash
python -m OSC_Reader.OSC_Viewer path/to/example.osc
```

Open the viewer and choose a file from the dialog:

```bash
python -m OSC_Reader.OSC_Viewer
```

Launch through the public helper:

```python
from OSC_Reader import visualize_detector_data

visualize_detector_data("example.tif")
```

Backward-compatible alias:

```python
from OSC_Reader import visualize_osc_data

visualize_osc_data("example.osc")
```

### Windows launchers

Start the viewer with the bundled batch file:

```bat
Run_OSC_Viewer.bat
```

Or drag a detector image onto `Run_OSC_Viewer.bat`.

Register `.osc` files to open with the viewer for the current Windows user:

```bat
Register_OSC_Default_App.bat
```

## Desktop Viewer Guide

The main interactive application lives in `OSC_Reader/OSC_Viewer.py`. It is
the canonical GUI for this repository.

### Layout overview

The window is organized into four main regions:

1. A top toolbar with `File`, `View`, and contextual `Tools` groups.
2. A settings shelf with `Geometry`, `Sampling`, and `Profiles`.
3. A central image panel with histogram/LUT controls.
4. Linked bottom and side line-profile plots plus status-bar feedback.

### File controls

| Control | Shortcut | Purpose |
| --- | --- | --- |
| `Open Image` | `Ctrl+O` | Open a detector image through the shared file loader |
| `Save Image` | `Ctrl+S` | Save the rendered main image plot |
| `Reset View` | `R` | Reset the current image view to full extent |

### View controls

| Control | Shortcut | Purpose |
| --- | --- | --- |
| `Detector` | `Ctrl+1` | Show raw detector pixel coordinates |
| `φ/2θ` | `Ctrl+2` | Convert the loaded detector image into angle space |
| `q-space` | `Ctrl+3` | Convert the current detector image into reciprocal-space display via the angle-space pipeline |

### Tool controls

These are contextual and depend on the active view:

| Control | Shortcut | Availability | Purpose |
| --- | --- | --- | --- |
| `Pick Beam Center` | `B` | Detector view | Click near the direct beam or bright reference peak and snap to the nearest local peak top |
| `ROI` | `A` | `φ/2θ` view | Draw one or more ROIs for integrated `2θ` and `φ` comparisons |

### Geometry settings

`Geometry` is always expanded in the current UI and contains the parameters
that define the coordinate conversion:

| Setting | Default | Meaning |
| --- | --- | --- |
| `Center X (px)` | detector midpoint on load | Beam center column |
| `Center Y (px)` | detector midpoint on load | Beam center row |
| `Distance` | `75.0 mm` | Sample-to-detector distance |
| `Pixel Size` | `0.1 mm` | Detector pixel pitch |
| `Wavelength (Å)` | `1.54060` | Radiation wavelength used for `q-space` conversion |
| `Incident Angle` | `0.0 deg` | Incident angle used for `q-space` conversion |
| `φ Zero Direction` | `Up` | Which detector direction should map to `φ = 0` in the GUI display |

### Sampling settings

`Sampling` is always expanded and controls the output resolution for both
angle-space and `q-space` rebins:

| Setting | Default on startup | Behavior on image load |
| --- | --- | --- |
| `2θ Bins` | `1000` | Resets to detector width after loading an image |
| `φ Bins` | `720` | Resets to detector height after loading an image |

### Profile settings

`Profiles` is always expanded and contains display-only toggles:

| Setting | Purpose |
| --- | --- |
| `Image Log View` | Render the main image with logarithmic intensity scaling |
| `Bottom Profile Log Y` | Use a log Y axis for the bottom line profile |
| `Side Profile Log X` | Use a log X axis for the side line profile |

`Image Log View` changes the rendered image intensity mapping only. It does not
modify the underlying detector data used for conversions.

### Typical detector-to-angle workflow

1. Open a detector image.
2. Stay in `Detector` view.
3. Enable `Pick Beam Center`.
4. Click near the direct beam or other intended bright reference.
5. Let the viewer snap to the nearest local peak top.
6. Fine-tune `Center X` and `Center Y` if necessary.
7. Enter `Distance`, `Pixel Size`, and optionally `φ Zero Direction`.
8. Adjust `2θ Bins` and `φ Bins`.
9. Switch to `φ/2θ`.
10. Optionally switch again to `q-space` after setting `Wavelength (Å)` and
    `Incident Angle`.

### ROI workflow

ROI selection is available only in `φ/2θ` view because the ROI summaries are
integrated `2θ` and `φ` profiles.

#### Creating the first ROI

1. Switch to `φ/2θ`.
2. Toggle `ROI`.
3. Left-click and drag in empty image space.
4. Release the mouse to place the rectangle.

The ROI profile figure stays hidden until that first rectangle is completed.

#### Editing and comparing ROIs

- Each ROI is a pyqtgraph `RectROI` with draggable corner handles.
- You can resize or move existing ROIs after placement.
- Additional drags in empty space create additional ROIs.
- Multiple ROIs are overlaid in a dedicated profile window.
- When more than one ROI is present, the integrated curves are shifted so each
  ROI peak is centered at `2θ = 0` and `φ = 0` in the comparison plot.

#### ROI profile window controls

The ROI profile dialog includes:

| Control | Purpose |
| --- | --- |
| `Add Region` | Arms the main image so the next drag adds another ROI |
| `Delete Selected` | Removes the selected ROI or ROIs |
| `2θ Log Y` | Log-scale the integrated `2θ` plot |
| `φ Log Y` | Log-scale the integrated `φ` plot |
| `Save Figure` | Save the ROI comparison figure |

### Status-bar behavior

The status bar reports:

- the active view (`Detector`, `φ/2θ`, or `q-space`),
- the render target (`60 FPS`),
- cursor coordinates and intensity,
- beam-center summary,
- ROI summary when ROI mode is active.

### Save behavior

- `Save Image` exports the currently rendered main image plot.
- The ROI profile window has its own `Save Figure` button for ROI overlays.

## Python API Overview

### Core detector I/O

`OSC_Reader/__init__.py` exposes the main import and export helpers:

```python
from OSC_Reader import (
    ShapeError,
    convert_to_asc,
    detector2jpg,
    load_detector_image,
    read_detector_image,
    read_osc,
    supported_detector_extensions,
)
```

### `load_detector_image(...)`

This returns a `DetectorImageLoadResult` dataclass:

```python
from OSC_Reader import load_detector_image

result = load_detector_image("example.edf", frame_index=0)

print(result.path)
print(result.format_name)
print(result.loader_name)
print(result.frame_index, result.frame_count)
print(result.header)
print(result.data.shape)
```

Fields:

- `data`
- `path`
- `format_name`
- `loader_name`
- `frame_index`
- `frame_count`
- `header`

### Detector-to-angle conversion API

```python
from OSC_Reader import (
    convert_image_to_phi_2theta_space,
    prepare_gui_phi_display,
    read_detector_image,
)

image = read_detector_image("example.osc")

result = convert_image_to_phi_2theta_space(
    image,
    distance_mm=75.0,
    pixel_size_mm=0.1,
    center_row_px=(image.shape[0] - 1) / 2.0,
    center_col_px=(image.shape[1] - 1) / 2.0,
    radial_bins=image.shape[1],
    azimuth_bins=image.shape[0],
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

### q-space conversion API

```python
from OSC_Reader import (
    convert_image_to_phi_2theta_space,
    prepare_gui_phi_display,
    read_detector_image,
)
from OSC_Reader.angle_space import convert_phi_2theta_to_qr_qz_space

image = read_detector_image("example.osc")
angle_result = convert_image_to_phi_2theta_space(
    image,
    distance_mm=75.0,
    pixel_size_mm=0.1,
    center_row_px=(image.shape[0] - 1) / 2.0,
    center_col_px=(image.shape[1] - 1) / 2.0,
)

cake, two_theta_deg, phi_deg = prepare_gui_phi_display(angle_result)
q_result = convert_phi_2theta_to_qr_qz_space(
    cake,
    two_theta_deg,
    phi_deg,
    wavelength_angstrom=1.5406,
    incident_angle_deg=0.0,
)

print(q_result.qr.shape, q_result.qz.shape, q_result.intensity.shape)
```

`convert_phi_2theta_to_qr_qz_space(...)` returns a `QSpaceResult` with:

- `qr`
- `qz`
- `intensity`

### Optional diffraction-analysis helpers

These are exposed lazily from `OSC_Reader.__init__`. They are available only if
their scientific dependencies are installed.

Examples:

```python
from OSC_Reader import (
    setup_azimuthal_integrator,
    display,
    integrate_spec,
    plot_qz_vs_qr,
    fit_pvoigt_peaks,
)
```

Important optional modules:

- `OSC_Reader/tools.py`
- `OSC_Reader/peak_analysis.py`
- `OSC_Reader/viewer.py`

## Architecture

### Public package surface

`OSC_Reader/__init__.py` is the package facade:

- core detector I/O is imported eagerly,
- GUI entry points are imported on demand,
- heavy scientific modules are loaded lazily,
- missing optional dependencies surface as targeted `ImportError`s only when
  those features are accessed.

That design keeps `import OSC_Reader` lightweight even when optional analysis
packages are not installed.

### Native `.osc` parser

`OSC_Reader/OSC_Reader.py` contains the in-repo Rigaku parser:

- it verifies that the file begins with the `RAXIS` signature,
- determines byte order from the version field,
- reads detector width and height from the header,
- slices the pixel payload starting at byte offset `6000`,
- reshapes the detector frame and performs the expected signed-value fixup.

The same module also provides:

- `convert_to_asc(...)`
- `detector2jpg(...)`
- `osc2jpg(...)`

### Shared detector loader

`OSC_Reader/image_import.py` is the canonical detector import layer used by the
viewer and the public API.

Responsibilities:

- pick the native `.osc` path when appropriate,
- fall back to FabIO for other detector formats,
- normalize arrays to a strict 2D detector frame,
- preserve basic import metadata for callers that need provenance.

### Desktop viewer

`OSC_Reader/OSC_Viewer.py` is the main application:

- file loading happens through a Qt worker object when the Qt signal stack is
  available,
- detector-to-angle conversions run in a worker thread so the UI stays
  responsive,
- the central image panel shares a view with linked horizontal and vertical
  line profiles,
- a histogram LUT controls the rendered image window,
- contextual tools are shown only when they are relevant to the active view,
- ROI overlays and ROI-integrated profile plots are managed entirely inside the
  same viewer module.

### Conversion engine

`OSC_Reader/angle_space.py` contains the geometry and rebinning logic:

- `DetectorCakeGeometry` stores detector geometry in metric units,
- `DetectorCakeResult` stores the rebinned `φ/2θ` result,
- `QSpaceResult` stores the rebinned reciprocal-space result,
- a pure-Python path exists for environments without numba,
- a numba-backed path parallelizes work across chunks when available,
- `warm_angle_space_engine(...)` primes the numba path to reduce the first
  conversion cost in the GUI.

### Optional analysis modules

`OSC_Reader/tools.py` and `OSC_Reader/peak_analysis.py` are not part of the
core viewer path, but they remain part of the repository for richer diffraction
analysis workflows:

- pyFAI-based detector calibration and integration helpers,
- plotting helpers,
- reciprocal-space plotting,
- peak fitting with Gaussian, Lorentzian, and pseudo-Voigt models.

## Core Math and Domain Conventions

This repository is algorithm-heavy, so the user-facing coordinate conventions
matter.

### Coordinate system

- Detector-space coordinates are pixel indices.
- Beam center is stored as `(center_row_px, center_col_px)`.
- Pixel size and distance are supplied in millimetres through the public
  conversion API and converted internally to metres.
- `φ Zero Direction` is a GUI/display convention, not a detector metadata
  file field.
- The GUI allows `Up`, `Left`, `Down`, or `Right` as the `φ = 0` direction.

### Detector to angle-space transform

The angle-space path models a flat detector and computes the polar coordinates
of detector-pixel corners relative to the chosen beam center.

For a detector corner at metric offsets `x` and `y` from the beam center and a
sample-to-detector distance `d`, the core corner conversion is:

```math
2\theta = \arctan\left(\frac{\sqrt{x^2 + y^2}}{d}\right)
```

```math
\chi = \operatorname{atan2}(y, x)
```

The implementation then performs exact pixel splitting into uniformly spaced
`2θ` and azimuth bins instead of assigning each detector pixel to just one
output bin. That exact splitter is what powers
`integrate_detector_to_cake_exact(...)`.

### GUI phi convention

The raw azimuth axis is transformed for display with the selected `φ = 0`
direction:

```math
\phi_{gui} = \mathrm{wrap}_{[-180, 180)}\left(\phi_0 - \chi\right)
```

where `phi_0` is determined by the chosen zero direction:

- `Right -> 0 deg`
- `Down -> 90 deg`
- `Left -> 180 deg`
- `Up -> -90 deg`

`prepare_gui_phi_display(...)` sorts and wraps the raw azimuth output into this
GUI convention.

### q-space conversion

The viewer's `q-space` view is derived from the current `φ/2θ` image, not
from a separate detector-space geometry path. The implementation converts the
current angle-space grid into `qr` and `qz` using the configured wavelength and
incident angle, then rebins those coordinates with `numpy.histogram2d`.

As implemented in `convert_phi_2theta_to_qr_qz_space(...)`, with

- `k = 2 pi / lambda`,
- `alpha = incident angle`,
- `theta = deg2rad(radial_deg)`,
- `phi = deg2rad(phi_deg)`,

the intermediate components are:

```math
q_y = \left(\cos\alpha(\cos\theta - 1) + \sin\alpha\sin\theta\cos\phi\right)k
```

```math
q_z = \left(-\sin\alpha(\cos\theta - 1) + \cos\alpha\sin\theta\cos\phi\right)k
```

```math
q_r = \mathrm{sign}(\phi)\sqrt{(k\sin\theta\sin\phi)^2 + q_y^2}
```

The final displayed `q-space` image is an averaged rebin of those samples onto
uniform `qr` and `qz` centers.

### Performance behavior

- The GUI targets `60 FPS` for cursor/profile updates.
- `warm_angle_space_engine(...)` primes the numba path in a background thread.
- The angle-space engine defaults to `8` workers and caps against CPU count.
- The first conversion can be slower than later conversions because of warm-up.
- ROI profile overlays are computed from the currently visible angle-space ROIs
  in Python on demand.

### Scope and assumptions

- The main viewer exposes flat-detector geometry controls only.
- `correct_solid_angle` exists in the Python API but is not currently exposed
  as a GUI toggle.
- The GUI is intentionally a local desktop tool; there is no server-side or
  browser-based execution path in this repository.

## Available Commands

### Development and inspection

| Command | Purpose |
| --- | --- |
| `python -m compileall OSC_Reader` | Compile-check the package |
| `python -m OSC_Reader.OSC_Viewer` | Launch the GUI and open a file picker |
| `python -m OSC_Reader.OSC_Viewer path/to/file` | Launch the GUI with a specific detector image |
| `python -c "from OSC_Reader import read_detector_image; ..."` | Script detector I/O from Python |

### Windows helpers

| Command | Purpose |
| --- | --- |
| `Run_OSC_Viewer.bat` | Launch the desktop viewer using a discovered Python installation |
| `Register_OSC_Default_App.bat` | Register `.osc` to open with the viewer for the current user |

## Testing and Validation

The repository does not currently ship a formal automated test suite.

### Minimum maintenance checks

When changing core parsing, conversion, or GUI code, a practical minimum is:

```bash
python -m compileall OSC_Reader
```

Then manually verify:

1. A detector image opens successfully.
2. `Detector`, `φ/2θ`, and `q-space` view switches all work.
3. Beam-center picking lands on the intended feature.
4. A first ROI can be drawn in `φ/2θ` view.
5. Additional ROIs can be added, selected, deleted, and saved from the ROI
   profile window.
6. `Save Image` and ROI `Save Figure` both export expected output.

### Validation materials

The `Validation/` directory contains supporting validation artifacts:

- `Validation/Methodology.md`
- `Validation/Validator.ipynb`

These are useful when checking detector export behavior and conversion results.

## Packaging and Distribution

This project is best thought of as:

- a Python library for detector-image I/O and coordinate conversion,
- a local desktop GUI for interactive inspection.

It is not a deployable web application, and the repository does not contain
Docker, Kubernetes, PaaS, or service deployment configuration.

### Build a distribution

```bash
pip install build
python -m build
```

### Internal distribution guidance

If you want to distribute the tool internally:

1. Build and ship a wheel or source distribution.
2. Install it into a managed Python environment.
3. On Windows, include `Run_OSC_Viewer.bat` if you want a double-click GUI
   entry point.
4. Optionally run `Register_OSC_Default_App.bat` for user-level `.osc`
   association.

## Troubleshooting

### The GUI does not start

Check that the GUI stack is installed:

```bash
pip show pyside6 pyqtgraph
```

### Non-`.osc` images fail to open

FabIO is required for the non-native formats:

```bash
pip show fabio
```

Install it if necessary:

```bash
pip install fabio
```

### Optional analysis functions raise `ImportError`

Functions coming from `tools.py`, `peak_analysis.py`, or the legacy
matplotlib viewer require optional scientific packages that are not part of
the minimal install.

Install the missing extras, for example:

```bash
pip install matplotlib pandas scipy pyfai lmfit datashader
```

### The first conversion is slower than later ones

That is expected. The viewer warms the numba-backed conversion engine in the
background, and the first real conversion still carries some compilation and
allocation cost.

### The ROI button is disabled

ROI selection is only enabled in `φ/2θ` view after a detector image is
loaded.

### The `q-space` button does not do anything

`q-space` depends on the detector image and the angle-space conversion path.
Load an image first, then ensure `Distance`, `Pixel Size`, `Wavelength (Å)`,
and `Incident Angle` are set sensibly.

### Windows still opens `.osc` files in another app

Run:

```bat
Register_OSC_Default_App.bat
```

If Windows still prefers the old handler, sign out and back in or restart
Explorer.

## Contributing

Pull requests and issue reports are welcome.

When changing the repository:

1. Keep the README aligned with the actual UI labels and API surface.
2. Treat `OSC_Reader/OSC_Viewer.py` and `OSC_Reader/angle_space.py` as coupled
   when changing view-mode behavior.
3. Validate GUI behavior manually when touching beam-center picking, ROI
   drawing, or view switching.
4. Avoid documenting optional modules as mandatory dependencies.

## License

Released under the GPL-3.0 License. See [LICENSE](LICENSE).

## Contact

David Beckwitt - david.beckwitt@gmail.com
