# OSC Reader and Diffraction Toolkit

`OSC_Reader` is a lightweight toolkit for working with Rigaku RAXIS `.osc`
images.  It now bundles the diffraction utilities from the former
`DVB_pack` project, providing a single, streamlined package for loading raw
detector frames, visual exploration, azimuthal integration and peak
fitting.

---

## Features

- **Read `.osc` files** directly into memory for further processing.
- **Convert to ASCII grid files (`.asc`)** for numerical analysis.
- **Convert to JPEG images** for easy sharing or documentation.
- **Interactive viewer** with cross‑hair inspection, pixel intensity
  readouts, and adjustable intensity scaling.
- **Diffraction utilities** (`OSC_Reader.tools`) for azimuthal integration,
  reciprocal space plotting and detector corrections.
- **Peak analysis helpers** (`OSC_Reader.peak_analysis`) for fitting and
  analysing pseudo-Voigt peaks.

## Installation

```bash
pip install .
```

This installs the core `OSC_Reader` package along with its minimal
dependencies.  The diffraction and peak-analysis utilities rely on
additional scientific packages such as `matplotlib`, `pandas`, `pyFAI`,
`fabio`, `scipy`, and `lmfit`.  Install them with your preferred package
manager when you need those features.

## Quick Start

### Read a file

```python
from OSC_Reader import read_osc

data = read_osc("example.osc")
print(data.shape)
```

### Convert to an ASCII grid

```python
from OSC_Reader import convert_to_asc

convert_to_asc("example.osc")  # creates example.asc
```

### Convert to a JPEG image

```python
from OSC_Reader import osc2jpg

osc2jpg("example.osc")  # creates example.jpg
```

### Visualize interactively

Run the viewer as a script to explore pixel values and cross sections:

```bash
python -m OSC_Reader.OSC_Viewer path/to/example.osc
```

or call it directly from Python:

```python
from OSC_Reader import visualize_osc_data

visualize_osc_data("example.osc")
```

### Use the diffraction utilities

```python
from OSC_Reader import (
    setup_azimuthal_integrator,
    display,
    integrate_spec,
    plot_qz_vs_qr,
)

ai = setup_azimuthal_integrator("calibration.poni")
data = display("sample.osc", ai, show=False)
spec = integrate_spec(data, d=0.3, c=(1500, 1500), th_range=(5, 50), phi_range=(-90, 90))
plot_qz_vs_qr(spec)
```

### Peak analysis

```python
from OSC_Reader import fit_pvoigt_peaks, process_data

results = process_data(ai, data, regions=[[7, 12, -20, 20, "003"]])
fits = fit_pvoigt_peaks(results)
```

## Contributing

Pull requests and issue reports are welcome.  If you have an idea or find
 a bug, please open an issue so we can discuss it.

## License

Released under the GPL‑3.0 License.  See the [LICENSE](LICENSE) file for
details.

## Contact

David Beckwitt – david.beckwitt@gmail.com

