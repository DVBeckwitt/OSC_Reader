# OSC Reader and Viewer

OSC_Reader is a lightweight toolkit for working with RAXIS `.osc` image
files produced by Rigaku area detectors.  It can load raw detector data
into NumPy arrays, export the data to common formats, and provide an
interactive viewer for quick inspection.

---

## Features

- **Read `.osc` files** directly into memory for further processing.
- **Convert to ASCII grid files (`.asc`)** for numerical analysis.
- **Convert to JPEG images** for easy sharing or documentation.
- **Interactive viewer** with cross‑hair inspection, pixel intensity
  readouts, and adjustable intensity scaling.

## Installation

```bash
pip install .
```

This installs the `OSC_Reader` package along with all required
dependencies (NumPy, tifffile, docopt and logbook).

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
from OSC_Reader.OSC_Reader import osc2jpg

osc2jpg("example.osc")  # creates example.jpg
```

### Visualize interactively

Run the viewer as a script to explore pixel values and cross sections:

```bash
python -m OSC_Reader.OSC_Viewer path/to/example.osc
```

## Contributing

Pull requests and issue reports are welcome.  If you have an idea or find
 a bug, please open an issue so we can discuss it.

## License

Released under the GPL‑3.0 License.  See the [LICENSE](LICENSE) file for
details.

## Contact

David Beckwitt – david.beckwitt@gmail.com

