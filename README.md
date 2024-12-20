# R-AXIS Area Detector Image Converter

A utility to convert `.osc` files generated by Rigaku R-Axis area detectors into various output formats, such as TIFF or ASCII grid files (`.asc`). This tool is designed to streamline data processing for scientific applications.

---

## Features

- **Convert `.osc` files to ASCII grid files (`.asc`)** for numerical or scientific analysis.
- **Convert '.osc' files to Images (.jpg)**
- **Read `.osc` files without making a file**.
- **Visualize `.osc` files interactively** using the new `OSC_Viewer` utility for improved analysis and interpretation.

## New Feature: OSC_Viewer

The `OSC_Viewer` script provides an interactive visualization of `.osc` files. It allows users to explore the image data with adjustable parameters such as minimum and maximum intensity (using sliders). Additionally, it offers interactive crosshair navigation to display pixel-specific information, including intensity and cross-sectional plots.

### Key Features of OSC_Viewer:
- **Interactive Visualization**: View `.osc` files as 2D images with cross-sectional plots along the X and Y axes.
- **Crosshair Interaction**: Navigate through the image interactively to see pixel-specific information including X, Y coordinates and intensity.
- **Adjustable Visualization Parameters**: Use sliders to adjust the visualization range (vmin and vmax) for optimal viewing.

## Commands

### Functions
1. **`read_osc(filename, RAW=False)`**
   - Reads a `.osc` file into a NumPy array.
   - **Parameters**:
     - `filename`: The path to the `.osc` file.
     - `RAW`: (Optional) If `True`, returns both the parsed data and the raw binary data.

2. **`convert_to_asc(filename, force=False)`**
   - Converts a `.osc` file into an ASCII grid file (`.asc`).
   - **Parameters**:
     - `filename`: The path to the `.osc` file to convert.
     - `force`: (Optional) If `True`, overwrites the destination file if it already exists.

3. **`visualize_osc_data(filename)`** (in `OSC_Viewer.py`)
   - Visualizes an `.osc` file as an interactive plot.
   - **Parameters**:
     - `filename`: The path to the `.osc` file to be visualized.

# Contributing
We welcome contributions to improve this utility! Feel free to submit pull requests or open issues for feature requests or bugs.

# Contact
For any questions or issues, please reach out to: David Beckwitt
Email: david.beckwitt@gmail.com
