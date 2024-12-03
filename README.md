# R-AXIS Area Detector Image Converter

A utility to convert `.osc` files generated by Rigaku R-Axis area detectors into various output formats, 
such as TIFF or ASCII grid files (`.asc`). 
This tool is designed to streamline data processing for scientific applications.

---

## Features

- **Convert `.osc` files to ASCII grid files (`.asc`)** for numerical or scientific analysis.
- **Read** '.osc' files without making a file.


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


# Contributing
We welcome contributions to improve this utility! Feel free to submit pull requests or open issues for feature requests or bugs.

# Contact
For any questions or issues, please reach out to: David Beckwitt
Email: david.beckwitt@gmail.com