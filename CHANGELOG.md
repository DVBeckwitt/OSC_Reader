# Changelog

## Unreleased

- **Fixed detector full-extent reset in the Qt viewer:**
  - `Detector` view now keeps enough pyqtgraph range limit slack for aspect-locked images.
  - `Reset View` now returns to the full detector extent instead of clipping wide or tall detector frames.
  - Status: fixed and covered by `test_detector_full_extent_covers_edges_after_load_and_reset`.

- **Hardened log display controls:**
  - `Image Log View`, `Bottom Profile Log Y`, and `Side Profile Log X` now tolerate zero, negative, `NaN`, and infinite detector values without breaking the viewer.
  - Cached log-toggle state is restored safely before and after loading detector data.
  - Status: fixed and covered by viewer regression tests for image log data, profile log data, and cached log state.

- **Validation:**
  - Verified with `python -m unittest discover -s tests -p "test_*.py"`.
  - Verified with `python -m compileall OSC_Reader`.

## Changes Made in the New Version 0.3.1

- **Data Reading in `convert_to_asc`:**
  - Old: Used `read_osc(filename)` to interpret the RAXIS file.
  - New: Directly reads raw file bytes using `np.fromfile` and calls `_interpret` internally, removing dependency on `read_osc` here.

- **File Validation in `convert_to_asc`:**
  - Old: Validation and error checks were deferred to `read_osc`.
  - New: Checks the first five bytes (`raw[:5]`) to verify the file is a RAXIS file before interpretation. Raises `IOError` if invalid.

- **Writing the `.asc` File:**
  - Old: Wrote all pixel data on one line per row with simple spacing.
  - New: Aligns columns based on the largest value’s width and formats the file with 10 columns per line for readability.

- **Data Interpretation with `_interpret`:**
  - Old: Pixel data was extracted without a specified offset.
  - New: Uses a `data_start_offset = 6000` to locate pixel data. Pixel data is reshaped into `(height, width)` after slicing the array from this offset.
  - Maintains the signed 16-bit interpretation logic but now tied to the known file structure.

- **Image Display in `visualize_osc_data`:**
  - Old: `imshow` used `extent=[0, data.shape[1], data.shape[0], 0]`.
  - New: `imshow` now uses `extent=[0, data.shape[1], 0, data.shape[0]]`, flipping the Y-axis for a more standard coordinate system.

- **Error Handling:**
  - Old: Depended on errors raised by `read_osc`.
  - New: Raises `IOError` earlier for non-RAXIS files and isolates interpretation errors more cleanly.

These updates streamline data reading, improve `.asc` output formatting, and clarify the axis orientation in image visualization.
