from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

import numpy as np
import logbook
import cv2

from os.path import splitext, exists
import logging

class ShapeError(Exception):
    """An error encountered while attempting to slice and reshape the
       data to fit the file-specified dimensions."""

    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception


_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "plot_qr": ("tools", "plot_qr"),
}


def osc2jpg(input_file):
    # Read the OSC file into a numpy array
    data = read_osc(input_file)
    # Take the log of the data to enhance contrast
    data = np.log(data)

    # Normalize data to 0-255 for display as an 8-bit image
    min_val, max_val = data.min(), data.max()
    # If the image is uniform, handle that edge case
    if min_val == max_val:
        image_array = np.zeros_like(data, dtype=np.uint8)
    else:
        image_array = (255 * (data - min_val) / (max_val - min_val)).astype(np.uint8)
    output_file = f"{splitext(input_file)[0]}.jpg"
    # Write the image to a JPG file
    cv2.imwrite(output_file, image_array)
    print(f"Converted {input_file} to {output_file}")

def convert_to_asc(filename, force=False):
    """Converts a RAXIS file to an ASCII grid file (.asc).

    Parameters
    ----------
    filename : str
        The filename of the RAXIS file to convert.
    force : bool, optional
        Whether to overwrite an existing file of the same name.
        Default is False.
    """
    print(f"Converting {filename}")

    newfilename = f"{splitext(filename)[0]}.asc"
    if exists(newfilename) and not force:
        print(f"The file {newfilename} already exists. Re-run with --force "
              "to overwrite it.")
        return

    try:
        # Directly read raw bytes and interpret them using _interpret.
        raw = np.fromfile(filename, dtype='u1')
        if raw[:5].tobytes() != b'RAXIS':
            raise IOError("This file doesn't seem to be a RAXIS file at all. Aborting!")
        data = _interpret(raw)
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return

    try:
        with open(newfilename, 'w') as asc_file:
            asc_file.write('(DAS)^2 2-D text output of image region:\n')
            asc_file.write(f'Number of pixels in X direction =       {data.shape[1]}\n')
            asc_file.write(f'Number of pixels in Y direction =       {data.shape[0]}\n')
            asc_file.write('X starting pixel of ROI =          1\n')
            asc_file.write('Y starting pixel of ROI =          1\n')
            asc_file.write('ROI pixel values follow (X-direction changing fastest, bottom left first):\n')

            max_val = data.max()
            width = len(str(max_val)) + 1  # One space more than the largest number's length
            # get the rows
            rows, cols = data.shape
            
            for row_idx in range(rows):
                row_data = data[row_idx, :]
                for start in range(0, cols, 10):
                    line_values = row_data[start:start+10]
                    asc_file.write(''.join(f"{val:{width}d}" for val in line_values) + '\n')

        print(f'Converted {filename} to {newfilename}')
    except Exception as e:
        logging.error(f"Error writing {newfilename}: {e}")

        
def _interpret(arr):
    """Adapted from libmagic and modified to apply the discovered transformation.

       Parameters
       ----------
       arr : ndarray (uint8)
         1-dimensional ndarray of dtype uint8 containing the whole RAXIS file

       Returns
       -------
       arr : ndarray (int32)
         2-dimensional int32 array of transformed RAXIS detector data
    """
    # Determine endianness from the version field
    version = arr[796:800].view('>u4')[0]
    endian = '>' if version < 20 else '<'

    # Read width and height
    width = int(arr[768:772].view(endian + 'u4')[0])
    height = int(arr[772:776].view(endian + 'u4')[0])

    logbook.info('Interpreting as {}-endian with dimensions {}x{}'
                 .format('big' if endian == '>' else 'little',
                         width, height))

    diagnostics = f"""
Diagnostic information:
    length of the raw array (in bytes): {len(arr)}
    length of the raw array / 4 :       {len(arr) // 4}
    width:                              {width}
    height:                             {height}
    len / 4 - (width * height):         {len(arr) // 4 - (width * height)}
    """
    logbook.debug(diagnostics)

    try:
        # Adjust this offset based on the known RAXIS file structure:
        data_start_offset = 6000
        pixel_count = width * height
        # Extract pixel data starting at the known offset
        pixel_data = arr[data_start_offset:].view(endian + 'u2')[:pixel_count].reshape((height, width))

        # Convert to int32 for safe arithmetic
        int32_arr = pixel_data.astype('int32')
        # Correctly interpret signed 16-bit:
        # Values >= 0x8000 represent negative numbers and require subtracting 0x10000.
        mask = int32_arr >= 0x8000
        int32_arr[mask] -= 0x10000
        int32_arr[mask] += 0x8000
        int32_arr[mask] *= 32
        
        # Apply the discovered transformation: (signed_val + 0x8000) * 32
        final_arr = int32_arr

        return final_arr

    except ValueError as err:
        raise ShapeError(
            """Couldn't convert this array because of a problem interpreting
               its shape. This file may be corrupt.
            """ + diagnostics, original_exception=err) from err


def read_osc(filename, RAW = False):
    """Reads a RAXIS OSC file into an ndarray

       Parameters
       ----------
       filename : string
         The filename to read.

       Returns
       -------
       arr : ndarray
         2-dimensional ndarray of dtype uint16
    """
    # Attempt to interpret the file
    raw = np.fromfile(filename, dtype='u1')
    if raw[:5].tobytes() != b'RAXIS':
        raise IOError("This file doesn't seem to be a RAXIS file at all. "
                      "(A check that the first five characters are 'RAXIS'"
                      " failed). Aborting!")
    if RAW:
        return  _interpret(raw), raw
    else:
        return _interpret(raw)


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = import_module(f".{module_name}", __package__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ShapeError",
    "convert_to_asc",
    "osc2jpg",
    "read_osc",
    *_OPTIONAL_EXPORTS.keys(),
]
