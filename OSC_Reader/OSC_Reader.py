import numpy as np
import logbook

from os.path import splitext, exists
import logging

class ShapeError(Exception):
    """An error encountered while attempting to slice and reshape the
       data to fit the file-specified dimensions."""

    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception


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
        # Read the RAXIS file to get data as a 2D array
        data = read_osc(filename)
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return

    try:
        # Write data to .asc format
        with open(newfilename, 'w') as asc_file:
            for row in data:
                asc_file.write(' '.join(map(str, row)) + '\n')
        print(f'Converted {filename} to {newfilename}')
    except Exception as e:
        logging.error(f"Error writing {newfilename}: {e}")
def _interpret(arr):
    """Adapted from libmagic.

       See file-5.14/magic/Magdir/scientific
       available at ftp://ftp.astron.com/pub/file/file-5.14.tar.gz

       Parameters
       ----------
       arr : ndarray
         1-dimensional ndarray of dtype uint8 containing the
         whole RAXIS file

       Returns
       -------
       arr : ndarray
         2-dimensional uint16 array of RAXIS detector data
    """
    # Check the version field to determine the endianess
    version = arr[796:800].view('>u4')[0]
    endian = '>' if version < 20 else '<'

    # Width and height must be cast to at least a uint64 to
    # safely multiply them. (Otherwise default numpy rules
    # result in multiplication modulo 2 ** 32.)
    width = int(arr[768:772].view(endian + 'u4')[0])
    height = int(arr[772:776].view(endian + 'u4')[0])

    logbook.info('Interpreting as {}-endian with dimensions {}x{}'
                 .format('big' if endian == '>' else 'little',
                         width, height))

    diagnostics = """
Diagnostic information:
    length of the raw array (in bytes): {}
    length of the raw array / 4 :       {}
    width:                              {}
    height:                             {}
    len / 4 - (width * height):         {}
""".format(len(arr), len(arr) // 4, width, height,
           len(arr) // 4 - (width * height))

    logbook.debug(diagnostics)

    try:
        reshaped_arr = (arr.view(endian + 'u2')[-(width * height):]
                        .reshape((width, height)))
        
        # Apply bit-shifting for values greater than 32767, excluding the already-handled values
        mask = reshaped_arr >= 32767
        reshaped_arr[mask] = (reshaped_arr[mask] << 5) & 0xFFFF  # Perform the shift safely

        return reshaped_arr
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
