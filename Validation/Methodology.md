# Methodology of Validation

To ensure all 16-bit values are interpreted correctly, the following steps were performed:

1. Using the `Validator.ipynb` script, a sample file named `generated_file.osc` was created. This file includes every possible 16-bit hex value and uses a real sample header for proper simulation.
2. The generated `.osc` file was then converted to `.asc` format twice: once using `OSC_Reader` and once using Rigaku CrystalClear, both outputting the data as 32-bit integers.
3. The `Validator.ipynb` script was subsequently used to compare pixel values from both `.asc` files, ensuring that all pixels align consistently between the two interpretations.

**Finding:** All 16-bit values are processed identically by both `OSC_Reader` and Rigaku CrystalClear.
