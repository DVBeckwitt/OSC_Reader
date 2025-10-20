"""Utility functions for XRD analysis and image processing."""

import json
from pathlib import Path

import fabio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI
from scipy.ndimage import rotate
from scipy.optimize import curve_fit

from .OSC_Reader import read_osc
from . import peak_analysis as pa

try:
    import datashader as ds
    from datashader.mpl_ext import dsshow
    from datashader import transfer_functions as tf
    _HAS_DATASHADER = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_DATASHADER = False


def parse_poni_file(file_path):
    """Parse a .poni calibration file into a dictionary of values."""
    parameters = {}
    with open(file_path, "r") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                key, value = line.split(":", 1)
                value = value.strip()
                try:
                    parameters[key.strip()] = float(value)
                except ValueError:
                    parameters[key.strip()] = value
    return parameters

def _auto_alpha(sample_minus_dark: np.ndarray,
                substrate: np.ndarray) -> float:
    # --- force both arrays to float without copying if already float ---
    sm  = sample_minus_dark.astype(float, copy=False)
    sub = substrate.astype(float,       copy=False)

    mask = sub > 0
    if not np.any(mask):
        return 1.0

    # create a float output array for the division
    ratios = np.divide(
        sm[mask], sub[mask],
        out=np.full_like(sm[mask], np.inf, dtype=float),
        where=sub[mask] != 0
    )

    alpha = ratios[ratios > 0].min(initial=1.0)
    return min(alpha, 1.0)



def display(sample_path, ai, dark_path=None, substrate_path=None, vmax=200,
            title=None, show=True):
    """Plot a dark-subtracted OSC image with the beam center highlighted.

    Parameters
    ----------
    sample_path : str
        Path to the sample ``.osc`` file.
    dark_path : str
        Path to the dark ``.osc`` file.
    ai : :class:`pyFAI.AzimuthalIntegrator`
        Integrator containing calibration parameters.
    vmax : float, optional
        Upper bound for the colormap. Defaults to ``200``.
    title : str, optional
        Title for the plot. Defaults to the sample file name.
    show : bool, optional
        When ``True`` the plot is displayed. Set to ``False`` to skip
        calling :func:`matplotlib.pyplot.show`. Defaults to ``True``.

    Returns
    -------
    numpy.ndarray
        The dark-subtracted image array.
    """
    sample_image = read_osc(sample_path)
    sample_image = sample_image.astype(float, copy=False)

    if dark_path is not None:
        dark_image  = read_osc(dark_path).astype(float, copy=False)
        data        = sample_image - dark_image
    else:
        data = sample_image

    if substrate_path is not None:
        substrate_image = read_osc(substrate_path).astype(float, copy=False)

        # ---------- new bits ----------
        alpha = _auto_alpha(data, substrate_image)
        data  -= alpha * substrate_image


    x_center = ai.poni2 / ai.pixel1
    y_center = ai.poni1 / ai.pixel2

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap="turbo", vmin=0, vmax=vmax, origin="lower")
    ax.scatter(x_center, y_center, c="red", s=100, marker="o", label="PONI center")
    if title is None:
        title = Path(sample_path).stem
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend()
    plt.colorbar(im, ax=ax)
    if show:
        plt.show()
    return data


def setup_azimuthal_integrator(file_path, base_distance=0.075):
    """Return a :class:`pyFAI.AzimuthalIntegrator` using values from a .poni file."""
    params = parse_poni_file(file_path)
    detector_config = json.loads(params.get("Detector_config", "{}"))
    pixel1 = float(detector_config.get("pixel1", 1e-4))
    pixel2 = float(detector_config.get("pixel2", 1e-4))

    ai = pyFAI.AzimuthalIntegrator(
        dist=base_distance,
        poni1=params.get("Poni1", 0.0),
        poni2=params.get("Poni2", 0.0),
        pixel1=pixel1,
        pixel2=pixel2,
        rot1=params.get("Rot1", 0.0),
        rot2=params.get("Rot2", 0.0),
        rot3=params.get("Rot3", 0.0),
        wavelength=params.get("Wavelength", 0.0),
    )
    return ai


def xrd_peaks(file_path, peak_type, intensity_threshold=0, L_limit=None, plot=None):
    """Load an XRD table and optionally plot filtered peaks."""
    # Load data specifying column names and handling whitespace
    columns = ["h", "k", "l", "d", "F(real)", "F(imag)", "|F|", "2θ", "I", "M"]
    data = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=1)

    # Convert 'I' and '2θ' columns to numeric, coercing errors
    data['I'] = pd.to_numeric(data['I'], errors='coerce')
    data['2θ'] = pd.to_numeric(data['2θ'], errors='coerce')
    data['I'] = pd.to_numeric(data['|F|'], errors='coerce')

    # same for h k l 
# Convert 'h', 'k', 'l' columns to integers, handling NaNs by replacing them with 0
    data['h'] = pd.to_numeric(data['h'], errors='coerce').fillna(0).astype(int)
    data['k'] = pd.to_numeric(data['k'], errors='coerce').fillna(0).astype(int)
    data['l'] = pd.to_numeric(data['l'], errors='coerce').fillna(0).astype(int)

    # Drop rows with NaN values in 'I' column
    data.dropna(subset=['I'], inplace=True)
    data.dropna(subset=['|F|'], inplace=True)

    # Filter data based on intensity threshold
    filtered_data = data[data['|F|'] > intensity_threshold]

    # Further filter to select specific h and k values
    # Further filter to select specific h and k values
    if peak_type == "Spec":
        filtered_data = filtered_data[(filtered_data['h'] == 0) & (filtered_data['k'] == 0) & 
                            (filtered_data['l'] > 0) & (filtered_data['l'] < 16)]
    if L_limit == 1:
        filtered_data = filtered_data[(filtered_data['l'] >0)]

    # filter anything above 2 theta of 65
    cif_pd = filtered_data[filtered_data['2θ'] < 65]

    # Normalize the intensity
    cif_pd['I'] = (cif_pd['|F|'] / cif_pd['|F|'].max())

    if plot == True:
        # plot them together
        # Plotting
        plt.figure(figsize=(12, 8))
        plt.scatter(cif_pd['2θ'], cif_pd['I'], label='Data', marker = 'x')

        # Adding vertical dashed lines that stop at each point
        for i, row in cif_pd.iterrows():
            plt.vlines(x=row["2θ"], ymin=0, ymax=row["I"], colors='gray', linestyles='--', linewidth=0.5)
            plt.text(row["2θ"], row["I"], f'({row["h"]} {row["k"]} {row["l"]})', fontsize=9, ha='right', rotation=45)

        plt.title('XRD Peaks with Precise Vertical Lines')
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Normalized Intensity (I)')
        plt.show() 

    return cif_pd

def rotate_image(image, angle, center):
    """
    Rotate the image by the specified angle around the given center.
    
    Parameters:
    - image: 2D numpy array, the image to be rotated
    - angle: float, the rotation angle in degrees (positive for counterclockwise)
    - center: tuple of (x, y), the center of rotation
    
    Returns:
    - final_image: 2D numpy array, the rotated image
    """
    # Calculate the padding needed
    h, w = image.shape
    diagonal = int(np.ceil(np.sqrt(h**2 + w**2)))
    padding = (diagonal - min(h, w)) // 2
    
    # Pad the image
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    
    # Calculate the new center after padding
    new_center = (center[0] + padding, center[1] + padding)
    
    # Rotate the padded image
    rotated_padded_image = rotate(padded_image, angle, reshape=False, order=1, mode='constant', cval=0)
    
    # Calculate the cropping indices
    start_x = (rotated_padded_image.shape[1] - w) // 2
    start_y = (rotated_padded_image.shape[0] - h) // 2
    end_x = start_x + w
    end_y = start_y + h
    
    # Crop the image back to the original size
    final_image = rotated_padded_image[start_y:end_y, start_x:end_x]
    
    return final_image
def get_radial(d, c, file_name):
    """Return radial and angular coordinates for each pixel."""
    # Constants
    pixel_size = 0.1  # in mm
    
    # flip the file_name along the x-axis
    file_name = np.flipud(file_name)
    
    # Compute the coordinates of each pixel relative to the center
    y_coords, x_coords = np.indices(file_name.shape)
    x_coords = (x_coords - c[1]) * pixel_size
    y_coords = (y_coords - c[0]) * pixel_size

    # Compute r, theta, phi for each pixel
    r = np.sqrt(x_coords**2 + y_coords**2)
    theta = np.arctan(r / d) * 180 / np.pi
    phi = np.arctan2(y_coords, x_coords) * 180 / np.pi
    
    # Adjust phi values to be in the range [0, 360)
    phi = np.where(phi < 0, phi + 360, phi)
    
    return theta, phi

def load_data(file_name):
    """Load an image from .npy or .asc format."""
    num_pixels_x, num_pixels_y = 3000, 3000

    if file_name.endswith('.npy'):
        image = np.load(file_name)
        
        return np.flipud(image) 
    elif file_name.endswith('.asc'):
        return np.loadtxt(file_name, skiprows=6).reshape(num_pixels_y, num_pixels_x)

    else:
        raise ValueError("Unsupported file format")
    
def image(d, c, gamma, file_name, orientation=None, rotate_angle=None, Gamma=0):
    """Correct image distortion and optionally rotate."""
    # Constants
    num_pixels_x, num_pixels_y = 3000, 3000
    pixel_size = 0.1  # in mm
    
    # Load the pixel data

    # Load the pixel data
    pixel_data = load_data(file_name)
    
    # Convert degrees to radians
    gamma_rad = np.deg2rad(gamma)
    Gamma_rad = np.deg2rad(Gamma)
    
    # Compute the coordinates of each pixel relative to the center
    y_coords, x_coords = np.indices(pixel_data.shape)
    x_coords = (x_coords - c[1]) * pixel_size
    y_coords = (y_coords - c[0]) * pixel_size

    # Compute r, theta, phi for each pixel
    r = np.sqrt(x_coords**2 + y_coords**2)
    theta = np.arctan(r / d)
    phi = np.arctan2(y_coords, x_coords)

    # Calculate C, x', and y'
    C = d * np.sin(theta) / (np.cos(gamma_rad) * np.cos(theta)*np.cos(Gamma_rad) +  np.sin(theta)  * (np.cos(gamma_rad)*np.sin(phi) * np.sin(Gamma_rad) -np.sin(gamma_rad) * np.cos(phi)))
    y_prime = (C * (np.sin(phi) * np.cos(gamma_rad) - np.cos(phi)* np.sin(gamma_rad) *np.sin(Gamma_rad) ) / pixel_size + c[0]).astype(int)
    x_prime = (C *np.cos(gamma_rad) * np.cos(phi) / pixel_size + c[1]).astype(int)

    # Create an empty array for the corrected image
    corrected_image = np.zeros_like(pixel_data)
    valid_indices = (y_prime >= 0) & (y_prime < num_pixels_y) & (x_prime >= 0) & (x_prime < num_pixels_x)
    corrected_image[y_prime[valid_indices], x_prime[valid_indices]] = pixel_data[valid_indices]

    # Rotate the image if required
    if rotate_angle is not None:
        corrected_image = rotate_image(corrected_image, rotate_angle, (c[1], c[0]))  # Negative for counter-clockwise rotation

    # Trim and rotate the image based on type
    if orientation == True:
        final_image = corrected_image
    else:
        rotated_image = np.rot90(corrected_image, k=-1)    
        final_image = np.fliplr(rotated_image)
        final_image = final_image[:(c[0]), :]
    
    return final_image



def integrate_spec(final_image, d, c, th_range, phi_range, orientation=None, integration_method = None):
    """Integrate intensity over selected theta and phi ranges."""
        # Constants
    num_pixels_x, num_pixels_y = 3000, 3000
    pixel_size = 0.1  # in mm
    
    y_coords, x_coords = np.indices(final_image.shape)

    # Compute the coordinates of each pixel relative to the center
    if orientation == "T":
        x_coords = (x_coords - c[0]) * pixel_size
        y_coords = (c[1] - y_coords ) * pixel_size
    else:
        x_coords = (x_coords - c[1]) * pixel_size
        y_coords = (c[0] - y_coords ) * pixel_size

    # Compute r, theta, phi for each pixel
    r = np.sqrt(x_coords**2 + y_coords**2)
    theta = np.arctan(r / d)
    phi = np.arctan2(y_coords,x_coords)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    # Filter by theta and phi ranges

    th_min = np.deg2rad(th_range[0])
    th_max = np.deg2rad(th_range[1])

    phi_min = np.deg2rad(phi_range[0])
    phi_max = np.deg2rad(phi_range[1])
    
    # Flatten theta and phi for filtering
    flat_theta = theta.flatten()
    flat_phi = phi.flatten()
    
    # Filter the DataFrame
    filtered_df = pd.DataFrame({
        'x': x_coords.flatten(),
        'y': y_coords.flatten(),
        'theta': flat_theta,
        'phi': flat_phi,
        'intensity_pre': final_image.flatten()
    })
    
    condition = (flat_theta >= th_min) & (flat_theta <= th_max) & (flat_phi >= phi_min) & (flat_phi <= phi_max)
    filtered_df = filtered_df[condition]

    # Assuming th_min, th_max, d, and pixel_size are defined elsewhere
    # Create more dynamic bins based on actual pixel geometry
    def calculate_pixel_weight(th, d, pixel_size):
        # Example function to calculate weights (this is a placeholder)
        # A proper implementation would involve geometric intersection calculations
        return pixel_size / (d * np.cos(th))  # Simplified assumption

    # Recalculate pixels to reflect weighted distribution
    pixels = (np.tan(th_max) - np.tan(th_min)) * d / pixel_size
    pixels = np.round(pixels).astype(int)

    # Define the theta array and calculate weights
    th_array = np.linspace(th_min, th_max, pixels)
    weights = calculate_pixel_weight(th_array, d, pixel_size)

    # Calculate weighted bin edges
    weighted_bin_widths = np.cumsum(weights)
    weighted_bin_widths = weighted_bin_widths / weighted_bin_widths[-1] * (th_max - th_min)
    bin_edges = np.concatenate(([th_min], th_min + weighted_bin_widths))

    # Adjust DataFrame processing
    filtered_df['weight'] = filtered_df['theta'].apply(lambda th: calculate_pixel_weight(th, d, pixel_size))
    filtered_df['weighted_intensity'] = filtered_df['intensity_pre'] * filtered_df['weight']

    # Use cut to assign each theta to a bin
    filtered_df['bin'] = pd.cut(filtered_df['theta'], bins=bin_edges, labels=np.arange(len(bin_edges)-1))

    # Aggregate the data with weights
    summed_df = filtered_df.groupby('bin').agg(
        intensity_sum=pd.NamedAgg(column='weighted_intensity', aggfunc='sum'),
        theta_mean=pd.NamedAgg(column='theta', aggfunc='mean')
    )
    return filtered_df, summed_df['intensity_sum'], summed_df['theta_mean'] * 180/np.pi

def plot_qz_vs_qr(
    thetai,
    ai,
    data,
    a1,
    c1,
    hkl_list=None,
    qr_adjust=0.15,
    qz_min=0.1,
    qz_max=3.5,
    plot=True,
):
    """Calculate ``qr`` and ``qz`` coordinates from an OSC image.

    Parameters
    ----------
    thetai : float
        Incident angle in degrees.
    ai : :class:`pyFAI.AzimuthalIntegrator`
        Integrator describing the detector geometry.
    data : array-like
        2D intensity image (OSC).
    a1, c1 : float
        Lattice parameters ``a`` and ``c``.
    hkl_list : sequence of ``(h, k)`` tuples, optional
        Miller indices for the in-plane Bragg rods. When ``None`` the
        historical default of ``[(1, 0), (1, 1), (2, 0)]`` is used.
    qr_adjust : float, optional
        Half-width of the ``qr`` integration box.
    qz_min, qz_max : float, optional
        Vertical limits of the integration boxes.

    plot : bool, optional
        When ``True`` the computed ``qr``/``qz`` map is displayed using
        :func:`plot_q`. Set to ``False`` to skip plotting. Defaults to
        ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``qr`` (``x``), ``qz`` (``y``) and ``intensity``.
    dict
        Dictionary with box definitions for each Bragg rod.
    """

    if hkl_list is None and qr_adjust == 0.15 and qz_min == 0.1 and qz_max == 3.5:
        # Backwards compatible defaults
        boxes = {
            "L(10L)": {
                "qr_min": -q(1, 0, 0, a1, c1) - 0.15,
                "qr_max": -q(1, 0, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.5,
                "qrm2": -2,
                "qrM2": -1.95,
                "color": "purple",
                "linestyle": "-.",
            },
            "R(10L)": {
                "qr_min": q(1, 0, 0, a1, c1) - 0.15,
                "qr_max": q(1, 0, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.5,
                "qrm2": 1.95,
                "qrM2": 2,
                "color": "blue",
                "linestyle": "-",
            },
            "L(20L)": {
                "qr_min": -q(1, 1, 0, a1, c1) - 0.15,
                "qr_max": -q(1, 1, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.3,
                "qrm2": -3,
                "qrM2": -3.05,
                "color": "green",
                "linestyle": ":",
            },
            "R(20L)": {
                "qr_min": q(1, 1, 0, a1, c1) - 0.15,
                "qr_max": q(1, 1, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.3,
                "qrm2": 3.35,
                "qrM2": 3.4,
                "color": "orange",
                "linestyle": "--",
            },
            "L(2,0,L)": {
                "qr_min": -q(2, 0, 0, a1, c1) - 0.15,
                "qr_max": -q(2, 0, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.3,
                "qrm2": -4,
                "qrM2": -3.95,
                "color": "brown",
                "linestyle": "-.",
            },
            "R(2,0,L)": {
                "qr_min": q(2, 0, 0, a1, c1) - 0.15,
                "qr_max": q(2, 0, 0, a1, c1) + 0.15,
                "qz_min": 0.1,
                "qz_max": 3.3,
                "qrm2": 3.95,
                "qrM2": 4,
                "color": "brown",
                "linestyle": "-.",
            },
        }
    else:
        # Generic generation of boxes for user-specified HK0 rods
        if hkl_list is None:
            hkl_list = [(1, 0), (1, 1), (2, 0)]

        boxes = {}
        colors = [
            "purple",
            "blue",
            "green",
            "orange",
            "brown",
            "red",
            "cyan",
            "magenta",
            "black",
        ]
        for idx, (h, k, *_) in enumerate(hkl_list):
            color = colors[idx % len(colors)]
            label = f"{h}{k}L"
            q_val = q(h, k, 0, a1, c1)

            boxes[f"L({label})"] = {
                "qr_min": -q_val - qr_adjust,
                "qr_max": -q_val + qr_adjust,
                "qz_min": qz_min,
                "qz_max": qz_max,
                "color": color,
                "linestyle": "-.",
            }
            boxes[f"R({label})"] = {
                "qr_min": q_val - qr_adjust,
                "qr_max": q_val + qr_adjust,
                "qz_min": qz_min,
                "qz_max": qz_max,
                "color": color,
                "linestyle": "-",
            }

    regions = [
        [7, 12, -20, 20, "003"],
        [29, 32, 54, 66, "119"]
    ]
    intensity, phi, theta, region_data = pa.process_data(
        ai,
        data,
        regions,
        do_fitting=False,
        show_plots=False,
    )

    # ``phi`` and ``theta`` are returned in degrees as 1D arrays representing the
    # azimuthal and radial axes of the integrated image.  Create 2D grids in
    # radians so they broadcast correctly with ``intensity`` when calculating
    # reciprocal space coordinates.
    phi = np.deg2rad(phi)[:, None]
    theta = np.deg2rad(theta)[None, :]

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    k = 2 * np.pi / 1.5406
    thetai_rad = np.deg2rad(thetai)

    qx = sin_theta * sin_phi * k
    st_cp = sin_theta * cos_phi
    qy = (np.cos(thetai_rad) * (cos_theta - 1) + np.sin(thetai_rad) * st_cp) * k
    qz = (-np.sin(thetai_rad) * (cos_theta - 1) + np.cos(thetai_rad) * st_cp) * k

    # Determine the sign of ``qr`` using the broadcast ``phi`` array
    qr = np.where(phi >= 0, np.hypot(qx, qy), -np.hypot(qx, qy))

    df = pd.DataFrame({"x": qr.ravel(), "y": qz.ravel(), "intensity": intensity.ravel()})

    if plot:
        vm = df["intensity"].max()
        plot_q(df, boxes, "qz_vs_qr", vm)

    return df, boxes

# A function that returns the Qz or Qr for any given reciprocal rod of interest in a hexagonal crystal
def q(h, k, l, a, c):
    d = (4/3 * ( h**2 + h*k + k**2 ) / a**2)**(-1/2)

    return 2 * np.pi / d

def plot_q(s, boxes, plot_title, vm, use_datashader=True, point_size=10):
    """Plot qz versus qr using all data points.

    Parameters
    ----------
    s : pandas.DataFrame
        DataFrame containing ``x``, ``y`` and ``intensity`` columns.
    boxes : dict
        Dictionary describing regions of interest to draw.
    plot_title : str
        Title (and output filename) for the plot.
    vm : float
        Maximum value for colormap scaling.
    use_datashader : bool, optional
        Use datashader for rendering if available. This is much faster for
        large datasets.
    point_size : int, optional
        Size of the points in pixels. When datashader is available this
        controls the spreading radius. Defaults to ``10``.

    """

    fig, ax = plt.subplots(figsize=(10, 10))

    qr = s["x"].to_numpy()
    qz = s["y"].to_numpy()
    intensity = s["intensity"].to_numpy()

    if use_datashader and _HAS_DATASHADER:
        # Render using datashader for speed and spread points to avoid gaps
        img = dsshow(
            s,
            ds.Point("x", "y"),
            ds.mean("intensity"),
            plot_width=1000,
            plot_height=1000,
            x_range=(qr.min(), qr.max()),
            y_range=(qz.min(), qz.max()),
            ax=ax,
            vmin=0,
            vmax=vm,
            cmap="turbo",
            shade_hook=lambda im: tf.spread(im, px=point_size),

        )
        if hasattr(img, "set_interpolation"):
            img.set_interpolation("bilinear")

    else:
        # Use a scatter plot with large, semi-transparent points
        ax.scatter(
            qr,
            qz,
            c=intensity,
            s=point_size ** 2,
            cmap="turbo",
            vmin=0,
            vmax=vm,
            alpha=0.6,
            edgecolors="none",
        )

    # Drawing rectangles and adding labels
    for box_name, box in boxes.items():
        rect = plt.Rectangle(
            (box["qr_min"], box["qz_min"]),
            box["qr_max"] - box["qr_min"],
            box["qz_max"] - box["qz_min"],
            edgecolor=box["color"],
            facecolor="none",
            linestyle=box["linestyle"],
            lw=2,
        )
        ax.add_patch(rect)
        ax.text(
            box["qr_min"],
            box["qz_max"],
            box_name,
            color="red",
            fontsize=12,
            verticalalignment="top",
        )

    # Labels, title, and formatting
    ax.set_xlabel(r"qr (Å$^{-1}$)", fontsize=14)
    ax.set_ylabel(r"qz (Å$^{-1}$)", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(plot_title, fontsize=16)
    plt.savefig(f"{plot_title}.png", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.show()


def plot_qr(s, boxes, plot_title, vm, use_datashader=True, point_size=10):
    """Plot qz versus qr using all data points with optional box styling.

    Parameters
    ----------
    s : pandas.DataFrame
        DataFrame containing ``x``, ``y`` and ``intensity`` columns.
    boxes : dict
        Dictionary describing regions of interest to draw. Each entry must
        contain ``qr_min``, ``qr_max``, ``qz_min`` and ``qz_max``. Optional
        ``color`` and ``linestyle`` control the appearance of the rectangle.
        When omitted a default color cycle and solid line style are used.
    plot_title : str
        Title (and output filename) for the plot.
    vm : float
        Maximum value for colormap scaling.
    use_datashader : bool, optional
        Use datashader for rendering if available. This is much faster for
        large datasets.
    point_size : int, optional
        Size of the points in pixels. When datashader is available this
        controls the spreading radius. Defaults to ``10``.

    """

    fig, ax = plt.subplots(figsize=(10, 10))

    qr = s["x"]
    qz = s["y"]
    intensity = s["intensity"]

    if use_datashader and _HAS_DATASHADER:
        img = dsshow(
            s,
            ds.Point("x", "y"),
            ds.mean("intensity"),
            plot_width=1000,
            plot_height=1000,
            x_range=(qr.min(), qr.max()),
            y_range=(qz.min(), qz.max()),
            ax=ax,
            vmin=0,
            vmax=vm,
            cmap="turbo",
            shade_hook=lambda im: tf.spread(im, px=point_size),

        )
        if hasattr(img, "set_interpolation"):
            img.set_interpolation("bilinear")
    else:
        ax.scatter(
            qr,
            qz,
            c=intensity,
            s=point_size ** 2,
            cmap="turbo",
            vmin=0,
            vmax=vm,
            alpha=0.6,
            edgecolors="none",
        )

    color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    for box_name, box in boxes.items():
        color = box.get("color", next(color_cycle, "C0"))
        linestyle = box.get("linestyle", "-")
        rect = plt.Rectangle(
            (box["qr_min"], box["qz_min"]),
            box["qr_max"] - box["qr_min"],
            box["qz_max"] - box["qz_min"],
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            lw=2,
        )
        ax.add_patch(rect)
        ax.text(
            box["qr_min"],
            box["qz_max"],
            box_name,
            color="red",
            fontsize=12,
            verticalalignment="top",
        )

    ax.set_xlabel(r"qr (Å$^{-1}$)", fontsize=14)
    ax.set_ylabel(r"qz (Å$^{-1}$)", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(plot_title, fontsize=16)
    plt.savefig(f"{plot_title}.png", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.show()


def plot_q_3d(
    df,
    plot_title="q-space 3D",
    interactive=False,
    qr_limits=None,
    qz_limits=None,
    max_intensity=None,
    output_html=None,
    max_points=None,
    surface=False,
):
    """Plot ``qr`` versus ``qz`` in 3D with intensity as the z-axis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``x`` (qr), ``y`` (qz) and ``intensity`` columns.
    plot_title : str, optional
        Title for the plot. Defaults to ``"q-space 3D"``.

    interactive : bool, optional
        When ``True`` the plot is displayed immediately using Plotly.
        The figure is rendered in the browser (or inline when running
        inside Jupyter) allowing smooth rotation without the expensive
        re-draws required by Matplotlib. ``None`` is returned in this
        mode. Defaults to ``False`` which creates a Matplotlib plot and
        returns the ``Axes`` object without showing the figure.

    qr_limits : tuple of float, optional
        ``(min_qr, max_qr)`` range to display. Data outside this range is
        discarded before plotting.

    qz_limits : tuple of float, optional
        ``(min_qz, max_qz)`` range to display. Data outside this range is
        discarded before plotting.

    max_intensity : float, optional
        Upper limit for the intensity color scale. Intensities above this
        value are clipped. When ``None`` the maximum value in ``df`` is used.

    output_html : str, optional
        Path to write a standalone HTML file containing the interactive
        Plotly figure. Only used when ``interactive`` is ``True``.
        When provided, the file is written in addition to displaying the
        figure so it can be viewed later without rerunning the code.

    max_points : int, optional
        Maximum number of points to plot when ``interactive`` is ``True``.
        When the data contains more rows than ``max_points`` only a subset
        is displayed to keep the browser responsive. The subset is chosen
        by stepping through the DataFrame at regular intervals. ``None``
        uses all points.

    surface : bool, optional
        When ``True`` a surface mesh is drawn instead of individual points.
        The mesh is generated from Delaunay triangulation. Defaults to ``False``
        which displays a scatter plot.

    Returns
    -------
    matplotlib.axes.Axes or None
        The plot object containing the visualization when ``interactive`` is
        ``False``. ``None`` is returned in interactive mode. The minimum of the
        color scale is always fixed at ``0``.
    """

    # Optionally restrict the data range. Previously the function always limited
    # ``qr`` to ``[-5, 5]`` and ``qz`` to ``[0, 6]`` which resulted in empty
    # plots when the data fell outside those bounds. The limits can now be
    # provided explicitly and default to showing the full data range.
    if qr_limits is not None:
        df = df[(df["x"] >= qr_limits[0]) & (df["x"] <= qr_limits[1])]
    if qz_limits is not None:
        df = df[(df["y"] >= qz_limits[0]) & (df["y"] <= qz_limits[1])]

    if df.empty:
        raise ValueError("No points fall within the specified q-range limits")

    x_range = qr_limits if qr_limits is not None else [df["x"].min(), df["x"].max()]
    y_range = qz_limits if qz_limits is not None else [df["y"].min(), df["y"].max()]
    if max_intensity is None:
        max_intensity = df["intensity"].max()
    z_range = [0, max_intensity]
    clipped_intensity = np.clip(df["intensity"], 0, max_intensity)

    if max_points is not None and len(df) > max_points:
        step = int(np.ceil(len(df) / max_points))
        df = df.iloc[::step]
        clipped_intensity = np.clip(df["intensity"], 0, max_intensity)

    if interactive:
        import plotly.graph_objects as go

        if surface:
            from scipy.spatial import Delaunay

            tri = Delaunay(df[["x", "y"]].to_numpy())
            mesh = go.Mesh3d(
                x=df["x"].to_numpy(),
                y=df["y"].to_numpy(),
                z=clipped_intensity,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                intensity=clipped_intensity,
                colorscale="Turbo",
                cmin=0,
                cmax=max_intensity,
                showscale=True,
            )
            fig = go.Figure(data=mesh)
        else:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=df["x"].to_numpy(),
                    y=df["y"].to_numpy(),
                    z=clipped_intensity,
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=clipped_intensity,
                        colorscale="Turbo",
                        cmin=0,
                        cmax=max_intensity,
                    ),
                )
            )
        fig.update_layout(
            title=plot_title,
            scene=dict(
                xaxis_title="qr (Å⁻¹)",
                yaxis_title="qz (Å⁻¹)",
                zaxis_title="Intensity",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
            ),
        )
        if output_html is not None:
            # Include plotly.js directly so the file can be viewed offline
            fig.write_html(
                output_html,
                include_plotlyjs=True,
                auto_open=False,
            )
        fig.show(renderer="browser")

        return None
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D plot

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        x = df["x"]
        y = df["y"]
        z = clipped_intensity

        if surface:
            surf = ax.plot_trisurf(
                x,
                y,
                z,
                cmap="turbo",
                linewidth=0.2,
                antialiased=True,
                vmin=0,
                vmax=max_intensity,
            )
        else:
            sc = ax.scatter(x, y, z, c=z, cmap="turbo", s=1,
                            vmin=0, vmax=max_intensity)
        ax.set_xlabel(r"qr (Å$^{-1}$)")
        ax.set_ylabel(r"qz (Å$^{-1}$)")
        ax.set_zlabel("Intensity")
        ax.set_title(plot_title)
        if surface:
            fig.colorbar(surf, ax=ax, label="Intensity")
        else:
            fig.colorbar(sc, ax=ax, label="Intensity")
        ax.set_zlim(z_range)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        return ax


def integrate_qr_vs_qz(df, boxes, num_bins=200, plot=True):
    """Integrate intensity along qr for each box as a function of qz.

    When ``plot`` is ``True`` each box is displayed in its own subplot for
    clearer comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with ``x`` (qr), ``y`` (qz) and ``intensity`` columns.
    boxes : dict
        Dictionary describing regions of interest with ``qr_min``, ``qr_max``,
        ``qz_min`` and ``qz_max`` entries.
    num_bins : int, optional
        Number of ``qz`` bins for the integration. Defaults to ``200``.
    plot : bool, optional
        If ``True``, plot the integrated intensities. Defaults to ``True``.

    Returns
    -------
    dict
        Mapping of ``box_name`` to a DataFrame with ``qz`` and
        ``integrated_intensity`` columns.
    """

    results = {}
    valid_items = [item for item in boxes.items() if item[1].get("qr_max", 0) > 0]

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    inten = df["intensity"].to_numpy()

    axes = None
    if plot:
        fig, axes = plt.subplots(len(valid_items), 1, figsize=(6, 3 * len(valid_items)))
        if len(valid_items) == 1:
            axes = [axes]

    for idx, (name, box) in enumerate(valid_items):
        mask = (
            (x >= box["qr_min"]) & (x <= box["qr_max"]) &
            (y >= box["qz_min"]) & (y <= box["qz_max"])
        )

        if not np.any(mask):
            continue

        bins = np.linspace(box["qz_min"], box["qz_max"], num_bins + 1)
        intensity, edges = np.histogram(y[mask], bins=bins, weights=inten[mask])
        centers = 0.5 * (edges[:-1] + edges[1:])
        results[name] = pd.DataFrame({
            "qz": centers,
            "integrated_intensity": intensity,
        })

        if plot:
            ax = axes[idx]
            ax.plot(centers, intensity, label=name)
            ax.set_xlabel(r"qz (Å$^{-1}$)")
            ax.set_ylabel("Integrated Intensity")
            ax.set_title(name)
            ax.legend()

    if plot and valid_items:
        plt.tight_layout()
        plt.show()

    return results


def _gaussian(x, amplitude, center, sigma):
    """Return a Gaussian line shape."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def integrate_qz_vs_qr(
    df,
    qr_range,
    qz_range=None,
    num_bins=200,
    plot=True,
    fit=True,
):

    """Integrate intensity along qz for one or more qr ranges.

    ``qr_range`` and ``qz_range`` may specify a single region, or ``qr_range``
    can be a dictionary of multiple regions like ``integrate_qr_vs_qz``. When a
    dictionary is provided, ``qz_range`` should be ``None`` and each entry must
    contain ``qr_min``, ``qr_max``, ``qz_min`` and ``qz_max`` values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with ``x`` (qr), ``y`` (qz) and ``intensity`` columns.
    qr_range : tuple or dict
        Either ``(qr_min, qr_max)`` for a single region or a dictionary mapping
        names to regions. Each region dictionary must include ``qr_min``,
        ``qr_max``, ``qz_min`` and ``qz_max``.
    qz_range : tuple, optional
        Required only when ``qr_range`` is a tuple. Defines ``(qz_min, qz_max)``
        for that single region.
    num_bins : int, optional
        Number of ``qr`` bins for the integration. Defaults to ``200``.
    plot : bool, optional
        When ``True`` display the integrated profile. Defaults to ``True``.
    fit : bool, optional
        If ``plot`` is ``True`` and ``fit`` is ``True`` the profile is fit with
        a Gaussian function and only the fit curve is shown. Set to ``False`` to
        plot the raw integrated intensities. Defaults to ``True``.


    Returns
    -------
    dict or pandas.DataFrame
        If multiple regions are supplied a dictionary mapping region names to
        DataFrames with ``qr`` and ``integrated_intensity`` columns is
        returned. For a single region the DataFrame is returned directly.
    """

    if isinstance(qr_range, dict) and qz_range is None:
        boxes = qr_range
    else:
        boxes = {
            "region_0": {
                "qr_min": qr_range[0],
                "qr_max": qr_range[1],
                "qz_min": qz_range[0],
                "qz_max": qz_range[1],
            }
        }

    results = {}
    valid_items = [
        item for item in boxes.items() if item[1].get("qr_max", 0) > item[1].get("qr_min", 0)
    ]

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    inten = df["intensity"].to_numpy()

    axes = None
    if plot:
        fig, axes = plt.subplots(len(valid_items), 1, figsize=(6, 3 * len(valid_items)))
        if len(valid_items) == 1:
            axes = [axes]

    for idx, (name, box) in enumerate(valid_items):
        mask = (
            (x >= box["qr_min"]) & (x <= box["qr_max"])
            & (y >= box["qz_min"]) & (y <= box["qz_max"])
        )

        if not np.any(mask):
            continue

        bins = np.linspace(box["qr_min"], box["qr_max"], num_bins + 1)
        intensity, edges = np.histogram(x[mask], bins=bins, weights=inten[mask])
        centers = 0.5 * (edges[:-1] + edges[1:])
        results[name] = pd.DataFrame({"qr": centers, "integrated_intensity": intensity})

        if plot:
            ax = axes[idx]
            if fit:
                # Fit a Gaussian profile and plot only the fit curve
                p0 = [
                    intensity.max(),
                    centers[np.argmax(intensity)],
                    (centers.max() - centers.min()) / 6.0,
                ]
                try:
                    popt, _ = curve_fit(
                        _gaussian,
                        centers,
                        intensity,
                        p0=p0,
                        maxfev=10000,
                    )
                except Exception:
                    popt = p0
                x_fit = np.linspace(centers.min(), centers.max(), 400)
                y_fit = _gaussian(x_fit, *popt)
                ax.plot(x_fit, y_fit, label=name)
            else:
                ax.plot(centers, intensity, label=name)

            ax.set_xlabel(r"qr (Å$^{-1}$)")
            ax.set_ylabel("Integrated Intensity")
            ax.set_title(name)
            ax.legend()

    if plot and valid_items:
        plt.tight_layout()
        plt.show()

    return next(iter(results.values())) if len(results) == 1 else results



def clean_data(data):
    data = np.nan_to_num(data, nan=np.nanmean(data), posinf=np.nanmax(data[np.isfinite(data)]), neginf=np.nanmin(data[np.isfinite(data)]))
    if (data == 0).any():
        data[data == 0] = np.nanmin(data[data != 0])  # Replace zeros with the smallest non-zero value
    return data

def background_spec(Left_Background, Right_Background, Lth, Rth):
    """Fit a polynomial background from left and right slices."""
    Left_Background = clean_data(Left_Background)
    Right_Background = clean_data(Right_Background)

    valid_left_indices = ~np.isnan(Left_Background)
    valid_right_indices = ~np.isnan(Right_Background)
    Left_Background_clean = Left_Background[valid_left_indices]
    Right_Background_clean = Right_Background[valid_right_indices]
    Lth_clean = Lth[valid_left_indices]
    Rth_clean = Rth[valid_right_indices]

    combined_th = np.concatenate((Lth_clean, Rth_clean))
    combined_background = np.concatenate((Left_Background_clean, Right_Background_clean))

    # Ensure that the filtering is applied correctly
    valid_indices = combined_th > 0
    combined_th = combined_th[valid_indices]
    combined_background = combined_background[valid_indices]

    coefficients = np.polyfit(combined_th, combined_background, 3)  # Polynomial degree
    polynomial = np.poly1d(coefficients)

    return polynomial


# Backwards compatibility
integrate_Spec = integrate_spec
background_Spec = background_spec
plot_Q = plot_q
Get_radial = get_radial
