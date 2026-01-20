"""Peak fitting and analysis utilities."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _load_lmfit_models():
    from lmfit.models import (
        GaussianModel,
        LorentzianModel,
        LinearModel,
        ConstantModel,
        PseudoVoigtModel,
        PolynomialModel,
    )
    return {
        "GaussianModel": GaussianModel,
        "LorentzianModel": LorentzianModel,
        "LinearModel": LinearModel,
        "ConstantModel": ConstantModel,
        "PseudoVoigtModel": PseudoVoigtModel,
        "PolynomialModel": PolynomialModel,
    }


def _load_lmfit_lineshapes():
    from lmfit.lineshapes import gaussian, lorentzian
    return gaussian, lorentzian

##############################################################################
# 1) Helper functions
##############################################################################
def estimate_initial_parameters(x, y):
    """
    Estimate initial parameters: amplitude, center, and sigma.
    amplitude ~ max of y,
    center ~ x value at peak,
    sigma ~ 1/10 of the x-range as a naive guess.
    """
    amp = y.max()
    cen = x[np.argmax(y)]
    sigma = (x.max() - x.min()) / 10
    return amp, cen, sigma

def compute_fwhm(x, y):
    """
    Compute the full width at half maximum (FWHM) of a discrete curve y(x).
    Uses the difference between the first and last x where y >= half of the peak max.
    """
    peak_val = np.max(y)
    half_max = 0.5 * peak_val
    indices = np.where(y >= half_max)[0]
    if indices.size < 2:
        return np.nan
    return x[indices[-1]] - x[indices[0]]

def derivative_wrt_sigma(sigma, frac, A, B, X):
    """
    Compute the partial derivative of FWHM with respect to sigma.
    """
    if X <= 0:
        return 0.0
    coeff = (1 - frac) * A**5 + frac * B**5
    return (coeff * sigma**4) / (X**(4/5))

def derivative_wrt_fraction(sigma, frac, A, B, X):
    """
    Compute the partial derivative of FWHM with respect to fraction.
    """
    if X <= 0:
        return 0.0
    top = (B * sigma)**5 - (A * sigma)**5
    return top / (5 * X**(4/5))

def compute_fwhm_and_error_from_cov(result, prefix='pv_'):
    """
    Compute the pseudo-Voigt FWHM and its uncertainty using partial derivatives
    and the covariance of (sigma, fraction). Uses a diagonal approximation if needed.
    """
    # lmfit's PseudoVoigtModel defines ``sigma`` such that the FWHM of the
    # Gaussian and Lorentzian components (as well as of the resulting
    # pseudo-Voigt profile) is simply ``2*sigma``.  Using the more familiar
    # 2.3548 factor for a standard Gaussian would therefore give an incorrect
    # value here.  Both components share the same width in this parameterisation
    # so the factors are identical.
    A = 2.0  # Gaussian FWHM factor from sigma
    B = 2.0  # Lorentzian FWHM factor from sigma

    p_sigma = result.params.get(prefix + 'sigma', None)
    p_frac  = result.params.get(prefix + 'fraction', None)
    if (p_sigma is None) or (p_frac is None):
        return np.nan, np.nan

    sigma_val = p_sigma.value
    frac_val = p_frac.value

    X = (1 - frac_val) * (A * sigma_val)**5 + frac_val * (B * sigma_val)**5
    fwhm = X**0.2

    if result.covar is None:
        dsigma = p_sigma.stderr if p_sigma.stderr else 0.0
        dfrac = p_frac.stderr if p_frac.stderr else 0.0
        df_dsigma = derivative_wrt_sigma(sigma_val, frac_val, A, B, X)
        df_dfrac  = derivative_wrt_fraction(sigma_val, frac_val, A, B, X)
        fwhm_err = np.sqrt((df_dsigma * dsigma)**2 + (df_dfrac * dfrac)**2)
        return fwhm, fwhm_err

    parnames = list(result.var_names)
    try:
        i_sigma = parnames.index(prefix + 'sigma')
        i_frac  = parnames.index(prefix + 'fraction')
    except ValueError:
        dsigma = p_sigma.stderr if p_sigma.stderr else 0.0
        dfrac = p_frac.stderr if p_frac.stderr else 0.0
        df_dsigma = derivative_wrt_sigma(sigma_val, frac_val, A, B, X)
        df_dfrac  = derivative_wrt_fraction(sigma_val, frac_val, A, B, X)
        fwhm_err = np.sqrt((df_dsigma * dsigma)**2 + (df_dfrac * dfrac)**2)
        return fwhm, np.nan

    df_dsigma = derivative_wrt_sigma(sigma_val, frac_val, A, B, X)
    df_dfrac  = derivative_wrt_fraction(sigma_val, frac_val, A, B, X)
    grad = np.array([df_dsigma, df_dfrac])
    cov_matrix = np.array([
        [result.covar[i_sigma, i_sigma], result.covar[i_sigma, i_frac]],
        [result.covar[i_frac, i_sigma],  result.covar[i_frac, i_frac]]
    ])
    fwhm_var = grad @ cov_matrix @ grad
    fwhm_err = np.sqrt(fwhm_var) if fwhm_var >= 0 else np.nan
    return fwhm, fwhm_err

##############################################################################
# 2) Fit function using Gaussian + Lorentzian + Linear background
##############################################################################
def perform_fit(x, y, lower_bound, upper_bound, equal_weight_fit=False):
    """
    Build a composite model (Gaussian + Lorentzian + Linear) using ``lmfit``.
    The Gaussian and Lorentzian widths are independent, but their centers are
    constrained to be the same. Optional weighting can be applied.
    """
    if y.size == 0 or not np.any(y > 0):
        return None

    models = _load_lmfit_models()
    GaussianModel = models["GaussianModel"]
    LorentzianModel = models["LorentzianModel"]
    LinearModel = models["LinearModel"]

    amp, cen, sigma = estimate_initial_parameters(x, y)

    gauss = GaussianModel(prefix='g_')
    lor   = LorentzianModel(prefix='l_')
    lin   = LinearModel(prefix='lin_')
    composite = gauss + lor + lin

    params = composite.make_params()
    params['g_amplitude'].set(value=amp * 0.5, min=0)
    params['g_center'].set(value=cen, min=lower_bound, max=upper_bound)
    params['g_sigma'].set(value=sigma, min=1e-6)

    params['l_amplitude'].set(value=amp * 0.5, min=0)
    params['l_center'].set(expr='g_center')
    params['l_sigma'].set(value=sigma, min=1e-6)

    params['lin_slope'].set(value=0)
    params['lin_intercept'].set(value=np.median(y), min=np.min(y), max=np.max(y))

    fit_kws = {}
    if equal_weight_fit:
        safe_y = np.where(y <= 0, 1e-6, y)
        fit_kws["weights"] = 1.0 / safe_y

    try:
        result = composite.fit(y, params, x=x, **fit_kws)
        return result
    except Exception as e:
        print("Fit error:", e)
        return None

##############################################################################
# 3) Plotting the Fit with Explicit Gaussian and Lorentzian Components
##############################################################################
def plot_fit_with_components(ax_obj, x_data, y_data, fit_result, x_label, use_log=False):
    """
    Plot raw data, total composite fit, and explicitly computed Gaussian and
    Lorentzian contributions. The underlying model is a sum of a Gaussian and a
    Lorentzian peak that share the same center but have independent widths. The
    linear background (from the additional linear model) is added back.  The
    plot is annotated with the percent Lorentzian and FWHM values.
    """
    ax_obj.plot(x_data, y_data, 'ko', label='Raw Data', markersize=4, zorder=1)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
    y_fit_total = fit_result.eval(x=x_smooth)

    gaussian, lorentzian = _load_lmfit_lineshapes()
    
    # Extract parameters from the fit result
    g_amp = fit_result.params['g_amplitude'].value
    g_cen = fit_result.params['g_center'].value
    g_sigma = fit_result.params['g_sigma'].value

    l_amp = fit_result.params['l_amplitude'].value
    l_sigma = fit_result.params['l_sigma'].value

    # Compute Gaussian and Lorentzian contributions using lmfit lineshapes
    gaussian_comp = gaussian(x_smooth, amplitude=g_amp, center=g_cen, sigma=g_sigma)
    lorentzian_comp = lorentzian(x_smooth, amplitude=l_amp, center=g_cen, sigma=l_sigma)
    
    # Evaluate linear background from the composite fit.
    # Depending on the lmfit version the key may be 'lin' or 'lin_'.
    comps = fit_result.eval_components(x=x_smooth)
    lin_bkg = comps.get('lin', comps.get('lin_', np.zeros_like(x_smooth)))
    
    # Plot the curves
    ax_obj.plot(x_smooth, y_fit_total, 'r-', lw=2, alpha=0.8, label='Total Fit', zorder=2)
    ax_obj.plot(x_smooth, gaussian_comp + lin_bkg, 'm-', lw=2, label='Gaussian + bkg', zorder=3)
    ax_obj.plot(x_smooth, lorentzian_comp + lin_bkg, 'c-', lw=2, label='Lorentzian + bkg', zorder=3)
    ax_obj.plot(x_smooth, lin_bkg, 'g--', lw=2, label='Linear bkg', zorder=2)
    if use_log:
        ax_obj.set_yscale('log')
    ax_obj.set_xlabel(x_label)
    ax_obj.set_ylabel('Scaled Intensity')
    ax_obj.legend(loc='best')
    
    # Compute FWHM values for each component
    FWHM_gauss = 2.3548200 * g_sigma
    FWHM_lor = 2.0 * l_sigma

    # FWHM of the combined peak (Gaussian + Lorentzian) without background
    overall_fwhm = compute_fwhm(x_smooth, gaussian_comp + lorentzian_comp)
    overall_fwhm_err = np.nan

    # Calculate percent Lorentzian based on amplitudes
    percent_lor = 100.0 * l_amp / (g_amp + l_amp) if (g_amp + l_amp) != 0 else np.nan
    
    # Create annotation text.
    annotation_lines = [
         f"Percent Lorentzian: {percent_lor:.1f}%",
         f"Overall FWHM: {overall_fwhm:.3f}" + (f" ± {overall_fwhm_err:.3f}" if not np.isnan(overall_fwhm_err) else ""),
         f"FWHM (Gaussian): {FWHM_gauss:.3f}",
         f"FWHM (Lorentzian): {FWHM_lor:.3f}"
    ]
    annotation_text = "\n".join(annotation_lines)
    
    # Add annotation to the top-left of the axes.
    ax_obj.text(0.05, 0.95, annotation_text, transform=ax_obj.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

###############################################################################
# 4) Fit pseudo-Voigt peaks on qr_vs_qz integration results
###############################################################################
def fit_pvoigt_peaks(
    data,
    peak_positions,
    background_degree=2,
    background_type="polynomial",
    show=True,
    start=None,
    end=None,
    peak_names=None,
    *,
    return_pattern=False,
    wavelength=1.5406,
):
    """Fit multiple pseudo-Voigt peaks to integrated qr-vs-qz data.

    Parameters
    ----------
    data : pandas.DataFrame
        Single region output from :func:`integrate_qr_vs_qz` with ``qz`` and
        ``integrated_intensity`` columns.
    peak_positions : list of float
        Approximate peak center positions.
    peak_names : list of str, optional
        Labels for each peak. Must be the same length as ``peak_positions``.
        When ``None`` use generic names ``"Peak 1"``, ``"Peak 2"``, ...
    background_degree : int, optional
        Degree of the background polynomial when ``background_type`` is
        ``"polynomial"``. Defaults to ``2``.
    background_type : {"polynomial", "linear", "constant"}, optional
        Type of background model to include. Defaults to ``"polynomial"``.
    show : bool, optional
        When ``True`` display the fit result. Defaults to ``True``.
    start : float, optional
        Minimum ``qz`` value to include in the fit. When ``None`` use the
        beginning of the data.
    end : float, optional
        Maximum ``qz`` value to include in the fit. When ``None`` use the end
        of the data.
    return_pattern : bool, optional
        Deprecated. Ignored in the current implementation. Kept for
        backward compatibility so that existing calls do not fail.
    wavelength : float, optional
        X-ray wavelength in Å used to convert ``qz`` positions to 2θ. Defaults
        to the Cu Kα1 wavelength of ``1.5406`` Å.

    Returns
    -------
    lmfit.model.ModelResult
        The fitted model result.
    numpy.ndarray
        Array of the fitted peak center positions converted to 2θ
        (degrees).
    numpy.ndarray
        Intensity values used in the fit.
    numpy.ndarray
        Integrated intensity (area) of each peak, in the same order as
        ``peak_positions``.
    numpy.ndarray
        2θ positions (degrees) for the evaluated fit pattern.
    numpy.ndarray
        Intensity of the total fit with the background component removed.
    """

    qz = data["qz"].values
    intensity = data["integrated_intensity"].values

    if peak_names is None:
        peak_names = [f"Peak {i+1}" for i in range(len(peak_positions))]
    elif len(peak_names) != len(peak_positions):
        raise ValueError("peak_names must match length of peak_positions")

    if start is not None or end is not None:
        mask = np.ones_like(qz, dtype=bool)
        if start is not None:
            mask &= qz >= start
        if end is not None:
            mask &= qz <= end
        qz = qz[mask]
        intensity = intensity[mask]

    models = _load_lmfit_models()
    PolynomialModel = models["PolynomialModel"]
    LinearModel = models["LinearModel"]
    ConstantModel = models["ConstantModel"]
    PseudoVoigtModel = models["PseudoVoigtModel"]

    if background_type == "polynomial":
        background = PolynomialModel(background_degree, prefix="bkg_")
    elif background_type == "linear":
        background = LinearModel(prefix="bkg_")
    elif background_type == "constant":
        background = ConstantModel(prefix="bkg_")
    else:
        raise ValueError(f"Unknown background_type '{background_type}'")

    model = background
    params = background.guess(intensity, x=qz)

    sigma_guess = (qz.max() - qz.min()) / (20 * max(1, len(peak_positions)))
    for i, cen in enumerate(peak_positions):
        pv = PseudoVoigtModel(prefix=f"p{i}_")
        model += pv
        params.update(pv.make_params())

        idx = int(np.argmin(np.abs(qz - cen)))
        amp_guess = max(intensity[idx], 1.0)

        params[f"p{i}_amplitude"].set(value=amp_guess, min=0)
        params[f"p{i}_center"].set(value=cen)
        params[f"p{i}_sigma"].set(value=sigma_guess, min=1e-6)
        params[f"p{i}_fraction"].set(value=0.5, min=0, max=1)

    try:
        result = model.fit(intensity, params, x=qz)
    except Exception as exc:  # pragma: no cover - runtime fit issues
        print("Fit error:", exc)
        return None

    if show:
        # Plot the fit and components
        fig, ax_fit = plt.subplots(figsize=(8, 5))

        ax_fit.plot(qz, intensity, "ko", label="Data", markersize=4)
        q_smooth = np.linspace(qz.min(), qz.max(), 600)
        ax_fit.plot(q_smooth, result.eval(x=q_smooth), "r-", label="Total Fit")
        comps = result.eval_components(x=q_smooth)
        bkg_key = "bkg_"
        if bkg_key in comps:
            ax_fit.plot(q_smooth, comps[bkg_key], "g-", label="Background")
        for i, name in enumerate(peak_names):
            key = f"p{i}_"
            if key in comps:
                ax_fit.plot(q_smooth, comps[key], "--", label=name)

        ax_fit.set_xlabel(r"qz (Å$^{-1}$)")
        ax_fit.set_ylabel("Integrated Intensity")
        ax_fit.set_title("Pseudo-Voigt Peak Fit")
        ax_fit.legend()

        # Store fit summary values for table output
        table_rows = []
        for i, name in enumerate(peak_names):
            prefix = f"p{i}_"
            p = result.params
            height = p[f"{prefix}height"].value
            height_err = p[f"{prefix}height"].stderr
            amp = p[f"{prefix}amplitude"].value
            amp_err = p[f"{prefix}amplitude"].stderr
            cen = p[f"{prefix}center"].value
            cen_err = p[f"{prefix}center"].stderr
            fwhm, fwhm_err = compute_fwhm_and_error_from_cov(result, prefix)

            # Integrated intensity corresponds to the peak area, which is
            # given directly by the ``amplitude`` parameter of the
            # pseudo-Voigt profile.
            int_int = amp
            int_int_err = amp_err

            # Convert FWHM from qz (Å⁻¹) to degrees
            arg = cen * wavelength / (4.0 * np.pi)
            two_theta_deriv = (wavelength / (2.0 * np.pi)) / np.sqrt(1 - arg**2)
            deg_per_qz = np.degrees(two_theta_deriv)
            fwhm_deg = fwhm * deg_per_qz
            fwhm_deg_err = fwhm_err * deg_per_qz if not np.isnan(fwhm_err) else np.nan

            height_str = f"{height:.2f}" + (
                f"±{height_err:.2f}" if height_err is not None else ""
            )

            cen_str = f"{cen:.4f}" + (f"±{cen_err:.4f}" if cen_err is not None else "")
            fwhm_str = f"{fwhm_deg:.4f}" + (
                f"±{fwhm_deg_err:.4f}" if not np.isnan(fwhm_deg_err) else ""
            )
            int_int_str = f"{int_int:.2f}" + (
                f"±{int_int_err:.2f}" if int_int_err is not None else ""
            )
            table_rows.append([name, height_str, cen_str, fwhm_str, int_int_str])

        # Annotate each peak label above the fitted center
        for i, name in enumerate(peak_names):
            cen = result.params[f"p{i}_center"].value
            y_val = result.eval(x=np.array([cen]))[0]
            ax_fit.text(cen, y_val, name, va="bottom", ha="center")

        fig.tight_layout()
        plt.show()

        # Display fit results in a separate table figure
        fig_table, ax_table = plt.subplots(
            figsize=(7, 1 + 0.4 * len(peak_names))
        )
        ax_table.axis("off")
        table = ax_table.table(
            cellText=table_rows,
            colLabels=[
                "Peak",
                "Height",

                "Center",
                "FWHM (deg)",
                "Integrated Intensity",
            ],
            colLoc="center",
            cellLoc="center",
            colWidths=[0.25] * 5,

            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        table.auto_set_column_width(col=list(range(5)))
        ax_table.set_title(f"Reduced χ²: {result.redchi:.3g}")

        fig_table.tight_layout()
        plt.show()

    # Collect integrated intensity for each peak (area under the curve).
    integrated_dict = {
        name: result.params[f"p{i}_amplitude"].value
        for i, name in enumerate(peak_names)
    }

    # Attach integrated intensities to the result for convenience.
    result.peak_areas = integrated_dict

    # Also provide the values as a numeric array in the same order as the
    # fitted peak centers.
    integrated_intensities = np.array(list(integrated_dict.values()), dtype=float)

    # Calculate the fitted peak center positions in 2\N{DEGREE SIGN}.
    centers_qz = [result.params[f"p{i}_center"].value for i in range(len(peak_names))]
    centers_qz = np.asarray(centers_qz, dtype=float)
    centers_two_theta = np.degrees(2.0 * np.arcsin(centers_qz * wavelength / (4.0 * np.pi)))

    # Evaluate the total fit over the data range and remove the background
    fit_total = result.eval(x=qz)
    comps = result.eval_components(x=qz)
    bkg_key = "bkg_"
    background_fit = comps.get(bkg_key, np.zeros_like(fit_total))
    fit_minus_bkg = fit_total - background_fit

    # Convert qz positions to 2θ for the returned pattern
    two_theta = np.degrees(2.0 * np.arcsin(qz * wavelength / (4.0 * np.pi)))

    return (
        result,
        centers_two_theta,
        intensity,
        integrated_intensities,
        two_theta,
        fit_minus_bkg,
    )

##############################################################################
# 4) Main data processing function
##############################################################################
def process_data(
    ai,
    data,
    regions_of_interest,
    npt_rad=3000,
    npt_azim=1000,
    vmin=0,
    vmax=500,
    output_filename='integrated_peak_intensities_real.npy',
    scaling_factor=100.0,
    do_fitting=True,  # If False, no fitting is performed but integration results are still plotted.
    background=None,
    plot_log_rad=False,
    plot_log_az=False,
    equal_weight_fit=False,
    show_plots=True,
):
    """

    Process raw data: scale, perform 2D integration, subtract background,
    extract 1D profiles, and optionally perform fits. Regardless of fitting,
    a figure with the 2D integrated map and 1D integration plots is shown when
    ``show_plots`` is ``True``.

    Parameters
    ----------
    show_plots : bool, optional
        If ``True``, display the generated figures. Set to ``False`` to suppress
        all plotting (useful for non-interactive use).

    This modified version explicitly returns the φ vs 2θ 2D intensity map in the
    final output.
    """

    # Allow regions_of_interest to be specified as a list of lists in addition to
    # the documented list of dictionaries. If we detect a list format, convert
    # each entry to the canonical dictionary form:
    # [theta_min, theta_max, phi_min, phi_max, name, [color]] ->
    # {"name": name, "theta_min": theta_min, "theta_max": theta_max,
    #  "phi_min": phi_min, "phi_max": phi_max, "color": color}
    if regions_of_interest and not isinstance(regions_of_interest[0], dict):
        converted = []
        for region in regions_of_interest:
            if len(region) < 5:
                raise ValueError(
                    "Region lists must have at least five elements: "
                    "theta_min, theta_max, phi_min, phi_max and name"
                )
            theta_min, theta_max, phi_min, phi_max, name, *rest = region
            color = rest[0] if rest else "white"
            converted.append(
                {
                    "name": str(name),
                    "theta_min": float(theta_min),
                    "theta_max": float(theta_max),
                    "phi_min": float(phi_min),
                    "phi_max": float(phi_max),
                    "color": color,
                }
            )
        regions_of_interest = converted
    nrows = 1 + len(regions_of_interest)
    fig_height = 6 + 3 * len(regions_of_interest)
    fig = plt.figure(figsize=(12, fig_height))
    gs = gridspec.GridSpec(nrows=nrows, ncols=2, height_ratios=[6] + [3]*len(regions_of_interest))

    # Scale raw data
    scaled_data = data * scaling_factor

    # 2D integration using pyFAI
    integration_result = ai.integrate2d(
        scaled_data,
        npt_rad=npt_rad,
        correctSolidAngle=True,
        npt_azim=npt_azim,
        method="full",
        unit="2th_deg",
    )
    intensity = integration_result.intensity
    radial = integration_result.radial    # 2θ in degrees
    azimuthal = integration_result.azimuthal  # φ in degrees

    # Adjust and sort azimuthal angles (shift to range centered around 0)
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    if background is not None:
        if background.shape != intensity_sorted.shape:
            raise ValueError("Background shape does not match data shape.")
        intensity_sorted = intensity_sorted - background

    # Keep only rows with |φ| < 90°
    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    # --- New normalization step ---
    # Use the first region's boundaries for normalization.
    region_first = regions_of_interest[0]
    theta_min = region_first['theta_min']
    theta_max = region_first['theta_max']
    phi_min = region_first['phi_min']
    phi_max = region_first['phi_max']

    # Create masks for the radial (θ) and azimuthal (φ) dimensions.
    mask_rad = (radial >= theta_min) & (radial <= theta_max)
    mask_az = (azimuthal_adjusted_sorted >= phi_min) & (azimuthal_adjusted_sorted <= phi_max)

    # Extract the intensity values in the first region.
    intensity_first_region = intensity_sorted[np.ix_(mask_az, mask_rad)]
    max_val = intensity_first_region.max()

    # Normalize: Divide all intensity values by the maximum of the first region and then multiply by the scaling constant.
    intensity_sorted = (intensity_sorted / max_val) * scaling_factor
    # Plot 2D integrated intensity map
    ax_2d = fig.add_subplot(gs[0, :])
    extent = [radial.min(), radial.max(), azimuthal_adjusted_sorted.min(), azimuthal_adjusted_sorted.max()]
    im = ax_2d.imshow(intensity_sorted, extent=extent, cmap='turbo', vmin=vmin, vmax=vmax,
                      aspect='auto', origin='lower')
    title = 'Integrated 2D Intensity (Scaled + Background-Subtracted)' if background is not None else 'Integrated 2D Intensity (Scaled)'
    ax_2d.set_title(title)
    ax_2d.set_xlabel('2θ (degrees)')
    ax_2d.set_ylabel('Azimuthal angle φ (degrees)')
    cbar = fig.colorbar(im, ax=ax_2d, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Scaled Intensity')

    # Overlay region-of-interest boxes on the 2D map.
    for region in regions_of_interest:
        theta_min = region['theta_min']
        theta_max = region['theta_max']
        phi_min = region['phi_min']
        phi_max = region['phi_max']
        color = region.get('color', 'white')
        ax_2d.add_patch(plt.Rectangle((theta_min, phi_min), theta_max - theta_min, phi_max - phi_min,
                                      edgecolor=color, facecolor='none', linestyle='--', linewidth=2))
        ax_2d.text(theta_min, phi_max + 1, region['name'], color=color, fontsize=10,
                   ha='left', va='bottom')

    # Dictionary to store integration (and optionally fit) results by region.
    results_by_region = {}
    row_index = 1
    for region in regions_of_interest:
        region_name = region['name']
        theta_min = region['theta_min']
        theta_max = region['theta_max']
        phi_min = region['phi_min']
        phi_max = region['phi_max']

        # Create masks for integration ranges.
        mask_az = (azimuthal_adjusted_sorted >= phi_min) & (azimuthal_adjusted_sorted <= phi_max)
        intensity_filtered_az = intensity_sorted[mask_az, :]
        azimuthal_filtered = azimuthal_adjusted_sorted[mask_az]

        mask_rad = (radial >= theta_min) & (radial <= theta_max)
        intensity_filtered = intensity_filtered_az[:, mask_rad]
        radial_filtered = radial[mask_rad]

        if intensity_filtered.size == 0:
            continue

        # 1D integrations: integrate intensity over φ to produce a radial profile;
        # and over 2θ to produce an azimuthal profile.
        intensity_1d = np.trapz(intensity_filtered, x=azimuthal_filtered, axis=0)
        intensity_1d_phi = np.trapz(intensity_filtered, x=radial_filtered, axis=1)

        # Store integration results.
        results_by_region[region_name] = {
            "Radial": {"2θ": radial_filtered.tolist(), "Intensity": intensity_1d.tolist()},
            "Azimuthal": {"φ": azimuthal_filtered.tolist(), "Intensity": intensity_1d_phi.tolist()},
            "UnbinnedData": {"Intensity2D": intensity_filtered.tolist(),
                             "RadialAxis": radial_filtered.tolist(),
                             "AzimuthalAxis": azimuthal_filtered.tolist()}
        }

        # Create subplots for each region: two columns for Radial and Azimuthal profiles.
        ax_rad = fig.add_subplot(gs[row_index, 0])
        ax_az = fig.add_subplot(gs[row_index, 1])
        
        if do_fitting:
            # Perform fits if enabled.
            x_fit_radial, y_fit_radial = radial_filtered, intensity_1d
            result_radial = perform_fit(x_fit_radial, y_fit_radial, theta_min, theta_max, equal_weight_fit=equal_weight_fit)

            x_fit_azimuthal, y_fit_azimuthal = azimuthal_filtered, intensity_1d_phi
            result_azimuthal = perform_fit(x_fit_azimuthal, y_fit_azimuthal, phi_min, phi_max, equal_weight_fit=equal_weight_fit)

            # Plot the fit and its components for both radial and azimuthal profiles.
            if result_radial is not None:
                plot_fit_with_components(ax_rad, x_fit_radial, y_fit_radial, result_radial, '2θ (degrees)', use_log=plot_log_rad)
            else:
                ax_rad.text(0.5, 0.5, 'No radial fit', ha='center', va='center', transform=ax_rad.transAxes)
            if result_azimuthal is not None:
                plot_fit_with_components(ax_az, x_fit_azimuthal, y_fit_azimuthal, result_azimuthal, 'Azimuthal φ (degrees)', use_log=plot_log_az)
            else:
                ax_az.text(0.5, 0.5, 'No azimuthal fit', ha='center', va='center', transform=ax_az.transAxes)

            # Save fit parameters
            results_by_region[region_name]["FittedParams"] = {
                "Radial": result_radial.params.valuesdict() if result_radial else {},
                "Azimuthal": result_azimuthal.params.valuesdict() if result_azimuthal else {}
            }
        else:
            # When fitting is disabled, simply plot the integration results.
            ax_rad.plot(radial_filtered, intensity_1d, 'ko-', label='Radial Integration', markersize=4)
            ax_rad.set_xlabel('2θ (degrees)')
            ax_rad.set_ylabel('Scaled Intensity')
            ax_rad.set_title(f"{region_name} - Radial Integration")
            ax_rad.legend(loc='best')
            
            ax_az.plot(azimuthal_filtered, intensity_1d_phi, 'ko-', label='Azimuthal Integration', markersize=4)
            ax_az.set_xlabel('Azimuthal φ (degrees)')
            ax_az.set_ylabel('Scaled Intensity')
            ax_az.set_title(f"{region_name} - Azimuthal Integration")
            ax_az.legend(loc='best')
        row_index += 1

    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Package and save main integrated data for output.
    main_data = {
        "Intensity_Sorted": intensity_sorted,
        "Radial": radial,
        "Azimuthal_Adjusted_Sorted": azimuthal_adjusted_sorted
    }
    # Explicitly include the φ vs 2θ 2D intensity map in the output.
    phi_vs_2theta_map = intensity_sorted

    final_output = {
        "Regions": results_by_region,
        "MainData": main_data,
        "phi_vs_2theta_map": phi_vs_2theta_map,
        "Regions_of_Interest": regions_of_interest,
        "ScalingFactor": scaling_factor,
    }

    # Save complete output including any fitted parameters for reproducibility.
    file_location = "C:\\Users\\Kenpo\\Downloads\\" + output_filename
    np.save(file_location, final_output)
    print(f"Data saved to {file_location}")

    # Remove fitted parameters from the return value so that the caller receives
    # only the information needed to reproduce the plots.
    results_no_fit = {
        k: {kk: vv for kk, vv in v.items() if kk != "FittedParams"}
        for k, v in results_by_region.items()
    }

    return intensity_sorted, azimuthal_adjusted_sorted, radial, results_no_fit

##############################################################################
# Example usage:
##############################################################################
# Uncomment and modify the following example usage as needed.
# 
# ai = pyFAI.AzimuthalIntegrator()  # Set up your pyFAI integrator with proper calibration.
# data = np.load('path_to_your_raw_data.npy')  # Load your raw 2D data array.
# regions_of_interest = [
#     {'name': 'Region 1', 'theta_min': 20, 'theta_max': 40, 'phi_min': 10, 'phi_max': 30, 'color': 'yellow'},
#     {'name': 'Region 2', 'theta_min': 40, 'theta_max': 60, 'phi_min': 10, 'phi_max': 30, 'color': 'cyan'},
# ]
#
# To process data with fitting:
# results = process_data(ai, data, regions_of_interest, do_fitting=True, output_filename='results_fitted.npy')
#
# To process data without fitting (integration only, but figures are still shown):
# results = process_data(ai, data, regions_of_interest, do_fitting=False, output_filename='results_no_fit.npy')
