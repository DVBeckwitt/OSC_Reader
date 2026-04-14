"""Exact detector-to-angle-space conversion helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _HAS_NUMBA = False


def _optional_njit(*jit_args, **jit_kwargs):
    def _decorate(fn):
        if not _HAS_NUMBA:
            return fn
        return njit(*jit_args, **jit_kwargs)(fn)

    return _decorate


EPS32 = 1.0 + np.finfo(np.float32).eps
BEAM_CENTER_CHI_DEG = 135.0
DEFAULT_ANGLE_SPACE_WORKERS = 24
DEFAULT_ANGLE_SPACE_ENGINE = "numba"
DEFAULT_TWO_THETA_MIN_DEG = 0.0
DEFAULT_TWO_THETA_MAX_DEG = 90.0
DEFAULT_PHI_MIN_DEG = -180.0
DEFAULT_PHI_MAX_DEG = 180.0
PHI_ZERO_OFFSET_DEGREES = -90.0


@dataclass(frozen=True)
class DetectorCakeGeometry:
    pixel_size_m: float
    distance_m: float
    center_row_px: float
    center_col_px: float


@dataclass(frozen=True)
class DetectorCakeResult:
    radial_deg: np.ndarray
    azimuthal_deg: np.ndarray
    intensity: np.ndarray
    sum_signal: np.ndarray
    sum_normalization: np.ndarray
    count: np.ndarray


def _default_worker_count() -> int:
    return max(
        1,
        min(int(DEFAULT_ANGLE_SPACE_WORKERS), os.cpu_count() or DEFAULT_ANGLE_SPACE_WORKERS),
    )


def _validate_axes(radial_deg: np.ndarray, azimuthal_deg: np.ndarray) -> None:
    if radial_deg.ndim != 1 or azimuthal_deg.ndim != 1:
        raise ValueError("radial_deg and azimuthal_deg must be 1D arrays.")
    if radial_deg.size < 2 or azimuthal_deg.size < 2:
        raise ValueError("radial_deg and azimuthal_deg need at least 2 bins each.")
    radial_step = np.diff(radial_deg)
    azimuthal_step = np.diff(azimuthal_deg)
    if not np.all(radial_step > 0.0):
        raise ValueError("radial_deg must be strictly increasing.")
    if not np.all(azimuthal_step > 0.0):
        raise ValueError("azimuthal_deg must be strictly increasing.")
    if not np.allclose(radial_step, radial_step[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("radial_deg must be uniformly spaced.")
    if not np.allclose(azimuthal_step, azimuthal_step[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("azimuthal_deg must be uniformly spaced.")


def _prepare_inputs(
    image: np.ndarray,
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    normalization: np.ndarray | None,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    signal = np.asarray(image, dtype=np.float32)
    radial = np.asarray(radial_deg, dtype=np.float64)
    azimuthal = np.asarray(azimuthal_deg, dtype=np.float64)
    _validate_axes(radial, azimuthal)
    if normalization is None:
        norm = np.ones(signal.shape, dtype=np.float32)
    else:
        norm = np.asarray(normalization, dtype=np.float32)
        if norm.shape != signal.shape:
            raise ValueError("normalization must match image shape.")
    if mask is None:
        return signal, norm, radial, azimuthal, np.zeros((1, 1), dtype=np.int8), False
    mask_array = np.asarray(mask, dtype=np.int8)
    if mask_array.shape != signal.shape:
        raise ValueError("mask must match image shape.")
    return signal, norm, radial, azimuthal, mask_array, True


def _prepare_selection(
    shape: tuple[int, int],
    rows: np.ndarray | None,
    cols: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if (rows is None) != (cols is None):
        raise ValueError("rows and cols must both be provided or both be omitted.")
    if rows is None or cols is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), False
    rows_array = np.asarray(rows, dtype=np.int64).ravel()
    cols_array = np.asarray(cols, dtype=np.int64).ravel()
    if rows_array.shape != cols_array.shape:
        raise ValueError("rows and cols must have the same shape.")
    height, width = int(shape[0]), int(shape[1])
    if rows_array.size and (
        np.any(rows_array < 0)
        or np.any(rows_array >= height)
        or np.any(cols_array < 0)
        or np.any(cols_array >= width)
    ):
        raise ValueError("rows/cols contain indices outside the image bounds.")
    return rows_array, cols_array, True


def _row_col_edges(
    shape: tuple[int, int],
    geometry: DetectorCakeGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    row_edges = (
        np.arange(height + 1, dtype=np.float64) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_edges = (
        np.arange(width + 1, dtype=np.float64) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    return row_edges, col_edges


def _resolve_engine(engine: str) -> str:
    engine_name = str(engine).strip().lower()
    if engine_name == "auto":
        return "numba" if _HAS_NUMBA else "python"
    if engine_name not in {"python", "numba"}:
        raise ValueError("engine must be one of: auto, python, numba.")
    if engine_name == "numba" and not _HAS_NUMBA:
        raise RuntimeError("engine='numba' requested, but numba is unavailable.")
    return engine_name


def _resolve_workers(workers: int | str | None, work_items: int, engine: str) -> int:
    if work_items <= 1 or engine == "python":
        return 1
    if workers is None:
        return 1
    if isinstance(workers, str):
        if workers != "auto":
            raise ValueError("workers must be an int, 'auto', or None.")
        resolved = _default_worker_count()
    else:
        resolved = int(workers)
    return max(1, min(resolved, int(work_items)))


def _chunk_ranges(length: int, workers: int) -> list[tuple[int, int]]:
    if workers <= 1 or length <= 1:
        return [(0, int(length))]
    chunk_edges = np.linspace(0, int(length), int(workers) + 1, dtype=np.int64)
    ranges: list[tuple[int, int]] = []
    for index in range(int(workers)):
        start = int(chunk_edges[index])
        stop = int(chunk_edges[index + 1])
        if stop > start:
            ranges.append((start, stop))
    return ranges or [(0, int(length))]


def _inverse_calc_upper_bound(value: float) -> float:
    if value > 0.0:
        return float(value / EPS32)
    if value < 0.0:
        return float(value * EPS32)
    return 0.0


def _calc_area(i1: float, i2: float, slope: float, intercept: float) -> float:
    return (i2 - i1) * (0.5 * slope * (i2 + i1) + intercept)


def _area4p(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    c0: float,
    c1: float,
    d0: float,
    d1: float,
) -> float:
    return 0.5 * ((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0))


def _recenter_helper(azim: float, period: float, chi_disc_at_pi: bool = True) -> float:
    if (chi_disc_at_pi and azim < 0.0) or ((not chi_disc_at_pi) and azim < 0.5 * period):
        return azim + period
    return azim


def _corner_to_polar_deg(y: float, x: float, distance: float) -> tuple[float, float]:
    radial = math.degrees(math.atan2(math.hypot(x, y), distance))
    if x == 0.0 and y == 0.0:
        azimuth = BEAM_CENTER_CHI_DEG
    else:
        azimuth = math.degrees(math.atan2(y, x))
    return radial, azimuth


def _integrate_edge(box: np.ndarray, start0: float, start1: float, stop0: float, stop1: float) -> None:
    if start0 == stop0:
        return
    slope = (stop1 - start1) / (stop0 - start0)
    intercept = stop1 - slope * stop0
    if start0 < stop0:
        p_value = math.ceil(start0)
        delta_p = p_value - start0
        if p_value > stop0:
            segment_area = _calc_area(start0, stop0, slope, intercept)
            if segment_area != 0.0:
                abs_area = abs(segment_area)
                delta_a = stop0 - start0
                height = 0
                while abs_area > 0.0 and height < box.shape[1]:
                    if delta_a > abs_area:
                        delta_a = abs_area
                        abs_area = -1.0
                    box[int(start0), height] += math.copysign(delta_a, segment_area)
                    abs_area -= delta_a
                    height += 1
        else:
            if delta_p > 0.0:
                segment_area = _calc_area(start0, p_value, slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = delta_p
                    while abs_area > 0.0 and height < box.shape[1]:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[int(p_value) - 1, height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
            for index0 in range(int(math.floor(p_value)), int(math.floor(stop0))):
                segment_area = _calc_area(float(index0), float(index0 + 1), slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = 1.0
                    while abs_area > 0.0 and height < box.shape[1]:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[index0, height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
            p_value = math.floor(stop0)
            delta_p = stop0 - p_value
            if delta_p > 0.0:
                segment_area = _calc_area(p_value, stop0, slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = abs(delta_p)
                    while abs_area > 0.0 and height < box.shape[1]:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[int(p_value), height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
        return
    p_value = math.floor(start0)
    if stop0 > p_value:
        segment_area = _calc_area(start0, stop0, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            delta_a = start0 - stop0
            height = 0
            while abs_area > 0.0 and height < box.shape[1]:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(start0), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
        return
    delta_p = p_value - start0
    if delta_p < 0.0:
        segment_area = _calc_area(start0, p_value, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = abs(delta_p)
            while abs_area > 0.0 and height < box.shape[1]:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(p_value), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
    for index0 in range(int(start0), int(math.ceil(stop0)), -1):
        segment_area = _calc_area(float(index0), float(index0 - 1), slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = 1.0
            while abs_area > 0.0 and height < box.shape[1]:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[index0 - 1, height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
    p_value = math.ceil(stop0)
    delta_p = stop0 - p_value
    if delta_p < 0.0:
        segment_area = _calc_area(p_value, stop0, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = abs(delta_p)
            while abs_area > 0.0 and height < box.shape[1]:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(stop0), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1


def _accumulate_pixel_python(
    row: int,
    col: int,
    signal_value: float,
    norm_value: float,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    n_rad: int,
    n_az: int,
    delta0: float,
    delta1: float,
    pos0_min: float,
    pos0_maxin: float,
    pos1_min: float,
    pos1_maxin: float,
    pos1_period: float,
    sum_signal: np.ndarray,
    sum_normalization: np.ndarray,
    count: np.ndarray,
    box: np.ndarray,
) -> None:
    y0 = float(row_edges[row])
    y1 = float(row_edges[row + 1])
    x0 = float(col_edges[col])
    x1 = float(col_edges[col + 1])
    a0, a1 = _corner_to_polar_deg(y0, x0, distance)
    b0, b1 = _corner_to_polar_deg(y1, x0, distance)
    c0, c1 = _corner_to_polar_deg(y1, x1, distance)
    d0, d1 = _corner_to_polar_deg(y0, x1, distance)
    area = _area4p(a0, a1, b0, b1, c0, c1, d0, d1)
    if pos1_period > 0.0 and area > 0.0:
        a1 = _recenter_helper(a1, pos1_period, True)
        b1 = _recenter_helper(b1, pos1_period, True)
        c1 = _recenter_helper(c1, pos1_period, True)
        d1 = _recenter_helper(d1, pos1_period, True)
        center1 = 0.25 * (a1 + b1 + c1 + d1)
        if center1 > 0.5 * pos1_period:
            a1 -= pos1_period
            b1 -= pos1_period
            c1 -= pos1_period
            d1 -= pos1_period
    a0 = (min(max(a0, pos0_min), pos0_maxin) - pos0_min) / delta0
    b0 = (min(max(b0, pos0_min), pos0_maxin) - pos0_min) / delta0
    c0 = (min(max(c0, pos0_min), pos0_maxin) - pos0_min) / delta0
    d0 = (min(max(d0, pos0_min), pos0_maxin) - pos0_min) / delta0
    a1 = (min(max(a1, pos1_min), pos1_maxin) - pos1_min) / delta1
    b1 = (min(max(b1, pos1_min), pos1_maxin) - pos1_min) / delta1
    c1 = (min(max(c1, pos1_min), pos1_maxin) - pos1_min) / delta1
    d1 = (min(max(d1, pos1_min), pos1_maxin) - pos1_min) / delta1
    min0 = min(a0, b0, c0, d0)
    max0 = max(a0, b0, c0, d0)
    min1 = min(a1, b1, c1, d1)
    max1 = max(a1, b1, c1, d1)
    foffset0 = math.floor(min0)
    foffset1 = math.floor(min1)
    ioffset0 = int(foffset0)
    ioffset1 = int(foffset1)
    width0 = int(math.ceil(max0) - foffset0)
    width1 = int(math.ceil(max1) - foffset1)
    if width0 <= 0 or width1 <= 0:
        return
    for index0 in range(width0 + 1):
        for index1 in range(width1 + 1):
            box[index0, index1] = 0.0
    a0 -= foffset0
    b0 -= foffset0
    c0 -= foffset0
    d0 -= foffset0
    a1 -= foffset1
    b1 -= foffset1
    c1 -= foffset1
    d1 -= foffset1
    _integrate_edge(box, a0, a1, b0, b1)
    _integrate_edge(box, b0, b1, c0, c1)
    _integrate_edge(box, c0, c1, d0, d1)
    _integrate_edge(box, d0, d1, a0, a1)
    sum_area = 0.0
    for index0 in range(width0):
        for index1 in range(width1):
            sum_area += float(box[index0, index1])
    if sum_area == 0.0 or not math.isfinite(sum_area):
        return
    inv_area = 1.0 / sum_area
    for index0 in range(width0):
        bin_rad = ioffset0 + index0
        if bin_rad < 0 or bin_rad >= n_rad:
            continue
        for index1 in range(width1):
            bin_az = ioffset1 + index1
            if bin_az < 0 or bin_az >= n_az:
                continue
            weight = float(box[index0, index1]) * inv_area
            if weight == 0.0:
                continue
            sum_signal[bin_az, bin_rad] += signal_value * weight
            sum_normalization[bin_az, bin_rad] += norm_value * weight
            count[bin_az, bin_rad] += weight


def _run_chunk_python(
    signal: np.ndarray,
    normalization: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk_start: int,
    chunk_stop: int,
    use_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rad = int(radial.size)
    n_az = int(azimuthal.size)
    delta0 = float(radial[1] - radial[0])
    delta1 = float(azimuthal[1] - azimuthal[0])
    pos0_min = float(radial[0] - 0.5 * delta0)
    pos1_min = float(azimuthal[0] - 0.5 * delta1)
    pos0_max = float(radial[-1] + 0.5 * delta0)
    pos1_max = float(azimuthal[-1] + 0.5 * delta1)
    pos0_maxin = _inverse_calc_upper_bound(pos0_max)
    pos1_maxin = _inverse_calc_upper_bound(pos1_max)
    sum_signal = np.zeros((n_az, n_rad), dtype=np.float64)
    sum_normalization = np.zeros((n_az, n_rad), dtype=np.float64)
    count = np.zeros((n_az, n_rad), dtype=np.float64)
    box = np.zeros((n_rad + 1, n_az + 1), dtype=np.float32)
    if use_selection:
        for selection_index in range(int(chunk_start), int(chunk_stop)):
            row = int(rows[selection_index])
            col = int(cols[selection_index])
            if has_mask and mask[row, col] != 0:
                continue
            signal_value = float(signal[row, col])
            norm_value = float(normalization[row, col])
            if (not math.isfinite(signal_value)) or (not math.isfinite(norm_value)) or norm_value == 0.0:
                continue
            _accumulate_pixel_python(
                row,
                col,
                signal_value,
                norm_value,
                row_edges,
                col_edges,
                distance,
                n_rad,
                n_az,
                delta0,
                delta1,
                pos0_min,
                pos0_maxin,
                pos1_min,
                pos1_maxin,
                360.0,
                sum_signal,
                sum_normalization,
                count,
                box,
            )
        return sum_signal, sum_normalization, count
    for row in range(int(chunk_start), int(chunk_stop)):
        for col in range(signal.shape[1]):
            if has_mask and mask[row, col] != 0:
                continue
            signal_value = float(signal[row, col])
            norm_value = float(normalization[row, col])
            if (not math.isfinite(signal_value)) or (not math.isfinite(norm_value)) or norm_value == 0.0:
                continue
            _accumulate_pixel_python(
                row,
                col,
                signal_value,
                norm_value,
                row_edges,
                col_edges,
                distance,
                n_rad,
                n_az,
                delta0,
                delta1,
                pos0_min,
                pos0_maxin,
                pos1_min,
                pos1_maxin,
                360.0,
                sum_signal,
                sum_normalization,
                count,
                box,
            )
    return sum_signal, sum_normalization, count


@_optional_njit(cache=True, nogil=True, inline="always")
def _inverse_calc_upper_bound_numba(value: float) -> float:
    if value > 0.0:
        return value / EPS32
    if value < 0.0:
        return value * EPS32
    return 0.0


@_optional_njit(cache=True, nogil=True, inline="always")
def _calc_area_numba(i1: float, i2: float, slope: float, intercept: float) -> float:
    return (i2 - i1) * (0.5 * slope * (i2 + i1) + intercept)


@_optional_njit(cache=True, nogil=True, inline="always")
def _area4p_numba(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    c0: float,
    c1: float,
    d0: float,
    d1: float,
) -> float:
    return 0.5 * ((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0))


@_optional_njit(cache=True, nogil=True, inline="always")
def _recenter_helper_numba(azim: float, period: float, chi_disc_at_pi: bool = True) -> float:
    if (chi_disc_at_pi and azim < 0.0) or ((not chi_disc_at_pi) and azim < 0.5 * period):
        return azim + period
    return azim


@_optional_njit(cache=True, nogil=True, inline="always")
def _corner_to_polar_deg_numba(y: float, x: float, distance: float) -> tuple[float, float]:
    radial = math.degrees(math.atan2(math.hypot(x, y), distance))
    if x == 0.0 and y == 0.0:
        azimuth = BEAM_CENTER_CHI_DEG
    else:
        azimuth = math.degrees(math.atan2(y, x))
    return radial, azimuth


@_optional_njit(cache=True, nogil=True, inline="always")
def _integrate_edge_numba(
    box: np.ndarray,
    start0: float,
    start1: float,
    stop0: float,
    stop1: float,
    box_height: int,
) -> None:
    if start0 == stop0:
        return
    slope = (stop1 - start1) / (stop0 - start0)
    intercept = stop1 - slope * stop0
    if start0 < stop0:
        p_value = math.ceil(start0)
        delta_p = p_value - start0
        if p_value > stop0:
            segment_area = _calc_area_numba(start0, stop0, slope, intercept)
            if segment_area != 0.0:
                abs_area = abs(segment_area)
                delta_a = stop0 - start0
                height = 0
                while abs_area > 0.0 and height < box_height:
                    if delta_a > abs_area:
                        delta_a = abs_area
                        abs_area = -1.0
                    box[int(start0), height] += math.copysign(delta_a, segment_area)
                    abs_area -= delta_a
                    height += 1
        else:
            if delta_p > 0.0:
                segment_area = _calc_area_numba(start0, p_value, slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = delta_p
                    while abs_area > 0.0 and height < box_height:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[int(p_value) - 1, height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
            for index0 in range(int(math.floor(p_value)), int(math.floor(stop0))):
                segment_area = _calc_area_numba(float(index0), float(index0 + 1), slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = 1.0
                    while abs_area > 0.0 and height < box_height:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[index0, height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
            p_value = math.floor(stop0)
            delta_p = stop0 - p_value
            if delta_p > 0.0:
                segment_area = _calc_area_numba(p_value, stop0, slope, intercept)
                if segment_area != 0.0:
                    abs_area = abs(segment_area)
                    height = 0
                    delta_a = abs(delta_p)
                    while abs_area > 0.0 and height < box_height:
                        if delta_a > abs_area:
                            delta_a = abs_area
                            abs_area = -1.0
                        box[int(p_value), height] += math.copysign(delta_a, segment_area)
                        abs_area -= delta_a
                        height += 1
        return
    p_value = math.floor(start0)
    if stop0 > p_value:
        segment_area = _calc_area_numba(start0, stop0, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            delta_a = start0 - stop0
            height = 0
            while abs_area > 0.0 and height < box_height:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(start0), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
        return
    delta_p = p_value - start0
    if delta_p < 0.0:
        segment_area = _calc_area_numba(start0, p_value, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = abs(delta_p)
            while abs_area > 0.0 and height < box_height:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(p_value), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
    for index0 in range(int(start0), int(math.ceil(stop0)), -1):
        segment_area = _calc_area_numba(float(index0), float(index0 - 1), slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = 1.0
            while abs_area > 0.0 and height < box_height:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[index0 - 1, height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1
    p_value = math.ceil(stop0)
    delta_p = stop0 - p_value
    if delta_p < 0.0:
        segment_area = _calc_area_numba(p_value, stop0, slope, intercept)
        if segment_area != 0.0:
            abs_area = abs(segment_area)
            height = 0
            delta_a = abs(delta_p)
            while abs_area > 0.0 and height < box_height:
                if delta_a > abs_area:
                    delta_a = abs_area
                    abs_area = -1.0
                box[int(stop0), height] += math.copysign(delta_a, segment_area)
                abs_area -= delta_a
                height += 1


@_optional_njit(cache=True, nogil=True, inline="always")
def _accumulate_pixel_numba(
    row: int,
    col: int,
    signal_value: float,
    norm_value: float,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    n_rad: int,
    n_az: int,
    delta0: float,
    delta1: float,
    pos0_min: float,
    pos0_maxin: float,
    pos1_min: float,
    pos1_maxin: float,
    pos1_period: float,
    sum_signal: np.ndarray,
    sum_normalization: np.ndarray,
    count: np.ndarray,
    box: np.ndarray,
) -> None:
    y0 = row_edges[row]
    y1 = row_edges[row + 1]
    x0 = col_edges[col]
    x1 = col_edges[col + 1]
    a0, a1 = _corner_to_polar_deg_numba(y0, x0, distance)
    b0, b1 = _corner_to_polar_deg_numba(y1, x0, distance)
    c0, c1 = _corner_to_polar_deg_numba(y1, x1, distance)
    d0, d1 = _corner_to_polar_deg_numba(y0, x1, distance)
    area = _area4p_numba(a0, a1, b0, b1, c0, c1, d0, d1)
    if pos1_period > 0.0 and area > 0.0:
        a1 = _recenter_helper_numba(a1, pos1_period, True)
        b1 = _recenter_helper_numba(b1, pos1_period, True)
        c1 = _recenter_helper_numba(c1, pos1_period, True)
        d1 = _recenter_helper_numba(d1, pos1_period, True)
        center1 = 0.25 * (a1 + b1 + c1 + d1)
        if center1 > 0.5 * pos1_period:
            a1 -= pos1_period
            b1 -= pos1_period
            c1 -= pos1_period
            d1 -= pos1_period
    a0 = (min(max(a0, pos0_min), pos0_maxin) - pos0_min) / delta0
    b0 = (min(max(b0, pos0_min), pos0_maxin) - pos0_min) / delta0
    c0 = (min(max(c0, pos0_min), pos0_maxin) - pos0_min) / delta0
    d0 = (min(max(d0, pos0_min), pos0_maxin) - pos0_min) / delta0
    a1 = (min(max(a1, pos1_min), pos1_maxin) - pos1_min) / delta1
    b1 = (min(max(b1, pos1_min), pos1_maxin) - pos1_min) / delta1
    c1 = (min(max(c1, pos1_min), pos1_maxin) - pos1_min) / delta1
    d1 = (min(max(d1, pos1_min), pos1_maxin) - pos1_min) / delta1
    min0 = min(a0, b0, c0, d0)
    max0 = max(a0, b0, c0, d0)
    min1 = min(a1, b1, c1, d1)
    max1 = max(a1, b1, c1, d1)
    foffset0 = math.floor(min0)
    foffset1 = math.floor(min1)
    ioffset0 = int(foffset0)
    ioffset1 = int(foffset1)
    width0 = int(math.ceil(max0) - foffset0)
    width1 = int(math.ceil(max1) - foffset1)
    if width0 <= 0 or width1 <= 0:
        return
    for index0 in range(width0 + 1):
        for index1 in range(width1 + 1):
            box[index0, index1] = 0.0
    a0 -= foffset0
    b0 -= foffset0
    c0 -= foffset0
    d0 -= foffset0
    a1 -= foffset1
    b1 -= foffset1
    c1 -= foffset1
    d1 -= foffset1
    _integrate_edge_numba(box, a0, a1, b0, b1, width1 + 1)
    _integrate_edge_numba(box, b0, b1, c0, c1, width1 + 1)
    _integrate_edge_numba(box, c0, c1, d0, d1, width1 + 1)
    _integrate_edge_numba(box, d0, d1, a0, a1, width1 + 1)
    sum_area = 0.0
    for index0 in range(width0):
        for index1 in range(width1):
            sum_area += box[index0, index1]
    if sum_area == 0.0 or not math.isfinite(sum_area):
        return
    inv_area = 1.0 / sum_area
    for index0 in range(width0):
        bin_rad = ioffset0 + index0
        if bin_rad < 0 or bin_rad >= n_rad:
            continue
        for index1 in range(width1):
            bin_az = ioffset1 + index1
            if bin_az < 0 or bin_az >= n_az:
                continue
            weight = float(box[index0, index1]) * inv_area
            if weight == 0.0:
                continue
            sum_signal[bin_az, bin_rad] += signal_value * weight
            sum_normalization[bin_az, bin_rad] += norm_value * weight
            count[bin_az, bin_rad] += weight


@_optional_njit(cache=True, nogil=True)
def _run_chunk_numba(
    signal: np.ndarray,
    normalization: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk_start: int,
    chunk_stop: int,
    use_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rad = int(radial.size)
    n_az = int(azimuthal.size)
    delta0 = float(radial[1] - radial[0])
    delta1 = float(azimuthal[1] - azimuthal[0])
    pos0_min = float(radial[0] - 0.5 * delta0)
    pos1_min = float(azimuthal[0] - 0.5 * delta1)
    pos0_max = float(radial[n_rad - 1] + 0.5 * delta0)
    pos1_max = float(azimuthal[n_az - 1] + 0.5 * delta1)
    pos0_maxin = _inverse_calc_upper_bound_numba(pos0_max)
    pos1_maxin = _inverse_calc_upper_bound_numba(pos1_max)
    sum_signal = np.zeros((n_az, n_rad), dtype=np.float64)
    sum_normalization = np.zeros((n_az, n_rad), dtype=np.float64)
    count = np.zeros((n_az, n_rad), dtype=np.float64)
    box = np.zeros((n_rad + 1, n_az + 1), dtype=np.float32)
    if use_selection:
        for selection_index in range(int(chunk_start), int(chunk_stop)):
            row = int(rows[selection_index])
            col = int(cols[selection_index])
            if has_mask and mask[row, col] != 0:
                continue
            signal_value = float(signal[row, col])
            norm_value = float(normalization[row, col])
            if (not math.isfinite(signal_value)) or (not math.isfinite(norm_value)) or norm_value == 0.0:
                continue
            _accumulate_pixel_numba(
                row,
                col,
                signal_value,
                norm_value,
                row_edges,
                col_edges,
                distance,
                n_rad,
                n_az,
                delta0,
                delta1,
                pos0_min,
                pos0_maxin,
                pos1_min,
                pos1_maxin,
                360.0,
                sum_signal,
                sum_normalization,
                count,
                box,
            )
        return sum_signal, sum_normalization, count
    for row in range(int(chunk_start), int(chunk_stop)):
        for col in range(signal.shape[1]):
            if has_mask and mask[row, col] != 0:
                continue
            signal_value = float(signal[row, col])
            norm_value = float(normalization[row, col])
            if (not math.isfinite(signal_value)) or (not math.isfinite(norm_value)) or norm_value == 0.0:
                continue
            _accumulate_pixel_numba(
                row,
                col,
                signal_value,
                norm_value,
                row_edges,
                col_edges,
                distance,
                n_rad,
                n_az,
                delta0,
                delta1,
                pos0_min,
                pos0_maxin,
                pos1_min,
                pos1_maxin,
                360.0,
                sum_signal,
                sum_normalization,
                count,
                box,
            )
    return sum_signal, sum_normalization, count


def _run_python(
    signal: np.ndarray,
    normalization: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    use_selection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work_length = int(rows.size) if use_selection else int(signal.shape[0])
    return _run_chunk_python(
        signal,
        normalization,
        mask,
        has_mask,
        row_edges,
        col_edges,
        float(distance),
        radial,
        azimuthal,
        rows,
        cols,
        0,
        work_length,
        use_selection,
    )


def _run_numba(
    signal: np.ndarray,
    normalization: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    use_selection: bool,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work_length = int(rows.size) if use_selection else int(signal.shape[0])
    chunks = _chunk_ranges(work_length, workers)
    _run_chunk_numba(
        signal,
        normalization,
        mask,
        has_mask,
        row_edges,
        col_edges,
        float(distance),
        radial,
        azimuthal,
        rows,
        cols,
        int(chunks[0][0]),
        int(chunks[0][0]),
        bool(use_selection),
    )

    def _worker(chunk: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _run_chunk_numba(
            signal,
            normalization,
            mask,
            has_mask,
            row_edges,
            col_edges,
            float(distance),
            radial,
            azimuthal,
            rows,
            cols,
            int(chunk[0]),
            int(chunk[1]),
            bool(use_selection),
        )

    if workers <= 1 or len(chunks) <= 1:
        partials = [_worker(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            partials = list(executor.map(_worker, chunks))
    sum_signal = np.zeros((azimuthal.size, radial.size), dtype=np.float64)
    sum_normalization = np.zeros_like(sum_signal)
    count = np.zeros_like(sum_signal)
    for part_signal, part_norm, part_count in partials:
        sum_signal += part_signal
        sum_normalization += part_norm
        count += part_count
    return sum_signal, sum_normalization, count


def integrate_detector_to_cake_exact(
    image: np.ndarray,
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    normalization: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> DetectorCakeResult:
    signal, norm, radial, azimuthal, mask_array, has_mask = _prepare_inputs(
        image,
        radial_deg,
        azimuthal_deg,
        normalization,
        mask,
    )
    rows_array, cols_array, use_selection = _prepare_selection(signal.shape, rows, cols)
    row_edges, col_edges = _row_col_edges(signal.shape, geometry)
    engine_name = _resolve_engine(engine)
    work_items = int(rows_array.size) if use_selection else int(signal.shape[0])
    worker_count = _resolve_workers(workers, work_items, engine_name)
    if engine_name == "python":
        sum_signal, sum_normalization, count = _run_python(
            signal,
            norm,
            mask_array,
            has_mask,
            row_edges,
            col_edges,
            float(geometry.distance_m),
            radial,
            azimuthal,
            rows_array,
            cols_array,
            use_selection,
        )
    else:
        sum_signal, sum_normalization, count = _run_numba(
            signal,
            norm,
            mask_array,
            has_mask,
            row_edges,
            col_edges,
            float(geometry.distance_m),
            radial,
            azimuthal,
            rows_array,
            cols_array,
            use_selection,
            worker_count,
        )
    intensity = np.zeros_like(sum_signal, dtype=np.float32)
    valid = sum_normalization > 0.0
    intensity[valid] = (sum_signal[valid] / sum_normalization[valid]).astype(np.float32, copy=False)
    return DetectorCakeResult(
        radial_deg=np.array(radial, copy=True),
        azimuthal_deg=np.array(azimuthal, copy=True),
        intensity=intensity,
        sum_signal=sum_signal,
        sum_normalization=sum_normalization,
        count=count,
    )


def build_angle_axes(
    *,
    npt_rad: int,
    npt_azim: int,
    tth_min_deg: float = DEFAULT_TWO_THETA_MIN_DEG,
    tth_max_deg: float = DEFAULT_TWO_THETA_MAX_DEG,
    azimuth_min_deg: float = DEFAULT_PHI_MIN_DEG,
    azimuth_max_deg: float = DEFAULT_PHI_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    radial_edges = np.linspace(
        float(tth_min_deg),
        float(tth_max_deg),
        int(max(2, npt_rad)) + 1,
        dtype=np.float64,
    )
    azimuth_edges = np.linspace(
        float(azimuth_min_deg),
        float(azimuth_max_deg),
        int(max(2, npt_azim)) + 1,
        dtype=np.float64,
    )
    radial_deg = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    azimuthal_deg = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    return radial_deg, azimuthal_deg


def flat_solid_angle_normalization(
    image_shape: tuple[int, int],
    geometry: DetectorCakeGeometry,
) -> np.ndarray:
    row_coords = (
        (np.arange(int(image_shape[0]), dtype=np.float64) + 0.5)
        - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_coords = (
        (np.arange(int(image_shape[1]), dtype=np.float64) + 0.5)
        - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    yy = row_coords[:, None]
    xx = col_coords[None, :]
    path = np.sqrt(xx * xx + yy * yy + float(geometry.distance_m) ** 2)
    return np.asarray((float(geometry.distance_m) / path) ** 3, dtype=np.float32)


def convert_image_to_phi_2theta_space(
    image: np.ndarray,
    *,
    distance_mm: float,
    pixel_size_mm: float,
    center_row_px: float,
    center_col_px: float,
    radial_bins: int | None = None,
    azimuth_bins: int | None = None,
    two_theta_min_deg: float = DEFAULT_TWO_THETA_MIN_DEG,
    two_theta_max_deg: float = DEFAULT_TWO_THETA_MAX_DEG,
    phi_min_deg: float = DEFAULT_PHI_MIN_DEG,
    phi_max_deg: float = DEFAULT_PHI_MAX_DEG,
    correct_solid_angle: bool = False,
    engine: str = DEFAULT_ANGLE_SPACE_ENGINE,
    workers: int | str | None = DEFAULT_ANGLE_SPACE_WORKERS,
) -> DetectorCakeResult:
    signal = np.asarray(image)
    if signal.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if float(distance_mm) <= 0.0:
        raise ValueError("distance_mm must be > 0.")
    if float(pixel_size_mm) <= 0.0:
        raise ValueError("pixel_size_mm must be > 0.")

    radial_bins = int(radial_bins if radial_bins is not None else signal.shape[1])
    azimuth_bins = int(azimuth_bins if azimuth_bins is not None else signal.shape[0])
    if radial_bins < 2 or azimuth_bins < 2:
        raise ValueError("radial_bins and azimuth_bins must each be >= 2.")

    geometry = DetectorCakeGeometry(
        pixel_size_m=float(pixel_size_mm) / 1000.0,
        distance_m=float(distance_mm) / 1000.0,
        center_row_px=float(center_row_px),
        center_col_px=float(center_col_px),
    )
    radial_deg, azimuthal_deg = build_angle_axes(
        npt_rad=radial_bins,
        npt_azim=azimuth_bins,
        tth_min_deg=two_theta_min_deg,
        tth_max_deg=two_theta_max_deg,
        azimuth_min_deg=phi_min_deg,
        azimuth_max_deg=phi_max_deg,
    )
    normalization = None
    if correct_solid_angle:
        normalization = flat_solid_angle_normalization(signal.shape, geometry)
    return integrate_detector_to_cake_exact(
        signal,
        radial_deg,
        azimuthal_deg,
        geometry,
        normalization=normalization,
        engine=engine,
        workers=workers,
    )


def prepare_gui_phi_display(
    result: DetectorCakeResult,
    *,
    phi_min_deg: float = DEFAULT_PHI_MIN_DEG,
    phi_max_deg: float = DEFAULT_PHI_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gui_phi = (
        (
            PHI_ZERO_OFFSET_DEGREES
            - np.asarray(result.azimuthal_deg, dtype=np.float64)
            + 180.0
        )
        % 360.0
    ) - 180.0
    order = np.argsort(gui_phi)
    gui_phi = gui_phi[order]
    cake = np.asarray(result.intensity, dtype=np.float64)[order, :]
    if float(phi_min_deg) <= float(phi_max_deg):
        mask = (gui_phi >= float(phi_min_deg)) & (gui_phi <= float(phi_max_deg))
    else:
        mask = (gui_phi >= float(phi_min_deg)) | (gui_phi <= float(phi_max_deg))
    return cake[mask, :], np.asarray(result.radial_deg, dtype=np.float64), gui_phi[mask]


__all__ = [
    "DEFAULT_ANGLE_SPACE_ENGINE",
    "DEFAULT_ANGLE_SPACE_WORKERS",
    "DEFAULT_PHI_MAX_DEG",
    "DEFAULT_PHI_MIN_DEG",
    "DEFAULT_TWO_THETA_MAX_DEG",
    "DEFAULT_TWO_THETA_MIN_DEG",
    "DetectorCakeGeometry",
    "DetectorCakeResult",
    "build_angle_axes",
    "convert_image_to_phi_2theta_space",
    "flat_solid_angle_normalization",
    "integrate_detector_to_cake_exact",
    "prepare_gui_phi_display",
]
