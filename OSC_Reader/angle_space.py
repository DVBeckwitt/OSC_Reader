"""Exact detector-to-angle-space conversion helpers."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import os
import threading

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
DEFAULT_ANGLE_SPACE_WORKERS = 8
DEFAULT_ANGLE_SPACE_ENGINE = "numba"
DEFAULT_TWO_THETA_MIN_DEG = 0.0
DEFAULT_TWO_THETA_MAX_DEG = 90.0
DEFAULT_PHI_MIN_DEG = -180.0
DEFAULT_PHI_MAX_DEG = 180.0
DEFAULT_GUI_PHI_MIN_DEG = -180.0
DEFAULT_GUI_PHI_MAX_DEG = 180.0
DEFAULT_PHI_ZERO_DIRECTION = "up"
PHI_ZERO_DIRECTIONS = ("up", "left", "down", "right")
_PHI_ZERO_DIRECTION_TO_AZIMUTH_DEG = {
    "right": 0.0,
    "down": 90.0,
    "left": 180.0,
    "up": -90.0,
}
_ANGLE_SPACE_WARMUP_LOCK = threading.Lock()
_ANGLE_SPACE_WARMED = False


@dataclass(frozen=True)
class DetectorCakeGeometry:
    pixel_size_m: float
    distance_m: float
    center_row_px: float
    center_col_px: float


@dataclass(frozen=True)
class DetectorCakeGeometryUncertainty:
    """Covariance of geometry parameters for coordinate calibration error.

    Parameter order is ``(center_row_px, center_col_px, pixel_size_m, distance_m)``.
    """

    covariance: np.ndarray

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.shape != (4, 4):
            raise ValueError(
                "geometry uncertainty covariance must have shape (4, 4) for "
                "(center_row_px, center_col_px, pixel_size_m, distance_m)."
            )
        cov = 0.5 * (cov + cov.T)
        if not np.all(np.isfinite(cov)):
            raise ValueError("geometry uncertainty covariance must be finite.")
        object.__setattr__(self, "covariance", cov)

    @classmethod
    def from_sigmas(
        cls,
        *,
        sigma_center_row_px: float = 0.0,
        sigma_center_col_px: float = 0.0,
        sigma_pixel_size_m: float = 0.0,
        sigma_distance_m: float = 0.0,
    ) -> "DetectorCakeGeometryUncertainty":
        return cls(
            covariance=np.diag(
                np.array(
                    [
                        float(sigma_center_row_px) ** 2,
                        float(sigma_center_col_px) ** 2,
                        float(sigma_pixel_size_m) ** 2,
                        float(sigma_distance_m) ** 2,
                    ],
                    dtype=np.float64,
                )
            )
        )


@dataclass(frozen=True)
class DetectorCakeCalibrationStats:
    radial_sigma_deg: np.ndarray
    azimuthal_sigma_deg: np.ndarray
    radial_total_sigma_deg: np.ndarray
    azimuthal_total_sigma_deg: np.ndarray
    radial_label_total_sigma_deg: np.ndarray
    azimuthal_label_total_sigma_deg: np.ndarray


@dataclass(frozen=True)
class DetectorCakeSubpixelErrorStats:
    refinement_grid: int
    radial_mean_error_deg: np.ndarray
    azimuthal_mean_error_deg: np.ndarray
    radial_sigma_error_deg: np.ndarray
    azimuthal_sigma_error_deg: np.ndarray
    radial_label_sigma_error_deg: np.ndarray
    azimuthal_label_sigma_error_deg: np.ndarray


@dataclass(frozen=True)
class DetectorCakeCoordinateStats:
    area_deg2: np.ndarray
    radial_mean_deg: np.ndarray
    azimuthal_mean_deg: np.ndarray
    radial_sigma_deg: np.ndarray
    azimuthal_sigma_deg: np.ndarray
    radial_label_sigma_deg: np.ndarray
    azimuthal_label_sigma_deg: np.ndarray
    calibration: DetectorCakeCalibrationStats | None = None
    subpixel_error: DetectorCakeSubpixelErrorStats | None = None


@dataclass(frozen=True)
class DetectorCakeResult:
    radial_deg: np.ndarray
    azimuthal_deg: np.ndarray
    intensity: np.ndarray
    sum_signal: np.ndarray
    sum_normalization: np.ndarray
    count: np.ndarray
    coordinate_stats: DetectorCakeCoordinateStats | None = None
    intensity_sem: np.ndarray | None = None


@dataclass(frozen=True)
class QSpaceResult:
    qr: np.ndarray
    qz: np.ndarray
    intensity: np.ndarray


# Coordinate statistics depend on geometry/binning, not detector intensities.
_COORDINATE_STATISTICS_CACHE_MAXSIZE = 4
_COORDINATE_STATISTICS_CACHE_LOCK = threading.Lock()
_COORDINATE_STATISTICS_CACHE: OrderedDict[
    tuple[object, ...],
    DetectorCakeCoordinateStats,
] = OrderedDict()


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


def _hash_array_contents(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _mask_cache_key(mask: np.ndarray | None, shape: tuple[int, int]) -> str | None:
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.shape != shape:
        raise ValueError("mask must match image shape.")
    return _hash_array_contents(np.asarray(mask_array != 0, dtype=np.uint8))


def _selection_cache_key(
    shape: tuple[int, int],
    rows: np.ndarray | None,
    cols: np.ndarray | None,
) -> tuple[str, str] | None:
    rows_array, cols_array, use_selection = _prepare_selection(shape, rows, cols)
    if not use_selection:
        return None
    order = np.lexsort((cols_array, rows_array))
    return (
        _hash_array_contents(rows_array[order].astype(np.int64, copy=False)),
        _hash_array_contents(cols_array[order].astype(np.int64, copy=False)),
    )


def _geometry_uncertainty_cache_key(
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None,
) -> str | None:
    if geometry_uncertainty is None:
        return None
    return _hash_array_contents(np.asarray(geometry_uncertainty.covariance, dtype=np.float64))


def _coordinate_statistics_cache_key(
    image_or_shape: np.ndarray | tuple[int, int] | list[int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None = None,
    numerical_subpixel_grid: int | None = None,
) -> tuple[object, ...]:
    shape = _resolve_image_shape(image_or_shape)
    radial = np.asarray(radial_deg, dtype=np.float64)
    azimuthal = np.asarray(azimuthal_deg, dtype=np.float64)
    return (
        shape,
        _hash_array_contents(radial),
        _hash_array_contents(azimuthal),
        float(geometry.distance_m),
        float(geometry.pixel_size_m),
        float(geometry.center_row_px),
        float(geometry.center_col_px),
        _mask_cache_key(mask, shape),
        _selection_cache_key(shape, rows, cols),
        _geometry_uncertainty_cache_key(geometry_uncertainty),
        1 if numerical_subpixel_grid is None else int(numerical_subpixel_grid),
    )


def _get_cached_coordinate_statistics(
    cache_key: tuple[object, ...],
) -> DetectorCakeCoordinateStats | None:
    with _COORDINATE_STATISTICS_CACHE_LOCK:
        cached = _COORDINATE_STATISTICS_CACHE.get(cache_key)
        if cached is None:
            return None
        _COORDINATE_STATISTICS_CACHE.move_to_end(cache_key)
        return cached


def _store_cached_coordinate_statistics(
    cache_key: tuple[object, ...],
    stats: DetectorCakeCoordinateStats,
) -> DetectorCakeCoordinateStats:
    with _COORDINATE_STATISTICS_CACHE_LOCK:
        _COORDINATE_STATISTICS_CACHE[cache_key] = stats
        _COORDINATE_STATISTICS_CACHE.move_to_end(cache_key)
        while len(_COORDINATE_STATISTICS_CACHE) > _COORDINATE_STATISTICS_CACHE_MAXSIZE:
            _COORDINATE_STATISTICS_CACHE.popitem(last=False)
    return stats


def _wrap_signed_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _normalize_phi_zero_direction(zero_direction: str) -> str:
    direction = str(zero_direction).strip().lower()
    if direction not in _PHI_ZERO_DIRECTION_TO_AZIMUTH_DEG:
        choices = ", ".join(option.title() for option in PHI_ZERO_DIRECTIONS)
        raise ValueError(f"zero_direction must be one of: {choices}.")
    return direction


def _phi_zero_azimuth_deg(zero_direction: str) -> float:
    return float(
        _PHI_ZERO_DIRECTION_TO_AZIMUTH_DEG[
            _normalize_phi_zero_direction(zero_direction)
        ]
    )


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
    cpu_limit = int(os.cpu_count() or resolved)
    return max(1, min(resolved, int(work_items), cpu_limit))


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


def _clip_polygon_halfplane(
    polygon: list[tuple[float, float]],
    *,
    axis: int,
    bound: float,
    keep_lower: bool,
) -> list[tuple[float, float]]:
    if len(polygon) < 3:
        return []

    def _inside(point: tuple[float, float]) -> bool:
        value = point[axis]
        return value <= bound if keep_lower else value >= bound

    def _intersect(
        start: tuple[float, float],
        stop: tuple[float, float],
    ) -> tuple[float, float]:
        sx, sy = start
        ex, ey = stop
        delta = (ex - sx) if axis == 0 else (ey - sy)
        if delta == 0.0:
            return (float(bound if axis == 0 else sx), float(bound if axis == 1 else sy))
        factor = ((bound - sx) / delta) if axis == 0 else ((bound - sy) / delta)
        if axis == 0:
            return float(bound), float(sy + factor * (ey - sy))
        return float(sx + factor * (ex - sx)), float(bound)

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = _inside(previous)
    for current in polygon:
        current_inside = _inside(current)
        if current_inside:
            if not previous_inside:
                output.append(_intersect(previous, current))
            output.append((float(current[0]), float(current[1])))
        elif previous_inside:
            output.append(_intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _clip_polygon_rectangle(
    polygon: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    clipped = _clip_polygon_halfplane(polygon, axis=0, bound=float(x_min), keep_lower=False)
    if len(clipped) < 3:
        return []
    clipped = _clip_polygon_halfplane(clipped, axis=0, bound=float(x_max), keep_lower=True)
    if len(clipped) < 3:
        return []
    clipped = _clip_polygon_halfplane(clipped, axis=1, bound=float(y_min), keep_lower=False)
    if len(clipped) < 3:
        return []
    clipped = _clip_polygon_halfplane(clipped, axis=1, bound=float(y_max), keep_lower=True)
    if len(clipped) < 3:
        return []
    return clipped


def _polygon_area_axis_moments(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float, float]:
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    signed_area = 0.0
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        signed_area += x0 * y1 - x1 * y0
    signed_area *= 0.5
    if (not math.isfinite(signed_area)) or abs(signed_area) <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if signed_area < 0.0:
        polygon = list(reversed(polygon))
    area = 0.0
    first_x = 0.0
    first_y = 0.0
    second_x = 0.0
    second_y = 0.0
    for index, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(index + 1) % len(polygon)]
        cross = x0 * y1 - x1 * y0
        area += cross
        first_x += (x0 + x1) * cross
        first_y += (y0 + y1) * cross
        second_x += (x0 * x0 + x0 * x1 + x1 * x1) * cross
        second_y += (y0 * y0 + y0 * y1 + y1 * y1) * cross
    area *= 0.5
    if area <= 0.0 or (not math.isfinite(area)):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return area, first_x / 6.0, first_y / 6.0, second_x / 12.0, second_y / 12.0


def _accumulate_coordinate_stats_quad(
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    area_deg2: np.ndarray,
    radial_first_moment: np.ndarray,
    azimuthal_first_moment: np.ndarray,
    radial_second_moment: np.ndarray,
    azimuthal_second_moment: np.ndarray,
) -> None:
    delta0 = float(radial[1] - radial[0])
    delta1 = float(azimuthal[1] - azimuthal[0])
    pos0_min = float(radial[0] - 0.5 * delta0)
    pos1_min = float(azimuthal[0] - 0.5 * delta1)
    pos0_max = float(radial[-1] + 0.5 * delta0)
    pos1_max = float(azimuthal[-1] + 0.5 * delta1)
    pos0_maxin = _inverse_calc_upper_bound(pos0_max)
    pos1_maxin = _inverse_calc_upper_bound(pos1_max)
    a0, a1 = _corner_to_polar_deg(y0, x0, distance)
    b0, b1 = _corner_to_polar_deg(y1, x0, distance)
    c0, c1 = _corner_to_polar_deg(y1, x1, distance)
    d0, d1 = _corner_to_polar_deg(y0, x1, distance)
    area = _area4p(a0, a1, b0, b1, c0, c1, d0, d1)
    if 360.0 > 0.0 and area > 0.0:
        a1 = _recenter_helper(a1, 360.0, True)
        b1 = _recenter_helper(b1, 360.0, True)
        c1 = _recenter_helper(c1, 360.0, True)
        d1 = _recenter_helper(d1, 360.0, True)
        center1 = 0.25 * (a1 + b1 + c1 + d1)
        if center1 > 180.0:
            a1 -= 360.0
            b1 -= 360.0
            c1 -= 360.0
            d1 -= 360.0
    polygon_abs = [
        (
            (min(max(a0, pos0_min), pos0_maxin) - pos0_min) / delta0,
            (min(max(a1, pos1_min), pos1_maxin) - pos1_min) / delta1,
        ),
        (
            (min(max(b0, pos0_min), pos0_maxin) - pos0_min) / delta0,
            (min(max(b1, pos1_min), pos1_maxin) - pos1_min) / delta1,
        ),
        (
            (min(max(c0, pos0_min), pos0_maxin) - pos0_min) / delta0,
            (min(max(c1, pos1_min), pos1_maxin) - pos1_min) / delta1,
        ),
        (
            (min(max(d0, pos0_min), pos0_maxin) - pos0_min) / delta0,
            (min(max(d1, pos1_min), pos1_maxin) - pos1_min) / delta1,
        ),
    ]
    min0 = min(point[0] for point in polygon_abs)
    max0 = max(point[0] for point in polygon_abs)
    min1 = min(point[1] for point in polygon_abs)
    max1 = max(point[1] for point in polygon_abs)
    foffset0 = math.floor(min0)
    foffset1 = math.floor(min1)
    width0 = int(math.ceil(max0) - foffset0)
    width1 = int(math.ceil(max1) - foffset1)
    if width0 <= 0 or width1 <= 0:
        return
    ioffset0 = int(foffset0)
    ioffset1 = int(foffset1)
    scale_area = delta0 * delta1
    for local0 in range(width0):
        bin_rad = ioffset0 + local0
        if bin_rad < 0 or bin_rad >= radial.size:
            continue
        cell_min0 = float(bin_rad)
        cell_max0 = float(bin_rad + 1)
        for local1 in range(width1):
            bin_az = ioffset1 + local1
            if bin_az < 0 or bin_az >= azimuthal.size:
                continue
            clipped = _clip_polygon_rectangle(
                polygon_abs,
                cell_min0,
                cell_max0,
                float(bin_az),
                float(bin_az + 1),
            )
            area_uv, first_u, first_v, second_u, second_v = _polygon_area_axis_moments(clipped)
            if area_uv <= 0.0:
                continue
            scaled_area = scale_area * area_uv
            radial_first = scale_area * (pos0_min * area_uv + delta0 * first_u)
            azimuthal_first = scale_area * (pos1_min * area_uv + delta1 * first_v)
            radial_second = scale_area * (
                (pos0_min * pos0_min * area_uv)
                + (2.0 * pos0_min * delta0 * first_u)
                + (delta0 * delta0 * second_u)
            )
            azimuthal_second = scale_area * (
                (pos1_min * pos1_min * area_uv)
                + (2.0 * pos1_min * delta1 * first_v)
                + (delta1 * delta1 * second_v)
            )
            area_deg2[bin_az, bin_rad] += scaled_area
            radial_first_moment[bin_az, bin_rad] += radial_first
            azimuthal_first_moment[bin_az, bin_rad] += azimuthal_first
            radial_second_moment[bin_az, bin_rad] += radial_second
            azimuthal_second_moment[bin_az, bin_rad] += azimuthal_second


def _accumulate_coordinate_stats_pixel(
    row: int,
    col: int,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    distance: float,
    radial: np.ndarray,
    azimuthal: np.ndarray,
    area_deg2: np.ndarray,
    radial_first_moment: np.ndarray,
    azimuthal_first_moment: np.ndarray,
    radial_second_moment: np.ndarray,
    azimuthal_second_moment: np.ndarray,
    *,
    subdivide_pixels: int = 1,
) -> None:
    y0 = float(row_edges[row])
    y1 = float(row_edges[row + 1])
    x0 = float(col_edges[col])
    x1 = float(col_edges[col + 1])
    subdivisions = max(1, int(subdivide_pixels))
    if subdivisions == 1:
        _accumulate_coordinate_stats_quad(
            y0,
            y1,
            x0,
            x1,
            distance,
            radial,
            azimuthal,
            area_deg2,
            radial_first_moment,
            azimuthal_first_moment,
            radial_second_moment,
            azimuthal_second_moment,
        )
        return
    dy = (y1 - y0) / float(subdivisions)
    dx = (x1 - x0) / float(subdivisions)
    for sub_row in range(subdivisions):
        sub_y0 = y0 + float(sub_row) * dy
        sub_y1 = sub_y0 + dy
        for sub_col in range(subdivisions):
            sub_x0 = x0 + float(sub_col) * dx
            sub_x1 = sub_x0 + dx
            _accumulate_coordinate_stats_quad(
                sub_y0,
                sub_y1,
                sub_x0,
                sub_x1,
                distance,
                radial,
                azimuthal,
                area_deg2,
                radial_first_moment,
                azimuthal_first_moment,
                radial_second_moment,
                azimuthal_second_moment,
            )


def _resolve_image_shape(
    image_or_shape: np.ndarray | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(image_or_shape, (tuple, list)):
        if len(image_or_shape) != 2:
            raise ValueError("image_or_shape must provide exactly two dimensions.")
        return int(image_or_shape[0]), int(image_or_shape[1])
    image = np.asarray(image_or_shape)
    if image.ndim != 2:
        raise ValueError("image_or_shape must be a 2D array or a (rows, cols) shape tuple.")
    return int(image.shape[0]), int(image.shape[1])


def _compute_coordinate_moment_arrays(
    image_or_shape: np.ndarray | tuple[int, int] | list[int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    subdivide_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = _resolve_image_shape(image_or_shape)
    radial = np.asarray(radial_deg, dtype=np.float64)
    azimuthal = np.asarray(azimuthal_deg, dtype=np.float64)
    _validate_axes(radial, azimuthal)
    if mask is None:
        mask_array = np.zeros((1, 1), dtype=np.int8)
        has_mask = False
    else:
        mask_array = np.asarray(mask, dtype=np.int8)
        if mask_array.shape != shape:
            raise ValueError("mask must match image shape.")
        has_mask = True
    rows_array, cols_array, use_selection = _prepare_selection(shape, rows, cols)
    row_edges, col_edges = _row_col_edges(shape, geometry)
    area_deg2 = np.zeros((azimuthal.size, radial.size), dtype=np.float64)
    radial_first_moment = np.zeros_like(area_deg2)
    azimuthal_first_moment = np.zeros_like(area_deg2)
    radial_second_moment = np.zeros_like(area_deg2)
    azimuthal_second_moment = np.zeros_like(area_deg2)
    subdivisions = max(1, int(subdivide_pixels))

    if use_selection:
        for selection_index in range(int(rows_array.size)):
            row = int(rows_array[selection_index])
            col = int(cols_array[selection_index])
            if has_mask and mask_array[row, col] != 0:
                continue
            _accumulate_coordinate_stats_pixel(
                row,
                col,
                row_edges,
                col_edges,
                float(geometry.distance_m),
                radial,
                azimuthal,
                area_deg2,
                radial_first_moment,
                azimuthal_first_moment,
                radial_second_moment,
                azimuthal_second_moment,
                subdivide_pixels=subdivisions,
            )
    else:
        for row in range(shape[0]):
            for col in range(shape[1]):
                if has_mask and mask_array[row, col] != 0:
                    continue
                _accumulate_coordinate_stats_pixel(
                    row,
                    col,
                    row_edges,
                    col_edges,
                    float(geometry.distance_m),
                    radial,
                    azimuthal,
                    area_deg2,
                    radial_first_moment,
                    azimuthal_first_moment,
                    radial_second_moment,
                    azimuthal_second_moment,
                    subdivide_pixels=subdivisions,
                )
    return (
        area_deg2,
        radial_first_moment,
        azimuthal_first_moment,
        radial_second_moment,
        azimuthal_second_moment,
    )


def _finalize_coordinate_statistics(
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    area_deg2: np.ndarray,
    radial_first_moment: np.ndarray,
    azimuthal_first_moment: np.ndarray,
    radial_second_moment: np.ndarray,
    azimuthal_second_moment: np.ndarray,
    *,
    calibration: DetectorCakeCalibrationStats | None = None,
    subpixel_error: DetectorCakeSubpixelErrorStats | None = None,
) -> DetectorCakeCoordinateStats:
    radial = np.asarray(radial_deg, dtype=np.float64)
    azimuthal = np.asarray(azimuthal_deg, dtype=np.float64)
    radial_mean = np.full_like(area_deg2, np.nan, dtype=np.float64)
    azimuthal_mean = np.full_like(area_deg2, np.nan, dtype=np.float64)
    radial_sigma = np.full_like(area_deg2, np.nan, dtype=np.float64)
    azimuthal_sigma = np.full_like(area_deg2, np.nan, dtype=np.float64)
    radial_label_sigma = np.full_like(area_deg2, np.nan, dtype=np.float64)
    azimuthal_label_sigma = np.full_like(area_deg2, np.nan, dtype=np.float64)
    valid = area_deg2 > 0.0
    if np.any(valid):
        radial_mean[valid] = radial_first_moment[valid] / area_deg2[valid]
        azimuthal_mean[valid] = azimuthal_first_moment[valid] / area_deg2[valid]
        radial_variance = (radial_second_moment[valid] / area_deg2[valid]) - (radial_mean[valid] ** 2)
        azimuthal_variance = (azimuthal_second_moment[valid] / area_deg2[valid]) - (azimuthal_mean[valid] ** 2)
        radial_sigma[valid] = np.sqrt(np.maximum(radial_variance, 0.0))
        azimuthal_sigma[valid] = np.sqrt(np.maximum(azimuthal_variance, 0.0))
        radial_centers_2d = np.broadcast_to(radial[np.newaxis, :], area_deg2.shape)
        azimuthal_centers_2d = np.broadcast_to(azimuthal[:, np.newaxis], area_deg2.shape)
        radial_label_variance = (
            (radial_second_moment[valid] / area_deg2[valid])
            - (2.0 * radial_centers_2d[valid] * radial_first_moment[valid] / area_deg2[valid])
            + (radial_centers_2d[valid] ** 2)
        )
        azimuthal_label_variance = (
            (azimuthal_second_moment[valid] / area_deg2[valid])
            - (2.0 * azimuthal_centers_2d[valid] * azimuthal_first_moment[valid] / area_deg2[valid])
            + (azimuthal_centers_2d[valid] ** 2)
        )
        radial_label_sigma[valid] = np.sqrt(np.maximum(radial_label_variance, 0.0))
        azimuthal_label_sigma[valid] = np.sqrt(np.maximum(azimuthal_label_variance, 0.0))
    return DetectorCakeCoordinateStats(
        area_deg2=area_deg2,
        radial_mean_deg=radial_mean,
        azimuthal_mean_deg=azimuthal_mean,
        radial_sigma_deg=radial_sigma,
        azimuthal_sigma_deg=azimuthal_sigma,
        radial_label_sigma_deg=radial_label_sigma,
        azimuthal_label_sigma_deg=azimuthal_label_sigma,
        calibration=calibration,
        subpixel_error=subpixel_error,
    )


def _analytic_coordinate_sigma_deg(
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    radial_eval, azimuthal_eval = np.broadcast_arrays(
        np.asarray(radial_deg, dtype=np.float64),
        np.asarray(azimuthal_deg, dtype=np.float64),
    )
    radial_sigma = np.full_like(radial_eval, np.nan, dtype=np.float64)
    azimuthal_sigma = np.full_like(radial_eval, np.nan, dtype=np.float64)
    valid = np.isfinite(radial_eval) & np.isfinite(azimuthal_eval)
    if not np.any(valid):
        return radial_sigma, azimuthal_sigma

    sigma_tth, sigma_phi = fast_display_sigma_profiles(
        radial_eval[valid],
        pixel_size_m=geometry.pixel_size_m,
        distance_m=geometry.distance_m,
    )
    radial_sigma[valid] = sigma_tth.astype(np.float64, copy=False)
    azimuthal_sigma[valid] = sigma_phi.astype(np.float64, copy=False)
    return radial_sigma, azimuthal_sigma


def fast_display_sigma_profiles(
    radial_deg: np.ndarray,
    *,
    pixel_size_m: float,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(np.asarray(radial_deg, dtype=np.float64))
    sigma_tth = np.full_like(theta, np.nan, dtype=np.float64)
    sigma_phi = np.full_like(theta, np.nan, dtype=np.float64)

    distance = float(distance_m)
    pixel_size = float(pixel_size_m)
    valid = (
        np.isfinite(theta)
        & np.isfinite(distance)
        & np.isfinite(pixel_size)
        & (distance > 0.0)
        & (pixel_size > 0.0)
    )
    if not np.any(valid):
        return sigma_tth.astype(np.float32), sigma_phi.astype(np.float32)

    prefactor = np.rad2deg(pixel_size / (np.sqrt(12.0) * distance))
    valid_theta = theta[valid]
    sigma_tth[valid] = prefactor * np.cos(valid_theta) ** 2

    tan_theta = np.tan(valid_theta)
    phi_valid = np.abs(tan_theta) > 1.0e-15
    if np.any(phi_valid):
        valid_indices = np.flatnonzero(valid.reshape(-1))
        sigma_phi_flat = sigma_phi.reshape(-1)
        sigma_phi_flat[valid_indices[phi_valid]] = prefactor / np.abs(tan_theta[phi_valid])
    return sigma_tth.astype(np.float32), sigma_phi.astype(np.float32)


def fast_display_sigma_maps(
    result: DetectorCakeResult,
    *,
    pixel_size_m: float,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_tth_profile, sigma_phi_profile = fast_display_sigma_profiles(
        result.radial_deg,
        pixel_size_m=pixel_size_m,
        distance_m=distance_m,
    )
    display_shape = np.asarray(result.intensity).shape
    sigma_tth = np.array(
        np.broadcast_to(sigma_tth_profile[np.newaxis, :], display_shape),
        dtype=np.float32,
        copy=True,
    )
    sigma_phi = np.array(
        np.broadcast_to(sigma_phi_profile[np.newaxis, :], display_shape),
        dtype=np.float32,
        copy=True,
    )
    invalid = (
        ~np.isfinite(np.asarray(result.sum_normalization, dtype=np.float64))
        | (np.asarray(result.sum_normalization, dtype=np.float64) <= 0.0)
    )
    sigma_tth[invalid] = np.nan
    sigma_phi[invalid] = np.nan
    return sigma_tth, sigma_phi


def _apply_analytic_coordinate_uncertainty(
    stats: DetectorCakeCoordinateStats,
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    calibration: DetectorCakeCalibrationStats | None = None,
    subpixel_error: DetectorCakeSubpixelErrorStats | None = None,
) -> DetectorCakeCoordinateStats:
    radial_centers = np.broadcast_to(
        np.asarray(radial_deg, dtype=np.float64)[None, :],
        stats.area_deg2.shape,
    )
    azimuthal_centers = np.broadcast_to(
        np.asarray(azimuthal_deg, dtype=np.float64)[:, None],
        stats.area_deg2.shape,
    )
    radial_eval = np.where(
        np.isfinite(stats.radial_mean_deg),
        stats.radial_mean_deg,
        radial_centers,
    )
    azimuthal_eval = np.where(
        np.isfinite(stats.azimuthal_mean_deg),
        stats.azimuthal_mean_deg,
        azimuthal_centers,
    )
    radial_sigma, azimuthal_sigma = _analytic_coordinate_sigma_deg(
        radial_eval,
        azimuthal_eval,
        geometry,
    )
    radial_label_sigma, azimuthal_label_sigma = _analytic_coordinate_sigma_deg(
        radial_centers,
        azimuthal_centers,
        geometry,
    )
    valid = stats.area_deg2 > 0.0
    radial_sigma[~valid] = np.nan
    azimuthal_sigma[~valid] = np.nan
    radial_label_sigma[~valid] = np.nan
    azimuthal_label_sigma[~valid] = np.nan
    return DetectorCakeCoordinateStats(
        area_deg2=stats.area_deg2,
        radial_mean_deg=stats.radial_mean_deg,
        azimuthal_mean_deg=stats.azimuthal_mean_deg,
        radial_sigma_deg=radial_sigma,
        azimuthal_sigma_deg=azimuthal_sigma,
        radial_label_sigma_deg=radial_label_sigma,
        azimuthal_label_sigma_deg=azimuthal_label_sigma,
        calibration=calibration,
        subpixel_error=subpixel_error,
    )


def _compute_calibration_coordinate_statistics(
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    coordinate_stats: DetectorCakeCoordinateStats,
    geometry_uncertainty: DetectorCakeGeometryUncertainty,
) -> DetectorCakeCalibrationStats:
    radial_centers = np.broadcast_to(
        np.asarray(radial_deg, dtype=np.float64)[None, :],
        coordinate_stats.area_deg2.shape,
    )
    azimuthal_centers = np.broadcast_to(
        np.asarray(azimuthal_deg, dtype=np.float64)[:, None],
        coordinate_stats.area_deg2.shape,
    )
    radial_eval = np.where(
        np.isfinite(coordinate_stats.radial_mean_deg),
        coordinate_stats.radial_mean_deg,
        radial_centers,
    )
    azimuthal_eval = np.where(
        np.isfinite(coordinate_stats.azimuthal_mean_deg),
        coordinate_stats.azimuthal_mean_deg,
        azimuthal_centers,
    )

    t_rad = np.deg2rad(radial_eval)
    phi_rad = np.deg2rad(azimuthal_eval)
    distance = float(geometry.distance_m)
    pixel_size = float(geometry.pixel_size_m)
    rho = distance * np.tan(t_rad)
    x = rho * np.cos(phi_rad)
    y = rho * np.sin(phi_rad)
    rho2 = x * x + y * y
    denom = distance * distance + rho2

    radial_sigma = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)
    azimuthal_sigma = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)
    radial_total = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)
    azimuthal_total = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)
    radial_label_total = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)
    azimuthal_label_total = np.full_like(coordinate_stats.area_deg2, np.nan, dtype=np.float64)

    valid = (
        (coordinate_stats.area_deg2 > 0.0)
        & np.isfinite(rho)
        & np.isfinite(denom)
        & (rho > 1.0e-15)
        & (denom > 0.0)
        & np.isfinite(pixel_size)
        & (pixel_size > 0.0)
        & np.isfinite(distance)
        & (distance > 0.0)
    )
    if np.any(valid):
        dtdx = np.zeros_like(coordinate_stats.area_deg2, dtype=np.float64)
        dtdy = np.zeros_like(coordinate_stats.area_deg2, dtype=np.float64)
        dtdd = np.zeros_like(coordinate_stats.area_deg2, dtype=np.float64)
        dphidx = np.zeros_like(coordinate_stats.area_deg2, dtype=np.float64)
        dphidy = np.zeros_like(coordinate_stats.area_deg2, dtype=np.float64)
        dtdx[valid] = distance * x[valid] / (rho[valid] * denom[valid])
        dtdy[valid] = distance * y[valid] / (rho[valid] * denom[valid])
        dtdd[valid] = -rho[valid] / denom[valid]
        dphidx[valid] = -y[valid] / rho2[valid]
        dphidy[valid] = x[valid] / rho2[valid]

        jt = np.zeros(coordinate_stats.area_deg2.shape + (4,), dtype=np.float64)
        jphi = np.zeros_like(jt)
        jt[..., 0] = dtdy * pixel_size
        jt[..., 1] = -dtdx * pixel_size
        jt[..., 2] = (dtdx * x + dtdy * y) / pixel_size
        jt[..., 3] = dtdd
        jphi[..., 0] = dphidy * pixel_size
        jphi[..., 1] = -dphidx * pixel_size
        jt_deg = np.rad2deg(jt)
        jphi_deg = np.rad2deg(jphi)

        covariance = np.asarray(geometry_uncertainty.covariance, dtype=np.float64)
        jt_valid = jt_deg[valid]
        jphi_valid = jphi_deg[valid]
        radial_var = np.einsum("ni,ij,nj->n", jt_valid, covariance, jt_valid)
        azimuthal_var = np.einsum("ni,ij,nj->n", jphi_valid, covariance, jphi_valid)
        radial_sigma[valid] = np.sqrt(np.maximum(radial_var, 0.0))
        azimuthal_sigma[valid] = np.sqrt(np.maximum(azimuthal_var, 0.0))

        combine = valid & np.isfinite(coordinate_stats.radial_sigma_deg) & np.isfinite(radial_sigma)
        radial_total[combine] = np.sqrt(coordinate_stats.radial_sigma_deg[combine] ** 2 + radial_sigma[combine] ** 2)
        combine = valid & np.isfinite(coordinate_stats.azimuthal_sigma_deg) & np.isfinite(azimuthal_sigma)
        azimuthal_total[combine] = np.sqrt(
            coordinate_stats.azimuthal_sigma_deg[combine] ** 2 + azimuthal_sigma[combine] ** 2
        )
        combine = valid & np.isfinite(coordinate_stats.radial_label_sigma_deg) & np.isfinite(radial_sigma)
        radial_label_total[combine] = np.sqrt(
            coordinate_stats.radial_label_sigma_deg[combine] ** 2 + radial_sigma[combine] ** 2
        )
        combine = valid & np.isfinite(coordinate_stats.azimuthal_label_sigma_deg) & np.isfinite(azimuthal_sigma)
        azimuthal_label_total[combine] = np.sqrt(
            coordinate_stats.azimuthal_label_sigma_deg[combine] ** 2 + azimuthal_sigma[combine] ** 2
        )

    return DetectorCakeCalibrationStats(
        radial_sigma_deg=radial_sigma,
        azimuthal_sigma_deg=azimuthal_sigma,
        radial_total_sigma_deg=radial_total,
        azimuthal_total_sigma_deg=azimuthal_total,
        radial_label_total_sigma_deg=radial_label_total,
        azimuthal_label_total_sigma_deg=azimuthal_label_total,
    )


def _compute_subpixel_refinement_error(
    base_stats: DetectorCakeCoordinateStats,
    refined_stats: DetectorCakeCoordinateStats,
    *,
    refinement_grid: int,
) -> DetectorCakeSubpixelErrorStats:
    return DetectorCakeSubpixelErrorStats(
        refinement_grid=int(refinement_grid),
        radial_mean_error_deg=np.abs(refined_stats.radial_mean_deg - base_stats.radial_mean_deg),
        azimuthal_mean_error_deg=np.abs(refined_stats.azimuthal_mean_deg - base_stats.azimuthal_mean_deg),
        radial_sigma_error_deg=np.abs(refined_stats.radial_sigma_deg - base_stats.radial_sigma_deg),
        azimuthal_sigma_error_deg=np.abs(refined_stats.azimuthal_sigma_deg - base_stats.azimuthal_sigma_deg),
        radial_label_sigma_error_deg=np.abs(
            refined_stats.radial_label_sigma_deg - base_stats.radial_label_sigma_deg
        ),
        azimuthal_label_sigma_error_deg=np.abs(
            refined_stats.azimuthal_label_sigma_deg - base_stats.azimuthal_label_sigma_deg
        ),
    )


def _compute_intensity_sem(
    sum_normalization: np.ndarray,
    count: np.ndarray,
    *,
    variance_per_effective_pixel: float,
) -> np.ndarray:
    sem = np.full_like(sum_normalization, np.nan, dtype=np.float64)
    variance = float(variance_per_effective_pixel)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("variance_per_effective_pixel must be finite and >= 0.")
    valid = (
        np.isfinite(sum_normalization)
        & np.isfinite(count)
        & (sum_normalization > 0.0)
        & (count > 0.0)
    )
    if np.any(valid):
        sem[valid] = np.sqrt(variance * count[valid]) / sum_normalization[valid]
    return sem


def _compute_detector_to_cake_coordinate_statistics_uncached(
    image_or_shape: np.ndarray | tuple[int, int] | list[int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None = None,
    numerical_subpixel_grid: int | None = None,
) -> DetectorCakeCoordinateStats:
    base_moments = _compute_coordinate_moment_arrays(
        image_or_shape,
        radial_deg,
        azimuthal_deg,
        geometry,
        mask=mask,
        rows=rows,
        cols=cols,
        subdivide_pixels=1,
    )
    base_polygon_stats = _finalize_coordinate_statistics(radial_deg, azimuthal_deg, *base_moments)
    base_stats = _apply_analytic_coordinate_uncertainty(
        base_polygon_stats,
        radial_deg,
        azimuthal_deg,
        geometry,
    )

    calibration_stats = None
    if geometry_uncertainty is not None:
        calibration_stats = _compute_calibration_coordinate_statistics(
            radial_deg,
            azimuthal_deg,
            geometry,
            base_stats,
            geometry_uncertainty,
        )

    subpixel_error = None
    refinement_grid = 1 if numerical_subpixel_grid is None else int(numerical_subpixel_grid)
    if refinement_grid > 1:
        refined_moments = _compute_coordinate_moment_arrays(
            image_or_shape,
            radial_deg,
            azimuthal_deg,
            geometry,
            mask=mask,
            rows=rows,
            cols=cols,
            subdivide_pixels=refinement_grid,
        )
        refined_stats = _finalize_coordinate_statistics(radial_deg, azimuthal_deg, *refined_moments)
        subpixel_error = _compute_subpixel_refinement_error(
            base_polygon_stats,
            refined_stats,
            refinement_grid=refinement_grid,
        )

    return _apply_analytic_coordinate_uncertainty(
        base_polygon_stats,
        radial_deg,
        azimuthal_deg,
        geometry,
        calibration=calibration_stats,
        subpixel_error=subpixel_error,
    )


def compute_detector_to_cake_coordinate_statistics(
    image_or_shape: np.ndarray | tuple[int, int] | list[int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    *,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None = None,
    numerical_subpixel_grid: int | None = None,
) -> DetectorCakeCoordinateStats:
    radial = np.asarray(radial_deg, dtype=np.float64)
    azimuthal = np.asarray(azimuthal_deg, dtype=np.float64)
    cache_key = _coordinate_statistics_cache_key(
        image_or_shape,
        radial,
        azimuthal,
        geometry,
        mask=mask,
        rows=rows,
        cols=cols,
        geometry_uncertainty=geometry_uncertainty,
        numerical_subpixel_grid=numerical_subpixel_grid,
    )
    cached = _get_cached_coordinate_statistics(cache_key)
    if cached is not None:
        return cached
    stats = _compute_detector_to_cake_coordinate_statistics_uncached(
        image_or_shape,
        radial,
        azimuthal,
        geometry,
        mask=mask,
        rows=rows,
        cols=cols,
        geometry_uncertainty=geometry_uncertainty,
        numerical_subpixel_grid=numerical_subpixel_grid,
    )
    return _store_cached_coordinate_statistics(cache_key, stats)


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
    compute_coordinate_statistics: bool = False,
    compute_intensity_sem: bool = False,
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None = None,
    numerical_subpixel_grid: int | None = None,
    intensity_variance_per_effective_pixel: float = 1.0,
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
    intensity_sem = None
    if compute_intensity_sem:
        intensity_sem = _compute_intensity_sem(
            sum_normalization,
            count,
            variance_per_effective_pixel=float(intensity_variance_per_effective_pixel),
        )
    coordinate_stats = None
    if compute_coordinate_statistics:
        coordinate_stats = compute_detector_to_cake_coordinate_statistics(
            signal.shape,
            radial,
            azimuthal,
            geometry,
            mask=mask_array if has_mask else None,
            rows=rows_array if use_selection else None,
            cols=cols_array if use_selection else None,
            geometry_uncertainty=geometry_uncertainty,
            numerical_subpixel_grid=numerical_subpixel_grid,
        )
    return DetectorCakeResult(
        radial_deg=np.array(radial, copy=True),
        azimuthal_deg=np.array(azimuthal, copy=True),
        intensity=intensity,
        sum_signal=sum_signal,
        sum_normalization=sum_normalization,
        count=count,
        intensity_sem=intensity_sem,
        coordinate_stats=coordinate_stats,
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
    compute_coordinate_statistics: bool = False,
    compute_intensity_sem: bool = False,
    geometry_uncertainty: DetectorCakeGeometryUncertainty | None = None,
    sigma_center_row_px: float = 0.0,
    sigma_center_col_px: float = 0.0,
    sigma_pixel_size_mm: float = 0.0,
    sigma_distance_mm: float = 0.0,
    numerical_subpixel_grid: int | None = None,
    intensity_variance_per_effective_pixel: float = 1.0,
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
    built_uncertainty = geometry_uncertainty
    provided_sigma = any(
        float(value) != 0.0
        for value in (
            sigma_center_row_px,
            sigma_center_col_px,
            sigma_pixel_size_mm,
            sigma_distance_mm,
        )
    )
    if built_uncertainty is not None and provided_sigma:
        raise ValueError(
            "Pass either geometry_uncertainty or explicit sigma_* values, not both."
        )
    if built_uncertainty is None and provided_sigma:
        built_uncertainty = DetectorCakeGeometryUncertainty.from_sigmas(
            sigma_center_row_px=float(sigma_center_row_px),
            sigma_center_col_px=float(sigma_center_col_px),
            sigma_pixel_size_m=float(sigma_pixel_size_mm) / 1000.0,
            sigma_distance_m=float(sigma_distance_mm) / 1000.0,
        )
    return integrate_detector_to_cake_exact(
        signal,
        radial_deg,
        azimuthal_deg,
        geometry,
        normalization=normalization,
        engine=engine,
        workers=workers,
        compute_coordinate_statistics=compute_coordinate_statistics,
        compute_intensity_sem=compute_intensity_sem,
        geometry_uncertainty=built_uncertainty,
        numerical_subpixel_grid=numerical_subpixel_grid,
        intensity_variance_per_effective_pixel=float(intensity_variance_per_effective_pixel),
    )


def warm_angle_space_engine(*, workers: int | str | None = 1) -> None:
    if not _HAS_NUMBA:
        return
    global _ANGLE_SPACE_WARMED
    if _ANGLE_SPACE_WARMED:
        return
    with _ANGLE_SPACE_WARMUP_LOCK:
        if _ANGLE_SPACE_WARMED:
            return
        dummy_shape = (32, 32)
        dummy = np.zeros(dummy_shape, dtype=np.float32)
        convert_image_to_phi_2theta_space(
            dummy,
            distance_mm=75.0,
            pixel_size_mm=0.1,
            center_row_px=(dummy_shape[0] - 1) / 2.0,
            center_col_px=(dummy_shape[1] - 1) / 2.0,
            radial_bins=32,
            azimuth_bins=32,
            workers=workers,
        )
        _ANGLE_SPACE_WARMED = True


def _prepare_gui_phi_indices(
    result: DetectorCakeResult,
    *,
    phi_min_deg: float,
    phi_max_deg: float,
    zero_direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gui_phi = _wrap_signed_degrees(
        _phi_zero_azimuth_deg(zero_direction)
        - np.asarray(result.azimuthal_deg, dtype=np.float64)
    )
    order = np.argsort(gui_phi)
    gui_phi = gui_phi[order]
    if float(phi_min_deg) <= float(phi_max_deg):
        mask = (gui_phi >= float(phi_min_deg)) & (gui_phi <= float(phi_max_deg))
    else:
        mask = (gui_phi >= float(phi_min_deg)) | (gui_phi <= float(phi_max_deg))
    return order, mask, gui_phi


def prepare_gui_phi_display_data(
    result: DetectorCakeResult,
    values: np.ndarray,
    *,
    phi_min_deg: float = DEFAULT_GUI_PHI_MIN_DEG,
    phi_max_deg: float = DEFAULT_GUI_PHI_MAX_DEG,
    zero_direction: str = DEFAULT_PHI_ZERO_DIRECTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    expected_shape = np.asarray(result.intensity).shape
    if data.shape != expected_shape:
        raise ValueError("values must match result.intensity shape.")
    order, mask, gui_phi = _prepare_gui_phi_indices(
        result,
        phi_min_deg=phi_min_deg,
        phi_max_deg=phi_max_deg,
        zero_direction=zero_direction,
    )
    return data[order, :][mask, :], np.asarray(result.radial_deg, dtype=np.float64), gui_phi[mask]


def prepare_gui_phi_display(
    result: DetectorCakeResult,
    *,
    phi_min_deg: float = DEFAULT_GUI_PHI_MIN_DEG,
    phi_max_deg: float = DEFAULT_GUI_PHI_MAX_DEG,
    zero_direction: str = DEFAULT_PHI_ZERO_DIRECTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return prepare_gui_phi_display_data(
        result,
        result.intensity,
        phi_min_deg=phi_min_deg,
        phi_max_deg=phi_max_deg,
        zero_direction=zero_direction,
    )


def convert_phi_2theta_to_qr_qz_space(
    intensity: np.ndarray,
    radial_deg: np.ndarray,
    phi_deg: np.ndarray,
    *,
    wavelength_angstrom: float = 1.5406,
    incident_angle_deg: float = 0.0,
    qr_bins: int | None = None,
    qz_bins: int | None = None,
) -> QSpaceResult:
    cake = np.asarray(intensity, dtype=np.float64)
    radial = np.asarray(radial_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)

    if cake.ndim != 2:
        raise ValueError("intensity must be a 2D array.")
    if radial.ndim != 1 or phi.ndim != 1:
        raise ValueError("radial_deg and phi_deg must be 1D arrays.")
    if cake.shape != (phi.size, radial.size):
        raise ValueError("intensity shape must match (len(phi_deg), len(radial_deg)).")

    qr_bins = int(radial.size if qr_bins is None else qr_bins)
    qz_bins = int(phi.size if qz_bins is None else qz_bins)
    if qr_bins < 2 or qz_bins < 2:
        raise ValueError("qr_bins and qz_bins must be at least 2.")

    wavelength = float(wavelength_angstrom)
    if wavelength <= 0.0:
        raise ValueError("wavelength_angstrom must be positive.")

    phi_rad = np.deg2rad(phi)[:, None]
    theta_rad = np.deg2rad(radial)[None, :]

    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    wavevector = (2.0 * np.pi) / wavelength
    incident_rad = np.deg2rad(float(incident_angle_deg))
    sin_incident = np.sin(incident_rad)
    cos_incident = np.cos(incident_rad)

    delta_cos = cos_theta - 1.0
    sin_theta_cos_phi = sin_theta * cos_phi
    qy = (cos_incident * delta_cos + sin_incident * sin_theta_cos_phi) * wavevector
    qz = (-sin_incident * delta_cos + cos_incident * sin_theta_cos_phi) * wavevector
    qr_mag = np.hypot((sin_theta * sin_phi) * wavevector, qy)
    qr = np.where(phi_rad >= 0.0, qr_mag, -qr_mag)

    qr_flat = qr.reshape(-1)
    qz_flat = qz.reshape(-1)
    intensity_flat = cake.reshape(-1)
    valid = (
        np.isfinite(qr_flat)
        & np.isfinite(qz_flat)
        & np.isfinite(intensity_flat)
    )
    if not np.any(valid):
        raise ValueError("No finite q-space samples were produced from the current angle-space image.")

    qr_flat = qr_flat[valid]
    qz_flat = qz_flat[valid]
    intensity_flat = intensity_flat[valid]

    qr_min = float(np.min(qr_flat))
    qr_max = float(np.max(qr_flat))
    qz_min = float(np.min(qz_flat))
    qz_max = float(np.max(qz_flat))
    if np.isclose(qr_min, qr_max):
        qr_pad = max(abs(qr_min) * 0.01, 1.0e-3)
        qr_min -= qr_pad
        qr_max += qr_pad
    if np.isclose(qz_min, qz_max):
        qz_pad = max(abs(qz_min) * 0.01, 1.0e-3)
        qz_min -= qz_pad
        qz_max += qz_pad

    qr_edges = np.linspace(qr_min, qr_max, qr_bins + 1, dtype=np.float64)
    qz_edges = np.linspace(qz_min, qz_max, qz_bins + 1, dtype=np.float64)
    weighted_sum, _, _ = np.histogram2d(
        qz_flat,
        qr_flat,
        bins=(qz_edges, qr_edges),
        weights=intensity_flat,
    )
    sample_count, _, _ = np.histogram2d(
        qz_flat,
        qr_flat,
        bins=(qz_edges, qr_edges),
    )
    rebinned = np.divide(
        weighted_sum,
        sample_count,
        out=np.zeros_like(weighted_sum, dtype=np.float64),
        where=sample_count > 0.0,
    )

    qr_centers = 0.5 * (qr_edges[:-1] + qr_edges[1:])
    qz_centers = 0.5 * (qz_edges[:-1] + qz_edges[1:])
    return QSpaceResult(qr=qr_centers, qz=qz_centers, intensity=rebinned)


__all__ = [
    "DEFAULT_ANGLE_SPACE_ENGINE",
    "DEFAULT_ANGLE_SPACE_WORKERS",
    "DEFAULT_GUI_PHI_MAX_DEG",
    "DEFAULT_GUI_PHI_MIN_DEG",
    "DEFAULT_PHI_MAX_DEG",
    "DEFAULT_PHI_MIN_DEG",
    "DEFAULT_PHI_ZERO_DIRECTION",
    "DEFAULT_TWO_THETA_MAX_DEG",
    "DEFAULT_TWO_THETA_MIN_DEG",
    "DetectorCakeCalibrationStats",
    "DetectorCakeCoordinateStats",
    "DetectorCakeGeometry",
    "DetectorCakeGeometryUncertainty",
    "DetectorCakeResult",
    "DetectorCakeSubpixelErrorStats",
    "QSpaceResult",
    "PHI_ZERO_DIRECTIONS",
    "build_angle_axes",
    "compute_detector_to_cake_coordinate_statistics",
    "convert_phi_2theta_to_qr_qz_space",
    "convert_image_to_phi_2theta_space",
    "flat_solid_angle_normalization",
    "fast_display_sigma_maps",
    "fast_display_sigma_profiles",
    "integrate_detector_to_cake_exact",
    "prepare_gui_phi_display",
    "warm_angle_space_engine",
]
