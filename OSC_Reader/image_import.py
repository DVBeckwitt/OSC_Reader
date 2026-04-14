"""Detector-image import helpers.

The project keeps its validated native Rigaku R-AXIS ``.osc`` parser as the
preferred path for those files and uses FabIO as a compatibility layer for
other common 2D detector formats.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np


COMMON_DETECTOR_EXTENSIONS = (
    ".osc",
    ".cbf",
    ".edf",
    ".img",
    ".mccd",
    ".sfrm",
    ".gfrm",
    ".tif",
    ".tiff",
    ".h5",
    ".hdf5",
    ".nxs",
)


@dataclass(frozen=True)
class DetectorImageLoadResult:
    data: np.ndarray
    path: str
    format_name: str
    loader_name: str
    frame_index: int = 0
    frame_count: int = 1
    header: Optional[Mapping[str, Any]] = None


def supported_detector_extensions() -> Tuple[str, ...]:
    """Return the curated extension list shown in file-pickers and docs."""
    return COMMON_DETECTOR_EXTENSIONS


def get_detector_file_dialog_filter() -> str:
    """Return a Qt file-dialog filter for common detector-image formats."""
    patterns = " ".join(f"*{suffix}" for suffix in COMMON_DETECTOR_EXTENSIONS)
    return f"Detector images ({patterns});;OSC files (*.osc);;All files (*)"


def _normalize_detector_array(data: Any, source: str) -> np.ndarray:
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError(f"Loaded detector image '{source}' has no pixel data.")

    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(
            f"Loaded detector image '{source}' is {squeezed.ndim}D after "
            "squeezing; only 2D detector frames are supported."
        )
    return squeezed


def _load_native_osc(path: str, frame_index: int) -> DetectorImageLoadResult:
    if frame_index != 0:
        raise IndexError("Native .osc loading only supports frame_index=0.")

    from .OSC_Reader import read_osc

    data = _normalize_detector_array(read_osc(path), source=path)
    return DetectorImageLoadResult(
        data=data,
        path=path,
        format_name="Rigaku RAXIS OSC",
        loader_name="native_osc",
        frame_index=0,
        frame_count=1,
        header={"format": "RAXIS"},
    )


def _import_fabio():
    try:
        import fabio
    except Exception as exc:
        raise ImportError(
            "FabIO is required to import non-.osc detector images. "
            "Install it with: pip install fabio"
        ) from exc
    return fabio


def _load_with_fabio(path: str, frame_index: int) -> DetectorImageLoadResult:
    fabio = _import_fabio()
    image = fabio.open(path)
    frame_count = int(getattr(image, "nframes", 1) or 1)

    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(
            f"frame_index={frame_index} is out of range for '{path}' "
            f"(frame_count={frame_count})."
        )

    frame = image.getframe(frame_index) if frame_count > 1 else image
    data = _normalize_detector_array(getattr(frame, "data", None), source=path)
    header = dict(getattr(frame, "header", {}) or {})

    return DetectorImageLoadResult(
        data=data,
        path=path,
        format_name=type(frame).__name__,
        loader_name="fabio",
        frame_index=frame_index,
        frame_count=frame_count,
        header=header,
    )


def load_detector_image(
    filename: Union[str, Path],
    frame_index: int = 0,
    prefer_native_osc: bool = True,
) -> DetectorImageLoadResult:
    """Load a detector image and return both pixel data and import metadata."""
    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Detector image not found: {path}")

    normalized_path = str(path)
    if prefer_native_osc and path.suffix.lower() == ".osc":
        return _load_native_osc(normalized_path, frame_index=frame_index)
    return _load_with_fabio(normalized_path, frame_index=frame_index)


def read_detector_image(
    filename: Union[str, Path],
    frame_index: int = 0,
    prefer_native_osc: bool = True,
) -> np.ndarray:
    """Load a 2D detector image and return its pixel array."""
    return load_detector_image(
        filename,
        frame_index=frame_index,
        prefer_native_osc=prefer_native_osc,
    ).data


__all__ = [
    "COMMON_DETECTOR_EXTENSIONS",
    "DetectorImageLoadResult",
    "get_detector_file_dialog_filter",
    "load_detector_image",
    "read_detector_image",
    "supported_detector_extensions",
]
