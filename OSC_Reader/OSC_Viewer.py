import argparse
import os
import sys
from pathlib import Path
import threading

import numpy as np

from .angle_space import (
    DEFAULT_ANGLE_SPACE_WORKERS,
    DEFAULT_GUI_PHI_MAX_DEG,
    DEFAULT_GUI_PHI_MIN_DEG,
    DEFAULT_PHI_ZERO_DIRECTION,
    PHI_ZERO_DIRECTIONS,
    convert_phi_2theta_to_qr_qz_space,
    convert_image_to_phi_2theta_space,
    prepare_gui_phi_display,
    warm_angle_space_engine,
)
from .image_import import get_detector_file_dialog_filter, read_detector_image

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except Exception:  # pragma: no cover - depends on optional GUI stack
    pg = None
    QtCore = None
    QtWidgets = None


_ACTIVE_WINDOWS = []


if QtCore is not None:
    _QtSignal = getattr(QtCore, "Signal", getattr(QtCore, "pyqtSignal", None))
else:  # pragma: no cover - Qt missing
    _QtSignal = None


if QtCore is not None:
    try:
        _QtDownArrow = QtCore.Qt.ArrowType.DownArrow
        _QtRightArrow = QtCore.Qt.ArrowType.RightArrow
    except AttributeError:  # pragma: no cover - Qt5 fallback
        _QtDownArrow = QtCore.Qt.DownArrow
        _QtRightArrow = QtCore.Qt.RightArrow

    try:
        _QtToolButtonTextBesideIcon = QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    except AttributeError:  # pragma: no cover - Qt5 fallback
        _QtToolButtonTextBesideIcon = QtCore.Qt.ToolButtonTextBesideIcon

    try:
        _QtPointingHandCursor = QtCore.Qt.CursorShape.PointingHandCursor
    except AttributeError:  # pragma: no cover - Qt5 fallback
        _QtPointingHandCursor = QtCore.Qt.PointingHandCursor
else:  # pragma: no cover - Qt missing
    _QtDownArrow = None
    _QtRightArrow = None
    _QtToolButtonTextBesideIcon = None
    _QtPointingHandCursor = None


if QtCore is not None:
    try:
        _QtLeftButton = QtCore.Qt.MouseButton.LeftButton
    except AttributeError:  # pragma: no cover - Qt5 fallback
        _QtLeftButton = QtCore.Qt.LeftButton
else:  # pragma: no cover - Qt missing
    _QtLeftButton = None


_ROI_COLORS = (
    "#ffb020",
    "#4aa3ff",
    "#39d98a",
    "#ff6b6b",
    "#c084fc",
    "#ffd166",
)


if QtCore is not None and _QtSignal is not None:
    class _OSCLoadWorker(QtCore.QObject):
        loaded = _QtSignal(str, object)
        failed = _QtSignal(str, str)
        finished = _QtSignal()

        def __init__(self, filename):
            super().__init__()
            self.filename = filename

        def run(self):
            try:
                data = read_detector_image(self.filename)
            except Exception as exc:  # pragma: no cover - read error path
                self.failed.emit(self.filename, str(exc))
            else:
                self.loaded.emit(self.filename, data)
            finally:
                self.finished.emit()


if QtCore is not None and _QtSignal is not None:
    class _AngleSpaceWorker(QtCore.QObject):
        loaded = _QtSignal(object)
        failed = _QtSignal(str)
        finished = _QtSignal()

        def __init__(
            self,
            image,
            *,
            distance_mm,
            pixel_size_mm,
            center_row_px,
            center_col_px,
            radial_bins,
            azimuth_bins,
        ):
            super().__init__()
            self.image = np.asarray(image)
            self.distance_mm = float(distance_mm)
            self.pixel_size_mm = float(pixel_size_mm)
            self.center_row_px = float(center_row_px)
            self.center_col_px = float(center_col_px)
            self.radial_bins = int(radial_bins)
            self.azimuth_bins = int(azimuth_bins)

        def run(self):
            try:
                result = convert_image_to_phi_2theta_space(
                    self.image,
                    distance_mm=self.distance_mm,
                    pixel_size_mm=self.pixel_size_mm,
                    center_row_px=self.center_row_px,
                    center_col_px=self.center_col_px,
                    radial_bins=self.radial_bins,
                    azimuth_bins=self.azimuth_bins,
                )
            except Exception as exc:  # pragma: no cover - runtime/UI path
                self.failed.emit(str(exc))
            else:
                self.loaded.emit({"result": result})
            finally:
                self.finished.emit()


if QtWidgets is not None:
    class _CollapsibleSection(QtWidgets.QFrame):
        def __init__(self, title, *, expanded=True, collapsible=True, parent=None):
            super().__init__(parent)
            self.setObjectName("settingsSection")
            self._collapsible = bool(collapsible)

            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            self.toggle_button = QtWidgets.QToolButton(self)
            self.toggle_button.setObjectName("sectionToggle")
            self.toggle_button.setCheckable(self._collapsible)
            self.toggle_button.setChecked(bool(expanded) or not self._collapsible)
            self.toggle_button.setText(str(title))
            if _QtToolButtonTextBesideIcon is not None:
                self.toggle_button.setToolButtonStyle(_QtToolButtonTextBesideIcon)
            if self._collapsible and _QtPointingHandCursor is not None:
                self.toggle_button.setCursor(_QtPointingHandCursor)
            root_layout.addWidget(self.toggle_button)

            self.body = QtWidgets.QWidget(self)
            self.body.setObjectName("sectionBody")
            self.body_layout = QtWidgets.QVBoxLayout(self.body)
            self.body_layout.setContentsMargins(12, 4, 12, 12)
            self.body_layout.setSpacing(10)
            root_layout.addWidget(self.body)

            if self._collapsible:
                self.toggle_button.toggled.connect(self.set_expanded)
            self.set_expanded(bool(expanded) or not self._collapsible)

        def set_expanded(self, expanded):
            expanded = bool(expanded) or not self._collapsible
            self.toggle_button.blockSignals(True)
            if self._collapsible:
                self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
            if _QtDownArrow is not None and _QtRightArrow is not None:
                self.toggle_button.setArrowType(
                    _QtDownArrow if expanded or not self._collapsible else _QtRightArrow
                )
            self.body.setVisible(expanded)


if pg is not None:
    class _ImageViewBox(pg.ViewBox):
        def __init__(self, *, owner=None, **kwargs):
            super().__init__(enableMenu=False, **kwargs)
            self.owner = owner

        def mouseDragEvent(self, ev, axis=None):
            if self.owner is not None and self.owner._should_draw_roi_from_drag(ev):
                ev.accept()
                start_point = self.mapToView(ev.buttonDownPos())
                current_point = self.mapToView(ev.pos())
                self.owner._handle_roi_drag(
                    float(start_point.x()),
                    float(start_point.y()),
                    float(current_point.x()),
                    float(current_point.y()),
                    is_start=bool(ev.isStart()),
                    is_finish=bool(ev.isFinish()),
                )
                return
            super().mouseDragEvent(ev, axis=axis)


if QtWidgets is not None:
    class _ROIProfileWindow(QtWidgets.QDialog):
        def __init__(
            self,
            *,
            on_close=None,
            on_add_region=None,
            on_delete_regions=None,
            on_selection_changed=None,
            parent=None,
        ):
            super().__init__(parent)
            self._on_close = on_close
            self._on_add_region = on_add_region
            self._on_delete_regions = on_delete_regions
            self._on_selection_changed = on_selection_changed
            self._default_stem = "roi_profiles"
            self.setWindowTitle("ROI Integrated Profiles")
            self.resize(920, 680)

            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(10)

            toolbar = QtWidgets.QHBoxLayout()
            toolbar.setContentsMargins(0, 0, 0, 0)
            toolbar.setSpacing(10)
            self.summary_label = QtWidgets.QLabel("Enable ROI selection in φ/2θ view.")
            self.summary_label.setWordWrap(True)
            toolbar.addWidget(self.summary_label, 1)
            self.add_region_button = QtWidgets.QPushButton("Add Region")
            toolbar.addWidget(self.add_region_button, 0)
            self.delete_region_button = QtWidgets.QPushButton("Delete Selected")
            toolbar.addWidget(self.delete_region_button, 0)
            self.theta_log_checkbox = QtWidgets.QCheckBox("2θ Log Y")
            toolbar.addWidget(self.theta_log_checkbox, 0)
            self.phi_log_checkbox = QtWidgets.QCheckBox("φ Log Y")
            toolbar.addWidget(self.phi_log_checkbox, 0)
            self.save_button = QtWidgets.QPushButton("Save Figure")
            toolbar.addWidget(self.save_button, 0)
            root_layout.addLayout(toolbar)

            content_layout = QtWidgets.QHBoxLayout()
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(10)
            root_layout.addLayout(content_layout, 1)

            side_panel = QtWidgets.QFrame()
            side_panel.setMinimumWidth(180)
            side_panel.setMaximumWidth(240)
            side_layout = QtWidgets.QVBoxLayout(side_panel)
            side_layout.setContentsMargins(0, 0, 0, 0)
            side_layout.setSpacing(6)
            side_layout.addWidget(QtWidgets.QLabel("Regions"))
            self.region_list = QtWidgets.QListWidget()
            self.region_list.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )
            side_layout.addWidget(self.region_list, 1)
            content_layout.addWidget(side_panel, 0)

            self.graphics = pg.GraphicsLayoutWidget()
            self.graphics.ci.layout.setRowStretchFactor(0, 1)
            self.graphics.ci.layout.setRowStretchFactor(1, 1)
            content_layout.addWidget(self.graphics, 1)

            self.theta_plot = self.graphics.addPlot(row=0, col=0)
            self.theta_plot.setLabel("left", "Integrated Intensity")
            self.theta_plot.setLabel("bottom", "2θ (degrees)")
            self.theta_plot.showGrid(x=True, y=True, alpha=0.25)
            self.theta_legend = self.theta_plot.addLegend(offset=(8, 8))
            self.theta_curves = []

            self.phi_plot = self.graphics.addPlot(row=1, col=0)
            self.phi_plot.setLabel("left", "Integrated Intensity")
            self.phi_plot.setLabel("bottom", "φ (degrees)")
            self.phi_plot.showGrid(x=True, y=True, alpha=0.25)
            self.phi_legend = self.phi_plot.addLegend(offset=(8, 8))
            self.phi_curves = []
            self._current_profiles = []
            self._current_summary = "Enable ROI selection in φ/2θ view."
            self._current_aligned = False
            self._log_eps = 1e-3

            self.add_region_button.clicked.connect(self._handle_add_region_clicked)
            self.delete_region_button.clicked.connect(self._handle_delete_selected_clicked)
            self.region_list.itemSelectionChanged.connect(self._handle_selection_changed)
            self.theta_log_checkbox.toggled.connect(self._handle_plot_log_toggled)
            self.phi_log_checkbox.toggled.connect(self._handle_plot_log_toggled)
            self.save_button.clicked.connect(self.save_figure)
            self.delete_region_button.setEnabled(False)

        def set_default_stem(self, stem):
            stem = str(stem).strip()
            self._default_stem = stem or "roi_profiles"

        @staticmethod
        def _clear_plot_curves(plot, legend, curves):
            for curve in curves:
                plot.removeItem(curve)
            legend.clear()
            curves.clear()

        def set_regions(self, regions):
            selected_names = {
                self.region_list.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
                for row in range(self.region_list.count())
                if self.region_list.item(row).isSelected()
            }
            self.region_list.blockSignals(True)
            self.region_list.clear()
            for region in regions:
                item = QtWidgets.QListWidgetItem(str(region["name"]))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(region["name"]))
                item.setForeground(pg.mkColor(region["color"]))
                if not region.get("valid", True):
                    item.setText(f"{region['name']} (empty)")
                self.region_list.addItem(item)
                if region.get("selected") or region["name"] in selected_names:
                    item.setSelected(True)
            self.region_list.blockSignals(False)
            self.delete_region_button.setEnabled(bool(self.selected_region_names()))

        def selected_region_names(self):
            names = []
            for item in self.region_list.selectedItems():
                names.append(str(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            return names

        def _display_curve_y(self, values, *, log_enabled):
            array = np.asarray(values, dtype=np.float64)
            if not log_enabled:
                return array
            return np.maximum(array, self._log_eps)

        def _render_profiles(self):
            self.summary_label.setText(str(self._current_summary))
            self._clear_plot_curves(self.theta_plot, self.theta_legend, self.theta_curves)
            self._clear_plot_curves(self.phi_plot, self.phi_legend, self.phi_curves)
            self.theta_plot.setLabel(
                "bottom",
                "Δ2θ from peak (degrees)" if self._current_aligned else "2θ (degrees)",
            )
            self.phi_plot.setLabel(
                "bottom",
                "Δφ from peak (degrees)" if self._current_aligned else "φ (degrees)",
            )
            self.theta_plot.setLogMode(x=False, y=self.theta_log_checkbox.isChecked())
            self.phi_plot.setLogMode(x=False, y=self.phi_log_checkbox.isChecked())
            for profile in self._current_profiles:
                theta_curve = self.theta_plot.plot(
                    profile["theta_x"],
                    self._display_curve_y(
                        profile["theta_y"],
                        log_enabled=self.theta_log_checkbox.isChecked(),
                    ),
                    pen=pg.mkPen(profile["color"], width=2),
                    name=profile["name"],
                )
                phi_curve = self.phi_plot.plot(
                    profile["phi_x"],
                    self._display_curve_y(
                        profile["phi_y"],
                        log_enabled=self.phi_log_checkbox.isChecked(),
                    ),
                    pen=pg.mkPen(profile["color"], width=2),
                    name=profile["name"],
                )
                self.theta_legend.addItem(theta_curve, profile["name"])
                self.phi_legend.addItem(phi_curve, profile["name"])
                self.theta_curves.append(theta_curve)
                self.phi_curves.append(phi_curve)
            self.theta_plot.enableAutoRange()
            self.phi_plot.enableAutoRange()

        def update_profiles(self, profiles, summary, *, aligned=False):
            self._current_profiles = list(profiles)
            self._current_summary = str(summary)
            self._current_aligned = bool(aligned)
            self._render_profiles()

        def clear_profiles(self, summary):
            self._current_profiles = []
            self._current_summary = str(summary)
            self._current_aligned = False
            self._render_profiles()

        def _handle_add_region_clicked(self):
            if callable(self._on_add_region):
                self._on_add_region()

        def _handle_delete_selected_clicked(self):
            if callable(self._on_delete_regions):
                names = self.selected_region_names()
                if names:
                    self._on_delete_regions(names)

        def _handle_selection_changed(self):
            self.delete_region_button.setEnabled(bool(self.selected_region_names()))
            if callable(self._on_selection_changed):
                self._on_selection_changed(self.selected_region_names())

        def _handle_plot_log_toggled(self, _checked):
            self._render_profiles()

        def save_figure(self):
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save ROI integrated profiles",
                str(Path.cwd() / f"{self._default_stem}.png"),
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*)",
            )
            if not selected:
                return
            self.graphics.grab().save(selected)

        def closeEvent(self, event):
            super().closeEvent(event)
            if callable(self._on_close):
                self._on_close()

class OSCViewerWindow(QtWidgets.QMainWindow):
    TARGET_FPS = 60.0
    MAX_PROFILE_POINTS = 800
    LOG_EPS = 1e-3
    BEAM_CENTER_SEARCH_RADIUS_PX = 25

    def __init__(self, filename):
        super().__init__()
        self.setWindowTitle("OSC Reader (High FPS Viewer)")
        self.resize(1680, 1020)
        self._apply_always_on_top()

        self.filename = None
        self.data = None
        self.detector_data = None
        self.angle_space_result = None
        self.angle_space_cake = None
        self.q_space_result = None
        self.height = 0
        self.width = 0
        self.data_min = 0.0
        self.data_max = 1.0
        self.current_view_mode = "detector"
        self.display_x_full = np.array([], dtype=np.float64)
        self.display_y_full = np.array([], dtype=np.float64)
        self.display_x_edges = np.array([], dtype=np.float64)
        self.display_y_edges = np.array([], dtype=np.float64)
        self.display_x_label = "X pixels"
        self.display_y_label = "Y pixels"

        self.pending_x = None
        self.pending_y = None
        self.pending_dirty = False
        self.last_ix = None
        self.last_iy = None
        self.last_bottom_range = (None, None)
        self.last_left_range = (None, None)

        self.x_profile_coords = None
        self.y_profile_coords = None
        self.step_x = 1
        self.step_y = 1
        self.bottom_log_enabled = False
        self.left_log_enabled = False
        self.image_log_enabled = False
        self._loading = False
        self._converting = False
        self._loader_thread = None
        self._loader_worker = None
        self._conversion_thread = None
        self._conversion_worker = None
        self._conversion_target_view = "angle_space"
        self.beam_center_row = None
        self.beam_center_col = None
        self._warmup_thread = None
        self.roi_items = []
        self._drawing_roi_item = None
        self._next_roi_index = 1
        self._selected_roi_names = set()
        self._updating_roi = False
        self.roi_profile_window = None
        self._roi_window_pending_initial_draw = False

        self._build_ui()
        self._start_angle_space_warmup()
        if filename:
            QtCore.QTimer.singleShot(0, lambda: self.load_file(filename))

    def _apply_always_on_top(self):
        top_flag = None
        try:
            top_flag = QtCore.Qt.WindowType.WindowStaysOnTopHint
        except AttributeError:
            top_flag = getattr(QtCore.Qt, "WindowStaysOnTopHint", None)
        if top_flag is not None:
            self.setWindowFlag(top_flag, True)

    @staticmethod
    def _safe_limits_linear(min_value, max_value):
        if np.isclose(min_value, max_value):
            padding = max(abs(min_value) * 0.01, 1.0)
            return min_value - padding, max_value + padding
        return min_value, max_value

    def _safe_limits_log(self, min_value, max_value):
        min_value = max(float(min_value), self.LOG_EPS)
        max_value = max(float(max_value), min_value)
        if np.isclose(min_value, max_value):
            # Log axes need multiplicative padding, not additive.
            low = max(min_value / 1.25, self.LOG_EPS)
            high = max(max_value * 1.25, low * 1.01)
            return low, high
        low = max(min_value / 1.05, self.LOG_EPS)
        high = max(max_value * 1.05, low * 1.01)
        return low, high

    @staticmethod
    def _smooth_peak_window(window):
        padded = np.pad(np.asarray(window, dtype=np.float64), 1, mode="edge")
        return (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0

    @staticmethod
    def _parabolic_peak_offset(left, center, right):
        denominator = float(left) - 2.0 * float(center) + float(right)
        if np.isclose(denominator, 0.0):
            return 0.0
        offset = 0.5 * (float(left) - float(right)) / denominator
        if not np.isfinite(offset):
            return 0.0
        return float(np.clip(offset, -0.5, 0.5))

    def _compute_axis_range(self, values, log_enabled):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            if log_enabled:
                return self.LOG_EPS, self.LOG_EPS * 10.0
            return 0.0, 1.0

        if log_enabled:
            positive = values[values > 0]
            if positive.size == 0:
                return self.LOG_EPS, self.LOG_EPS * 10.0
            return self._safe_limits_log(np.min(positive), np.max(positive))
        return self._safe_limits_linear(np.min(values), np.max(values))

    @staticmethod
    def _coord_edges(coords):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.size == 0:
            return np.array([0.0, 1.0], dtype=np.float64)
        if coords.size == 1:
            value = float(coords[0])
            return np.array([value - 0.5, value + 0.5], dtype=np.float64)
        deltas = np.diff(coords)
        edges = np.empty(coords.size + 1, dtype=np.float64)
        edges[1:-1] = coords[:-1] + 0.5 * deltas
        edges[0] = coords[0] - 0.5 * deltas[0]
        edges[-1] = coords[-1] + 0.5 * deltas[-1]
        return edges

    @staticmethod
    def _index_from_edges(value, edges):
        index = int(np.searchsorted(edges, value, side="right") - 1)
        return max(0, min(index, len(edges) - 2))

    def _x_text(self, ix):
        if self.current_view_mode == "detector":
            return str(ix)
        return f"{float(self.display_x_full[ix]):.3f}"

    def _y_text(self, iy):
        if self.current_view_mode == "detector":
            return str(iy)
        return f"{float(self.display_y_full[iy]):.3f}"

    def _beam_center_summary(self):
        if self.beam_center_row is None or self.beam_center_col is None:
            return "Beam center: unset"
        return f"Beam center: ({self.beam_center_col:.2f}, {self.beam_center_row:.2f}) px"

    def _roi_supported_in_current_view(self):
        return self.current_view_mode == "angle_space" and self.data is not None

    def _visible_roi_items(self):
        return [item for item in self.roi_items if item.isVisible()]

    def _clear_roi_items(self):
        for roi_item in self.roi_items:
            try:
                self.image_plot.removeItem(roi_item)
            except Exception:
                pass
        self.roi_items = []
        self._drawing_roi_item = None
        self._next_roi_index = 1
        self._selected_roi_names = set()
        self._roi_window_pending_initial_draw = False
        self._close_roi_profile_window()

    def _make_roi_name(self, index):
        return f"ROI {int(index)}"

    @staticmethod
    def _roi_color_for_index(index):
        return _ROI_COLORS[(int(index) - 1) % len(_ROI_COLORS)]

    def _set_selected_roi_names(self, names, *, refresh_window=False):
        valid_names = {getattr(roi_item, "_roi_name", "") for roi_item in self.roi_items}
        self._selected_roi_names = {str(name) for name in names if str(name) in valid_names}
        self._apply_roi_selection_styles()
        if refresh_window and self.roi_button.isChecked():
            self._update_roi_profile_window()

    def _apply_roi_selection_styles(self):
        selected_names = set(self._selected_roi_names)
        for roi_item in self.roi_items:
            color = getattr(roi_item, "_roi_color", _ROI_COLORS[0])
            is_selected = getattr(roi_item, "_roi_name", "") in selected_names
            roi_item.setPen(pg.mkPen(color, width=4 if is_selected else 2))
            roi_item.hoverPen = pg.mkPen(color, width=5 if is_selected else 3)
            roi_index = int(getattr(roi_item, "_roi_index", 0))
            roi_item.setZValue((120 if is_selected else 20) + roi_index)

    def _create_roi_item(self, x_pos, y_pos, width, height):
        roi_index = self._next_roi_index
        self._next_roi_index += 1
        color = self._roi_color_for_index(roi_index)
        roi_item = pg.RectROI(
            [x_pos, y_pos],
            [width, height],
            movable=True,
            pen=pg.mkPen(color, width=2),
            hoverPen=pg.mkPen(color, width=3),
        )
        roi_item.setZValue(20 + roi_index)
        roi_item.addScaleHandle([0, 0], [1, 1])
        roi_item.addScaleHandle([1, 0], [0, 1])
        roi_item.addScaleHandle([0, 1], [1, 0])
        roi_item.addScaleHandle([1, 1], [0, 0])
        roi_item._roi_index = roi_index
        roi_item._roi_name = self._make_roi_name(roi_index)
        roi_item._roi_color = color
        roi_item.sigRegionChanged.connect(
            lambda *_, item=roi_item: self._on_roi_region_changed(item)
        )
        self.image_plot.addItem(roi_item)
        self.roi_items.append(roi_item)
        self._set_selected_roi_names([roi_item._roi_name])
        return roi_item

    def _set_roi_overlay_visible(self, visible, *, refresh_profiles=True):
        if not visible or not self._roi_supported_in_current_view():
            for roi_item in self.roi_items:
                roi_item.hide()
            return
        for roi_item in self.roi_items:
            roi_item.show()
        if refresh_profiles and self.roi_items:
            self._on_roi_region_changed()

    def _ensure_roi_profile_window(self):
        if self.roi_profile_window is None:
            self.roi_profile_window = _ROIProfileWindow(
                on_close=self._on_roi_window_closed,
                on_add_region=self._start_roi_add_mode_from_profile_window,
                on_delete_regions=self._delete_roi_names,
                on_selection_changed=self._on_profile_window_selection_changed,
                parent=self,
            )
        default_stem = Path(self.filename).stem if self.filename else "osc_view"
        self.roi_profile_window.set_default_stem(f"{default_stem}_roi_profiles")
        return self.roi_profile_window

    def _close_roi_profile_window(self):
        if self.roi_profile_window is not None:
            self.roi_profile_window.hide()

    def _on_roi_window_closed(self):
        if self.roi_button.isChecked():
            self.roi_button.blockSignals(True)
            self.roi_button.setChecked(False)
            self.roi_button.blockSignals(False)
        self._set_roi_overlay_visible(False)
        if self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _start_roi_add_mode_from_profile_window(self):
        if not self._roi_supported_in_current_view():
            return
        if not self.roi_button.isChecked():
            self.roi_button.setChecked(True)
        self._roi_window_pending_initial_draw = True
        self._set_status_message(
            "Left-click and drag in empty space on the φ/2θ image to add another ROI."
        )
        self.raise_()
        self.activateWindow()

    def _delete_roi_names(self, names):
        remove_names = {str(name) for name in names}
        if not remove_names:
            return
        remaining = []
        for roi_item in self.roi_items:
            if getattr(roi_item, "_roi_name", "") in remove_names:
                try:
                    self.image_plot.removeItem(roi_item)
                except Exception:
                    pass
            else:
                remaining.append(roi_item)
        self.roi_items = remaining
        self._selected_roi_names -= remove_names
        self._drawing_roi_item = None
        self._apply_roi_selection_styles()
        self._update_roi_profile_window(show_window=self.roi_profile_window is not None and self.roi_profile_window.isVisible())
        if self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _on_profile_window_selection_changed(self, names):
        self._set_selected_roi_names(names)

    def _roi_minimum_spans(self):
        x_span = max(abs(float(self.display_x_edges[-1] - self.display_x_edges[0])), 1e-6)
        y_span = max(abs(float(self.display_y_edges[-1] - self.display_y_edges[0])), 1e-6)
        return (
            max(x_span / max(4 * max(self.width, 1), 1), 1e-6),
            max(y_span / max(4 * max(self.height, 1), 1), 1e-6),
        )

    def _roi_contains_point(self, roi_item, x_value, y_value):
        state = roi_item.getState()
        pos = state["pos"]
        size = state["size"]
        x_low, x_high = sorted((float(pos[0]), float(pos[0] + size[0])))
        y_low, y_high = sorted((float(pos[1]), float(pos[1] + size[1])))
        margin_x, margin_y = self._roi_minimum_spans()
        return (
            (x_low - margin_x) <= float(x_value) <= (x_high + margin_x)
            and (y_low - margin_y) <= float(y_value) <= (y_high + margin_y)
        )

    def _set_roi_rect(
        self,
        roi_item,
        x0,
        y0,
        x1,
        y1,
        *,
        show_window=False,
        commit_geometry=False,
    ):
        if not self._roi_supported_in_current_view():
            return

        x_min = float(self.display_x_edges[0])
        x_max = float(self.display_x_edges[-1])
        y_min = float(self.display_y_edges[0])
        y_max = float(self.display_y_edges[-1])
        min_width, min_height = self._roi_minimum_spans()

        x0 = float(np.clip(x0, x_min, x_max))
        x1 = float(np.clip(x1, x_min, x_max))
        y0 = float(np.clip(y0, y_min, y_max))
        y1 = float(np.clip(y1, y_min, y_max))

        x_low, x_high = sorted((x0, x1))
        y_low, y_high = sorted((y0, y1))
        if np.isclose(x_low, x_high):
            x_high = min(x_max, x_low + min_width)
            x_low = max(x_min, x_high - min_width)
        if np.isclose(y_low, y_high):
            y_high = min(y_max, y_low + min_height)
            y_low = max(y_min, y_high - min_height)

        self._updating_roi = True
        roi_item.setPos((x_low, y_low), update=True, finish=False)
        roi_item.setSize(
            (x_high - x_low, y_high - y_low),
            update=True,
            finish=bool(commit_geometry),
        )
        self._updating_roi = False
        roi_item.show()
        self._update_roi_profile_window(show_window=show_window)
        if self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _should_draw_roi_from_drag(self, event):
        start_point = self.image_view.mapToView(event.buttonDownPos())
        return (
            self._roi_supported_in_current_view()
            and self.roi_button.isChecked()
            and _QtLeftButton is not None
            and event.button() == _QtLeftButton
            and (
                self._drawing_roi_item is not None
                or not any(
                    self._roi_contains_point(
                        roi_item,
                        float(start_point.x()),
                        float(start_point.y()),
                    )
                    for roi_item in self._visible_roi_items()
                )
            )
        )

    def _handle_roi_drag(self, start_x, start_y, current_x, current_y, *, is_start=False, is_finish=False):
        if not self._roi_supported_in_current_view() or not self.roi_button.isChecked():
            return
        if is_start or self._drawing_roi_item is None:
            min_width, min_height = self._roi_minimum_spans()
            self._drawing_roi_item = self._create_roi_item(
                float(start_x),
                float(start_y),
                min_width,
                min_height,
            )
        if self._drawing_roi_item is None:
            return
        self._set_roi_rect(
            self._drawing_roi_item,
            start_x,
            start_y,
            current_x,
            current_y,
            show_window=bool(is_finish),
            commit_geometry=bool(is_finish),
        )
        if is_start:
            self._set_status_message(
                "Release to place the ROI. Drag the corners afterward to refine the selection."
            )
        if is_finish:
            self._drawing_roi_item = None
            roi_count = len(self._visible_roi_items())
            if roi_count > 1:
                self._set_status_message(
                    f"Added {roi_count} ROIs. The profile window overlays them with peak-centered 2θ and φ axes."
                )

    def _compute_roi_profile_for_item(self, roi_item):
        if self.data is None:
            return None

        state = roi_item.getState()
        pos = state["pos"]
        size = state["size"]
        theta_min, theta_max = sorted((float(pos[0]), float(pos[0] + size[0])))
        phi_min, phi_max = sorted((float(pos[1]), float(pos[1] + size[1])))

        theta_mask = (self.display_x_full >= theta_min) & (self.display_x_full <= theta_max)
        phi_mask = (self.display_y_full >= phi_min) & (self.display_y_full <= phi_max)
        if not np.any(theta_mask) or not np.any(phi_mask):
            return None

        roi_image = np.asarray(self.data[np.ix_(phi_mask, theta_mask)], dtype=np.float64)
        if roi_image.size == 0:
            return None

        theta_profile = np.nansum(roi_image, axis=0)
        phi_profile = np.nansum(roi_image, axis=1)
        if theta_profile.size == 0 or phi_profile.size == 0:
            return None
        theta_finite = np.isfinite(theta_profile)
        phi_finite = np.isfinite(phi_profile)
        if not np.any(theta_finite) or not np.any(phi_finite):
            return None
        theta_peak_index = int(np.nanargmax(theta_profile))
        phi_peak_index = int(np.nanargmax(phi_profile))
        return {
            "name": getattr(roi_item, "_roi_name", f"ROI {len(self.roi_items)}"),
            "color": getattr(roi_item, "_roi_color", _ROI_COLORS[0]),
            "theta_x": self.display_x_full[theta_mask],
            "theta_y": theta_profile,
            "phi_x": self.display_y_full[phi_mask],
            "phi_y": phi_profile,
            "theta_min": theta_min,
            "theta_max": theta_max,
            "phi_min": phi_min,
            "phi_max": phi_max,
            "shape": roi_image.shape,
            "theta_peak": float(self.display_x_full[theta_mask][theta_peak_index]),
            "phi_peak": float(self.display_y_full[phi_mask][phi_peak_index]),
        }

    def _collect_roi_profiles(self):
        profiles = []
        for roi_item in self._visible_roi_items():
            profile = self._compute_roi_profile_for_item(roi_item)
            if profile is not None:
                profiles.append(profile)
        aligned = len(profiles) > 1
        if aligned:
            shifted = []
            for profile in profiles:
                shifted.append(
                    {
                        **profile,
                        "theta_x": np.asarray(profile["theta_x"], dtype=np.float64) - profile["theta_peak"],
                        "phi_x": np.asarray(profile["phi_x"], dtype=np.float64) - profile["phi_peak"],
                    }
                )
            profiles = shifted
        return profiles, aligned

    def _roi_descriptors(self, valid_names):
        descriptors = []
        selected_names = set(self._selected_roi_names)
        for roi_item in self._visible_roi_items():
            name = getattr(roi_item, "_roi_name", f"ROI {len(descriptors) + 1}")
            descriptors.append(
                {
                    "name": name,
                    "color": getattr(roi_item, "_roi_color", _ROI_COLORS[0]),
                    "selected": name in selected_names,
                    "valid": name in valid_names,
                }
            )
        return descriptors

    def _update_roi_profile_window(self, *, show_window=False):
        if not self.roi_button.isChecked():
            self._close_roi_profile_window()
            return

        profile_window = self._ensure_roi_profile_window()
        profiles, aligned = self._collect_roi_profiles()
        valid_names = {profile["name"] for profile in profiles}
        profile_window.set_regions(self._roi_descriptors(valid_names))
        if not profiles:
            profile_window.clear_profiles("ROI: select a non-empty region in φ/2θ space.")
        else:
            if aligned:
                summary = (
                    f"{len(profiles)} ROIs overlaid. Each integrated profile is shifted so its peak is at "
                    "2θ = 0 and φ = 0."
                )
            else:
                payload = profiles[0]
                summary = (
                    f"{payload['name']}    "
                    f"2θ: {payload['theta_min']:.3f}..{payload['theta_max']:.3f} deg    "
                    f"φ: {payload['phi_min']:.3f}..{payload['phi_max']:.3f} deg    "
                    f"Bins: {payload['shape'][1]} x {payload['shape'][0]}"
                )
            profile_window.update_profiles(
                profiles,
                summary,
                aligned=aligned,
            )
        if show_window:
            self._roi_window_pending_initial_draw = False
            profile_window.show()
            profile_window.raise_()
            profile_window.activateWindow()

    def _roi_summary(self):
        if self.data is None or not self.roi_button.isChecked() or not self._roi_supported_in_current_view():
            return None
        visible_rois = self._visible_roi_items()
        if not visible_rois:
            return None
        if len(visible_rois) > 1:
            return f"ROIs: {len(visible_rois)} selected (peak-aligned overlay)"
        state = visible_rois[0].getState()
        pos = state["pos"]
        size = state["size"]
        x_low, x_high = sorted((float(pos[0]), float(pos[0] + size[0])))
        y_low, y_high = sorted((float(pos[1]), float(pos[1] + size[1])))
        return (
            f"{getattr(visible_rois[0], '_roi_name', 'ROI')}  "
            f"{self.display_x_label}: {x_low:.3f}..{x_high:.3f}  "
            f"{self.display_y_label}: {y_low:.3f}..{y_high:.3f}"
        )

    def _estimate_data_stats(self):
        # Use a coarse sample for fast startup; profile axes are refined per-cursor.
        return self._estimate_sample_stats(self.data)

    @staticmethod
    def _estimate_sample_stats(array):
        array = np.asarray(array, dtype=np.float64)
        sample_target = 512
        sample_step_x = max(1, array.shape[1] // sample_target)
        sample_step_y = max(1, array.shape[0] // sample_target)
        sample = array[::sample_step_y, ::sample_step_x]
        sample_min = float(np.min(sample))
        sample_max = float(np.max(sample))
        if np.isclose(sample_min, sample_max):
            # Fallback for pathological cases.
            sample_min = float(np.min(array))
            sample_max = float(np.max(array))
        return sample_min, sample_max

    def _image_display_data(self):
        image = np.asarray(self.data, dtype=np.float64)
        if not self.image_log_enabled:
            return image
        return np.log10(np.maximum(image, self.LOG_EPS))

    def _refresh_image_display(self):
        if self.data is None:
            return
        image_display = self._image_display_data()
        self.image_item.setImage(image_display, autoLevels=False)
        display_min, display_max = self._estimate_sample_stats(image_display)
        slider_low = display_min if self.image_log_enabled else min(0.0, display_min)
        slider_high = display_max
        if np.isclose(slider_low, slider_high):
            slider_high = slider_low + 1.0
        default_fraction = 0.85 if self.image_log_enabled else 0.35
        vmin_default = float(slider_low if self.image_log_enabled else np.clip(0.0, slider_low, slider_high))
        vmax_default = float(slider_low + default_fraction * (slider_high - slider_low))
        if vmax_default <= vmin_default:
            vmax_default = slider_high
        self.image_item.setLevels((vmin_default, vmax_default))
        self.hist_lut.setLevels(vmin_default, vmax_default)

    def _apply_window_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f1115;
                color: #edf2f7;
            }
            QLabel {
                color: #edf2f7;
            }
            QFrame#toolbarGroup,
            QFrame#settingsShelf {
                background-color: #131923;
                border: 1px solid #242b36;
                border-radius: 12px;
            }
            QLabel#sectionCaption {
                color: #97a6b7;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#fieldLabel {
                color: #aeb9c6;
            }
            QPushButton#commandButton,
            QToolButton#viewButton,
            QToolButton#contextButton,
            QToolButton#profileToggle {
                background-color: #1b2330;
                border: 1px solid #2c3644;
                border-radius: 8px;
                color: #f3f7fb;
                padding: 6px 12px;
            }
            QPushButton#commandButton:hover,
            QToolButton#viewButton:hover,
            QToolButton#contextButton:hover,
            QToolButton#profileToggle:hover {
                background-color: #242e3d;
                border-color: #3a4758;
            }
            QToolButton#viewButton:checked,
            QToolButton#contextButton:checked,
            QToolButton#profileToggle:checked {
                background-color: #1f5f8b;
                border-color: #67b4e5;
                color: #ffffff;
            }
            QToolButton#sectionToggle {
                background-color: transparent;
                border: none;
                color: #f3f7fb;
                font-weight: 600;
                padding: 10px 12px 6px 12px;
                text-align: left;
            }
            QToolButton#sectionToggle:hover {
                color: #9bd1f3;
            }
            QFrame#settingsSection {
                background-color: #121720;
                border: 1px solid #242b36;
                border-radius: 12px;
            }
            QWidget#sectionBody {
                background-color: transparent;
                border: none;
            }
            QComboBox,
            QSpinBox,
            QDoubleSpinBox {
                background-color: #0e131a;
                border: 1px solid #2a3340;
                border-radius: 7px;
                color: #edf2f7;
                min-height: 28px;
                padding: 2px 8px;
            }
            QComboBox:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus {
                border-color: #67b4e5;
            }
            QStatusBar {
                background-color: #0c0f14;
                border-top: 1px solid #1f2630;
                color: #9fb0c1;
            }
            QStatusBar QLabel {
                color: #9fb0c1;
            }
            """
        )

    def _make_caption_label(self, text):
        label = QtWidgets.QLabel(str(text).upper())
        label.setObjectName("sectionCaption")
        return label

    def _make_field_label(self, text):
        label = QtWidgets.QLabel(str(text))
        label.setObjectName("fieldLabel")
        return label

    def _make_action_button(self, text, *, tooltip=None, shortcut=None):
        button = QtWidgets.QPushButton(str(text))
        button.setObjectName("commandButton")
        if tooltip:
            button.setToolTip(str(tooltip))
        if shortcut:
            button.setShortcut(str(shortcut))
        return button

    def _make_tool_button(
        self,
        text,
        *,
        object_name,
        checkable=False,
        tooltip=None,
        shortcut=None,
        minimum_width=None,
    ):
        button = QtWidgets.QToolButton(self)
        button.setText(str(text))
        button.setObjectName(str(object_name))
        button.setCheckable(bool(checkable))
        if _QtPointingHandCursor is not None:
            button.setCursor(_QtPointingHandCursor)
        if tooltip:
            button.setToolTip(str(tooltip))
        if shortcut:
            button.setShortcut(str(shortcut))
        if minimum_width is not None:
            button.setMinimumWidth(int(minimum_width))
        return button

    def _create_toolbar_group(self, title):
        frame = QtWidgets.QFrame()
        frame.setObjectName("toolbarGroup")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self._make_caption_label(title))
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        layout.addLayout(row)
        return frame, row

    def _add_labeled_field(self, layout, row, column, label_text, widget):
        layout.addWidget(self._make_field_label(label_text), row, column)
        layout.addWidget(widget, row, column + 1)

    def _set_status_message(self, message):
        self.statusBar().showMessage(str(message))

    def _sync_view_controls(self):
        button_map = {
            "detector": self.detector_view_button,
            "angle_space": self.angle_space_view_button,
            "q_space": self.q_space_view_button,
        }
        active_mode = self.current_view_mode if self.current_view_mode in button_map else "detector"
        for mode, button in button_map.items():
            button.blockSignals(True)
            button.setChecked(mode == active_mode)
            button.blockSignals(False)

        pick_visible = self.detector_data is not None and self.current_view_mode == "detector"
        roi_visible = self._roi_supported_in_current_view()
        if not roi_visible and self.roi_button.isChecked():
            self.roi_button.blockSignals(True)
            self.roi_button.setChecked(False)
            self.roi_button.blockSignals(False)
        self.pick_center_button.setVisible(pick_visible)
        self.roi_button.setVisible(roi_visible)
        self.context_tools_frame.setVisible(pick_visible or roi_visible)
        self._set_roi_overlay_visible(roi_visible and self.roi_button.isChecked())
        if not roi_visible:
            self._close_roi_profile_window()
        self.status_view_label.setText(
            {
                "detector": "View: Detector",
                "angle_space": "View: φ/2θ",
                "q_space": "View: q-space",
            }.get(self.current_view_mode, "View: Detector")
        )

    def _request_view_mode(self, view_mode):
        if self._loading or self._converting or self.detector_data is None:
            self._sync_view_controls()
            return

        view_mode = str(view_mode)
        if view_mode == "detector":
            self.show_detector_view()
            return
        if view_mode == "angle_space":
            if self.angle_space_result is not None:
                self._update_angle_space_display(force_show=True)
                return
            self.convert_active_image()
            return
        if view_mode == "q_space":
            if self.q_space_result is not None:
                self.show_q_space_view(
                    self.q_space_result.intensity,
                    self.q_space_result.qr,
                    self.q_space_result.qz,
                )
                return
            if self.angle_space_result is not None:
                self._update_q_space_display(force_show=True)
                return
            self.convert_active_image_to_q_space()

    def _build_ui(self):
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
        self._apply_window_style()

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 8)
        main_layout.setSpacing(10)
        self.setCentralWidget(central)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(10)

        file_group, file_row = self._create_toolbar_group("File")
        self.open_button = self._make_action_button(
            "Open Image",
            tooltip="Open a detector image. (Ctrl+O)",
            shortcut="Ctrl+O",
        )
        self.save_button = self._make_action_button(
            "Save Image",
            tooltip="Save the current main image view. (Ctrl+S)",
            shortcut="Ctrl+S",
        )
        self.reset_button = self._make_action_button(
            "Reset View",
            tooltip="Reset the zoom to the full image extent. (R)",
            shortcut="R",
        )
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.save_button)
        file_row.addWidget(self.reset_button)

        view_group, view_row = self._create_toolbar_group("View")
        self.detector_view_button = self._make_tool_button(
            "Detector",
            object_name="viewButton",
            checkable=True,
            tooltip="Show the raw detector pixel view. (Ctrl+1)",
            shortcut="Ctrl+1",
            minimum_width=102,
        )
        self.angle_space_view_button = self._make_tool_button(
            "φ/2θ",
            object_name="viewButton",
            checkable=True,
            tooltip="Show the angle-space conversion, creating it if needed. (Ctrl+2)",
            shortcut="Ctrl+2",
            minimum_width=86,
        )
        self.q_space_view_button = self._make_tool_button(
            "q-space",
            object_name="viewButton",
            checkable=True,
            tooltip="Show the reciprocal-space conversion, creating it if needed. (Ctrl+3)",
            shortcut="Ctrl+3",
            minimum_width=96,
        )
        self.view_button_group = QtWidgets.QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        self.view_button_group.addButton(self.detector_view_button)
        self.view_button_group.addButton(self.angle_space_view_button)
        self.view_button_group.addButton(self.q_space_view_button)
        self.detector_view_button.setChecked(True)
        view_row.addWidget(self.detector_view_button)
        view_row.addWidget(self.angle_space_view_button)
        view_row.addWidget(self.q_space_view_button)

        self.context_tools_frame, tools_row = self._create_toolbar_group("Tools")
        self.pick_center_button = self._make_tool_button(
            "Pick Beam Center",
            object_name="contextButton",
            checkable=True,
            tooltip="Click the detector image to place the beam center. (B)",
            shortcut="B",
            minimum_width=148,
        )
        self.roi_button = self._make_tool_button(
            "ROI",
            object_name="contextButton",
            checkable=True,
            tooltip="Draw one or more φ/2θ ROIs and compare integrated 2θ and φ plots. (A)",
            shortcut="A",
            minimum_width=88,
        )
        tools_row.addWidget(self.pick_center_button)
        tools_row.addWidget(self.roi_button)

        top_bar.addWidget(file_group)
        top_bar.addWidget(view_group)
        top_bar.addWidget(self.context_tools_frame)
        top_bar.addStretch(1)
        main_layout.addLayout(top_bar)

        self.center_col_spin = QtWidgets.QDoubleSpinBox()
        self.center_row_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.incident_angle_spin = QtWidgets.QDoubleSpinBox()
        self.radial_bins_spin = QtWidgets.QSpinBox()
        self.azimuth_bins_spin = QtWidgets.QSpinBox()
        self.phi_zero_direction_combo = QtWidgets.QComboBox()

        for spin in (self.center_col_spin, self.center_row_spin):
            spin.setDecimals(2)
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setSingleStep(1.0)
            spin.setMinimumWidth(132)

        self.distance_spin.setDecimals(4)
        self.distance_spin.setRange(0.0001, 1_000_000.0)
        self.distance_spin.setSingleStep(0.1)
        self.distance_spin.setSuffix(" mm")
        self.distance_spin.setValue(75.0)
        self.distance_spin.setMinimumWidth(132)

        self.pixel_size_spin.setDecimals(5)
        self.pixel_size_spin.setRange(0.00001, 1000.0)
        self.pixel_size_spin.setSingleStep(0.001)
        self.pixel_size_spin.setSuffix(" mm")
        self.pixel_size_spin.setValue(0.1)
        self.pixel_size_spin.setMinimumWidth(132)

        self.wavelength_spin.setDecimals(5)
        self.wavelength_spin.setRange(0.00001, 1000.0)
        self.wavelength_spin.setSingleStep(0.001)
        self.wavelength_spin.setSuffix(" Å")
        self.wavelength_spin.setValue(1.54060)
        self.wavelength_spin.setKeyboardTracking(False)
        self.wavelength_spin.setMinimumWidth(132)

        self.incident_angle_spin.setDecimals(4)
        self.incident_angle_spin.setRange(-90.0, 90.0)
        self.incident_angle_spin.setSingleStep(0.01)
        self.incident_angle_spin.setSuffix(" deg")
        self.incident_angle_spin.setValue(0.0)
        self.incident_angle_spin.setAccelerated(True)
        self.incident_angle_spin.setKeyboardTracking(False)
        self.incident_angle_spin.setMinimumWidth(132)

        for spin in (self.radial_bins_spin, self.azimuth_bins_spin):
            spin.setRange(2, 10000)
            spin.setSingleStep(10)
            spin.setMinimumWidth(132)
        self.radial_bins_spin.setValue(1000)
        self.azimuth_bins_spin.setValue(720)

        self.phi_zero_direction_combo.setMinimumWidth(132)
        self.phi_zero_direction_combo.setToolTip(
            "Choose which detector direction is treated as φ = 0 during φ/2θ conversion."
        )
        for direction in PHI_ZERO_DIRECTIONS:
            self.phi_zero_direction_combo.addItem(direction.title())
        self.phi_zero_direction_combo.setCurrentText(DEFAULT_PHI_ZERO_DIRECTION.title())

        self.bottom_log_button = self._make_tool_button(
            "Bottom Profile Log Y",
            object_name="profileToggle",
            checkable=True,
            tooltip="Use a log Y axis for the bottom profile.",
        )
        self.left_log_button = self._make_tool_button(
            "Side Profile Log X",
            object_name="profileToggle",
            checkable=True,
            tooltip="Use a log X axis for the side profile.",
        )
        self.image_log_button = self._make_tool_button(
            "Image Log View",
            object_name="profileToggle",
            checkable=True,
            tooltip="Display the main image with logarithmic intensity scaling.",
        )

        settings_shelf = QtWidgets.QFrame()
        settings_shelf.setObjectName("settingsShelf")
        settings_layout = QtWidgets.QHBoxLayout(settings_shelf)
        settings_layout.setContentsMargins(12, 10, 12, 10)
        settings_layout.setSpacing(10)

        self.geometry_section = _CollapsibleSection("Geometry", expanded=True)
        geometry_widget = QtWidgets.QWidget()
        geometry_grid = QtWidgets.QGridLayout(geometry_widget)
        geometry_grid.setContentsMargins(0, 0, 0, 0)
        geometry_grid.setHorizontalSpacing(12)
        geometry_grid.setVerticalSpacing(8)
        self._add_labeled_field(geometry_grid, 0, 0, "Center X (px)", self.center_col_spin)
        self._add_labeled_field(geometry_grid, 0, 2, "Center Y (px)", self.center_row_spin)
        self._add_labeled_field(geometry_grid, 1, 0, "Distance", self.distance_spin)
        self._add_labeled_field(geometry_grid, 1, 2, "Pixel Size", self.pixel_size_spin)
        self._add_labeled_field(geometry_grid, 2, 0, "Wavelength (Å)", self.wavelength_spin)
        self._add_labeled_field(geometry_grid, 2, 2, "Incident Angle", self.incident_angle_spin)
        self._add_labeled_field(geometry_grid, 3, 0, "φ Zero Direction", self.phi_zero_direction_combo)
        self.geometry_section.body_layout.addWidget(geometry_widget)

        self.sampling_section = _CollapsibleSection(
            "Sampling",
            expanded=True,
            collapsible=False,
        )
        sampling_widget = QtWidgets.QWidget()
        sampling_grid = QtWidgets.QGridLayout(sampling_widget)
        sampling_grid.setContentsMargins(0, 0, 0, 0)
        sampling_grid.setHorizontalSpacing(12)
        sampling_grid.setVerticalSpacing(8)
        self._add_labeled_field(sampling_grid, 0, 0, "2θ Bins", self.radial_bins_spin)
        self._add_labeled_field(sampling_grid, 1, 0, "φ Bins", self.azimuth_bins_spin)
        self.sampling_section.body_layout.addWidget(sampling_widget)

        self.profiles_section = _CollapsibleSection(
            "Profiles",
            expanded=True,
            collapsible=False,
        )
        profiles_widget = QtWidgets.QWidget()
        profiles_layout = QtWidgets.QVBoxLayout(profiles_widget)
        profiles_layout.setContentsMargins(0, 0, 0, 0)
        profiles_layout.setSpacing(8)
        profiles_layout.addWidget(self.image_log_button)
        profiles_layout.addWidget(self.bottom_log_button)
        profiles_layout.addWidget(self.left_log_button)
        profiles_layout.addStretch(1)
        self.profiles_section.body_layout.addWidget(profiles_widget)

        settings_layout.addWidget(self.geometry_section, 2)
        settings_layout.addWidget(self.sampling_section, 1)
        settings_layout.addWidget(self.profiles_section, 1)
        main_layout.addWidget(settings_shelf)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.ci.layout.setColumnStretchFactor(0, 1)
        self.graphics.ci.layout.setColumnStretchFactor(1, 5)
        self.graphics.ci.layout.setColumnStretchFactor(2, 1)
        self.graphics.ci.layout.setRowStretchFactor(0, 5)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        main_layout.addWidget(self.graphics, 1)

        self.left_plot = self.graphics.addPlot(row=0, col=0)
        self.image_view = _ImageViewBox(owner=self)
        self.image_plot = self.graphics.addPlot(row=0, col=1, viewBox=self.image_view)
        self.bottom_plot = self.graphics.addPlot(row=1, col=1)
        self.hist_lut = pg.HistogramLUTItem()
        self.graphics.addItem(self.hist_lut, row=0, col=2, rowspan=2)

        self.image_plot.setLabel("bottom", "X pixels")
        self.image_plot.setLabel("left", "Y pixels")
        self.image_plot.setAspectLocked(True)
        self.image_plot.showGrid(x=True, y=True, alpha=0.2)
        self.image_view = self.image_plot.getViewBox()
        self.image_view.setMouseMode(pg.ViewBox.RectMode)

        self.bottom_plot.setLabel("left", "Intensity")
        self.bottom_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bottom_plot.setXLink(self.image_plot)

        self.left_plot.setLabel("bottom", "Intensity")
        self.left_plot.showGrid(x=True, y=True, alpha=0.2)
        self.left_plot.setYLink(self.image_plot)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setAutoDownsample(True)
        self.image_plot.addItem(self.image_item)
        self.hist_lut.setImageItem(self.image_item)
        try:
            self.hist_lut.gradient.loadPreset("turbo")
        except Exception:
            pass

        crosshair_pen = pg.mkPen(color="#ff3b30", width=1)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.image_plot.addItem(self.v_line)
        self.image_plot.addItem(self.h_line)
        self.beam_center_marker = pg.ScatterPlotItem(
            size=14,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen("#ffd60a", width=2),
            symbol="+",
        )
        self.image_plot.addItem(self.beam_center_marker)

        self.bottom_curve = self.bottom_plot.plot([], [], pen=pg.mkPen("#1e90ff", width=1))
        self.left_curve = self.left_plot.plot([], [], pen=pg.mkPen("#2ecc71", width=1))

        self.bottom_marker = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("#ff3b30"), pen=None)
        self.left_marker = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("#ff3b30"), pen=None)
        self.bottom_plot.addItem(self.bottom_marker)
        self.left_plot.addItem(self.left_marker)

        self.status_view_label = QtWidgets.QLabel("View: Detector")
        self.status_fps_label = QtWidgets.QLabel(f"Render Target: {int(self.TARGET_FPS)} FPS")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.status_view_label)
        self.statusBar().addPermanentWidget(self.status_fps_label)

        self.open_button.clicked.connect(self.open_file_dialog)
        self.save_button.clicked.connect(self.save_current_view)
        self.reset_button.clicked.connect(self.reset_zoom)
        self.detector_view_button.clicked.connect(
            lambda checked: self._request_view_mode("detector") if checked else None
        )
        self.angle_space_view_button.clicked.connect(
            lambda checked: self._request_view_mode("angle_space") if checked else None
        )
        self.q_space_view_button.clicked.connect(
            lambda checked: self._request_view_mode("q_space") if checked else None
        )
        self.pick_center_button.toggled.connect(self._on_pick_center_toggled)
        self.roi_button.toggled.connect(self._on_roi_toggled)
        self.image_log_button.toggled.connect(self.set_image_log_mode)
        self.bottom_log_button.toggled.connect(self.set_bottom_log_mode)
        self.left_log_button.toggled.connect(self.set_left_log_mode)
        self.center_col_spin.valueChanged.connect(self._on_beam_center_spin_changed)
        self.center_row_spin.valueChanged.connect(self._on_beam_center_spin_changed)
        self.wavelength_spin.valueChanged.connect(self._on_wavelength_changed)
        self.incident_angle_spin.valueChanged.connect(self._on_incident_angle_changed)
        self.phi_zero_direction_combo.currentTextChanged.connect(
            self._on_phi_zero_direction_changed
        )

        self.mouse_proxy = pg.SignalProxy(
            self.image_plot.scene().sigMouseMoved,
            rateLimit=240,
            slot=self.on_scene_mouse_moved,
        )
        self.image_plot.scene().sigMouseClicked.connect(self.on_scene_mouse_clicked)

        refresh_interval = max(1, int(round(1000.0 / self.TARGET_FPS)))
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(refresh_interval)
        self.refresh_timer.timeout.connect(self.render_pending_cursor)
        self.refresh_timer.start()
        self._set_interaction_enabled(False)
        self._set_status_message("Open a detector image to begin.")

    def _set_interaction_enabled(self, enabled):
        has_detector = enabled and self.detector_data is not None
        has_display = enabled and self.data is not None
        self.save_button.setEnabled(has_display)
        self.reset_button.setEnabled(has_display)
        self.detector_view_button.setEnabled(has_detector)
        self.angle_space_view_button.setEnabled(has_detector)
        self.q_space_view_button.setEnabled(has_detector)
        self.pick_center_button.setEnabled(has_detector and self.current_view_mode == "detector")
        self.roi_button.setEnabled(has_display and self.current_view_mode == "angle_space")
        self.image_log_button.setEnabled(has_display)
        self.bottom_log_button.setEnabled(has_display)
        self.left_log_button.setEnabled(has_display)
        self.center_col_spin.setEnabled(has_detector)
        self.center_row_spin.setEnabled(has_detector)
        self.distance_spin.setEnabled(has_detector)
        self.pixel_size_spin.setEnabled(has_detector)
        self.wavelength_spin.setEnabled(has_detector)
        self.incident_angle_spin.setEnabled(has_detector)
        self.radial_bins_spin.setEnabled(has_detector)
        self.azimuth_bins_spin.setEnabled(has_detector)
        self.phi_zero_direction_combo.setEnabled(has_detector)
        self._sync_view_controls()

    def _set_beam_center(self, row, col, update_spins=True):
        if self.detector_data is not None:
            row = float(np.clip(row, 0.0, self.detector_data.shape[0] - 1))
            col = float(np.clip(col, 0.0, self.detector_data.shape[1] - 1))
        self.beam_center_row = float(row)
        self.beam_center_col = float(col)
        if update_spins:
            self.center_row_spin.blockSignals(True)
            self.center_col_spin.blockSignals(True)
            self.center_row_spin.setValue(self.beam_center_row)
            self.center_col_spin.setValue(self.beam_center_col)
            self.center_row_spin.blockSignals(False)
            self.center_col_spin.blockSignals(False)
        self._update_beam_center_marker()

    def _update_beam_center_marker(self):
        visible = (
            self.current_view_mode == "detector"
            and self.detector_data is not None
            and self.beam_center_row is not None
            and self.beam_center_col is not None
        )
        self.beam_center_marker.setVisible(visible)
        if visible:
            self.beam_center_marker.setData(
                [self.beam_center_col],
                [self.beam_center_row],
            )

    def _on_beam_center_spin_changed(self, _value):
        if self.detector_data is None:
            return
        self._set_beam_center(
            self.center_row_spin.value(),
            self.center_col_spin.value(),
            update_spins=False,
        )

    def _on_pick_center_toggled(self, enabled):
        if enabled and self.current_view_mode != "detector":
            self.pick_center_button.blockSignals(True)
            self.pick_center_button.setChecked(False)
            self.pick_center_button.blockSignals(False)
            return
        if enabled:
            self._set_status_message(
                "Click the detector image to place the beam center. The click snaps to the nearest peak top."
            )
        elif self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _on_roi_toggled(self, enabled):
        if enabled and not self._roi_supported_in_current_view():
            self.roi_button.blockSignals(True)
            self.roi_button.setChecked(False)
            self.roi_button.blockSignals(False)
            self._set_roi_overlay_visible(False)
            self._close_roi_profile_window()
            self._roi_window_pending_initial_draw = False
            return
        if enabled:
            self._set_status_message(
                "Left-click and drag in empty space to create another ROI. Drag existing corners to refine selections."
            )
            if self.roi_items:
                self._roi_window_pending_initial_draw = False
                self._set_roi_overlay_visible(True, refresh_profiles=False)
                self._update_roi_profile_window(show_window=True)
            else:
                self._roi_window_pending_initial_draw = True
                self._close_roi_profile_window()
        else:
            self._set_roi_overlay_visible(False)
            self._close_roi_profile_window()
            self._roi_window_pending_initial_draw = False
        if self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _on_roi_region_changed(self, _item=None):
        if self._updating_roi or self.data is None:
            return
        self._update_roi_profile_window(
            show_window=bool(
                self.roi_button.isChecked()
                and self._roi_window_pending_initial_draw
                and bool(self._visible_roi_items())
            )
        )
        self.update_cursor(
            self.last_ix if self.last_ix is not None else self.width // 2,
            self.last_iy if self.last_iy is not None else self.height // 2,
            force=True,
        )

    def _snap_beam_center_to_peak(self, row, col):
        if self.detector_data is None:
            return float(row), float(col)

        data = np.asarray(self.detector_data, dtype=np.float64)
        row = float(np.clip(row, 0.0, data.shape[0] - 1))
        col = float(np.clip(col, 0.0, data.shape[1] - 1))
        radius = int(max(1, self.BEAM_CENTER_SEARCH_RADIUS_PX))
        row_index = int(round(row))
        col_index = int(round(col))

        row_min = max(0, row_index - radius)
        row_max = min(data.shape[0] - 1, row_index + radius)
        col_min = max(0, col_index - radius)
        col_max = min(data.shape[1] - 1, col_index + radius)
        window = data[row_min : row_max + 1, col_min : col_max + 1]
        if window.size == 0:
            return row, col

        smooth = self._smooth_peak_window(window)
        valid = np.isfinite(smooth)
        if not np.any(valid):
            return row, col

        peak_map = valid.copy()
        padded = np.pad(smooth, 1, mode="edge")
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbour = padded[
                    1 + row_offset : 1 + row_offset + smooth.shape[0],
                    1 + col_offset : 1 + col_offset + smooth.shape[1],
                ]
                peak_map &= smooth >= neighbour
        candidates = np.argwhere(peak_map)
        if candidates.size == 0:
            flat_index = int(np.nanargmax(np.where(valid, smooth, -np.inf)))
            candidates = np.array([np.unravel_index(flat_index, smooth.shape)])

        local_row = row - float(row_min)
        local_col = col - float(col_min)
        distances_sq = (
            (candidates[:, 0].astype(np.float64) - local_row) ** 2
            + (candidates[:, 1].astype(np.float64) - local_col) ** 2
        )
        intensities = smooth[candidates[:, 0], candidates[:, 1]]
        best_index = np.lexsort((-intensities, distances_sq))[0]
        peak_row = int(candidates[best_index, 0])
        peak_col = int(candidates[best_index, 1])

        row_offset = 0.0
        col_offset = 0.0
        if 0 < peak_row < smooth.shape[0] - 1:
            row_offset = self._parabolic_peak_offset(
                smooth[peak_row - 1, peak_col],
                smooth[peak_row, peak_col],
                smooth[peak_row + 1, peak_col],
            )
        if 0 < peak_col < smooth.shape[1] - 1:
            col_offset = self._parabolic_peak_offset(
                smooth[peak_row, peak_col - 1],
                smooth[peak_row, peak_col],
                smooth[peak_row, peak_col + 1],
            )

        snapped_row = float(np.clip(row_min + peak_row + row_offset, 0.0, data.shape[0] - 1))
        snapped_col = float(np.clip(col_min + peak_col + col_offset, 0.0, data.shape[1] - 1))
        return snapped_row, snapped_col

    def _current_phi_zero_direction(self):
        direction = self.phi_zero_direction_combo.currentText().strip().lower()
        return direction or DEFAULT_PHI_ZERO_DIRECTION

    def _start_angle_space_warmup(self):
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            return
        self._warmup_thread = threading.Thread(
            target=warm_angle_space_engine,
            kwargs={"workers": 1},
            daemon=True,
        )
        self._warmup_thread.start()

    def _update_angle_space_display(self, *, force_show=False):
        if self.angle_space_result is None:
            return
        cake, radial_deg, phi_deg = prepare_gui_phi_display(
            self.angle_space_result,
            phi_min_deg=DEFAULT_GUI_PHI_MIN_DEG,
            phi_max_deg=DEFAULT_GUI_PHI_MAX_DEG,
            zero_direction=self._current_phi_zero_direction(),
        )
        self.angle_space_cake = cake
        if force_show or self.current_view_mode == "angle_space":
            self.show_angle_space_view(cake, radial_deg, phi_deg)

    def _update_q_space_display(self, *, force_show=False):
        if self.angle_space_result is None:
            return
        cake, radial_deg, phi_deg = prepare_gui_phi_display(
            self.angle_space_result,
            phi_min_deg=DEFAULT_GUI_PHI_MIN_DEG,
            phi_max_deg=DEFAULT_GUI_PHI_MAX_DEG,
            zero_direction=self._current_phi_zero_direction(),
        )
        self.angle_space_cake = cake
        self.q_space_result = convert_phi_2theta_to_qr_qz_space(
            cake,
            radial_deg,
            phi_deg,
            wavelength_angstrom=self.wavelength_spin.value(),
            incident_angle_deg=self.incident_angle_spin.value(),
            qr_bins=self.radial_bins_spin.value(),
            qz_bins=self.azimuth_bins_spin.value(),
        )
        if force_show or self.current_view_mode == "q_space":
            self.show_q_space_view(
                self.q_space_result.intensity,
                self.q_space_result.qr,
                self.q_space_result.qz,
            )

    def _on_phi_zero_direction_changed(self, _text):
        self.q_space_result = None
        self._update_angle_space_display(force_show=self.current_view_mode == "angle_space")
        if self.current_view_mode == "q_space":
            self._update_q_space_display(force_show=True)

    def _on_wavelength_changed(self, _value):
        self.q_space_result = None
        if self.current_view_mode == "q_space" and self.angle_space_result is not None:
            self._update_q_space_display(force_show=True)

    def _on_incident_angle_changed(self, _value):
        self.q_space_result = None
        if self.current_view_mode == "q_space" and self.angle_space_result is not None:
            self._update_q_space_display(force_show=True)

    def _set_image_aspect_locked(self, locked):
        self.image_view.setAspectLocked(bool(locked))

    def _start_conversion(self, *, target_view="angle_space"):
        if self._converting or self.detector_data is None:
            return
        self._conversion_target_view = str(target_view)
        if self.beam_center_row is None or self.beam_center_col is None:
            self._set_beam_center(
                (self.detector_data.shape[0] - 1) / 2.0,
                (self.detector_data.shape[1] - 1) / 2.0,
            )
        if "_AngleSpaceWorker" not in globals():
            try:
                result = convert_image_to_phi_2theta_space(
                    self.detector_data,
                    distance_mm=self.distance_spin.value(),
                    pixel_size_mm=self.pixel_size_spin.value(),
                    center_row_px=self.beam_center_row,
                    center_col_px=self.beam_center_col,
                    radial_bins=self.radial_bins_spin.value(),
                    azimuth_bins=self.azimuth_bins_spin.value(),
                )
            except Exception as exc:  # pragma: no cover - fallback error path
                self._on_conversion_failed(str(exc))
            else:
                self._on_conversion_loaded({"result": result})
                self._set_interaction_enabled(self.data is not None)
            return

        self._converting = True
        self.open_button.setEnabled(False)
        self._set_interaction_enabled(False)
        if self._conversion_target_view == "q_space":
            self._set_status_message(
                f"Converting to q-space via φ/2θ with {DEFAULT_ANGLE_SPACE_WORKERS} workers ..."
            )
        else:
            self._set_status_message(
                f"Converting to φ/2θ with {DEFAULT_ANGLE_SPACE_WORKERS} workers ..."
            )

        self._conversion_thread = QtCore.QThread(self)
        self._conversion_worker = _AngleSpaceWorker(
            self.detector_data,
            distance_mm=self.distance_spin.value(),
            pixel_size_mm=self.pixel_size_spin.value(),
            center_row_px=self.beam_center_row,
            center_col_px=self.beam_center_col,
            radial_bins=self.radial_bins_spin.value(),
            azimuth_bins=self.azimuth_bins_spin.value(),
        )
        self._conversion_worker.moveToThread(self._conversion_thread)

        self._conversion_thread.started.connect(self._conversion_worker.run)
        self._conversion_worker.loaded.connect(self._on_conversion_loaded)
        self._conversion_worker.failed.connect(self._on_conversion_failed)
        self._conversion_worker.finished.connect(self._conversion_thread.quit)
        self._conversion_worker.finished.connect(self._conversion_worker.deleteLater)
        self._conversion_thread.finished.connect(self._on_conversion_finished)
        self._conversion_thread.finished.connect(self._conversion_thread.deleteLater)
        self._conversion_thread.start()

    def _on_conversion_loaded(self, payload):
        self.angle_space_result = payload["result"]
        self.q_space_result = None
        if self._conversion_target_view == "q_space":
            self._update_q_space_display(force_show=True)
        else:
            self._update_angle_space_display(force_show=True)

    def _on_conversion_failed(self, message):
        target_label = "q-space via φ/2θ" if self._conversion_target_view == "q_space" else "φ/2θ space"
        QtWidgets.QMessageBox.critical(
            self,
            "Conversion Error",
            f"Failed to convert the active image to {target_label}.\n\n{message}",
        )
        if self.data is not None:
            self._set_status_message(
                "Conversion failed. Adjust the geometry and try again."
            )

    def _on_conversion_finished(self):
        self._converting = False
        self.open_button.setEnabled(True)
        self._set_interaction_enabled(self.data is not None)
        self._conversion_worker = None
        self._conversion_thread = None

    def convert_active_image(self):
        self._start_conversion(target_view="angle_space")

    def convert_active_image_to_q_space(self):
        self._start_conversion(target_view="q_space")

    def _start_loader(self, filename):
        if self._loading:
            return
        if "_OSCLoadWorker" not in globals():
            # Fallback path if Qt signal bindings are unavailable.
            try:
                data = read_detector_image(filename)
            except Exception as exc:  # pragma: no cover - fallback error path
                self._on_loader_failed(filename, str(exc))
            else:
                self._on_loader_loaded(filename, data)
                self._set_interaction_enabled(self.data is not None)
            return

        self._loading = True
        self.open_button.setEnabled(False)
        self._set_interaction_enabled(False)
        self._set_status_message(f"Loading: {Path(filename).name} ...")

        self._loader_thread = QtCore.QThread(self)
        self._loader_worker = _OSCLoadWorker(filename)
        self._loader_worker.moveToThread(self._loader_thread)

        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.loaded.connect(self._on_loader_loaded)
        self._loader_worker.failed.connect(self._on_loader_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._on_loader_finished)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()

    def _on_loader_loaded(self, filename, data):
        self.filename = filename
        self.setWindowTitle(f"OSC Reader (High FPS Viewer) - {Path(filename).name}")
        print("Data loaded successfully.")
        print("Preparing profile sampling grid...")
        self.set_data(data)
        print("Profile sampling ready.")

    def _on_loader_failed(self, filename, message):
        QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to read:\n{filename}\n\n{message}")
        self._set_status_message("Load failed. Choose another detector image with Open Image.")

    def _on_loader_finished(self):
        self._loading = False
        self.open_button.setEnabled(True)
        self._set_interaction_enabled(self.data is not None)
        self._loader_worker = None
        self._loader_thread = None

    def open_file_dialog(self):
        if self._loading or self._converting:
            return
        start_dir = str(Path(self.filename).parent) if self.filename else str(Path.cwd())
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select a detector image",
            start_dir,
            get_detector_file_dialog_filter(),
        )
        if selected:
            self.load_file(selected)

    def load_file(self, filename):
        if self._converting:
            return
        if not filename or not os.path.isfile(filename):
            QtWidgets.QMessageBox.warning(self, "File Error", f"File not found:\n{filename}")
            return
        self._start_loader(filename)

    def set_data(self, data):
        detector_data = np.asarray(data)
        if detector_data.ndim != 2:
            raise ValueError("Detector data must be a 2D array.")
        self._clear_roi_items()
        self.detector_data = detector_data
        self.angle_space_result = None
        self.angle_space_cake = None
        self.q_space_result = None

        detector_height, detector_width = self.detector_data.shape
        self.center_col_spin.setRange(-1_000_000.0, max(1_000_000.0, float(detector_width)))
        self.center_row_spin.setRange(-1_000_000.0, max(1_000_000.0, float(detector_height)))
        self.radial_bins_spin.setValue(detector_width)
        self.azimuth_bins_spin.setValue(detector_height)
        self._set_beam_center(
            (detector_height - 1) / 2.0,
            (detector_width - 1) / 2.0,
        )
        self.show_detector_view()

    def _set_display_data(self, data, x_coords, y_coords, x_label, y_label):
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("Display data must be a 2D array.")

        self.height, self.width = self.data.shape
        self.display_x_full = np.asarray(x_coords, dtype=np.float64)
        self.display_y_full = np.asarray(y_coords, dtype=np.float64)
        if self.display_x_full.size != self.width or self.display_y_full.size != self.height:
            raise ValueError("Coordinate vectors must match the display data shape.")
        self.display_x_edges = self._coord_edges(self.display_x_full)
        self.display_y_edges = self._coord_edges(self.display_y_full)
        self.display_x_label = x_label
        self.display_y_label = y_label
        self.data_min, self.data_max = self._estimate_data_stats()

        self.prepare_profile_cache()

        self.image_plot.setLabel("bottom", self.display_x_label)
        self.image_plot.setLabel("left", self.display_y_label)
        self._refresh_image_display()
        self.image_item.setRect(
            QtCore.QRectF(
                float(self.display_x_edges[0]),
                float(self.display_y_edges[0]),
                float(self.display_x_edges[-1] - self.display_x_edges[0]),
                float(self.display_y_edges[-1] - self.display_y_edges[0]),
            )
        )

        self.image_plot.setLimits(
            xMin=float(self.display_x_edges[0]),
            xMax=float(self.display_x_edges[-1]),
            yMin=float(self.display_y_edges[0]),
            yMax=float(self.display_y_edges[-1]),
        )
        self.image_view.setRange(
            xRange=(float(self.display_x_edges[0]), float(self.display_x_edges[-1])),
            yRange=(float(self.display_y_edges[0]), float(self.display_y_edges[-1])),
            padding=0.0,
        )

        low, high = self._safe_limits_linear(self.data_min, self.data_max)
        self.bottom_plot.setYRange(low, high, padding=0.02)
        self.left_plot.setXRange(low, high, padding=0.02)

        self.pending_dirty = False
        self.last_ix = None
        self.last_iy = None
        self.last_bottom_range = (None, None)
        self.last_left_range = (None, None)
        self._update_beam_center_marker()
        self.update_cursor(self.width // 2, self.height // 2, force=True)
        self._set_interaction_enabled(True)

    def show_detector_view(self):
        if self.detector_data is None:
            return
        self.current_view_mode = "detector"
        self.pick_center_button.blockSignals(True)
        self.pick_center_button.setChecked(False)
        self.pick_center_button.blockSignals(False)
        self._set_image_aspect_locked(True)
        x_coords = np.arange(self.detector_data.shape[1], dtype=np.float64)
        y_coords = np.arange(self.detector_data.shape[0], dtype=np.float64)
        self._set_display_data(
            self.detector_data,
            x_coords,
            y_coords,
            "X pixels",
            "Y pixels",
        )

    def show_angle_space_view(self, cake, radial_deg, phi_deg):
        self.current_view_mode = "angle_space"
        self.pick_center_button.blockSignals(True)
        self.pick_center_button.setChecked(False)
        self.pick_center_button.blockSignals(False)
        self._set_image_aspect_locked(False)
        self._set_display_data(
            cake,
            radial_deg,
            phi_deg,
            "2θ (degrees)",
            "φ (degrees)",
        )
        self.image_plot.setLimits(
            yMin=DEFAULT_GUI_PHI_MIN_DEG,
            yMax=DEFAULT_GUI_PHI_MAX_DEG,
        )
        self.image_view.setYRange(
            DEFAULT_GUI_PHI_MIN_DEG,
            DEFAULT_GUI_PHI_MAX_DEG,
            padding=0.0,
        )

    def show_q_space_view(self, q_image, qr, qz):
        self.current_view_mode = "q_space"
        self.pick_center_button.blockSignals(True)
        self.pick_center_button.setChecked(False)
        self.pick_center_button.blockSignals(False)
        self._set_image_aspect_locked(False)
        self._set_display_data(
            q_image,
            qr,
            qz,
            "qr (1/Å)",
            "qz (1/Å)",
        )

    def prepare_profile_cache(self):
        self.step_x = max(1, self.width // self.MAX_PROFILE_POINTS)
        self.step_y = max(1, self.height // self.MAX_PROFILE_POINTS)

        self.x_profile_coords = self.display_x_full[::self.step_x].astype(np.float32, copy=False)
        self.y_profile_coords = self.display_y_full[::self.step_y].astype(np.float32, copy=False)

    def _positive_only(self, values):
        clipped = np.asarray(values, dtype=np.float64)
        clipped = clipped[clipped > 0]
        if clipped.size == 0:
            return np.array([self.LOG_EPS], dtype=np.float64)
        return clipped

    def _profile_visible_slice(self):
        x_low, x_high = sorted(self.image_view.viewRange()[0])
        y_low, y_high = sorted(self.image_view.viewRange()[1])

        x_start_idx = np.searchsorted(self.x_profile_coords, x_low, side="left")
        x_end_idx = np.searchsorted(self.x_profile_coords, x_high, side="right")
        y_start_idx = np.searchsorted(self.y_profile_coords, y_low, side="left")
        y_end_idx = np.searchsorted(self.y_profile_coords, y_high, side="right")

        return x_start_idx, x_end_idx, y_start_idx, y_end_idx

    def set_bottom_log_mode(self, enabled):
        self.bottom_log_enabled = bool(enabled)
        self.bottom_plot.setLogMode(x=False, y=self.bottom_log_enabled)
        self.last_bottom_range = (None, None)
        if self.last_ix is not None and self.last_iy is not None:
            self.update_cursor(self.last_ix, self.last_iy, force=True)

    def set_left_log_mode(self, enabled):
        self.left_log_enabled = bool(enabled)
        self.left_plot.setLogMode(x=self.left_log_enabled, y=False)
        self.last_left_range = (None, None)
        if self.last_ix is not None and self.last_iy is not None:
            self.update_cursor(self.last_ix, self.last_iy, force=True)

    def set_image_log_mode(self, enabled):
        self.image_log_enabled = bool(enabled)
        if self.data is not None:
            self._refresh_image_display()

    def on_scene_mouse_moved(self, event):
        if self.data is None:
            return

        position = event[0]
        if not self.image_plot.sceneBoundingRect().contains(position):
            return

        mouse_point = self.image_view.mapSceneToView(position)
        x_data = float(mouse_point.x())
        y_data = float(mouse_point.y())
        if (
            self.display_x_edges[0] <= x_data <= self.display_x_edges[-1]
            and self.display_y_edges[0] <= y_data <= self.display_y_edges[-1]
        ):
            self.pending_x = x_data
            self.pending_y = y_data
            self.pending_dirty = True

    def on_scene_mouse_clicked(self, event):
        if self.data is None:
            return

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        position = event.scenePos()
        if not self.image_plot.sceneBoundingRect().contains(position):
            return

        mouse_point = self.image_view.mapSceneToView(position)
        x_data = float(mouse_point.x())
        y_data = float(mouse_point.y())
        if self.current_view_mode == "detector" and self.pick_center_button.isChecked():
            snapped_row, snapped_col = self._snap_beam_center_to_peak(y_data, x_data)
            self._set_beam_center(snapped_row, snapped_col)
            x_data = snapped_col
            y_data = snapped_row
            self.pick_center_button.blockSignals(True)
            self.pick_center_button.setChecked(False)
            self.pick_center_button.blockSignals(False)
        self.pending_x = x_data
        self.pending_y = y_data
        self.pending_dirty = True
        self.render_pending_cursor()

    def render_pending_cursor(self):
        if self.data is None or not self.pending_dirty:
            return

        ix = self._index_from_edges(self.pending_x, self.display_x_edges)
        iy = self._index_from_edges(self.pending_y, self.display_y_edges)
        self.pending_dirty = False
        self.update_cursor(ix, iy)

    def update_cursor(self, ix, iy, force=False):
        if self.data is None:
            return
        if not force and ix == self.last_ix and iy == self.last_iy:
            return

        intensity = float(self.data[iy, ix])
        x_value = float(self.display_x_full[ix])
        y_value = float(self.display_y_full[iy])

        self.v_line.setPos(x_value)
        self.h_line.setPos(y_value)

        row_profile = self.data[iy, ::self.step_x]
        col_profile = self.data[::self.step_y, ix]

        if self.bottom_log_enabled:
            row_plot = np.maximum(np.asarray(row_profile, dtype=np.float64), self.LOG_EPS)
            bottom_marker_y = max(intensity, self.LOG_EPS)
        else:
            row_plot = row_profile
            bottom_marker_y = intensity

        if self.left_log_enabled:
            col_plot = np.maximum(np.asarray(col_profile, dtype=np.float64), self.LOG_EPS)
            left_marker_x = max(intensity, self.LOG_EPS)
        else:
            col_plot = col_profile
            left_marker_x = intensity

        self.bottom_curve.setData(self.x_profile_coords, row_plot)
        self.left_curve.setData(col_plot, self.y_profile_coords)
        self.bottom_marker.setData([x_value], [bottom_marker_y])
        self.left_marker.setData([left_marker_x], [y_value])

        x_start_idx, x_end_idx, y_start_idx, y_end_idx = self._profile_visible_slice()
        visible_row = row_profile[x_start_idx:x_end_idx]
        visible_col = col_profile[y_start_idx:y_end_idx]
        if visible_row.size == 0:
            visible_row = row_profile
        if visible_col.size == 0:
            visible_col = col_profile

        if self.bottom_log_enabled:
            row_for_range = self._positive_only(visible_row)
        else:
            row_for_range = np.asarray(visible_row, dtype=np.float64)
        if self.left_log_enabled:
            col_for_range = self._positive_only(visible_col)
        else:
            col_for_range = np.asarray(visible_col, dtype=np.float64)

        row_low, row_high = self._compute_axis_range(row_for_range, self.bottom_log_enabled)
        col_low, col_high = self._compute_axis_range(col_for_range, self.left_log_enabled)

        if (row_low, row_high) != self.last_bottom_range:
            self.bottom_plot.setYRange(row_low, row_high, padding=0.02)
            self.last_bottom_range = (row_low, row_high)
        if (col_low, col_high) != self.last_left_range:
            self.left_plot.setXRange(col_low, col_high, padding=0.02)
            self.last_left_range = (col_low, col_high)

        roi_summary = self._roi_summary()
        self._set_status_message(
            "Left-click+drag: zoom | Move mouse: inspect | "
            f"{self.display_x_label}: {self._x_text(ix)}  "
            f"{self.display_y_label}: {self._y_text(iy)}  "
            f"Intensity: {intensity:.0f}  "
            f"{self._beam_center_summary()}"
            + (f"  {roi_summary}" if roi_summary else "")
        )

        self.last_ix = ix
        self.last_iy = iy

    def reset_zoom(self):
        if self.data is None:
            return
        self.image_view.setRange(
            xRange=(float(self.display_x_edges[0]), float(self.display_x_edges[-1])),
            yRange=(float(self.display_y_edges[0]), float(self.display_y_edges[-1])),
            padding=0.0,
        )

    def save_current_view(self):
        if self.data is None:
            return

        default_stem = Path(self.filename).stem if self.filename else "osc_view"
        if self.current_view_mode == "angle_space":
            default_stem = f"{default_stem}_phi_2theta"
        elif self.current_view_mode == "q_space":
            default_stem = f"{default_stem}_q_space"
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save rendered image",
            str(Path.cwd() / f"{default_stem}.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*)",
        )
        if not selected:
            return

        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self.image_plot)
        try:
            exporter.parameters()["width"] = max(1200, int(self.image_plot.width() * 2))
        except Exception:
            pass
        exporter.export(selected)
        print(f"Saved image to {selected}")


def _require_qt_viewer_stack():
    if pg is None or QtWidgets is None or QtCore is None:
        raise ImportError(
            "High-FPS viewer requires pyqtgraph and a Qt backend.\n"
            "Install with: pip install pyqtgraph pyside6"
        )


def visualize_detector_data(filename=None):
    """Visualize a detector image in the high-FPS Qt viewer.

    Parameters
    ----------
    filename : str | None
        The path to the detector image. If omitted, a file picker is shown.
    """
    _require_qt_viewer_stack()

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    if filename is None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select a detector image",
            str(Path.cwd()),
            get_detector_file_dialog_filter(),
        )
        filename = selected
        if not filename:
            print("No detector image selected.")
            return

    if not os.path.isfile(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return

    window = OSCViewerWindow(filename)
    _ACTIVE_WINDOWS.append(window)
    window.showMaximized()

    try:
        app.exec()
    finally:
        if window in _ACTIVE_WINDOWS:
            _ACTIVE_WINDOWS.remove(window)


def visualize_osc_data(filename=None):
    """Backwards-compatible alias for :func:`visualize_detector_data`."""
    return visualize_detector_data(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a detector image.")
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Optional path to the detector image. If omitted, a file picker is shown.",
    )
    args = parser.parse_args()

    visualize_detector_data(args.filename)
