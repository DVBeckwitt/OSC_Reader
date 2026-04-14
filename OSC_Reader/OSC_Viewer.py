import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .angle_space import (
    DEFAULT_ANGLE_SPACE_WORKERS,
    convert_image_to_phi_2theta_space,
    prepare_gui_phi_display,
)
from .OSC_Reader import read_osc

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
                data = read_osc(self.filename)
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
                cake, radial_deg, phi_deg = prepare_gui_phi_display(result)
            except Exception as exc:  # pragma: no cover - runtime/UI path
                self.failed.emit(str(exc))
            else:
                self.loaded.emit(
                    {
                        "result": result,
                        "cake": cake,
                        "radial_deg": radial_deg,
                        "phi_deg": phi_deg,
                    }
                )
            finally:
                self.finished.emit()


class OSCViewerWindow(QtWidgets.QMainWindow):
    TARGET_FPS = 60.0
    MAX_PROFILE_POINTS = 800
    LOG_EPS = 1e-3

    def __init__(self, filename):
        super().__init__()
        self.setWindowTitle("OSC Reader (High FPS Viewer)")
        self._apply_always_on_top()

        self.filename = None
        self.data = None
        self.detector_data = None
        self.angle_space_result = None
        self.angle_space_cake = None
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
        self._loading = False
        self._converting = False
        self._loader_thread = None
        self._loader_worker = None
        self._conversion_thread = None
        self._conversion_worker = None
        self.beam_center_row = None
        self.beam_center_col = None

        self._build_ui()
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

    def _estimate_data_stats(self):
        # Use a coarse sample for fast startup; profile axes are refined per-cursor.
        sample_target = 512
        sample_step_x = max(1, self.width // sample_target)
        sample_step_y = max(1, self.height // sample_target)
        sample = self.data[::sample_step_y, ::sample_step_x]
        sample_min = float(np.min(sample))
        sample_max = float(np.max(sample))
        if np.isclose(sample_min, sample_max):
            # Fallback for pathological cases.
            sample_min = float(np.min(self.data))
            sample_max = float(np.max(self.data))
        return sample_min, sample_max

    def _build_ui(self):
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        controls_layout = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open OSC")
        self.save_button = QtWidgets.QPushButton("Save Image")
        self.reset_button = QtWidgets.QPushButton("Reset Zoom")
        self.show_detector_button = QtWidgets.QPushButton("Show Detector")
        self.pick_center_button = QtWidgets.QPushButton("Pick Beam Center")
        self.convert_button = QtWidgets.QPushButton("Convert to φ/2θ")
        self.bottom_log_button = QtWidgets.QPushButton("Bottom Log Y")
        self.left_log_button = QtWidgets.QPushButton("Side Log X")
        self.pick_center_button.setCheckable(True)
        self.bottom_log_button.setCheckable(True)
        self.left_log_button.setCheckable(True)
        fps_label = QtWidgets.QLabel(f"Render target: {int(self.TARGET_FPS)} FPS")
        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.show_detector_button)
        controls_layout.addWidget(self.pick_center_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.bottom_log_button)
        controls_layout.addWidget(self.left_log_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(fps_label)
        main_layout.addLayout(controls_layout)

        geometry_layout = QtWidgets.QHBoxLayout()

        self.center_col_spin = QtWidgets.QDoubleSpinBox()
        self.center_row_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.radial_bins_spin = QtWidgets.QSpinBox()
        self.azimuth_bins_spin = QtWidgets.QSpinBox()

        for spin in (self.center_col_spin, self.center_row_spin):
            spin.setDecimals(2)
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setSingleStep(1.0)

        self.distance_spin.setDecimals(4)
        self.distance_spin.setRange(0.0001, 1_000_000.0)
        self.distance_spin.setSingleStep(0.1)
        self.distance_spin.setSuffix(" mm")
        self.distance_spin.setValue(75.0)

        self.pixel_size_spin.setDecimals(5)
        self.pixel_size_spin.setRange(0.00001, 1000.0)
        self.pixel_size_spin.setSingleStep(0.001)
        self.pixel_size_spin.setSuffix(" mm")
        self.pixel_size_spin.setValue(0.1)

        for spin in (self.radial_bins_spin, self.azimuth_bins_spin):
            spin.setRange(2, 10000)
            spin.setSingleStep(10)
        self.radial_bins_spin.setValue(1000)
        self.azimuth_bins_spin.setValue(720)

        geometry_layout.addWidget(QtWidgets.QLabel("Center X"))
        geometry_layout.addWidget(self.center_col_spin)
        geometry_layout.addWidget(QtWidgets.QLabel("Center Y"))
        geometry_layout.addWidget(self.center_row_spin)
        geometry_layout.addWidget(QtWidgets.QLabel("Distance"))
        geometry_layout.addWidget(self.distance_spin)
        geometry_layout.addWidget(QtWidgets.QLabel("Pixel Size"))
        geometry_layout.addWidget(self.pixel_size_spin)
        geometry_layout.addWidget(QtWidgets.QLabel("2θ Bins"))
        geometry_layout.addWidget(self.radial_bins_spin)
        geometry_layout.addWidget(QtWidgets.QLabel("φ Bins"))
        geometry_layout.addWidget(self.azimuth_bins_spin)
        geometry_layout.addStretch(1)
        main_layout.addLayout(geometry_layout)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.ci.layout.setColumnStretchFactor(0, 1)
        self.graphics.ci.layout.setColumnStretchFactor(1, 4)
        self.graphics.ci.layout.setColumnStretchFactor(2, 1)
        self.graphics.ci.layout.setRowStretchFactor(0, 4)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        main_layout.addWidget(self.graphics, 1)

        self.left_plot = self.graphics.addPlot(row=0, col=0)
        self.image_plot = self.graphics.addPlot(row=0, col=1)
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

        self.info_label = QtWidgets.QLabel("Left-click+drag: zoom | Move mouse: inspect")
        main_layout.addWidget(self.info_label)

        self.open_button.clicked.connect(self.open_file_dialog)
        self.save_button.clicked.connect(self.save_current_view)
        self.reset_button.clicked.connect(self.reset_zoom)
        self.show_detector_button.clicked.connect(self.show_detector_view)
        self.pick_center_button.toggled.connect(self._on_pick_center_toggled)
        self.convert_button.clicked.connect(self.convert_active_image)
        self.bottom_log_button.toggled.connect(self.set_bottom_log_mode)
        self.left_log_button.toggled.connect(self.set_left_log_mode)
        self.center_col_spin.valueChanged.connect(self._on_beam_center_spin_changed)
        self.center_row_spin.valueChanged.connect(self._on_beam_center_spin_changed)

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

    def _set_interaction_enabled(self, enabled):
        self.save_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.show_detector_button.setEnabled(enabled and self.detector_data is not None)
        self.pick_center_button.setEnabled(enabled and self.current_view_mode == "detector")
        self.convert_button.setEnabled(enabled and self.detector_data is not None)
        self.bottom_log_button.setEnabled(enabled)
        self.left_log_button.setEnabled(enabled)
        self.center_col_spin.setEnabled(enabled and self.detector_data is not None)
        self.center_row_spin.setEnabled(enabled and self.detector_data is not None)
        self.distance_spin.setEnabled(enabled and self.detector_data is not None)
        self.pixel_size_spin.setEnabled(enabled and self.detector_data is not None)
        self.radial_bins_spin.setEnabled(enabled and self.detector_data is not None)
        self.azimuth_bins_spin.setEnabled(enabled and self.detector_data is not None)

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
            self.info_label.setText(
                "Click the detector image to place the beam center."
            )
        elif self.data is not None:
            self.update_cursor(
                self.last_ix if self.last_ix is not None else self.width // 2,
                self.last_iy if self.last_iy is not None else self.height // 2,
                force=True,
            )

    def _start_conversion(self):
        if self._converting or self.detector_data is None:
            return
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
                cake, radial_deg, phi_deg = prepare_gui_phi_display(result)
            except Exception as exc:  # pragma: no cover - fallback error path
                self._on_conversion_failed(str(exc))
            else:
                self._on_conversion_loaded(
                    {
                        "result": result,
                        "cake": cake,
                        "radial_deg": radial_deg,
                        "phi_deg": phi_deg,
                    }
                )
                self._set_interaction_enabled(self.data is not None)
            return

        self._converting = True
        self.open_button.setEnabled(False)
        self._set_interaction_enabled(False)
        self.info_label.setText(
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
        self.angle_space_cake = payload["cake"]
        self.show_angle_space_view(
            payload["cake"],
            payload["radial_deg"],
            payload["phi_deg"],
        )

    def _on_conversion_failed(self, message):
        QtWidgets.QMessageBox.critical(
            self,
            "Conversion Error",
            f"Failed to convert the active image to φ/2θ space.\n\n{message}",
        )
        if self.data is not None:
            self.info_label.setText(
                "Conversion failed. Adjust the geometry and try again."
            )

    def _on_conversion_finished(self):
        self._converting = False
        self.open_button.setEnabled(True)
        self._set_interaction_enabled(self.data is not None)
        self._conversion_worker = None
        self._conversion_thread = None

    def convert_active_image(self):
        self._start_conversion()

    def _start_loader(self, filename):
        if self._loading:
            return
        if "_OSCLoadWorker" not in globals():
            # Fallback path if Qt signal bindings are unavailable.
            try:
                data = read_osc(filename)
            except Exception as exc:  # pragma: no cover - fallback error path
                self._on_loader_failed(filename, str(exc))
            else:
                self._on_loader_loaded(filename, data)
                self._set_interaction_enabled(self.data is not None)
            return

        self._loading = True
        self.open_button.setEnabled(False)
        self._set_interaction_enabled(False)
        self.info_label.setText(f"Loading: {Path(filename).name} ...")

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
        self.info_label.setText("Load failed. Choose another file with Open OSC.")

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
            "Select an OSC file",
            start_dir,
            "OSC files (*.osc);;All files (*)",
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
            raise ValueError("OSC data must be a 2D array.")
        self.detector_data = detector_data
        self.angle_space_result = None
        self.angle_space_cake = None

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
        self.image_item.setImage(self.data, autoLevels=False)
        self.image_item.setRect(
            QtCore.QRectF(
                float(self.display_x_edges[0]),
                float(self.display_y_edges[0]),
                float(self.display_x_edges[-1] - self.display_x_edges[0]),
                float(self.display_y_edges[-1] - self.display_y_edges[0]),
            )
        )

        slider_low = min(0.0, self.data_min)
        slider_high = self.data_max
        if np.isclose(slider_low, slider_high):
            slider_high = slider_low + 1.0

        vmin_default = float(np.clip(0.0, slider_low, slider_high))
        vmax_default = float(slider_low + 0.35 * (slider_high - slider_low))
        if vmax_default <= vmin_default:
            vmax_default = slider_high

        self.image_item.setLevels((vmin_default, vmax_default))
        self.hist_lut.setLevels(vmin_default, vmax_default)

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
        self._set_display_data(
            cake,
            radial_deg,
            phi_deg,
            "2θ (degrees)",
            "φ (degrees)",
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
            self._set_beam_center(y_data, x_data)
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

        self.info_label.setText(
            "Left-click+drag: zoom | Move mouse: inspect | "
            f"{self.display_x_label}: {self._x_text(ix)}  "
            f"{self.display_y_label}: {self._y_text(iy)}  "
            f"Intensity: {intensity:.0f}  "
            f"{self._beam_center_summary()}"
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


def visualize_osc_data(filename=None):
    """Visualizes an OSC file in a high-FPS Qt viewer.

    Parameters
    ----------
    filename : str | None
        The path to the OSC file. If omitted, a file picker is shown.
    """
    _require_qt_viewer_stack()

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    if filename is None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select an OSC file",
            str(Path.cwd()),
            "OSC files (*.osc);;All files (*)",
        )
        filename = selected
        if not filename:
            print("No OSC file selected.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an OSC file.")
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Optional path to the OSC file. If omitted, a file picker is shown.",
    )
    args = parser.parse_args()

    visualize_osc_data(args.filename)
