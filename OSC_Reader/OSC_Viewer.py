import argparse
import os
import sys
from pathlib import Path

import numpy as np

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
        self.height = 0
        self.width = 0
        self.data_min = 0.0
        self.data_max = 1.0

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
        self._loader_thread = None
        self._loader_worker = None

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
        self.bottom_log_button = QtWidgets.QPushButton("Bottom Log Y")
        self.left_log_button = QtWidgets.QPushButton("Side Log X")
        self.bottom_log_button.setCheckable(True)
        self.left_log_button.setCheckable(True)
        fps_label = QtWidgets.QLabel(f"Render target: {int(self.TARGET_FPS)} FPS")
        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.bottom_log_button)
        controls_layout.addWidget(self.left_log_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(fps_label)
        main_layout.addLayout(controls_layout)

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
        self.bottom_log_button.toggled.connect(self.set_bottom_log_mode)
        self.left_log_button.toggled.connect(self.set_left_log_mode)

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
        self.bottom_log_button.setEnabled(enabled)
        self.left_log_button.setEnabled(enabled)

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
        if self._loading:
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
        if not filename or not os.path.isfile(filename):
            QtWidgets.QMessageBox.warning(self, "File Error", f"File not found:\n{filename}")
            return
        self._start_loader(filename)

    def set_data(self, data):
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("OSC data must be a 2D array.")

        self.height, self.width = self.data.shape
        self.data_min, self.data_max = self._estimate_data_stats()

        self.prepare_profile_cache()

        self.image_item.setImage(self.data, autoLevels=False)
        self.image_item.setRect(QtCore.QRectF(0, 0, self.width, self.height))

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

        self.image_plot.setLimits(xMin=0, xMax=self.width, yMin=0, yMax=self.height)
        self.image_view.setRange(xRange=(0, self.width), yRange=(0, self.height), padding=0.0)

        low, high = self._safe_limits_linear(self.data_min, self.data_max)
        self.bottom_plot.setYRange(low, high, padding=0.02)
        self.left_plot.setXRange(low, high, padding=0.02)

        self.pending_dirty = False
        self.last_ix = None
        self.last_iy = None
        self.last_bottom_range = (None, None)
        self.last_left_range = (None, None)
        self.update_cursor(self.width // 2, self.height // 2, force=True)

    def prepare_profile_cache(self):
        self.step_x = max(1, self.width // self.MAX_PROFILE_POINTS)
        self.step_y = max(1, self.height // self.MAX_PROFILE_POINTS)

        self.x_profile_coords = np.arange(0, self.width, self.step_x, dtype=np.float32)
        self.y_profile_coords = np.arange(0, self.height, self.step_y, dtype=np.float32)

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
        if 0 <= x_data < self.width and 0 <= y_data < self.height:
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
        self.pending_x = float(mouse_point.x())
        self.pending_y = float(mouse_point.y())
        self.pending_dirty = True
        self.render_pending_cursor()

    def render_pending_cursor(self):
        if self.data is None or not self.pending_dirty:
            return

        ix = int(np.clip(self.pending_x, 0, self.width - 1))
        iy = int(np.clip(self.pending_y, 0, self.height - 1))
        self.pending_dirty = False
        self.update_cursor(ix, iy)

    def update_cursor(self, ix, iy, force=False):
        if self.data is None:
            return
        if not force and ix == self.last_ix and iy == self.last_iy:
            return

        intensity = float(self.data[iy, ix])

        self.v_line.setPos(ix)
        self.h_line.setPos(iy)

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
        self.bottom_marker.setData([ix], [bottom_marker_y])
        self.left_marker.setData([left_marker_x], [iy])

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
            f"X: {ix}  Y: {iy}  Intensity: {intensity:.0f}"
        )

        self.last_ix = ix
        self.last_iy = iy

    def reset_zoom(self):
        if self.data is None:
            return
        self.image_view.setRange(xRange=(0, self.width), yRange=(0, self.height), padding=0.0)

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
