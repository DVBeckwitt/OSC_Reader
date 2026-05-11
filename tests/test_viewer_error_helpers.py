import os
import unittest
import warnings
from dataclasses import replace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from OSC_Reader.OSC_Viewer import (
    OSCViewerWindow,
    QtWidgets,
    _default_image_levels,
    _merge_angle_error_result,
    _nanmean_profile,
    _normalize_angle_display_mode,
)
from OSC_Reader.angle_space import (
    DetectorCakeCoordinateStats,
    DetectorCakeResult,
    fast_display_sigma_maps,
    prepare_gui_phi_display_data,
)


class ViewerErrorHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = None
        if QtWidgets is not None:
            cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self._cache_write_patch = mock.patch("OSC_Reader.OSC_Viewer._write_viewer_cache")
        self._cache_write_patch.start()

    def tearDown(self):
        self._cache_write_patch.stop()

    @staticmethod
    def _base_result():
        return DetectorCakeResult(
            radial_deg=np.array([10.0, 20.0], dtype=np.float64),
            azimuthal_deg=np.array([-45.0, 45.0], dtype=np.float64),
            intensity=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            sum_signal=np.zeros((2, 2), dtype=np.float64),
            sum_normalization=np.ones((2, 2), dtype=np.float64),
            count=np.ones((2, 2), dtype=np.float64),
        )

    @staticmethod
    def _coordinate_stats():
        shape = (2, 2)
        return DetectorCakeCoordinateStats(
            area_deg2=np.ones(shape, dtype=np.float64),
            radial_mean_deg=np.zeros(shape, dtype=np.float64),
            azimuthal_mean_deg=np.zeros(shape, dtype=np.float64),
            radial_sigma_deg=np.array([[0.10, 0.20], [0.30, 0.40]], dtype=np.float64),
            azimuthal_sigma_deg=np.array([[1.10, 1.20], [1.30, 1.40]], dtype=np.float64),
            radial_label_sigma_deg=np.zeros(shape, dtype=np.float64),
            azimuthal_label_sigma_deg=np.zeros(shape, dtype=np.float64),
        )

    def _assert_converted_view_preserves_negative_display(self, viewer):
        image_display = viewer._image_display_data()
        levels = viewer.image_item.getLevels()
        self.assertEqual(float(viewer.data[0, 0]), -2.0)
        self.assertFalse(np.any(image_display == 1.0e20))
        self.assertEqual(float(levels[0]), 0.0)
        self.assertLess(float(levels[1]), 1.0e20)

    def test_nanmean_profile_ignores_missing_bins(self):
        values = np.array(
            [
                [1.0, np.nan, 3.0],
                [3.0, 5.0, np.nan],
            ],
            dtype=np.float64,
        )

        collapsed_rows = _nanmean_profile(values, axis=0)
        collapsed_cols = _nanmean_profile(values, axis=1)

        np.testing.assert_allclose(
            collapsed_rows,
            np.array([2.0, 5.0, 3.0], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            collapsed_cols,
            np.array([2.0, 4.0], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_intensity_error_levels_clip_outlier_tail(self):
        values = np.array([0.0, 0.01, 0.02, 0.03, 1.0], dtype=np.float64)

        normal_low, normal_high = _default_image_levels(
            values,
            log_enabled=False,
            favor_low_intensity=False,
        )
        error_low, error_high = _default_image_levels(
            values,
            log_enabled=False,
            favor_low_intensity=True,
        )

        self.assertEqual(normal_low, 0.0)
        self.assertEqual(error_low, 0.0)
        self.assertLess(error_high, normal_high)
        self.assertGreater(error_high, 0.0)

    def test_default_image_levels_ignore_negative_sentinel_for_linear_scale(self):
        values = np.array([-5.0, 0.0, 10.0, 1.0e20], dtype=np.float64)

        low, high = _default_image_levels(
            values,
            log_enabled=False,
            favor_low_intensity=False,
        )

        self.assertEqual(low, 0.0)
        self.assertLess(high, 10.0)
        self.assertGreater(high, 0.0)

    def test_default_image_levels_fall_back_when_only_negative_sentinel_remains(self):
        values = np.array([-5.0, 1.0e20], dtype=np.float64)

        low, high = _default_image_levels(
            values,
            log_enabled=False,
            favor_low_intensity=False,
        )

        self.assertEqual((low, high), (0.0, 1.0))

    def test_default_log_image_levels_keep_zero_lower_bound(self):
        values = np.array([-3.0, -2.0, 0.5, 20.0], dtype=np.float64)

        low, high = _default_image_levels(
            values,
            log_enabled=True,
            favor_low_intensity=False,
        )

        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.5)

    def test_default_log_image_levels_fall_back_to_zero_when_only_sentinel_remains(self):
        values = np.array([20.0], dtype=np.float64)

        low, high = _default_image_levels(
            values,
            log_enabled=True,
            favor_low_intensity=False,
        )

        self.assertEqual((low, high), (0.0, 1.0))

    def test_intensity_sem_merge_does_not_force_coordinate_stats(self):
        base_result = self._base_result()
        loader = mock.Mock(side_effect=AssertionError("coordinate stats should stay lazy"))

        merged = _merge_angle_error_result(
            base_result,
            require_intensity_sem=True,
            coordinate_stats_loader=loader,
        )

        loader.assert_not_called()
        self.assertIsNone(merged.coordinate_stats)
        np.testing.assert_allclose(
            merged.intensity_sem,
            np.ones((2, 2), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_coordinate_stats_merge_uses_loader_only_when_requested(self):
        base_result = self._base_result()
        loader = mock.Mock(return_value=object())

        untouched = _merge_angle_error_result(base_result, coordinate_stats_loader=loader)
        loader.assert_not_called()
        self.assertIs(untouched, base_result)

        merged = _merge_angle_error_result(
            base_result,
            require_coordinate_stats=True,
            coordinate_stats_loader=loader,
        )
        loader.assert_called_once()
        self.assertIs(merged.coordinate_stats, loader.return_value)

    def test_old_coordinate_modes_normalize_to_merged_profiles(self):
        self.assertEqual(_normalize_angle_display_mode("theta_sigma"), "coordinate_profiles")
        self.assertEqual(_normalize_angle_display_mode("phi_sigma"), "coordinate_profiles")
        self.assertEqual(_normalize_angle_display_mode("intensity_sem"), "intensity_sem")

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_coordinate_profile_metric_reports_available_without_exact_stats(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.angle_space_result = self._base_result()
            viewer.angle_space_error_result = None

            self.assertTrue(viewer._angle_error_metrics_available("coordinate_profiles"))
            self.assertFalse(viewer._angle_error_metrics_available("intensity_sem"))
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_angle_error_profiles_fall_back_to_fast_sigma_maps_when_stats_missing(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.pixel_size_spin.setValue(0.1)
            viewer.distance_spin.setValue(200.0)
            base_result = self._base_result()
            viewer.angle_space_result = base_result
            viewer.angle_space_error_result = None

            expected_theta_source, expected_phi_source = fast_display_sigma_maps(
                base_result,
                pixel_size_m=viewer.pixel_size_spin.value() * 1.0e-3,
                distance_m=viewer.distance_spin.value() * 1.0e-3,
            )
            expected_theta_map, expected_radial_deg, _ = prepare_gui_phi_display_data(
                base_result,
                expected_theta_source,
                zero_direction=viewer._current_phi_zero_direction(),
            )
            expected_phi_map, _, expected_phi_deg = prepare_gui_phi_display_data(
                base_result,
                expected_phi_source,
                zero_direction=viewer._current_phi_zero_direction(),
            )

            with mock.patch.object(
                viewer,
                "_ensure_angle_space_coordinate_stats",
                side_effect=AssertionError("coordinate stats should stay lazy"),
            ):
                radial_deg, theta_profile, phi_deg, phi_profile, _ = viewer._angle_error_profiles()

            np.testing.assert_allclose(radial_deg, expected_radial_deg, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                theta_profile,
                _nanmean_profile(expected_theta_map, axis=0),
                rtol=1.0e-6,
                atol=0.0,
            )
            np.testing.assert_allclose(phi_deg, expected_phi_deg, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                phi_profile,
                _nanmean_profile(expected_phi_map, axis=1),
                rtol=1.0e-6,
                atol=0.0,
                equal_nan=True,
            )
            self.assertIs(viewer.angle_space_error_result, base_result)
            self.assertIsNone(viewer.angle_space_error_result.coordinate_stats)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_angle_error_profiles_prefer_precomputed_coordinate_stats(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            base_result = self._base_result()
            error_result = replace(base_result, coordinate_stats=self._coordinate_stats())
            viewer.angle_space_result = base_result
            viewer.angle_space_error_result = error_result

            expected_theta_map, expected_radial_deg, _ = prepare_gui_phi_display_data(
                error_result,
                error_result.coordinate_stats.radial_sigma_deg,
                zero_direction=viewer._current_phi_zero_direction(),
            )
            expected_phi_map, _, expected_phi_deg = prepare_gui_phi_display_data(
                error_result,
                error_result.coordinate_stats.azimuthal_sigma_deg,
                zero_direction=viewer._current_phi_zero_direction(),
            )

            with mock.patch(
                "OSC_Reader.OSC_Viewer.fast_display_sigma_maps",
                side_effect=AssertionError("fast sigma path should stay unused"),
            ):
                radial_deg, theta_profile, phi_deg, phi_profile, _ = viewer._angle_error_profiles()

            np.testing.assert_allclose(radial_deg, expected_radial_deg, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                theta_profile,
                _nanmean_profile(expected_theta_map, axis=0),
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(phi_deg, expected_phi_deg, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                phi_profile,
                _nanmean_profile(expected_phi_map, axis=1),
                rtol=0.0,
                atol=0.0,
            )
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_merged_profile_mode_uses_inline_uncertainty_profile_page(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            base_result = self._base_result()
            error_result = replace(
                base_result,
                coordinate_stats=self._coordinate_stats(),
                intensity_sem=np.full((2, 2), 0.25, dtype=np.float64),
            )
            viewer.angle_space_result = base_result
            viewer.angle_space_error_result = error_result
            viewer.show_angle_space_view(
                base_result.intensity,
                base_result.radial_deg,
                base_result.azimuthal_deg,
            )

            intensity_index = viewer.angle_display_combo.findData("intensity_sem")
            profiles_index = viewer.angle_display_combo.findData("coordinate_profiles")
            self.assertGreaterEqual(intensity_index, 0)
            self.assertGreaterEqual(profiles_index, 0)

            viewer.angle_display_combo.setCurrentIndex(profiles_index)
            viewer.angle_error_view_button.setChecked(True)
            self._app.processEvents()

            self.assertFalse(viewer._coordinate_error_profiles_requested())
            self.assertEqual(viewer.angle_display_combo.currentData(), "intensity_sem")
            self.assertIs(viewer.main_display_stack.currentWidget(), viewer.graphics)

            viewer.angle_display_combo.setCurrentIndex(profiles_index)
            self._app.processEvents()

            self.assertTrue(viewer._coordinate_error_profiles_requested())
            self.assertTrue(viewer._coordinate_error_profiles_active())
            self.assertIs(
                viewer.main_display_stack.currentWidget(),
                viewer.angle_error_profiles_widget,
            )

            expected_theta_x, expected_theta_y, expected_phi_x, expected_phi_y, _ = (
                viewer._angle_error_profiles()
            )
            theta_x, theta_y = viewer.angle_error_profiles_widget.theta_curve.getData()
            phi_x, phi_y = viewer.angle_error_profiles_widget.phi_curve.getData()
            np.testing.assert_allclose(theta_x, expected_theta_x, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(theta_y, expected_theta_y, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(phi_x, expected_phi_x, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(phi_y, expected_phi_y, rtol=0.0, atol=0.0)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_beam_center_change_invalidates_cached_converted_views(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.detector_data = np.zeros((8, 10), dtype=np.float32)
            viewer.beam_center_row = 3.0
            viewer.beam_center_col = 4.0
            viewer.angle_space_result = self._base_result()
            viewer.angle_space_error_result = replace(
                self._base_result(),
                intensity_sem=np.full((2, 2), 0.5, dtype=np.float64),
            )
            viewer.angle_space_cake = np.ones((2, 2), dtype=np.float32)
            viewer.q_space_result = object()

            viewer._set_beam_center(5.0, 6.0)

            self.assertIsNone(viewer.angle_space_result)
            self.assertIsNone(viewer.angle_space_error_result)
            self.assertIsNone(viewer.angle_space_cake)
            self.assertIsNone(viewer.q_space_result)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_beam_center_change_recomputes_active_angle_space_view(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.detector_data = np.zeros((8, 10), dtype=np.float32)
            viewer.current_view_mode = "angle_space"
            viewer.beam_center_row = 3.0
            viewer.beam_center_col = 4.0
            viewer.angle_space_result = self._base_result()
            viewer.q_space_result = object()

            with mock.patch.object(viewer, "convert_active_image") as convert_mock:
                viewer._set_beam_center(5.0, 6.0)

            convert_mock.assert_called_once_with()
            self.assertIsNone(viewer.angle_space_result)
            self.assertIsNone(viewer.q_space_result)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_small_beam_center_change_recomputes_active_angle_space_view(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.detector_data = np.zeros((2005, 2005), dtype=np.uint8)
            viewer.current_view_mode = "angle_space"
            viewer.beam_center_row = 1000.0
            viewer.beam_center_col = 1000.0
            viewer.angle_space_result = self._base_result()
            viewer.q_space_result = object()

            with mock.patch.object(viewer, "convert_active_image") as convert_mock:
                viewer._set_beam_center(1000.01, 1000.0)

            convert_mock.assert_called_once_with()
            self.assertIsNone(viewer.angle_space_result)
            self.assertIsNone(viewer.q_space_result)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_geometry_parameter_change_recomputes_active_q_space_view(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.detector_data = np.zeros((8, 10), dtype=np.float32)
            viewer.current_view_mode = "q_space"
            viewer.angle_space_result = self._base_result()
            viewer.q_space_result = object()

            with mock.patch.object(viewer, "convert_active_image_to_q_space") as convert_mock:
                viewer._on_geometry_parameter_changed(viewer.distance_spin.value())

            convert_mock.assert_called_once_with()
            self.assertIsNone(viewer.angle_space_result)
            self.assertIsNone(viewer.q_space_result)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_geometry_spin_boxes_disable_keyboard_tracking(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            for spin in (
                viewer.center_col_spin,
                viewer.center_row_spin,
                viewer.distance_spin,
                viewer.pixel_size_spin,
                viewer.radial_bins_spin,
                viewer.azimuth_bins_spin,
            ):
                self.assertFalse(spin.keyboardTracking())
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_cached_sampling_does_not_override_loaded_detector_dimensions(self):
        cache_state = {
            "viewer": {
                "sampling": {
                    "radial_bins": 2,
                    "azimuth_bins": 2,
                }
            }
        }

        with mock.patch("OSC_Reader.OSC_Viewer._load_viewer_cache", return_value=cache_state):
            viewer = OSCViewerWindow(filename=None)
        try:
            viewer.set_data(np.zeros((8, 10), dtype=np.float32))
            viewer._apply_cached_loaded_file_state()
            self._app.processEvents()

            self.assertEqual(viewer.radial_bins_spin.value(), 10)
            self.assertEqual(viewer.azimuth_bins_spin.value(), 8)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_detector_full_extent_covers_edges_after_load_and_reset(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            viewer.resize(1680, 1020)
            viewer.show()
            self._app.processEvents()

            detector = np.arange(100 * 400, dtype=np.float32).reshape(100, 400)
            viewer.set_data(detector)
            self._app.processEvents()

            x_edges = viewer.display_x_edges
            y_edges = viewer.display_y_edges
            x_range, y_range = viewer.image_view.viewRange()
            self.assertLessEqual(x_range[0], x_edges[0])
            self.assertGreaterEqual(x_range[1], x_edges[-1])
            self.assertLessEqual(y_range[0], y_edges[0])
            self.assertGreaterEqual(y_range[1], y_edges[-1])

            viewer.image_view.setRange(
                xRange=(100.0, 200.0),
                yRange=(20.0, 40.0),
                padding=0.0,
            )
            self._app.processEvents()
            viewer.reset_zoom()
            self._app.processEvents()

            x_range, y_range = viewer.image_view.viewRange()
            self.assertLessEqual(x_range[0], x_edges[0])
            self.assertGreaterEqual(x_range[1], x_edges[-1])
            self.assertLessEqual(y_range[0], y_edges[0])
            self.assertGreaterEqual(y_range[1], y_edges[-1])
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_image_log_view_uses_finite_display_data_for_invalid_values(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            detector = np.array(
                [
                    [0.0, -1.0],
                    [np.nan, np.inf],
                ],
                dtype=np.float64,
            )
            viewer.set_data(detector)

            viewer.image_log_button.setChecked(True)
            self._app.processEvents()

            image_display = viewer._image_display_data()
            levels = viewer.image_item.getLevels()
            self.assertTrue(np.all(np.isfinite(image_display)))
            self.assertTrue(np.all(np.isfinite(levels)))
            self.assertLess(float(levels[0]), float(levels[1]))
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_linear_image_view_marks_negative_pixels_without_scaling_to_them(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            detector = np.array(
                [
                    [-5.0, 0.0],
                    [10.0, 20.0],
                ],
                dtype=np.float64,
            )
            viewer.set_data(detector)
            viewer.image_log_button.setChecked(False)
            self._app.processEvents()

            image_display = viewer._image_display_data()
            levels = viewer.image_item.getLevels()
            self.assertEqual(float(image_display[0, 0]), 1.0e20)
            self.assertEqual(float(viewer.data[0, 0]), -5.0)
            self.assertGreaterEqual(float(levels[0]), 0.0)
            self.assertLess(float(levels[1]), 1.0e20)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_angle_space_view_does_not_mark_negative_pixels_as_detector_sentinel(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            cake = np.array(
                [
                    [-2.0, 0.0],
                    [1.0, 4.0],
                ],
                dtype=np.float64,
            )
            radial_deg = np.array([10.0, 20.0], dtype=np.float64)
            phi_deg = np.array([-45.0, 45.0], dtype=np.float64)

            viewer.image_log_button.setChecked(False)
            viewer.show_angle_space_view(cake, radial_deg, phi_deg)
            self._app.processEvents()

            self._assert_converted_view_preserves_negative_display(viewer)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_q_space_view_does_not_mark_negative_pixels_as_detector_sentinel(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            q_image = np.array(
                [
                    [-2.0, 0.0],
                    [1.0, 4.0],
                ],
                dtype=np.float64,
            )
            qr = np.array([-1.0, 1.0], dtype=np.float64)
            qz = np.array([-0.5, 0.5], dtype=np.float64)

            viewer.image_log_button.setChecked(False)
            viewer.show_q_space_view(q_image, qr, qz)
            self._app.processEvents()

            self._assert_converted_view_preserves_negative_display(viewer)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_log_image_view_marks_negative_pixels_without_scaling_to_them(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            detector = np.array(
                [
                    [-5.0, 0.001],
                    [0.01, 1.0],
                ],
                dtype=np.float64,
            )
            viewer.set_data(detector)

            viewer.image_log_button.setChecked(True)
            self._app.processEvents()

            image_display = viewer._image_display_data()
            levels = viewer.image_item.getLevels()
            self.assertEqual(float(image_display[0, 0]), 20.0)
            self.assertLess(float(levels[1]), 20.0)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_image_log_view_levels_keep_zero_lower_bound(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            detector = np.array(
                [
                    [0.001, 0.01],
                    [1.0, 10.0],
                ],
                dtype=np.float64,
            )
            viewer.set_data(detector)

            viewer.image_log_button.setChecked(True)
            self._app.processEvents()

            image_display = viewer._image_display_data()
            finite_display = image_display[
                np.isfinite(image_display) & (image_display != 20.0)
            ]
            levels = viewer.image_item.getLevels()
            self.assertEqual(float(levels[0]), 0.0)
            self.assertGreaterEqual(float(levels[1]), float(np.max(finite_display)))
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_profile_log_toggles_render_finite_curve_data_for_invalid_values(self):
        viewer = OSCViewerWindow(filename=None)
        try:
            detector = np.array(
                [
                    [0.0, -1.0],
                    [np.nan, np.inf],
                ],
                dtype=np.float64,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                viewer.set_data(detector)
                viewer.bottom_log_button.setChecked(True)
                viewer.left_log_button.setChecked(True)
                viewer.update_cursor(1, 1, force=True)
                self._app.processEvents()

            _, bottom_y = viewer.bottom_curve.getData()
            left_x, _ = viewer.left_curve.getData()
            self.assertTrue(np.all(np.isfinite(bottom_y)))
            self.assertTrue(np.all(np.isfinite(left_x)))
            self.assertTrue(viewer.bottom_log_enabled)
            self.assertTrue(viewer.left_log_enabled)
        finally:
            viewer.close()
            self._app.processEvents()

    @unittest.skipIf(QtWidgets is None, "Qt viewer stack unavailable")
    def test_cached_log_state_survives_loading_data(self):
        cache_state = {
            "viewer": {
                "profiles": {
                    "image_log_enabled": True,
                    "bottom_log_enabled": True,
                    "left_log_enabled": True,
                },
            },
        }
        with mock.patch("OSC_Reader.OSC_Viewer._load_viewer_cache", return_value=cache_state):
            viewer = OSCViewerWindow(filename=None)
        try:
            self.assertTrue(viewer.image_log_button.isChecked())
            self.assertTrue(viewer.bottom_log_button.isChecked())
            self.assertTrue(viewer.left_log_button.isChecked())

            viewer.set_data(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
            self._app.processEvents()

            self.assertTrue(viewer.image_log_enabled)
            self.assertTrue(viewer.bottom_log_enabled)
            self.assertTrue(viewer.left_log_enabled)
            levels = viewer.image_item.getLevels()
            self.assertTrue(np.all(np.isfinite(levels)))
            self.assertLess(float(levels[0]), float(levels[1]))
        finally:
            viewer.close()
            self._app.processEvents()


if __name__ == "__main__":
    unittest.main()
