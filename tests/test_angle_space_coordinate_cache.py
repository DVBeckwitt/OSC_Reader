import unittest
from unittest import mock

import numpy as np

import OSC_Reader.angle_space as angle_space
from OSC_Reader.angle_space import (
    DetectorCakeGeometry,
    DetectorCakeGeometryUncertainty,
)


class CoordinateStatisticsCacheTests(unittest.TestCase):
    def setUp(self):
        angle_space._COORDINATE_STATISTICS_CACHE.clear()

    def tearDown(self):
        angle_space._COORDINATE_STATISTICS_CACHE.clear()

    @staticmethod
    def _axes():
        radial = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        azimuthal = np.array([-45.0, 0.0, 45.0], dtype=np.float64)
        return radial, azimuthal

    @staticmethod
    def _geometry(*, center_row_px=1.5, center_col_px=2.0):
        return DetectorCakeGeometry(
            pixel_size_m=1.0e-4,
            distance_m=0.25,
            center_row_px=center_row_px,
            center_col_px=center_col_px,
        )

    def test_cache_key_ignores_detector_intensity_values(self):
        radial, azimuthal = self._axes()
        geometry = self._geometry()
        image_a = np.arange(20, dtype=np.float32).reshape(4, 5)
        image_b = np.full((4, 5), 999.0, dtype=np.float32)

        with mock.patch.object(
            angle_space,
            "_compute_coordinate_moment_arrays",
            wraps=angle_space._compute_coordinate_moment_arrays,
        ) as wrapped:
            stats_a = angle_space.compute_detector_to_cake_coordinate_statistics(
                image_a,
                radial,
                azimuthal,
                geometry,
            )
            stats_b = angle_space.compute_detector_to_cake_coordinate_statistics(
                image_b,
                radial,
                azimuthal,
                geometry,
            )

        self.assertEqual(wrapped.call_count, 1)
        np.testing.assert_allclose(stats_a.area_deg2, stats_b.area_deg2, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            stats_a.radial_sigma_deg,
            stats_b.radial_sigma_deg,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    def test_cache_invalidation_tracks_mask_and_geometry(self):
        radial, azimuthal = self._axes()
        image = np.ones((4, 5), dtype=np.float32)
        base_geometry = self._geometry()
        shifted_geometry = self._geometry(center_col_px=2.25)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1, 2] = 1

        with mock.patch.object(
            angle_space,
            "_compute_coordinate_moment_arrays",
            wraps=angle_space._compute_coordinate_moment_arrays,
        ) as wrapped:
            base_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                base_geometry,
            )
            angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                base_geometry,
            )
            masked_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                base_geometry,
                mask=mask,
            )
            shifted_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                shifted_geometry,
            )

        self.assertEqual(wrapped.call_count, 3)
        self.assertIsNot(base_stats, masked_stats)
        self.assertIsNot(base_stats, shifted_stats)

    def test_cache_invalidation_tracks_uncertainty_and_subpixel_grid(self):
        radial, azimuthal = self._axes()
        image = np.ones((4, 5), dtype=np.float32)
        geometry = self._geometry()
        uncertainty = DetectorCakeGeometryUncertainty.from_sigmas(
            sigma_center_row_px=0.25,
            sigma_center_col_px=0.15,
        )

        with mock.patch.object(
            angle_space,
            "_compute_coordinate_moment_arrays",
            wraps=angle_space._compute_coordinate_moment_arrays,
        ) as wrapped:
            base_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                geometry,
            )
            calibrated_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                geometry,
                geometry_uncertainty=uncertainty,
            )
            angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                geometry,
                geometry_uncertainty=uncertainty,
            )
            refined_stats = angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                geometry,
                geometry_uncertainty=uncertainty,
                numerical_subpixel_grid=2,
            )
            angle_space.compute_detector_to_cake_coordinate_statistics(
                image,
                radial,
                azimuthal,
                geometry,
                geometry_uncertainty=uncertainty,
                numerical_subpixel_grid=2,
            )

        self.assertEqual(wrapped.call_count, 4)
        self.assertIsNone(base_stats.calibration)
        self.assertIsNotNone(calibrated_stats.calibration)
        self.assertIsNotNone(refined_stats.subpixel_error)


if __name__ == "__main__":
    unittest.main()
