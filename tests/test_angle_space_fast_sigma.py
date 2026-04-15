import unittest

import numpy as np

from OSC_Reader.angle_space import (
    DetectorCakeGeometry,
    DetectorCakeResult,
    _analytic_coordinate_sigma_deg,
    fast_display_sigma_maps,
    fast_display_sigma_profiles,
)


class FastDisplaySigmaTests(unittest.TestCase):
    def test_profiles_match_closed_form(self):
        radial_deg = np.array([0.0, 30.0, 60.0], dtype=np.float64)
        pixel_size_m = 1.0e-4
        distance_m = 0.2

        sigma_tth, sigma_phi = fast_display_sigma_profiles(
            radial_deg,
            pixel_size_m=pixel_size_m,
            distance_m=distance_m,
        )

        prefactor = np.rad2deg(pixel_size_m / (np.sqrt(12.0) * distance_m))
        theta = np.deg2rad(radial_deg)
        expected_tth = prefactor * np.cos(theta) ** 2
        expected_phi = np.array(
            [np.nan, prefactor / np.abs(np.tan(theta[1])), prefactor / np.abs(np.tan(theta[2]))],
            dtype=np.float64,
        )

        np.testing.assert_allclose(sigma_tth, expected_tth.astype(np.float32), rtol=1.0e-6, atol=0.0)
        np.testing.assert_allclose(
            sigma_phi[1:],
            expected_phi[1:].astype(np.float32),
            rtol=1.0e-6,
            atol=0.0,
        )
        self.assertTrue(np.isnan(sigma_phi[0]))

    def test_analytic_sigma_is_phi_invariant(self):
        radial_deg = np.array([5.0, 15.0, 25.0], dtype=np.float64)
        azimuthal_deg = np.array([-90.0, 0.0, 135.0], dtype=np.float64)[:, None]
        radial_grid = np.broadcast_to(radial_deg[None, :], (azimuthal_deg.shape[0], radial_deg.size))
        azimuthal_grid = np.broadcast_to(azimuthal_deg, radial_grid.shape)
        geometry = DetectorCakeGeometry(
            center_row_px=0.0,
            center_col_px=0.0,
            distance_m=0.35,
            pixel_size_m=7.5e-5,
        )

        sigma_tth, sigma_phi = _analytic_coordinate_sigma_deg(
            radial_grid,
            azimuthal_grid,
            geometry,
        )
        expected_tth, expected_phi = fast_display_sigma_profiles(
            radial_deg,
            pixel_size_m=geometry.pixel_size_m,
            distance_m=geometry.distance_m,
        )

        np.testing.assert_allclose(
            sigma_tth,
            np.broadcast_to(expected_tth[None, :], radial_grid.shape),
            rtol=1.0e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            sigma_phi,
            np.broadcast_to(expected_phi[None, :], radial_grid.shape),
            rtol=1.0e-6,
            atol=0.0,
            equal_nan=True,
        )

    def test_sigma_maps_broadcast_and_mask_invalid_bins(self):
        result = DetectorCakeResult(
            radial_deg=np.array([10.0, 20.0, 30.0], dtype=np.float64),
            azimuthal_deg=np.array([-45.0, 45.0], dtype=np.float64),
            intensity=np.zeros((2, 3), dtype=np.float32),
            sum_signal=np.zeros((2, 3), dtype=np.float64),
            sum_normalization=np.array(
                [[1.0, 0.0, 2.0], [np.nan, 3.0, 4.0]],
                dtype=np.float64,
            ),
            count=np.ones((2, 3), dtype=np.int32),
        )

        sigma_tth, sigma_phi = fast_display_sigma_maps(
            result,
            pixel_size_m=8.0e-5,
            distance_m=0.25,
        )
        profile_tth, profile_phi = fast_display_sigma_profiles(
            result.radial_deg,
            pixel_size_m=8.0e-5,
            distance_m=0.25,
        )

        self.assertEqual(sigma_tth.shape, result.intensity.shape)
        self.assertEqual(sigma_phi.shape, result.intensity.shape)
        np.testing.assert_allclose(sigma_tth[0, 0], profile_tth[0], rtol=1.0e-6, atol=0.0)
        np.testing.assert_allclose(sigma_phi[1, 2], profile_phi[2], rtol=1.0e-6, atol=0.0)
        self.assertTrue(np.isnan(sigma_tth[0, 1]))
        self.assertTrue(np.isnan(sigma_tth[1, 0]))
        self.assertTrue(np.isnan(sigma_phi[0, 1]))
        self.assertTrue(np.isnan(sigma_phi[1, 0]))


if __name__ == "__main__":
    unittest.main()
