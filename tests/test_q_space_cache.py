import unittest
from unittest import mock

import numpy as np

import OSC_Reader.angle_space as angle_space
from OSC_Reader.angle_space import convert_phi_2theta_to_qr_qz_space

_NO_FINITE_QSPACE_IMAGE_MESSAGE = (
    r"^No finite q-space samples were produced from the current angle-space image\.$"
)


def _histogram_reference(
    intensity,
    radial_deg,
    phi_deg,
    *,
    wavelength_angstrom,
    incident_angle_deg,
    qr_bins,
    qz_bins,
):
    cake = np.asarray(intensity, dtype=np.float64)
    radial = np.asarray(radial_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)

    phi_rad = np.deg2rad(phi)[:, None]
    theta_rad = np.deg2rad(radial)[None, :]
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    wavevector = (2.0 * np.pi) / float(wavelength_angstrom)
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
    valid = np.isfinite(qr_flat) & np.isfinite(qz_flat) & np.isfinite(intensity_flat)

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

    qr_edges = np.linspace(qr_min, qr_max, int(qr_bins) + 1, dtype=np.float64)
    qz_edges = np.linspace(qz_min, qz_max, int(qz_bins) + 1, dtype=np.float64)
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
    return qr_centers, qz_centers, rebinned


class QSpaceCacheTests(unittest.TestCase):
    def setUp(self):
        angle_space._QSPACE_COORDINATE_CACHE.clear()

    def tearDown(self):
        angle_space._QSPACE_COORDINATE_CACHE.clear()

    def test_rebin_matches_histogram_reference_with_nans(self):
        radial = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float64)
        phi = np.array([-135.0, -45.0, 45.0], dtype=np.float64)
        intensity = np.array(
            [
                [1.0, np.nan, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, np.nan],
            ],
            dtype=np.float32,
        )

        expected_qr, expected_qz, expected_intensity = _histogram_reference(
            intensity,
            radial,
            phi,
            wavelength_angstrom=1.24,
            incident_angle_deg=0.35,
            qr_bins=5,
            qz_bins=4,
        )
        result = convert_phi_2theta_to_qr_qz_space(
            intensity,
            radial,
            phi,
            wavelength_angstrom=1.24,
            incident_angle_deg=0.35,
            qr_bins=5,
            qz_bins=4,
        )

        np.testing.assert_allclose(result.qr, expected_qr, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.qz, expected_qz, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            result.intensity,
            expected_intensity,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    def test_rebin_matches_histogram_reference_with_finite_values(self):
        radial = np.array(
            [
                4.757272111997083,
                9.535733042233272,
                14.314193972469461,
                19.092654902705647,
                23.871115832941836,
                28.649576763178025,
            ],
            dtype=np.float64,
        )
        phi = np.array(
            [
                -18.67824343380579,
                -5.115283629876151,
                8.447676174053488,
                22.010635977983128,
                35.573595781912765,
                49.1365555858424,
                62.69951538977204,
            ],
            dtype=np.float64,
        )
        intensity = np.arange(1.0, 43.0, dtype=np.float32).reshape(7, 6)

        expected_qr, expected_qz, expected_intensity = _histogram_reference(
            intensity,
            radial,
            phi,
            wavelength_angstrom=1.7294400786789155,
            incident_angle_deg=1.8328690600325714,
            qr_bins=2,
            qz_bins=4,
        )
        result = convert_phi_2theta_to_qr_qz_space(
            intensity,
            radial,
            phi,
            wavelength_angstrom=1.7294400786789155,
            incident_angle_deg=1.8328690600325714,
            qr_bins=2,
            qz_bins=4,
        )

        self.assertTrue(np.array_equal(result.qr, expected_qr))
        self.assertTrue(np.array_equal(result.qz, expected_qz))
        self.assertTrue(np.array_equal(result.intensity, expected_intensity))

    def test_coordinate_map_cache_reuses_axes_across_intensity_changes(self):
        radial = np.linspace(1.0, 40.0, 8, dtype=np.float64)
        phi = np.linspace(-160.0, 160.0, 6, dtype=np.float64)
        image_a = np.arange(48, dtype=np.float32).reshape(6, 8)
        image_b = np.full((6, 8), 9.0, dtype=np.float32)

        with mock.patch.object(
            angle_space,
            "_compute_q_space_coordinate_map_uncached",
            wraps=angle_space._compute_q_space_coordinate_map_uncached,
        ) as wrapped:
            convert_phi_2theta_to_qr_qz_space(
                image_a,
                radial,
                phi,
                wavelength_angstrom=1.5406,
                incident_angle_deg=0.25,
                qr_bins=9,
                qz_bins=7,
            )
            convert_phi_2theta_to_qr_qz_space(
                image_b,
                radial,
                phi,
                wavelength_angstrom=1.5406,
                incident_angle_deg=0.25,
                qr_bins=9,
                qz_bins=7,
            )

        self.assertEqual(wrapped.call_count, 1)

    def test_cache_invalidation_tracks_q_parameters(self):
        radial = np.linspace(2.0, 30.0, 5, dtype=np.float64)
        phi = np.linspace(-120.0, 120.0, 4, dtype=np.float64)
        image = np.ones((4, 5), dtype=np.float32)

        with mock.patch.object(
            angle_space,
            "_compute_q_space_coordinate_map_uncached",
            wraps=angle_space._compute_q_space_coordinate_map_uncached,
        ) as wrapped:
            convert_phi_2theta_to_qr_qz_space(
                image,
                radial,
                phi,
                wavelength_angstrom=1.5406,
                incident_angle_deg=0.0,
                qr_bins=5,
                qz_bins=4,
            )
            convert_phi_2theta_to_qr_qz_space(
                image,
                radial,
                phi,
                wavelength_angstrom=1.0000,
                incident_angle_deg=0.0,
                qr_bins=5,
                qz_bins=4,
            )
            convert_phi_2theta_to_qr_qz_space(
                image,
                radial,
                phi,
                wavelength_angstrom=1.0000,
                incident_angle_deg=0.0,
                qr_bins=6,
                qz_bins=4,
            )

        self.assertEqual(wrapped.call_count, 3)

    def test_nan_radial_axis_keeps_legacy_error_message(self):
        with self.assertRaisesRegex(ValueError, _NO_FINITE_QSPACE_IMAGE_MESSAGE):
            convert_phi_2theta_to_qr_qz_space(
                np.ones((2, 2), dtype=np.float32),
                np.array([np.nan, np.nan], dtype=np.float64),
                np.array([-1.0, 1.0], dtype=np.float64),
            )

    def test_nan_phi_axis_keeps_legacy_error_message(self):
        with self.assertRaisesRegex(ValueError, _NO_FINITE_QSPACE_IMAGE_MESSAGE):
            convert_phi_2theta_to_qr_qz_space(
                np.ones((2, 2), dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([np.nan, np.nan], dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
