import unittest

import numpy as np

from OSC_Reader.OSC_Viewer import _default_image_levels, _nanmean_profile


class ViewerErrorHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
