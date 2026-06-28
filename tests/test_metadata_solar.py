import unittest

import numpy as np

from skyfinder.analysis.metadata_ablations import solar_elevation


class SolarElevationTest(unittest.TestCase):
    def test_finite_and_bounded(self) -> None:
        lat = np.array([-89.0, -45.0, 0.0, 45.0, 89.0])
        month = np.array([1.0, 4.0, 6.0, 9.0, 12.0])
        hour = np.array([0.0, 6.0, 12.0, 18.0, 23.0])
        elev = solar_elevation(lat, month, hour)
        self.assertTrue(np.all(np.isfinite(elev)))
        self.assertTrue(np.all(elev >= -90.0))
        self.assertTrue(np.all(elev <= 90.0))

    def test_equator_noon_high(self) -> None:
        elev = solar_elevation([0.0], [6.0], [12.0])
        self.assertGreater(float(elev[0]), 40.0)

    def test_equator_midnight_negative(self) -> None:
        elev = solar_elevation([0.0], [6.0], [0.0])
        self.assertLess(float(elev[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
