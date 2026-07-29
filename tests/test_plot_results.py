from __future__ import annotations

import math
import unittest

import numpy as np

from eatr_rates.plot_results import convert_log_rates
from eatr_rates.time_units import resolve_time_unit


class PlotUnitTests(unittest.TestCase):
    def test_convert_log_rates_shifts_to_microseconds(self):
        log_rates_seconds = np.array([0.0, 2.5, 10.0], dtype=float)
        _, _, seconds_per_unit = resolve_time_unit("microseconds")

        converted = convert_log_rates(log_rates_seconds, seconds_per_unit)

        expected = log_rates_seconds + math.log(1e-6)
        self.assertTrue(np.allclose(converted, expected))

