"""Image registration tests."""

from __future__ import annotations

import numpy as np
import pytest

from imreg_dft import imreg


def test_odds() -> None:
    # low or almost-zero odds
    odds = imreg._get_odds(10, 20, 0)
    assert odds == pytest.approx(0)
    odds = imreg._get_odds(10, 20, 0.1)
    assert odds == pytest.approx(0)
    odds = imreg._get_odds(10, 20, 40)
    assert odds < 0.01

    # non-zero complementary odds
    odds = imreg._get_odds(10, 20, 100)
    assert odds < 0.6
    odds = imreg._get_odds(10, 200, 100)
    assert odds > 1.0 / 0.6

    # high (near-infinity) odds
    odds = imreg._get_odds(10, 200, 0)
    assert odds == -1
    odds = imreg._get_odds(10, 200, 0.1)
    assert odds == -1


def test_registration_contains_all_keys() -> None:
    """Verify that the registration result contains all expected return values."""
    im1 = np.ones((128,) * 2)
    im2 = np.ones((128,) * 2)

    result = imreg.similarity(im1, im2)
    expected_keys = {
        "tvec",
        "success",
        "angle",
        "scale",
        "Dscale",
        "Dangle",
        "Dt",
        "timg",
    }

    missing_keys = expected_keys - result.keys()

    assert not missing_keys, f"Result is missing keys: {missing_keys}"
