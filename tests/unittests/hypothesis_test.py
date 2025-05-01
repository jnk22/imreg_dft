"""Hypothesis tests for internal functions."""

from hypothesis import given
from hypothesis import strategies as st

from imreg_dft import imreg


@given(
    angle=st.floats(min_value=0, max_value=360),
    target=st.floats(min_value=0, max_value=360),
    stdev=st.floats(max_value=0.0),
)
def test_get_odds_stdev_leq_zero_returns_minus_one_or_zero_hypothesis(
    angle, target, stdev
) -> None:
    """Test the _get_odds function for various input angles, targets, and standard deviations."""
    odds = imreg._get_odds(angle, target, stdev)

    assert odds in (0.0, -1.0)


@given(
    angle=st.floats(min_value=0, max_value=360),
    target=st.floats(min_value=0, max_value=360),
)
def test_get_odds_stdev_none_always_returns_one_hypothesis(
    target: float, angle: float
) -> None:
    """TODO."""
    odds = imreg._get_odds(target, angle, None)

    assert odds == 1.0


@given(
    angle_target=st.floats(min_value=-360, max_value=360),
    stdev=st.floats(min_value=1e-6, max_value=1000),
)
def test_get_odds_angle_equals_target(angle_target, stdev) -> None:
    # When angle == target, the angle should always be preferred
    odds = imreg._get_odds(angle_target, angle_target, stdev)

    # odds should be between 0 and 1 (favoring the original angle)
    assert 0 <= odds < 1, (
        f"Unexpected odds: {odds} for stdev={stdev}, angle={angle_target}"
    )
