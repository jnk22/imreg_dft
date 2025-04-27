"""Reporting tests."""

import pytest

from imreg_dft import reporting


def test_wrapper() -> None:
    wrapper = reporting.ReportsWrapper()
    wrapper["one"] = 1
    wrapper.push_prefix("9")
    wrapper.pop_prefix("9")
    wrapper.push_prefix("1-")
    wrapper["two"] = 2
    wrapper.push_prefix("5-")
    wrapper["three"] = 3

    with pytest.raises(AssertionError):
        wrapper.pop_prefix("1-")
    wrapper.pop_prefix("5-")

    wrapper["four"] = 4
    wrapper.pop_prefix("1-")
    wrapper["five"] = 5

    assert "one" in wrapper._stuff[""]
    assert "two" in wrapper._stuff["1-"]
    assert "three" in wrapper._stuff["5-"]
    assert "four" in wrapper._stuff["1-"]
    assert "five" in wrapper._stuff[""]
