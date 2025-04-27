"""Transformation tests."""

from __future__ import annotations

import pytest

from imreg_dft import tform


@pytest.mark.parametrize(
    "instr",
    [
        "scale: 1.8 +-8.99\n angle:186 \nshift (x, y): 35,44.2 success:1",
        "scale: 1.8 angle:186 \nshift (x, y): 35, 44.2 +-0.5 success:1",
    ],
)
def test_parse(instr):
    res = tform._str2tform(instr)
    assert res["scale"] == pytest.approx(1.8)
    assert res["angle"] == pytest.approx(186)
    assert res["tvec"][0] == pytest.approx(44.2)  # y-component
    assert res["tvec"][1] == pytest.approx(35)  # x-component
