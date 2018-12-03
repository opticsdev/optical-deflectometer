import pytest
import deflectometer.deflectometer as defl
import os
import sys
import numpy as np


def test_import_module():
    """Check for working import"""
    assert('deflectometer' in sys.modules)


def test_slope_val():
    """Checks current impolementation of slope_val and will break if updated to ensure new tests created"""
    slopeval = defl.slope_val(cam_sep=2, screen_sep=5, zscreen_part=20, zcam_part=20, partsag=-0.2)
    assert(slopeval == (.1 + .25) / (1 + 1))
