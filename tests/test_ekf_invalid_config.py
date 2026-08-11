"""The (phi_std=0, dynamic_R=0) corner is degenerate and must fail by name.

`self.R *= phi_std**2` zeroes the measurement-noise matrix, and `dynamic_R=0`
selects that static R -- so the pair hands the update a singular innovation
covariance. Before this guard it surfaced as `numpy.linalg.LinAlgError: singular
matrix` from inside `trajectory()`, which killed an entire sweep 14 result files
in, with nothing naming the offending parameters.

The two valid forms are asserted here too, because the reason the corner is
invalid is exactly the reason they are the only sensible combinations:
`phi_std>0` with `dynamic_R=0` uses the static R, and `dynamic_R>0` ignores
`phi_std` entirely.
"""

import numpy as np
import pytest

from spf.filters.ekf_dualradio_filter import SPFPairedKalmanFilter
from spf.filters.ekf_single_radio_filter import SPFKalmanFilter


class FakeDS:
    """The minimum both EKF constructors read: matched antenna spacing and
    wavelength across the two receivers, plus their array angle offsets."""

    temp_file = True
    wavelengths = [0.052, 0.052]
    yaml_config = {
        "receivers": [
            {"antenna-spacing-m": 0.043, "theta-in-pis": 1.0},
            {"antenna-spacing-m": 0.043, "theta-in-pis": 0.5},
        ]
    }


def build(cls, **kw):
    ds = FakeDS()
    if cls is SPFPairedKalmanFilter:
        return cls(ds=ds, **kw)
    return cls(ds=ds, rx_idx=0, **kw)


@pytest.mark.parametrize("cls", [SPFPairedKalmanFilter, SPFKalmanFilter])
def test_zero_phi_std_with_static_r_is_rejected_by_name(cls):
    with pytest.raises(ValueError, match="phi_std=0 with dynamic_R=0"):
        build(cls, phi_std=0.0, dynamic_R=0.0)


@pytest.mark.parametrize("cls", [SPFPairedKalmanFilter, SPFKalmanFilter])
def test_the_guard_fires_before_touching_the_dataset(cls):
    """It must reject on parameters alone -- a sweep should not need to open a
    capture to learn the configuration is invalid."""
    if cls is SPFPairedKalmanFilter:
        with pytest.raises(ValueError, match="phi_std=0 with dynamic_R=0"):
            cls(ds=None, phi_std=0.0, dynamic_R=0.0)
    else:
        with pytest.raises(ValueError, match="phi_std=0 with dynamic_R=0"):
            cls(ds=None, rx_idx=0, phi_std=0.0, dynamic_R=0.0)


@pytest.mark.parametrize("cls", [SPFPairedKalmanFilter, SPFKalmanFilter])
@pytest.mark.parametrize("phi_std,dynamic_R", [(1.0, 0.0), (0.0, 0.1), (0.0, 1.0)])
def test_the_two_valid_forms_construct(cls, phi_std, dynamic_R):
    f = build(cls, phi_std=phi_std, dynamic_R=dynamic_R)
    assert f is not None


@pytest.mark.parametrize("cls", [SPFPairedKalmanFilter, SPFKalmanFilter])
def test_static_r_scales_with_phi_std_squared(cls):
    """Pins why zero is degenerate rather than merely small."""
    a = build(cls, phi_std=1.0, dynamic_R=0.0)
    b = build(cls, phi_std=2.0, dynamic_R=0.0)
    assert np.allclose(b.R, 4.0 * a.R)


@pytest.mark.parametrize("cls", [SPFPairedKalmanFilter, SPFKalmanFilter])
def test_dynamic_r_makes_phi_std_irrelevant(cls):
    """Why a full factorial over the two axes is duplicate work, not coverage:
    with dynamic_R>0 the update never reads self.R, so every phi_std value gives
    an identical filter."""
    import inspect

    src = inspect.getsource(cls.update)
    assert "self.dynamic_R == 0" in src, (
        "update() no longer switches on dynamic_R; the survey config's block "
        "structure assumes it does"
    )
