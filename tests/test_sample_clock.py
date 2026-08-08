import pytest

from spf.sdrpluto.direct_usb_protocol import TimeAnchorFlags, TimeAnchorV1
from spf.sdrpluto.sample_clock import (
    HostTimeAnchorMeasurement,
    capture_host_realtime_mapping,
    extend_low32_near,
    fit_sample_clock,
)


FLAGS = (
    TimeAnchorFlags.COUNTER_INTERVAL_VALID
    | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
    | TimeAnchorFlags.COUNTER_LOW32
    | TimeAnchorFlags.COUNTER_ADVANCED
)


def _measurement(*, request_id, counter, host_midpoint_ns, rtt_ns=200_000):
    anchor = TimeAnchorV1(
        flags=FLAGS,
        request_id=request_id,
        radio_monotonic_before_ns=request_id * 1_000_000,
        sample_counter_before=counter & 0xFFFFFFFF,
        sample_counter_after=(counter + 2) & 0xFFFFFFFF,
        radio_monotonic_after_ns=request_id * 1_000_000 + 1000,
    )
    return HostTimeAnchorMeasurement(
        anchor=anchor,
        host_monotonic_before_ns=host_midpoint_ns - rtt_ns // 2,
        host_monotonic_after_ns=host_midpoint_ns + rtt_ns // 2,
        transport="test",
    )


def test_low_counter_extends_near_frame_across_wrap():
    reference = 0x00000001_00000020
    assert extend_low32_near(reference, 0xFFFFFFF0) == 0x00000000_FFFFFFF0
    measurement = _measurement(
        request_id=1,
        counter=0x00000000_FFFFFFF0,
        host_midpoint_ns=1_000_000_000,
    )
    extended = measurement.extend_near(reference)
    assert extended.sample_counter_before == 0x00000000_FFFFFFF0
    assert extended.sample_counter_after == 0x00000000_FFFFFFF2


def test_affine_fit_recovers_sample_period_and_retains_rtt_uncertainty():
    sample_rate = 10_000_000
    period_ns = 100.0
    first_counter = 1_000_000
    first_host = 5_000_000_000
    measurements = [
        _measurement(
            request_id=index + 1,
            counter=first_counter + index * 100_000,
            # Counter midpoint is counter+1.
            host_midpoint_ns=int(first_host + index * 100_000 * period_ns),
            rtt_ns=200_000 + index * 20_000,
        )
        for index in range(4)
    ]
    extended = [item.extend_near(first_counter) for item in measurements]
    fit = fit_sample_clock(extended)

    assert fit.fitted_sample_rate_hz == pytest.approx(sample_rate, rel=1e-9)
    target = first_counter + 250_001
    assert fit.host_monotonic_ns(target) == first_host + 25_000_000
    assert fit.uncertainty_ns == 130_000
    lower, upper = fit.host_monotonic_bounds_ns(target)
    assert lower < first_host + 25_000_000 < upper


def test_one_anchor_requires_explicit_nominal_rate():
    measurement = _measurement(
        request_id=1,
        counter=100,
        host_midpoint_ns=1_000_000,
    ).extend_near(100)
    with pytest.raises(ValueError, match="counter-separated"):
        fit_sample_clock([measurement])
    fit = fit_sample_clock([measurement], nominal_sample_rate_hz=1_000_000)
    assert fit.nanoseconds_per_sample == 1000
    assert fit.anchor_count == 1


def test_nominal_rate_mapping_grows_uncertainty_outside_anchor_window():
    measurements = [
        _measurement(
            request_id=index + 1,
            counter=1_000_000 + index * 100_000,
            host_midpoint_ns=5_000_000_000 + index * 10_000_000,
            rtt_ns=200_000,
        )
        for index in range(2)
    ]
    fit = fit_sample_clock(
        [item.extend_near(1_000_000) for item in measurements],
        nominal_sample_rate_hz=10_000_000,
        lock_rate_to_nominal=True,
        maximum_rate_error_ppm=100,
    )
    inside = fit.uncertainty_ns_at(1_050_000)
    # One second beyond the last anchor adds 100 microseconds at 100 ppm.
    outside = fit.uncertainty_ns_at(11_100_001)
    assert outside == inside + 100_000
    assert fit.fitted_sample_rate_hz == pytest.approx(10_000_000)


def test_host_realtime_mapping_is_bracketed():
    mapping = capture_host_realtime_mapping()
    assert mapping.uncertainty_ns >= 0
    converted = mapping.realtime_ns(int(mapping.monotonic_midpoint_ns))
    assert converted == pytest.approx(mapping.realtime_ns_at_midpoint, abs=1)
