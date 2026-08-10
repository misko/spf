import pytest

from spf.scripts.mixed_transport_frequency_burn import (
    build_burn_schedule,
    parse_frequency_list,
)


def test_parse_frequency_list_supports_fractional_unit_values():
    assert parse_frequency_list("868M, 2467.1M, 5.8G") == (
        868_000_000,
        2_467_100_000,
        5_800_000_000,
    )


@pytest.mark.parametrize("value", ["", "1M,", "0", "1M,1M", "wat"])
def test_parse_frequency_list_fails_closed(value):
    with pytest.raises(ValueError):
        parse_frequency_list(value)


def test_schedule_interleaves_transport_and_does_not_repeat_epoch_boundary():
    frequencies = (868_000_000, 2_412_000_000, 5_800_000_000)
    schedule = build_burn_schedule(frequencies, epochs=4, seed=7)
    assert len(schedule) == 12
    assert all(
        step.transports in (("usb", "ip", "usb"), ("ip", "usb", "ip"))
        for step in schedule
    )
    assert all(
        step.gain_modes == ("manual_26", "slow_attack", "manual_41")
        for step in schedule
    )
    for epoch in range(4):
        block = schedule[epoch * 3 : (epoch + 1) * 3]
        assert {step.frequency_hz for step in block} == set(frequencies)
    for left, right in zip(schedule, schedule[1:]):
        if left.epoch != right.epoch:
            assert left.frequency_hz != right.frequency_hz
