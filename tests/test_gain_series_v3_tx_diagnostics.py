from types import SimpleNamespace

from tests.radio_hardware.test_gain_series_v3_tx_loopback_hardware import (
    TX_CORE_REGISTERS,
    _decode_tx_pipeline_debug,
    _tx_core_diagnostics,
)


class _Device:
    def __init__(self, *, debug_value=0xCFEF0003, fail_debug_read=False):
        self.values = {
            address: index + 10
            for index, address in enumerate(TX_CORE_REGISTERS.values())
        }
        self.values[TX_CORE_REGISTERS["timestamp_interval_control"]] = 0x2468
        self.debug_value = debug_value
        self.fail_debug_read = fail_debug_read
        self.debug_selected = False
        self.writes = []

    def reg_read(self, address):
        if address == TX_CORE_REGISTERS["timestamp_discard_count"] and self.debug_selected:
            if self.fail_debug_read:
                raise OSError("synthetic debug read failure")
            return self.debug_value
        return self.values[address]

    def reg_write(self, address, value):
        self.writes.append((address, value))
        if address == TX_CORE_REGISTERS["timestamp_interval_control"]:
            self.debug_selected = bool(value & 1)


def _radio(device):
    context = SimpleNamespace(find_device=lambda _name: device)
    return SimpleNamespace(sdr=SimpleNamespace(_ctx=context))


def test_decode_tx_pipeline_debug_names_each_sticky_boundary():
    decoded = _decode_tx_pipeline_debug(0xCFEF1234)

    assert decoded["dma_raw"] == 0xCF
    assert decoded["dac_raw"] == 0xEF
    assert decoded["timestamp_discard_count_low16"] == 0x1234
    assert decoded["dma"]["transfer_request_seen"]
    assert decoded["dma"]["fifo_write_seen"]
    assert decoded["dma"]["fifo_reset_released_seen"]
    assert decoded["dac"]["downstream_ready_seen"]
    assert decoded["dac"]["transfer_start_seen"]
    assert decoded["dac"]["upack_reset_released_seen"]


def test_tx_core_diagnostics_selects_debug_page_then_restores_control():
    device = _Device()

    result = _tx_core_diagnostics(_radio(device))

    assert result["tx_pipeline_debug"]["raw"] == 0xCFEF0003
    control = TX_CORE_REGISTERS["timestamp_interval_control"]
    assert device.writes == [(control, 0x2469), (control, 0x2468)]
    assert not device.debug_selected


def test_tx_core_diagnostics_restores_control_after_debug_read_failure():
    device = _Device(fail_debug_read=True)

    result = _tx_core_diagnostics(_radio(device))

    assert "synthetic debug read failure" in result["tx_pipeline_debug_error"]
    control = TX_CORE_REGISTERS["timestamp_interval_control"]
    assert device.writes == [(control, 0x2469), (control, 0x2468)]
    assert not device.debug_selected
