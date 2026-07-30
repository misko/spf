import numpy as np

from spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis import (
    circular_metrics,
    difference_of_differences,
    fit_delay,
    wrap_phase,
)


def test_wrap_phase_uses_negative_pi_to_pi_interval():
    wrapped = wrap_phase(np.asarray([-3 * np.pi, -np.pi, 0, np.pi, 3 * np.pi]))

    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi, 0, -np.pi, -np.pi])


def test_difference_of_differences_removes_control_drift():
    treated = [
        {
            "frequency_hz": 1,
            "gain_rx1_db": 2,
            "gain_rx2_db": 3,
            "phase_delta_rad": 0.4,
            "amplitude_ratio_delta_db": -10.5,
        }
    ]
    control = [
        {
            "frequency_hz": 1,
            "gain_rx1_db": 2,
            "gain_rx2_db": 3,
            "phase_delta_rad": 0.1,
            "amplitude_ratio_delta_db": 0.5,
        }
    ]

    result = difference_of_differences(treated, control)

    assert len(result) == 1
    assert np.isclose(result[0]["phase_delta_rad"], 0.3)
    assert np.isclose(result[0]["amplitude_ratio_delta_db"], -11.0)


def test_delay_fit_recovers_known_delay():
    delay_s = 1.25e-9
    frequencies = np.arange(400e6, 900e6, 50e6)
    rows = [
        {
            "frequency_hz": int(frequency),
            "phase_delta_rad": float(wrap_phase(-2 * np.pi * frequency * delay_s)),
        }
        for frequency in frequencies
    ]

    result = fit_delay(rows)

    assert np.isclose(result["delay_ps"], 1250.0, atol=1e-6)
    assert result["residual_rmse_deg"] < 1e-8


def test_circular_metrics_are_wrap_safe():
    result = circular_metrics(np.radians([179.0, -179.0]))

    assert np.isclose(abs(result["circular_bias_deg"]), 180.0)
    assert np.isclose(result["circular_mae_deg"], 179.0)
    assert result["circular_std_deg"] < 2.0
