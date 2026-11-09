import pytest

from spf.scripts.mute_pluto_tx import validate_loopback_safety


def test_20_db_path_requires_10_db_tx_derating():
    assert (
        validate_loopback_safety(
            physical_attenuation_db=20,
            strongest_tx_gain_db=-10,
        )
        == 30
    )


@pytest.mark.parametrize("strongest_tx_gain_db", [-9.75, 0])
def test_20_db_path_rejects_insufficient_tx_derating(strongest_tx_gain_db):
    with pytest.raises(ValueError, match="unsafe loopback"):
        validate_loopback_safety(
            physical_attenuation_db=20,
            strongest_tx_gain_db=strongest_tx_gain_db,
        )


def test_physical_attenuation_must_be_declared():
    with pytest.raises(ValueError, match="must be declared"):
        validate_loopback_safety(
            physical_attenuation_db=None,
            strongest_tx_gain_db=-80,
        )
