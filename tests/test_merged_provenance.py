"""Provenance carried by a merged dataset.

A merged store used to be unable to name the physical Pluto behind r0 -- the 31
receiver attrs holding `sdr_serial` were dropped -- or to say whether the capture
it came from had finalised, because the merged filename discards a `.tmp` suffix.
Five of the 24 datasets merged before 2026-08-06 turned out to have been built
from unfinalised captures with nothing recording it.

These tests pin the properties that fix relies on: the RX attrs land in the slots
zarr already provides, the merge's own root keys survive the copy, an unfinalised
source is visible afterwards, and a reader gets the same answer whether the
record is in-band or in a sidecar.
"""

from __future__ import annotations

import json

import pytest

from spf.dataset import provenance


# ------------------------------------------------------------ fake stores ---


class FakeGroup:
    def __init__(self, attrs=None, children=None):
        self.attrs = dict(attrs or {})
        self._children = children or {}

    def __getitem__(self, path):
        node = self
        for part in path.split("/"):
            node = node._children[part]
        return node

    def __contains__(self, path):
        try:
            self[path]
            return True
        except KeyError:
            return False


def _rx_store(*, root=None, r0_serial="SERIAL-R0", r1_serial="SERIAL-R1"):
    return FakeGroup(
        attrs=root
        if root is not None
        else {
            "capture_status": "complete",
            "capture_records_written_by_receiver": [3000, 3000],
            "sdr_identity_version": 1,
            "radio_metadata_schema_version": 1,
        },
        children={
            "receivers": FakeGroup(
                children={
                    "r0": FakeGroup({"sdr_serial": r0_serial, "firmware_verified": True}),
                    "r1": FakeGroup({"sdr_serial": r1_serial, "firmware_verified": True}),
                }
            )
        },
    )


def _tx_store(status="complete"):
    return FakeGroup(attrs={"capture_status": status})


def _merged_store():
    """A merged store as the merge builds it: receiver groups present, attrs empty."""
    return FakeGroup(
        attrs={"radio_metadata_schema_version": 2},
        children={
            "receivers": FakeGroup(
                children={"r0": FakeGroup(), "r1": FakeGroup()}
            )
        },
    )


def _build(rx_fn="/d/rx_tag_RO1.zarr", tx_fn="/d/tx_tag_RO2.zarr", **kwargs):
    return provenance.build_provenance(
        rx_zarr=kwargs.pop("rx_zarr", _rx_store()),
        rx_fn=rx_fn,
        tx_zarr=kwargs.pop("tx_zarr", _tx_store()),
        tx_fn=tx_fn,
        center_lat=37.8349747,
        center_lon=-122.4788380,
        **kwargs,
    )


# ------------------------------------------------------- the native slots ---


def test_rx_receiver_attrs_land_in_the_slots_zarr_already_provides():
    """No wrapper: receivers/r0/.zattrs already existed and was empty."""
    root, receivers = _build()
    merged = _merged_store()
    provenance.write_provenance(merged, root, receivers)

    assert merged["receivers/r0"].attrs["sdr_serial"] == "SERIAL-R0"
    assert merged["receivers/r1"].attrs["sdr_serial"] == "SERIAL-R1"


def test_the_merges_own_root_key_is_not_clobbered_by_the_source():
    """The RX source carries radio_metadata_schema_version=1; the merge sets 2.

    A blanket root-attr copy would silently downgrade it, which is why root
    metadata is namespaced under rx_source instead of copied flat.
    """
    root, receivers = _build()
    merged = _merged_store()
    provenance.write_provenance(merged, root, receivers)

    assert merged.attrs["radio_metadata_schema_version"] == 2
    assert merged.attrs["rx_source"]["attrs"]["radio_metadata_schema_version"] == 1


def test_projection_is_recorded_so_tx_pos_can_be_inverted_without_a_fit():
    root, _ = _build()
    assert root["projection"]["proj"] == "aeqd"
    assert root["projection"]["lat_0"] == pytest.approx(37.8349747)
    assert root["projection"]["lon_0"] == pytest.approx(-122.4788380)
    assert root["projection"]["position_units"] == "mm"


# --------------------------------------------------- the .tmp distinction ---


def test_an_unfinalized_rx_source_is_visible_after_the_merge():
    """The merged filename drops the .tmp; the record must not."""
    root, _ = _build(rx_fn="/d/rx_tag_RO1.zarr.tmp")
    assert root["rx_source"]["suffix"] == ".zarr.tmp"
    assert root["rx_source"]["finalized"] is False


def test_source_is_trustworthy_rejects_an_unfinalized_capture():
    root, _ = _build(rx_fn="/d/rx_tag_RO1.zarr.tmp")
    ok, reason = provenance.source_is_trustworthy({"root": root})
    assert ok is False
    assert ".zarr.tmp" in reason


def test_source_is_trustworthy_rejects_an_incomplete_capture_status():
    rx = _rx_store(root={"capture_status": "incomplete"})
    root, _ = _build(rx_zarr=rx)
    ok, reason = provenance.source_is_trustworthy({"root": root})
    assert ok is False
    assert "incomplete" in reason


def test_source_is_trustworthy_accepts_a_clean_capture():
    root, _ = _build()
    ok, _ = provenance.source_is_trustworthy({"root": root})
    assert ok is True


def test_missing_provenance_is_unknown_not_trusted():
    """A dataset merged before this change must not read as verified-good."""
    ok, reason = provenance.source_is_trustworthy(None)
    assert ok is None
    assert "no provenance" in reason


# -------------------------------------------------------------- TX record ---


def test_tx_record_is_traceability_only_not_radio_identity():
    """TX contributes GPS; its GPS is already in tx_pos_*_mm. No 31 attrs."""
    root, _ = _build()
    tx = root["tx_source"]
    assert set(tx) == {"store", "suffix", "finalized", "capture_status"}
    assert tx["store"] == "tx_tag_RO2.zarr"


def test_an_unfinalized_tx_is_recorded_too():
    root, _ = _build(tx_fn="/d/tx_tag_RO2.zarr.tmp")
    assert root["tx_source"]["finalized"] is False


# ---------------------------------------------------------------- reading ---


def test_radio_identity_answers_which_pluto_was_r0():
    root, receivers = _build()
    identity = provenance.radio_identity({"root": root, "receivers": receivers})
    assert identity == {"r0": "SERIAL-R0", "r1": "SERIAL-R1"}


def test_radio_identity_falls_back_to_the_transport_serial():
    rx = _rx_store()
    rx["receivers/r0"].attrs = {"direct_usb_serial": "USB-ONLY"}
    _, receivers = _build(rx_zarr=rx)
    assert provenance.radio_identity({"receivers": receivers})["r0"] == "USB-ONLY"


def test_in_band_is_preferred_over_a_sidecar(tmp_path):
    """Both may exist during the sidecar -> in-place promotion; in-band wins."""
    store = tmp_path / "merged.zarr"
    store.mkdir()
    (tmp_path / "merged.provenance.json").write_text(
        json.dumps({"root": {"rx_source": {"store": "FROM-SIDECAR"}}, "receivers": {}})
    )
    root, receivers = _build()
    merged = _merged_store()
    provenance.write_provenance(merged, root, receivers)

    def fake_open(path, mode="r"):
        return merged

    loaded = provenance.load_provenance(str(store), zarr_open=fake_open)
    assert loaded["source"] == "in-band"
    assert loaded["root"]["rx_source"]["store"] == "rx_tag_RO1.zarr"


def test_sidecar_is_used_when_the_store_has_no_in_band_record(tmp_path):
    store = tmp_path / "merged.zarr"
    store.mkdir()
    (tmp_path / "merged.provenance.json").write_text(
        json.dumps({"root": {"rx_source": {"store": "legacy.zarr"}}, "receivers": {}})
    )

    def fake_open(path, mode="r"):
        return _merged_store()  # no provenance_schema_version

    loaded = provenance.load_provenance(str(store), zarr_open=fake_open)
    assert loaded["source"] == "sidecar"
    assert loaded["root"]["rx_source"]["store"] == "legacy.zarr"


def test_no_record_anywhere_returns_none(tmp_path):
    store = tmp_path / "merged.zarr"
    store.mkdir()

    def fake_open(path, mode="r"):
        return _merged_store()

    assert provenance.load_provenance(str(store), zarr_open=fake_open) is None


# ----------------------------------------------------------- path helpers ---


@pytest.mark.parametrize(
    "store,expected",
    [
        ("/d/a.zarr", "/d/a.provenance.json"),
        ("/d/a.zarr.tmp", "/d/a.provenance.json"),
        ("/d/a.zarr/", "/d/a.provenance.json"),
    ],
)
def test_sidecar_path_ignores_the_tmp_suffix(store, expected):
    """A capture and its unfinalized twin share one provenance file."""
    assert provenance.sidecar_path_for(store) == expected


def test_sidecar_path_is_not_confused_by_zarr_in_a_directory_name():
    assert provenance.sidecar_path_for("/mnt/aug4.zarr_staging/m.zarr") == (
        "/mnt/aug4.zarr_staging/m.provenance.json"
    )


# ------------------------------------------------- merged-name resolution ---


def test_merged_name_splits_into_its_two_sources():
    from spf.scripts.merged_provenance_inventory import split_merged_name

    rx, tx = split_merged_name(
        "rover_2026_08_04_23_08_30_nRX2_bounce_spacing0p043_tag_RO3."
        "rover_2026_08_04_22_56_36_nRX1_circle_spacing0p05075_tag_RO2.zarr"
    )
    assert rx == "rover_2026_08_04_23_08_30_nRX2_bounce_spacing0p043_tag_RO3.zarr"
    assert tx == "rover_2026_08_04_22_56_36_nRX1_circle_spacing0p05075_tag_RO2.zarr"


def test_a_source_name_does_not_split():
    from spf.scripts.merged_provenance_inventory import split_merged_name

    assert split_merged_name("rover_2026_08_04_23_08_30_tag_RO3.zarr") is None
