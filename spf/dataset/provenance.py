"""Which physical radio produced a merged dataset, and did its source finish?

``v7_tx_rx_merge.py`` used to answer neither. It dropped 31 of 31 receiver attrs
and 3 of 4 root attrs, so a merged store could not name the Pluto behind ``r0``
(``sdr_serial`` lives in no other field) nor say whether the capture it came from
had finalised. Five of the 24 datasets merged before 2026-08-06 turned out to
have been built from ``.zarr.tmp`` sources -- three ``in_progress``, two
``incomplete`` -- and nothing in them said so, because the merged filename drops
the ``.tmp``.

WHERE THIS GOES IN THE STORE
----------------------------
Mostly into slots zarr already provides. A merged store already carries
``receivers/r0/.zattrs`` and ``receivers/r1/.zattrs``; the merge simply left them
empty while the source had 31 keys in each. So the RX radio identity is a dict
copy into the identical slot -- no new schema, no wrapper::

    receivers/r0/.zattrs   <- the RX source's r0 attrs, verbatim
    receivers/r1/.zattrs   <- the RX source's r1 attrs, verbatim

Only two root keys are new. Root-level source metadata is namespaced rather than
copied flat, because the merge owns ``radio_metadata_schema_version`` at root and
a blanket copy would silently overwrite it with the source's value::

    provenance_schema_version   this module's version
    projection                  {proj, lat_0, lon_0, units} -- the aeqd frame
    rx_source                   {store, suffix, finalized, attrs: <root attrs>}
    tx_source                   {store, suffix, finalized, capture_status}

WHY TX GETS FOUR FIELDS AND RX GETS THIRTY-ONE
----------------------------------------------
The TX contributes GPS and nothing else, and its GPS is already in the merged
store as ``tx_pos_x_mm`` / ``tx_pos_y_mm``. That projection is exactly invertible
without the TX source: the store also holds the RX's raw ``gps_lat``/``gps_long``
alongside its projected ``rx_pos_x_mm``/``rx_pos_y_mm``, which over-determines
the aeqd centre (recovered to 0.0000 mm RMS on a real dataset). So a TX used only
for its GPS has no radio identity worth keeping -- just enough to trace which
store it was and whether that store finalised.

``projection`` exists so that inversion does not require a least-squares fit, and
because the centre is derived per-merge from the mean TX GPS rather than fixed.

READING IT BACK
---------------
``load_provenance`` prefers in-band attrs and falls back to a
``<store>.provenance.json`` sidecar, so datasets written before and after this
change read identically.
"""

from __future__ import annotations

import json
import os

PROVENANCE_SCHEMA_VERSION = 1

# Root attrs the merge itself owns; never overwritten from a source store.
_MERGE_OWNED_ROOT_ATTRS = {"radio_metadata_schema_version"}

SIDECAR_SUFFIX = ".provenance.json"


def sidecar_path_for(store_path: str) -> str:
    """``a.zarr -> a.provenance.json`` (and ``a.zarr.tmp -> a.provenance.json``)."""
    base = store_path.rstrip("/")
    for suffix in (".zarr.tmp", ".zarr"):
        if base.endswith(suffix):
            return base[: -len(suffix)] + SIDECAR_SUFFIX
    return base + SIDECAR_SUFFIX


def store_suffix(store_path: str) -> str:
    """``.zarr`` or ``.zarr.tmp`` -- the distinction the merged filename destroys."""
    base = store_path.rstrip("/")
    return ".zarr.tmp" if base.endswith(".zarr.tmp") else ".zarr"


def is_finalized(store_path: str) -> bool:
    return store_suffix(store_path) == ".zarr"


def _source_record(store_path, root_attrs, *, full_attrs):
    record = {
        "store": os.path.basename(store_path.rstrip("/")),
        "suffix": store_suffix(store_path),
        "finalized": is_finalized(store_path),
    }
    if full_attrs:
        record["attrs"] = dict(root_attrs)
    else:
        # A TX used only for its GPS needs traceability, not identity.
        record["capture_status"] = root_attrs.get("capture_status")
    return record


def build_provenance(
    *,
    rx_zarr,
    rx_fn,
    tx_zarr,
    tx_fn,
    center_lat,
    center_lon,
    n_receivers=2,
):
    """Assemble the root attrs and per-receiver attrs a merged store should carry.

    Returns ``(root_attrs, receiver_attrs)`` where ``receiver_attrs`` maps
    ``"r0"``/``"r1"`` to the RX source's attrs for that receiver, verbatim.
    Nothing is written here -- see ``write_provenance``.
    """
    root = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "projection": {
            "proj": "aeqd",
            "lat_0": float(center_lat),
            "lon_0": float(center_lon),
            "units": "m",
            # tx_pos_*_mm / rx_pos_*_mm are millimetres in this frame.
            "position_units": "mm",
        },
        "rx_source": _source_record(rx_fn, dict(rx_zarr.attrs), full_attrs=True),
        "tx_source": _source_record(tx_fn, dict(tx_zarr.attrs), full_attrs=False),
    }
    receiver_attrs = {}
    for r in range(n_receivers):
        key = f"r{r}"
        try:
            receiver_attrs[key] = dict(rx_zarr[f"receivers/{key}"].attrs)
        except KeyError:
            receiver_attrs[key] = {}
    return root, receiver_attrs


def write_provenance(new_zarr, root_attrs, receiver_attrs):
    """Write provenance into a merged store, in its native slots.

    Root keys are namespaced (``rx_source`` / ``tx_source``) so nothing the merge
    owns is clobbered. Per-receiver attrs go into ``receivers/r*/.zattrs``, which
    already exist and are empty.
    """
    for key, value in root_attrs.items():
        if key in _MERGE_OWNED_ROOT_ATTRS:
            continue
        new_zarr.attrs[key] = value
    for name, attrs in receiver_attrs.items():
        group = new_zarr[f"receivers/{name}"]
        for key, value in attrs.items():
            group.attrs[key] = value


def load_provenance(store_path, zarr_open=None):
    """Read a merged store's provenance, in-band first, then sidecar.

    Returns ``None`` when neither exists -- a merged dataset written before this
    change and not yet backfilled.
    """
    if zarr_open is None:
        from spf.scripts.zarr_utils import zarr_open_from_lmdb_store as zarr_open

    root, receivers = {}, {}
    try:
        z = zarr_open(store_path, mode="r")
        root = dict(z.attrs)
        if "provenance_schema_version" in root:
            for name in ("r0", "r1"):
                try:
                    receivers[name] = dict(z[f"receivers/{name}"].attrs)
                except KeyError:
                    pass
            return {"source": "in-band", "root": root, "receivers": receivers}
    except Exception:
        pass

    sidecar = sidecar_path_for(store_path)
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("source", "sidecar")
        return payload
    return None


def radio_identity(provenance):
    """``{"r0": <serial>, "r1": <serial>}`` for the RX radios, or ``{}``.

    The question this module exists for: which physical Pluto was r0? Answering
    it from a merged dataset used to require the source store.
    """
    if not provenance:
        return {}
    out = {}
    for name, attrs in (provenance.get("receivers") or {}).items():
        serial = attrs.get("sdr_serial") or attrs.get("direct_usb_serial")
        if serial:
            out[name] = serial
    return out


def source_is_trustworthy(provenance):
    """``(ok, reason)`` -- did the RX capture behind this dataset finalise cleanly?

    Five of the 24 datasets merged before 2026-08-06 answer no, and the merged
    filename gives no hint of it.
    """
    if not provenance:
        return None, "no provenance recorded"
    rx = (provenance.get("root") or {}).get("rx_source") or {}
    if not rx:
        return None, "no rx_source recorded"
    if not rx.get("finalized", True):
        return False, f"RX source is {rx.get('suffix')} (capture never finalised)"
    status = (rx.get("attrs") or {}).get("capture_status")
    if status not in (None, "complete"):
        return False, f"RX source capture_status={status!r}"
    return True, f"RX source finalised, capture_status={status!r}"
