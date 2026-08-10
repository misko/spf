"""Capture-schema contracts shared by Rover boot and collection code."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureSchemaContract:
    data_version: int
    rx_transport: str
    direct_usb_protocol_version: int | None = None
    require_gain_metadata: bool = False


V7_CONTRACT = CaptureSchemaContract(
    data_version=7,
    rx_transport="direct_usb",
    direct_usb_protocol_version=2,
    require_gain_metadata=True,
)

V7_SHARED_DIRECT_USB_KEYS = frozenset(
    {
        "protocol-version",
        "frame-count-per-request",
        "gain-observation-interval-samples",
        "gain-observation-capacity",
        "gain-event-capacity",
        "require-gain-metadata",
    }
)


def normalize_capture_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy with implicit schema requirements materialized.

    Data version 7 is SPF's direct-radio gain/RSSI schema. Protocol v2 remains
    the compatibility default, while a top-level ``direct-usb`` mapping can
    select protocol v3 gain-series settings once for every receiver. Explicit
    conflicting values are rejected rather than overwritten.
    """

    normalized = deepcopy(config)
    if normalized.get("data-version") != V7_CONTRACT.data_version:
        return normalized

    receivers = normalized.get("receivers")
    if not isinstance(receivers, list) or not receivers:
        raise ValueError("data-version: 7 requires at least one receiver")

    shared_direct_usb = normalized.get("direct-usb", {})
    if not isinstance(shared_direct_usb, dict):
        raise ValueError("data-version: 7 top-level direct-usb must be a mapping")
    unsupported_shared_keys = set(shared_direct_usb) - V7_SHARED_DIRECT_USB_KEYS
    if unsupported_shared_keys:
        raise ValueError(
            "data-version: 7 top-level direct-usb contains receiver-specific "
            f"or unsupported fields: {sorted(unsupported_shared_keys)}"
        )

    for index, receiver in enumerate(receivers):
        if not isinstance(receiver, dict):
            raise ValueError(f"receivers[{index}] must be a mapping")

        explicit_transport = receiver.get("rx-transport")
        if explicit_transport not in (None, V7_CONTRACT.rx_transport):
            raise ValueError(
                "data-version: 7 requires rx-transport: direct_usb; "
                f"receivers[{index}] requests {explicit_transport!r}"
            )
        receiver["rx-transport"] = V7_CONTRACT.rx_transport

        direct_usb = receiver.setdefault("direct-usb", {})
        if not isinstance(direct_usb, dict):
            raise ValueError(f"receivers[{index}].direct-usb must be a mapping")
        for key, value in shared_direct_usb.items():
            if key in direct_usb and direct_usb[key] != value:
                raise ValueError(
                    f"receivers[{index}].direct-usb.{key} conflicts with "
                    "the top-level direct-usb default"
                )
            direct_usb.setdefault(key, value)

        explicit_protocol = direct_usb.get("protocol-version")
        if explicit_protocol not in (
            None,
            V7_CONTRACT.direct_usb_protocol_version,
            3,
        ):
            raise ValueError(
                "data-version: 7 supports direct-USB protocol version 2 or 3; "
                f"receivers[{index}] requests {explicit_protocol!r}"
            )
        direct_usb.setdefault(
            "protocol-version", V7_CONTRACT.direct_usb_protocol_version
        )
        if direct_usb["protocol-version"] == 3:
            direct_usb.setdefault("gain-observation-interval-samples", 32768)
            # One frame nominally contains 16 observations at the default
            # interval; retain headroom for reads that straddle a boundary.
            direct_usb.setdefault("gain-observation-capacity", 32)
            direct_usb.setdefault("gain-event-capacity", 0)

        explicit_metadata = direct_usb.get("require-gain-metadata")
        if explicit_metadata is False:
            raise ValueError(
                "data-version: 7 requires gain/RSSI metadata; "
                f"receivers[{index}] disables it"
            )
        direct_usb["require-gain-metadata"] = True
        direct_usb.setdefault("frame-count-per-request", 1)

    return normalized


def validate_transport_schema(config: dict[str, Any]) -> None:
    """Validate supported dataset/transport/wire-protocol combinations."""

    receivers = config.get("receivers")
    if not isinstance(receivers, list) or not receivers:
        raise ValueError("capture config requires at least one receiver")

    data_version = config.get("data-version")
    transports = {
        receiver.get("rx-transport", "iio")
        for receiver in receivers
        if isinstance(receiver, dict)
    }
    if len(transports) != 1:
        raise ValueError(
            "all receivers in one capture must use the same RX transport; "
            f"found {sorted(transports)}"
        )

    if data_version == V7_CONTRACT.data_version and transports != {
        V7_CONTRACT.rx_transport
    }:
        raise ValueError("data-version: 7 requires direct_usb transport")

    for receiver in receivers:
        if receiver.get("rx-transport", "iio") != "direct_usb":
            continue
        protocol_version = receiver.get("direct-usb", {}).get("protocol-version", 1)
        if protocol_version == 1 and data_version != 6:
            raise ValueError("direct_usb protocol v1 requires data-version: 6")
        if protocol_version == 2 and data_version not in (4, 7):
            raise ValueError(
                "direct_usb protocol v2 requires data-version: 4 "
                "(compatibility) or 7 (full metadata)"
            )
        if protocol_version == 3 and data_version != 7:
            raise ValueError("direct_usb protocol v3 requires data-version: 7")
        if protocol_version not in (1, 2, 3):
            raise ValueError(
                f"unsupported direct_usb protocol version: {protocol_version}"
            )
