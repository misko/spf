#!/usr/bin/env python3
"""Focused qualification probes for the optional libiio frame-metadata path.

Run this file with an isolated patched libiio 0.25 or 0.26 in
``LD_LIBRARY_PATH``/``PYTHONPATH``.  It is deliberately independent of pytest
so long stalls and throughput checks can be selected individually on a small
host.
"""

from __future__ import annotations

import argparse
import errno
import os
import socket
import time


def _enable_dual_rx(device) -> None:
    for channel in device.channels:
        if channel.scan_element:
            channel.enabled = True


def _metadata_refill(buffer, *, allow_startup_eagain: bool) -> bytes:
    for retry in range(65):
        try:
            return buffer.refill()
        except OSError as error:
            if (
                error.errno == errno.EAGAIN
                and allow_startup_eagain
                and retry < 64
            ):
                continue
            raise
    raise AssertionError("unreachable")


def _open_device(uri: str, kernel_buffers: int):
    import iio

    context = iio.Context(uri)
    context.set_timeout(10_000)
    device = context.find_device("cf-ad9361-lpc")
    if device is None:
        raise RuntimeError("cf-ad9361-lpc was not found")
    _enable_dual_rx(device)
    device.set_kernel_buffers_count(kernel_buffers)
    if device.kernel_buffers_count != kernel_buffers:
        raise RuntimeError("kernel-buffer count did not round-trip")
    return context, device


def capture(args: argparse.Namespace) -> None:
    import iio

    from spf.direct_radio.usb_protocol import RadioMetadataV3

    _context, device = _open_device(args.uri, args.kernel_buffers)
    buffer = iio.MetadataBuffer(device, args.samples)
    warmup_indices: list[int] = []
    for index in range(args.warmup_refills):
        raw = _metadata_refill(buffer, allow_startup_eagain=index == 0)
        warmup_indices.append(RadioMetadataV3.unpack(raw).buffer_sequence)
    if args.idle_seconds:
        time.sleep(args.idle_seconds)

    indices: list[int] = []
    observations: list[int] = []
    expected_iq_bytes = args.samples * 8
    for index in range(args.refills):
        raw = _metadata_refill(
            buffer,
            allow_startup_eagain=index == 0 and not args.warmup_refills,
        )
        iq = buffer.read()
        metadata = RadioMetadataV3.unpack(raw)
        if len(iq) != expected_iq_bytes:
            raise RuntimeError(f"wrong IQ size: {len(iq)}")
        if metadata.samples_per_channel != args.samples:
            raise RuntimeError("wrong samples_per_channel")
        if metadata.iq_payload_bytes != expected_iq_bytes:
            raise RuntimeError("wrong IQ payload size in metadata")
        if metadata.enabled_scan_mask != 0x0F:
            raise RuntimeError("wrong scan mask")
        if not metadata.gain_observations:
            raise RuntimeError("metadata has no gain observations")
        indices.append(metadata.buffer_sequence)
        observations.append(len(metadata.gain_observations))
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise RuntimeError(f"capture indices are not increasing: {indices}")
    print(
        f"PASS capture uri={args.uri} N={args.kernel_buffers} "
        f"warmup={warmup_indices} indices={indices} "
        f"observations={observations}"
    )


def throughput(args: argparse.Namespace) -> None:
    import iio

    _context, device = _open_device(args.uri, args.kernel_buffers)
    if args.mode == "metadata":
        buffer = iio.MetadataBuffer(device, args.samples)
    else:
        buffer = iio.Buffer(device, args.samples)

    for index in range(args.warmup_refills):
        if args.mode == "metadata":
            _metadata_refill(buffer, allow_startup_eagain=index == 0)
        else:
            buffer.refill()
        if len(buffer.read()) != args.samples * 8:
            raise RuntimeError("wrong warm-up IQ size")

    started = time.perf_counter_ns()
    metadata_bytes = 0
    iq_bytes = 0
    for _ in range(args.frames):
        if args.mode == "metadata":
            metadata_bytes += len(
                _metadata_refill(buffer, allow_startup_eagain=False)
            )
        else:
            buffer.refill()
        iq_bytes += len(buffer.read())
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    expected = args.frames * args.samples * 8
    if iq_bytes != expected:
        raise RuntimeError(f"wrong IQ total: {iq_bytes} != {expected}")
    print(
        f"PASS throughput uri={args.uri} mode={args.mode} "
        f"samples={args.samples} frames={args.frames} seconds={elapsed:.6f} "
        f"iq_MBps={iq_bytes / elapsed / 1_000_000:.3f} "
        f"metadata_bytes={metadata_bytes}"
    )


def _read_line(sock: socket.socket) -> bytes:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("iiOD closed before newline")
        data.extend(chunk)
    return bytes(data)


def _command(sock: socket.socket, value: str) -> bytes:
    sock.sendall(value.encode("ascii") + b"\r\n")
    return _read_line(sock)


def stalled_reader(args: argparse.Namespace) -> None:
    sock = socket.create_connection((args.host, 30431), timeout=10)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.receive_buffer)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.send_buffer)
    actual_receive = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    actual_send = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    reply = _command(
        sock, f"SET iio:device4 BUFFERS_COUNT {args.kernel_buffers}"
    )
    if reply != b"0\n":
        raise RuntimeError(f"SET failed: {reply!r}")
    reply = _command(sock, f"OPENM iio:device4 {args.samples} 0000000f")
    if reply != b"0\n":
        raise RuntimeError(f"OPENM failed: {reply!r}")
    iq_bytes = args.samples * 8
    sock.sendall(
        f"READBUFM iio:device4 {iq_bytes} 65536\r\n".encode("ascii")
    )
    print(
        f"STALLING pid={os.getpid()} local={sock.getsockname()} "
        f"peer={sock.getpeername()} requested_rcvbuf={args.receive_buffer} "
        f"actual_rcvbuf={actual_receive} requested_sndbuf={args.send_buffer} "
        f"actual_sndbuf={actual_send} iq_bytes={iq_bytes}",
        flush=True,
    )
    time.sleep(args.stall_seconds)
    sock.close()
    print("PASS stalled-reader closed-without-reading", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("uri")
    capture_parser.add_argument("--kernel-buffers", type=int, required=True)
    capture_parser.add_argument("--samples", type=int, default=1024)
    capture_parser.add_argument("--refills", type=int, default=10)
    capture_parser.add_argument("--warmup-refills", type=int, default=0)
    capture_parser.add_argument("--idle-seconds", type=float, default=0.0)
    capture_parser.set_defaults(function=capture)

    throughput_parser = subparsers.add_parser("throughput")
    throughput_parser.add_argument("uri")
    throughput_parser.add_argument("mode", choices=("ordinary", "metadata"))
    throughput_parser.add_argument("--samples", type=int, default=262144)
    throughput_parser.add_argument("--frames", type=int, default=64)
    throughput_parser.add_argument("--warmup-refills", type=int, default=2)
    throughput_parser.add_argument("--kernel-buffers", type=int, default=4)
    throughput_parser.set_defaults(function=throughput)

    stalled_parser = subparsers.add_parser("stalled-reader")
    stalled_parser.add_argument("host")
    stalled_parser.add_argument("--samples", type=int, default=262144)
    stalled_parser.add_argument("--kernel-buffers", type=int, default=4)
    stalled_parser.add_argument("--receive-buffer", type=int, default=4096)
    stalled_parser.add_argument("--send-buffer", type=int, default=4096)
    stalled_parser.add_argument("--stall-seconds", type=float, default=30.0)
    stalled_parser.set_defaults(function=stalled_reader)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
