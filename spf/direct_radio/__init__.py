"""Transport-neutral SPF direct-radio link: wire protocol, receivers, sample clock.

Extracted from ``spf.sdrpluto`` so that consumers outside this repository can depend
on the radio link without pulling in the DOA research stack. The five modules here
import only each other, the standard library, numpy, and (for the receivers) libusb1
— nothing else from ``spf``.

| Module | Role |
|---|---|
| ``usb_protocol`` | the frame, capability, status, start-request and time-anchor wire formats, plus ``RxFrameParser`` |
| ``usb_receiver`` | ``PlutoDirectUsbReceiver`` over a custom USB gadget interface |
| ``ip_protocol`` | UDP control/fragmentation layer wrapping the *same* inner frame |
| ``ip_receiver`` | ``PlutoDirectIpReceiver`` over acknowledged UDP |
| ``sample_clock`` | FPGA-counter to host-time fitting with an explicit error bound |

Both transports carry the identical inner frame, so both expose the same
``first_sample_sequence`` FPGA counter and the same ``TimeAnchorV1`` responses; only
the outer framing differs.

Nothing is re-exported at package level on purpose. ``usb_receiver`` imports libusb1
and ``ip_receiver`` opens sockets, so importing this package must not drag either in.
Import the module you need:

    from spf.direct_radio.usb_receiver import PlutoDirectUsbReceiver
    from spf.direct_radio.sample_clock import fit_sample_clock

The former ``spf.sdrpluto.direct_*``/``sample_clock`` paths remain as re-export shims
so existing callers keep working.

Two facts worth knowing before relying on the timing fields, both measured on RC17:

* ``first_sample_sequence`` is a **per-stream** counter that restarts at 0 under
  protocol v1/v2, and the **free-running FPGA counter** under v3 (flagged by
  ``MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID``). Continuity checks work on either;
  converting to absolute time requires v3, because only then does the frame counter
  share a timebase with the time anchors.
* ``fit_sample_clock`` needs anchors spanning much longer than the ~0.6 ms control
  round trip. Four anchors 5 ms apart fit a nonsense rate; six anchors 250 ms apart
  either side of a capture give ~±0.5 ms and a few ppm. Anchors queried immediately
  after a capture see ~81 ms round trips while the stream tears down — wait ~300 ms.
"""
