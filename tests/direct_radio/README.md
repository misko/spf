# `tests/direct_radio`

Contract tests for the `spf.direct_radio` package — the transport-neutral radio
link (`usb_protocol`, `usb_receiver`, `ip_protocol`, `ip_receiver`,
`sample_clock`) extracted out of `spf.sdrpluto`.

```bash
pytest tests/direct_radio/          # ~1 s, no hardware, no flags
```

## Why this directory exists separately

Behavioural coverage of these modules already exists and is substantial —
`tests/test_direct_usb_protocol.py`, `test_direct_usb_receiver.py`,
`test_direct_ip_protocol.py`, `test_direct_ip_receiver.py`,
`test_sample_clock.py` and the hardware suite under `tests/radio_hardware/`,
roughly 4 000 lines in total. All of it imports through the
`spf.sdrpluto.direct_*` shims.

That leaves one thing untested, and untestable from there: the **packaging
contract** the extraction was performed for. Because the shims are
`from spf.direct_radio.X import *`, both paths reach the same objects, so no
behavioural test can tell whether the boundary still holds. These tests check
the boundary itself:

| Test | What breaks if it fails |
|---|---|
| `test_every_module_imports_under_the_new_path` | new code cannot use `spf.direct_radio.X` |
| `test_package_imports_nothing_else_from_spf` | the package can no longer be vendored without the DOA stack |
| `test_package_third_party_dependencies_stay_minimal` | a consumer inherits a new install requirement |
| `test_importing_the_package_does_not_pull_in_libusb_or_sockets` | `import spf.direct_radio` stops being side-effect free |
| `test_shim_reexports_the_same_objects` | `isinstance` across the old and new paths stops holding |
| `test_shims_stay_thin` | logic re-accumulates in `spf.sdrpluto`, and the paths silently diverge |

## Notes for anyone editing these

**They are checked to fail.** Each assertion was verified against a deliberate
break: an `import spf.utils` added to the package, an extra statement in a
shim, an eager `usb_receiver` import in `__init__`, a `scipy` dependency, and a
rebound `TimeAnchorV1` in a shim. All five turned the relevant test red with
its intended message.

**The `spf` import check is AST-based, not textual.** The package docstring
contains literal `from spf.direct_radio.usb_receiver import ...` usage
examples, which a grep-style check reports as real imports.

**The isolation check runs in a subprocess.** By the time pytest reaches it the
session has already imported the receivers, so an in-process check would pass
no matter what the package does. It also compares against a baseline rather
than asserting absolute absence, so an unrelated start-up import cannot turn it
red.

**`MODULES` and `SHIMS` are spelled out rather than globbed.** Adding a module
is exactly the moment the dependency contract can be broken, so it should
require a deliberate edit here too.

**No hardware, no gating.** These are properties of the file layout, not of the
radio, so nothing here sits behind `--radio-hardware`. The attached-radio gates
live in `tests/radio_hardware/`.

## Known wart

Collection still loads the parent `tests/conftest.py`, which imports
`spf.dataset.spf_dataset` and the training stack — so running these pulls in
torch and wandb despite the package under test needing neither. It costs a few
seconds and does not affect what is asserted. Making the directory genuinely
standalone would mean moving it out from under `tests/`, which is a bigger call
than these tests warrant.
