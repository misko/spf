# Installing the SPF libiio host extension

The v5 radio firmware uses ordinary libiio IQ frames with an optional metadata
record containing capture index, sample sequence/time, gain history, gain
endpoints, and RSSI endpoints. Reading that record requires the matching SPF C
library and Python binding. Installing only `pylibiio` from PyPI is not enough:
that package does not contain the modified C protocol implementation.

## Supported versions

Use 0.25 unless an existing application specifically requires 0.26.

| Line | Immutable source tag | Commit | Role |
|---|---|---|---|
| 0.25 | `spf-frame-metadata-source/v0.25-final-v3` | `c26258bfa33098c2b215e19cf85d448e89499b1a` | recommended; same line as radio iiOD |
| 0.26 | `spf-frame-metadata-source/v0.26-final-v3` | `d5695c3eaa9cec99cc6f7b2c91565555044b907a` | supported host alternative |

Both were qualified over USB and standard libiio IP/TCP against
`v0.38-plutoplus-spf-libiio-metadata-v5`. Do not use an arbitrary branch tip.

## Ubuntu, Debian, or Raspberry Pi OS

Install build prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake make pkg-config flex bison python3-dev python3-venv \
  libxml2-dev libaio-dev libusb-1.0-0-dev
```

Create the SPF environment and install its Python dependencies first:

```bash
git clone https://github.com/misko/spf.git
cd spf
python3 -m venv ~/spf-virtualenv
~/spf-virtualenv/bin/python -m pip install --upgrade pip
~/spf-virtualenv/bin/python -m pip install -e .
```

Then install the patched library and binding. This step deliberately comes
last, because a later installation of the unmodified PyPI `pylibiio` package
could replace `iio.py` in the virtual environment.

```bash
./install_spf_libiio.sh \
  --series 0.25 \
  --python ~/spf-virtualenv/bin/python
```

The default prefix is `/usr/local`. The script verifies the immutable Git
commit, builds the local/USB/network backends and tools with the same DNS-SD-off
configuration used for hardware qualification, runs `ldconfig`, installs the
generated binding into the requested virtual environment, and fails unless
`iio.MetadataBuffer` and the exact Git version are present.

To use the supported 0.26 host instead:

```bash
./install_spf_libiio.sh --series 0.26 --python ~/spf-virtualenv/bin/python
```

Only one line should be installed into `/usr/local` at a time. The isolated
`--prefix /opt/libiio-spf-0.25` form is useful for testing two versions, but
then every process needs that prefix's `lib` directory in `LD_LIBRARY_PATH`.

## Verify an installation

```bash
~/spf-virtualenv/bin/python - <<'PY'
import iio
print(iio.version, iio.__file__)
print("MetadataBuffer:", hasattr(iio, "MetadataBuffer"))
assert iio.version == (0, 25, "c26258b")
assert hasattr(iio, "MetadataBuffer")
PY

/usr/local/bin/iio_info --version
/usr/local/bin/iio_info -S
```

For 0.26, the expected tuple is `(0, 26, "d5695c3")`.

The radio context must also advertise the capability:

```bash
/usr/local/bin/iio_attr -u ip:RADIO_ADDRESS -C iio,buffer-metadata
```

The value is `1` on the v5 firmware. An upstream/older radio still works with
the ordinary buffer API, but it cannot create `MetadataBuffer` captures.

## Updating an existing SPF checkout

After `git pull` or any dependency refresh, this is safe to rerun:

```bash
~/spf-virtualenv/bin/python -m pip install -e .
./install_spf_libiio.sh --series 0.25 --python ~/spf-virtualenv/bin/python
```

The source tags are immutable, so rerunning rebuilds the same reviewed code.
Do not set a global `PYTHONPATH` to a build directory; that makes the Python
binding and loaded C library easy to mismatch.
