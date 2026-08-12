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

### Recommended: prebuilt Debian 12 artifacts

Normal rover deployment should use the release bundle rather than compiling on
the rover. One `arm64` package supports both Pi 4 and Pi 5 when they run a
64-bit Debian 12 or Raspberry Pi OS 12 userland. Standard x86-64 hosts use the
`amd64` package. The Python wheel is `py3-none-any` and is shared by both.

Download these three files from one `libiio-artifacts-v*` SPF GitHub release:

- `spf-libiio_0.25+spfmeta3-1_arm64.deb` on Pi 4/Pi 5, or the `_amd64.deb`
  equivalent on x86-64
- `pylibiio-0.25+spfmeta3-py3-none-any.whl`
- `SHA256SUMS`

Place them in one directory and install them after SPF's ordinary Python
dependencies:

```bash
./install_spf_libiio_artifacts.sh \
  --bundle ~/Downloads/spf-libiio \
  --python ~/spf-virtualenv/bin/python
```

The installer checks every checksum, selects the package matching the local
Debian architecture, lets `apt` replace the distribution libiio packages, puts
the patched wheel in the selected virtual environment, and verifies
`MetadataBuffer`. Do not combine a `.deb`, wheel, and checksum file from
different releases.

The native package is intentionally limited to Debian 12's ABI. Build and test
a separate package line before supporting Debian 13, Ubuntu, 32-bit Raspberry
Pi OS (`armhf`), or another libc baseline. Pi CPU generation alone does not
require another build.

### Building artifacts locally

Artifact definitions and source locks live in `packaging/libiio/`. Build on a
native Debian 12 `amd64` or `arm64` machine:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates cmake dpkg-dev git pkg-config flex bison \
  python3 python3-dev python3-pip python3-setuptools python3-wheel \
  libaio-dev libusb-1.0-0-dev libxml2-dev

packaging/libiio/build_artifacts.sh \
  --series 0.25 \
  --output-dir dist/libiio
packaging/libiio/test_artifacts.sh --bundle dist/libiio
```

`packaging/libiio/versions.sh` is the single source of truth for immutable
libiio tags, commits, metadata revision, and Debian packaging revision. Change
the metadata revision when the protocol/binding changes; increment only the
package revision for packaging-only fixes. Review the source commit before
changing either lock.

The builder deliberately does not use libiio 0.25's legacy CPack dependency
discovery. It stages the normal CMake installation and creates explicit Debian
metadata, while building the pure Python binding as a separate wheel so it can
be installed into an isolated SPF virtual environment.

### CI, releases, and artifact tests

`.github/workflows/libiio-packages.yml` builds natively in clean Debian 12
containers on GitHub's `amd64` and `arm64` runners. Each architecture performs:

1. immutable tag/commit verification;
2. package checksum, architecture, file-content, and wheel API inspection;
3. installation of the exact `.deb` and wheel into the clean container;
4. dynamic-link, `iio_info`, backend scan, Python version, and
   `MetadataBuffer` smoke tests; and
5. the focused SPF dependency-contract tests.

Pull requests and changes on `main` retain the packages as workflow artifacts.
To publish an immutable GitHub release after both architecture jobs pass, tag
the reviewed SPF commit using a name such as:

```bash
git tag -a libiio-artifacts-v0.25-spfmeta3.1 -m "SPF libiio 0.25 metadata artifacts"
git push origin libiio-artifacts-v0.25-spfmeta3.1
```

CI then publishes the two `.deb` files, one wheel, and a release-level
`SHA256SUMS`. Hardware tests are intentionally a separate promotion gate: after
a new source revision, install the release bundle on at least one Pi 4/Pi 5 and
one x86-64 host, then run the existing `--radio-hardware` USB and IP metadata,
rate, retune, and soak tests. Packaging-only revisions need clean-container
tests plus one USB/IP smoke capture; they do not require repeating the complete
RF qualification unless native code changed.

### Source-build fallback

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
