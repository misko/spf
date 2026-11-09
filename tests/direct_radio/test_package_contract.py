"""Contract tests for the extracted :mod:`spf.direct_radio` package.

These deliberately do **not** re-test protocol or sample-clock behaviour. That
already lives in ``tests/test_direct_*.py`` and ``tests/test_sample_clock.py``,
which reach the very same objects through the ``spf.sdrpluto`` shims.

What is untested anywhere else is the *packaging* contract the extraction was
performed for, and which is invisible to every behavioural test precisely
because the shims make both paths equivalent:

* the new import path works, and is the one new code should use;
* the package imports nothing else from ``spf``, so a consumer outside this
  repository can depend on the radio link without the DOA research stack;
* importing the package does not drag in libusb or open sockets, which
  ``spf/direct_radio/__init__.py`` promises explicitly;
* the ``spf.sdrpluto`` shims re-export the *same objects*, so ``isinstance``
  holds across both paths;
* the shims stay thin, rather than slowly re-accumulating logic.

Every one of those is a property of the file layout rather than of the radio,
so none of this needs hardware and none of it is gated behind
``--radio-hardware``.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import subprocess
import sys
import sysconfig
import textwrap

import pytest

import spf
import spf.direct_radio


# The package modules are spelled out rather
# than globbed so that adding a module is a deliberate edit here too -- a new
# module is exactly the moment the dependency contract below can be broken.
MODULES = (
    "iio_metadata",
    "ip_protocol",
    "ip_receiver",
    "sample_clock",
    "tandem_agc",
    "usb_protocol",
    "usb_receiver",
)

# Old path -> new path. The old ones remain as re-export shims.
SHIMS = {
    "spf.sdrpluto.direct_ip_protocol": "spf.direct_radio.ip_protocol",
    "spf.sdrpluto.direct_ip_receiver": "spf.direct_radio.ip_receiver",
    "spf.sdrpluto.direct_usb_protocol": "spf.direct_radio.usb_protocol",
    "spf.sdrpluto.direct_usb_receiver": "spf.direct_radio.usb_receiver",
    "spf.sdrpluto.sample_clock": "spf.direct_radio.sample_clock",
}

# Third-party packages the extracted link is allowed to need. Keeping this list
# short is the entire point of the extraction: every addition is a new burden
# on any consumer vendoring the package.
ALLOWED_THIRD_PARTY = frozenset({"iio", "numpy", "usb1"})

# Modules that genuinely need libusb, and so cannot be imported where it is
# absent. The protocol and clock modules must never join this set.
NEEDS_LIBUSB = frozenset({"ip_receiver", "usb_receiver"})

PACKAGE_DIR = pathlib.Path(spf.direct_radio.__file__).resolve().parent
REPO_ROOT = pathlib.Path(spf.__file__).resolve().parent.parent


def _package_sources() -> list[pathlib.Path]:
    sources = sorted(PACKAGE_DIR.glob("*.py"))
    assert sources, f"no python sources found under {PACKAGE_DIR}"
    return sources


def _imported_module_names(tree: ast.AST) -> list[str]:
    """Every absolute module name imported by ``tree``.

    Uses the AST rather than a text search on purpose: the package docstring
    contains literal ``from spf.direct_radio.usb_receiver import ...`` usage
    examples, which a grep-style check reports as real imports.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot leave the package, so it is always fine.
            if node.level:
                continue
            if node.module:
                names.append(node.module)
    return names


@pytest.mark.parametrize("module_name", MODULES)
def test_every_module_imports_under_the_new_path(module_name):
    """New code must be able to say ``spf.direct_radio.X``, not the shim."""
    if module_name in NEEDS_LIBUSB:
        pytest.importorskip("usb1", reason="receivers require libusb1")
    module = importlib.import_module(f"spf.direct_radio.{module_name}")
    assert module.__name__ == f"spf.direct_radio.{module_name}"


def test_package_imports_nothing_else_from_spf():
    """The extraction's core promise: no dependency on the rest of ``spf``.

    If this fails, the package can no longer be vendored on its own, and the
    ``__init__`` docstring's claim about outside consumers has become false.
    """
    offenders: dict[str, list[str]] = {}
    for path in _package_sources():
        bad = [
            name
            for name in _imported_module_names(ast.parse(path.read_text()))
            if (name == "spf" or name.startswith("spf."))
            and not name.startswith("spf.direct_radio")
        ]
        if bad:
            offenders[path.name] = sorted(set(bad))
    assert offenders == {}, (
        "spf.direct_radio must not import the rest of spf; a consumer outside "
        f"this repository cannot satisfy these: {offenders}"
    )


def test_package_third_party_dependencies_stay_minimal():
    """Guard the dependency surface, not just the ``spf`` boundary.

    Standard-library imports are free; anything else is a package a consumer
    has to install. ``sys.stdlib_module_names`` is authoritative for the
    running interpreter, so this needs no hand-maintained stdlib list.
    """
    unexpected: dict[str, list[str]] = {}
    for path in _package_sources():
        tops = {
            name.split(".", 1)[0]
            for name in _imported_module_names(ast.parse(path.read_text()))
        }
        extra = sorted(
            top
            for top in tops
            if top not in sys.stdlib_module_names
            and top not in ALLOWED_THIRD_PARTY
            and top != "spf"
        )
        if extra:
            unexpected[path.name] = extra
    assert unexpected == {}, (
        "new third-party dependencies in spf.direct_radio; every one of these "
        f"becomes a requirement for anyone vendoring the package: {unexpected}"
    )


def test_importing_the_package_does_not_pull_in_libusb_or_sockets():
    """``__init__`` re-exports nothing so that this stays true.

    Run in a subprocess: by the time pytest reaches this test the session has
    already imported the receivers, so an in-process check would pass whatever
    the package does.  Compared against a baseline interpreter rather than
    asserting absolute absence, so an unrelated import at interpreter start-up
    cannot turn this red.
    """
    probe = textwrap.dedent(
        """
        import sys
        watched = ("usb1", "socket", "torch", "zarr")
        before = {name for name in watched if name in sys.modules}
        import spf.direct_radio
        after = {name for name in watched if name in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert leaked == [], (
        "importing spf.direct_radio must stay side-effect free, but it pulled "
        f"in {leaked}; the package __init__ deliberately re-exports nothing"
    )


@pytest.mark.parametrize("shim_name, target_name", sorted(SHIMS.items()))
def test_shim_reexports_the_same_objects(shim_name, target_name):
    """``isinstance`` must hold across the old and new paths.

    Both are live: every existing unit and hardware test still imports through
    the shim, so a name dropped from the new module would surface here as a
    missing re-export rather than as a confusing failure elsewhere.
    """
    pytest.importorskip("usb1", reason="the sdrpluto shims reach the receivers")
    shim = importlib.import_module(shim_name)
    target = importlib.import_module(target_name)

    public = sorted(name for name in vars(target) if not name.startswith("_"))
    assert public, f"{target_name} exports no public names"

    missing = [name for name in public if not hasattr(shim, name)]
    assert missing == [], f"{shim_name} does not re-export {missing}"

    rebound = [
        name for name in public if getattr(shim, name) is not getattr(target, name)
    ]
    assert rebound == [], (
        f"{shim_name} rebinds {rebound} to different objects than "
        f"{target_name}; isinstance across the two paths no longer holds"
    )


@pytest.mark.parametrize("shim_name, target_name", sorted(SHIMS.items()))
def test_shims_stay_thin(shim_name, target_name):
    """A shim is a docstring plus one star-import, and must stay that way.

    Logic reappearing in ``spf.sdrpluto`` is how the two paths would silently
    stop being equivalent, which is the failure the star-import check above
    cannot see.
    """
    source = pathlib.Path(
        importlib.import_module(shim_name).__file__
    ).resolve().read_text()
    body = ast.parse(source).body

    # Drop the module docstring if present.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    assert len(body) == 1, (
        f"{shim_name} should contain exactly one statement after its "
        f"docstring, found {len(body)}; move new logic into {target_name}"
    )
    statement = body[0]
    assert isinstance(statement, ast.ImportFrom)
    assert statement.module == target_name
    assert [alias.name for alias in statement.names] == ["*"]


def test_package_lives_outside_the_stdlib_and_site_packages():
    """Cheap sanity check that these tests bound the working tree.

    If ``spf`` resolves to an installed copy rather than the checkout, every
    assertion above would describe a different tree than the one being edited.
    """
    site_packages = sysconfig.get_paths().get("purelib")
    if site_packages:
        assert not str(PACKAGE_DIR).startswith(str(pathlib.Path(site_packages))), (
            f"spf.direct_radio resolved to an installed copy at {PACKAGE_DIR}; "
            "these contract tests would not be describing the working tree"
        )
    assert (REPO_ROOT / "spf" / "direct_radio").is_dir()
