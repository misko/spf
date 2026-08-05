"""Provenance header and logging setup for the per-capture ``.log`` sidecar.

Every capture writes three files next to each other: the Zarr store, a ``.yaml``
copy of the effective config, and a ``.log``. Until 2026-08-05 the ``.log`` was
always zero bytes. The cause was not a missing handler -- the collector does
attach a ``logging.FileHandler`` to it -- but that
``spf.mavlink.mavlink_controller`` calls ``logging.basicConfig(filename=...)``
at *import* time. Importing it installs a root handler, which makes the
collector's later ``basicConfig(...)`` a silent no-op: the sidecar was opened
(hence created, hence zero bytes) and every log line went to a relative
``logs.log`` in whatever directory the process happened to start in.

Two things follow, and this module owns both:

* ``configure_capture_logging`` uses ``force=True`` so the capture's own
  handlers always win, and resolves the sidecar to an ABSOLUTE path so the
  destination never depends on the working directory.
* Before any log line, the sidecar gets a provenance header answering "what run
  produced the store sitting next to this file?" -- argv, resolved config path
  and where that path came from, rover id, tag, host, git commit, and BOTH the
  local and UTC wall clock.

The timestamps are labelled and doubled on purpose. Capture filenames are
stamped in LOCAL time, and rovers do not agree on a timezone: on 2026-08-04
Rover 4 was on America/Los_Angeles while Rovers 1-3 were on Europe/London, so
two captures taken minutes apart were named eight hours and one calendar day
apart. A store's true instant must be readable without guessing which rover
wrote it.

The header is written only for the run currently in progress, into its own
``.log.tmp``. Nothing here ever opens or rewrites an already-recorded sidecar.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A reader (or a test) can find where provenance stops and log lines start.
CAPTURE_LOG_HEADER_START = "# ==== spf capture provenance ===="
CAPTURE_LOG_HEADER_END = "# ==== end capture provenance ===="

# Where the capture config path came from. drone_run.sh passes --yaml-config a
# path it got from spf.scripts.rover_capture_config, which uses the canonical
# per-rover config unless SPF_CAPTURE_CONFIG overrides it.
CONFIG_SOURCE_ENV = "SPF_CAPTURE_CONFIG"
CONFIG_SOURCE_CANONICAL = "canonical-resolver"
CONFIG_SOURCE_UNKNOWN = "unknown"

ROVER_ID_FILE = Path("/home/pi/rover_id")


def git_commit(repo_root: Path | str = REPO_ROOT) -> str:
    """Return the checkout's HEAD commit, with a ``-dirty`` suffix if modified.

    Never raises: a capture must not fail because git is missing or the
    checkout is an export. Provenance is best-effort but always present.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.decode().strip()
    except Exception:  # noqa: BLE001 - provenance must never break a capture
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.decode().strip()
    except Exception:  # noqa: BLE001
        return commit
    return f"{commit}-dirty" if dirty else commit


def resolve_config_source(
    config_path: str | Path | None, env: dict | None = None
) -> str:
    """Say whether the capture config came from the env override or the resolver.

    ``SPF_CAPTURE_CONFIG`` counts only when it actually names the config in
    use; an override left set but pointed elsewhere must not be reported as the
    source of this run's config.
    """
    env = os.environ if env is None else env
    override = env.get(CONFIG_SOURCE_ENV)
    if config_path is None:
        return CONFIG_SOURCE_UNKNOWN
    resolved = Path(config_path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        resolved = resolved.absolute()
    if override:
        override_path = Path(override).expanduser()
        try:
            override_path = override_path.resolve()
        except OSError:
            override_path = override_path.absolute()
        if override_path == resolved:
            return CONFIG_SOURCE_ENV
    return CONFIG_SOURCE_CANONICAL


def resolve_rover_id(
    tag: str | None = None,
    env: dict | None = None,
    rover_id_file: Path | str = ROVER_ID_FILE,
) -> str:
    """Best-effort rover identity: ``SPF_ROVER_ID``, then the tag, then the Pi file.

    The collector is never told the rover id directly -- drone_run.sh passes it
    only as the ``RO<n>`` tag -- so all three sources are consulted rather than
    recording nothing.
    """
    env = os.environ if env is None else env
    from_env = str(env.get("SPF_ROVER_ID", "")).strip()
    if from_env:
        return from_env
    if tag:
        stripped = tag.strip()
        # Tags look like RO4, BOOT_DIRECT_V7_RO3, ...
        for chunk in reversed(stripped.replace("-", "_").split("_")):
            if chunk.upper().startswith("RO") and chunk[2:].isdigit():
                return chunk[2:]
    try:
        text = Path(rover_id_file).read_text().strip()
    except OSError:
        return "unknown"
    return text if text else "unknown"


def _timestamps(run_started_at: float | None) -> list[tuple[str, str]]:
    when = time.time() if run_started_at is None else float(run_started_at)
    local = datetime.fromtimestamp(when).astimezone()
    utc = datetime.fromtimestamp(when, tz=timezone.utc)
    return [
        ("timestamp_unix", f"{when:.3f}"),
        ("timestamp_local", local.isoformat()),
        ("timestamp_local_timezone", str(local.tzname())),
        ("timestamp_local_utc_offset", str(local.strftime("%z"))),
        ("timestamp_utc", utc.isoformat().replace("+00:00", "Z")),
        # The filename stamp is LOCAL time, so record which clock named the store.
        ("filename_timestamp_local", local.strftime("%Y_%m_%d_%H_%M_%S")),
        ("filename_timestamp_is_local_time", "true"),
        ("host_timezone_name", str(env_timezone())),
    ]


def env_timezone() -> str:
    """The host's configured zone name, if the OS exposes one."""
    tz = os.environ.get("TZ")
    if tz:
        return tz
    for candidate in (Path("/etc/timezone"),):
        try:
            text = candidate.read_text().strip()
        except OSError:
            continue
        if text:
            return text
    link = Path("/etc/localtime")
    try:
        target = os.readlink(link)
    except OSError:
        return "unknown"
    parts = Path(target).parts
    if "zoneinfo" in parts:
        return "/".join(parts[parts.index("zoneinfo") + 1:])
    return "unknown"


def capture_provenance_lines(
    *,
    log_path: str | Path,
    config_path: str | Path | None = None,
    tag: str | None = None,
    argv: list[str] | None = None,
    run_started_at: float | None = None,
    data_filename: str | Path | None = None,
    env: dict | None = None,
    repo_root: Path | str = REPO_ROOT,
    rover_id_file: Path | str = ROVER_ID_FILE,
    extra: dict | None = None,
) -> list[str]:
    """Build the ``key: value`` provenance block for one capture."""
    env = os.environ if env is None else env
    argv = list(sys.argv if argv is None else argv)

    # A relative --yaml-config is only meaningful next to the cwd it was run
    # from, which is exactly the ambiguity this sidecar exists to remove.
    absolute_config = (
        str(Path(config_path).expanduser().absolute()) if config_path else "unknown"
    )

    fields: list[tuple[str, str]] = [
        ("argv", shlex.join(argv)),
        ("executable", sys.executable),
        ("cwd", os.getcwd()),
        ("capture_config_path", absolute_config),
        ("capture_config_source", resolve_config_source(config_path, env=env)),
        (
            "capture_config_env_override",
            env.get(CONFIG_SOURCE_ENV, "") or "(unset)",
        ),
        ("rover_id", resolve_rover_id(tag, env=env, rover_id_file=rover_id_file)),
        ("tag", tag if tag else "(none)"),
        ("hostname", platform.node()),
        ("pid", str(os.getpid())),
        ("git_commit", git_commit(repo_root)),
        ("repo_root", str(repo_root)),
        ("log_sidecar", str(log_path)),
    ]
    if data_filename is not None:
        fields.append(("data_filename", str(data_filename)))
    fields.extend(_timestamps(run_started_at))
    for key, value in (extra or {}).items():
        fields.append((str(key), str(value)))

    lines = [CAPTURE_LOG_HEADER_START]
    lines.extend(f"# {key}: {value}" for key, value in fields)
    lines.append(CAPTURE_LOG_HEADER_END)
    return lines


def write_capture_provenance(log_path: str | Path, **kwargs) -> Path:
    """Write the provenance header to ``log_path`` (absolute) and return the path.

    Appends rather than truncates so this is safe to call after handlers exist,
    and so a caller can never blank a file it did not create.
    """
    absolute = Path(log_path).expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    lines = capture_provenance_lines(log_path=absolute, **kwargs)
    with absolute.open("a") as sidecar:
        sidecar.write("\n".join(lines) + "\n")
        sidecar.flush()
        os.fsync(sidecar.fileno())
    return absolute


def configure_capture_logging(
    log_path: str | Path,
    *,
    level: str | int = "INFO",
    stream: bool = True,
    fmt: str = "%(asctime)s:%(levelname)s:%(message)s",
    **provenance,
) -> Path:
    """Point root logging at the capture's own sidecar and stamp its header.

    ``force=True`` is load-bearing: importing ``spf.mavlink.mavlink_controller``
    installs a root handler at import time, and without ``force`` this call is a
    no-op and the sidecar stays empty. The path is made absolute for the same
    reason that import-time handler is a hazard -- a relative log destination
    follows the working directory and has already caused a PermissionError on a
    root-owned ``logs.log``.
    """
    absolute = write_capture_provenance(log_path, **provenance)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []
    if stream:
        handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(str(absolute)))
    logging.basicConfig(handlers=handlers, format=fmt, level=level, force=True)
    logging.info(
        "capture log sidecar: %s (provenance header above)", str(absolute)
    )
    return absolute
