"""One rule, mechanically enforced, for every shell script under `pipefail`.

`rover sitl status` spent 105 commits reporting a bound SITL port as free. The
code read perfectly:

    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then

`grep -q` exits on the first match. `ss` was still writing, so it died of
SIGPIPE (141), and `set -o pipefail` turned "found" into a non-zero pipeline --
i.e. into "not found". Measured on the CI box: 13% of invocations. Always a
false NEGATIVE, always silent, and the same shape guarded two rover boot-path
decisions. See docs/learnings.md.

The rule this file enforces:

    In a script with `set -o pipefail`, no pipeline stage may be a consumer
    that can exit before its producer finishes (`grep -q`, `grep -m N`,
    `head`), unless the status is explicitly discarded.

Two ways to comply, both already used in these scripts:

    predicate   capture the producer, match with bash's own `[[ ]]` -- no pipe
                (data_collection/rover/rover_v3.1/rover, `cmd_sitl`)
    extraction  append `|| true`, because only stdout is wanted
                (data_collection/rover/rover_v3.1/rover:299)

An inline `# pipefail-safe: <reason>` opts a line out, in the spirit of the
`# shellcheck disable=` comments these scripts already carry.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PRAGMA = "# pipefail-safe:"
GUARD = re.compile(r"\|\|\s*(true|:)\s*[)\"']*\s*(#.*)?$")
# Split on a single `|`, never on `||`.
STAGE_SPLIT = re.compile(r"(?<!\|)\|(?!\|)")


def shell_scripts() -> list[Path]:
    """Every tracked file bash actually runs."""
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    scripts = []
    for name in tracked:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        if path.suffix and path.suffix != ".sh":
            continue
        try:
            head = path.open("rb").readline(200).decode("utf-8", "replace")
        except OSError:
            continue
        if path.suffix == ".sh" or re.match(r"^#!.*\b(ba)?sh\b", head):
            scripts.append(path)
    return scripts


def logical_lines(text: str):
    """Yield (line_number, joined_source). Pipelines here wrap on a trailing `|`.

    A line-at-a-time matcher misses more than half of the real sites, which is
    exactly how several of them survived review.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        start = index
        joined = lines[index]
        while (
            joined.rstrip().endswith(("|", "\\"))
            and not joined.rstrip().endswith("||")
            and index + 1 < len(lines)
        ):
            index += 1
            joined = joined.rstrip().rstrip("\\") + " " + lines[index].strip()
        yield start + 1, joined
        index += 1


def exits_early(stage: str) -> str | None:
    """Name the early-exit consumer in a pipeline stage, if it is one."""
    try:
        tokens = shlex.split(stage, comments=True)
    except ValueError:
        tokens = stage.split()
    if not tokens:
        return None
    command = Path(tokens[0]).name
    flags = [t for t in tokens[1:] if t.startswith("-")]
    if command == "head":
        # `head -c N` reads a byte budget but is still an early exit; only an
        # explicit "whole input" is safe, and there is no such flag. Flag all.
        return "head"
    if command == "grep":
        for flag in flags:
            if flag in ("--quiet", "--silent") or flag.startswith("--max-count"):
                return f"grep {flag}"
            if flag.startswith("-") and not flag.startswith("--"):
                if "q" in flag[1:] or "m" in flag[1:]:
                    return f"grep {flag}"
    return None


def violations_in(text: str) -> list[tuple[int, str, str]]:
    found = []
    for number, source in logical_lines(text):
        stripped = source.strip()
        if stripped.startswith("#") or not stripped:
            continue
        exemption_reason = source.partition(PRAGMA)[2].strip()
        if exemption_reason or GUARD.search(source):
            continue
        stages = STAGE_SPLIT.split(source)
        for stage in stages[1:]:
            consumer = exits_early(stage)
            if consumer:
                found.append((number, consumer, stripped))
                break
    return found


def test_no_unguarded_early_exit_pipelines_under_pipefail():
    offenders = []
    for script in shell_scripts():
        text = script.read_text(errors="replace")
        if "pipefail" not in text:
            continue
        for number, consumer, source in violations_in(text):
            offenders.append(
                f"{script.relative_to(REPO_ROOT)}:{number}: `{consumer}` can exit "
                f"before its producer, and pipefail turns that into failure\n"
                f"    {source}"
            )
    assert not offenders, (
        "Early-exit consumer ending a pipeline whose status is read, under "
        "`set -o pipefail`:\n\n"
        + "\n".join(offenders)
        + "\n\nCapture the producer and match with `[[ ]]` (predicate), or append "
        "`|| true` (extraction). If it is genuinely safe, say why with an inline "
        f"`{PRAGMA} <reason>`."
    )


# A linter that matches nothing passes forever. These pin the detector itself.


def test_scanner_catches_the_defect_that_motivated_it():
    """The exact line that shipped broken for 105 commits."""
    source = 'if ss -tlnp 2>/dev/null | grep -q ":${port} "; then\n'
    assert violations_in(source), "scanner blind to the original defect"


def test_scanner_catches_a_wrapped_pipeline():
    source = 'ser="$(udevadm info --name="$d" |\n    sed -n "s/^X=//p" | head -1)"\n'
    assert violations_in(source), "scanner blind to a line-wrapped pipeline"


def test_scanner_accepts_both_documented_idioms():
    assert not violations_in('fc="$(ls /dev/x* | head -1 || true)"\n')
    assert not violations_in(
        'listening="$(ss -tln || true)"\nif [[ "$listening" == *":80 "* ]]; then\n'
    )


def test_scanner_accepts_the_pragma_but_only_with_a_reason():
    assert not violations_in(f"cmd | head -1  {PRAGMA} producer emits one line\n")
    assert violations_in(f"cmd | head -1  {PRAGMA}\n")
    assert violations_in(f"cmd | head -1  {PRAGMA}   \n")
    assert violations_in("cmd | head -1  # unrelated comment\n")


def test_scanner_does_not_flag_a_logical_or():
    assert not violations_in("command -v grep >/dev/null || die 'no grep'\n")
