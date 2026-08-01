from spf.scripts.monitor_cgroup_resources import (
    parse_process_status,
    sample_cgroup_processes,
)


def test_parse_process_status_extracts_kib_fields():
    values = parse_process_status(
        "Name:\tpython\nVmSize:\t12345 kB\nVmRSS:\t456 kB\n"
        "RssAnon:\t321 kB\nRssFile:\t135 kB\n"
    )

    assert values == {
        "VmRSS": 456,
        "RssAnon": 321,
        "RssFile": 135,
        "VmSize": 12345,
    }


def test_sample_cgroup_sums_all_live_processes(tmp_path):
    cgroup = tmp_path / "cgroup"
    proc = tmp_path / "proc"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("11\n12\n13\n")
    for pid, rss, anon in ((11, 100, 70), (12, 200, 120)):
        process = proc / str(pid)
        process.mkdir(parents=True)
        (process / "status").write_text(
            f"VmSize:\t{rss * 3} kB\nVmRSS:\t{rss} kB\n"
            f"RssAnon:\t{anon} kB\nRssFile:\t{rss - anon} kB\n"
        )

    sample = sample_cgroup_processes(cgroup, proc_root=proc)

    assert sample == {
        "pid_count": 3,
        "VmRSS": 300,
        "RssAnon": 190,
        "RssFile": 110,
        "VmSize": 900,
    }
