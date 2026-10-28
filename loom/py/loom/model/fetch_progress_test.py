# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from unittest.mock import patch

import pytest

from loom.model.fetch_console import ConsoleProgress
from loom.model.fetch_progress import ByteProgress, ProgressEvent


@pytest.mark.parametrize("quiet", [False, True])
def test_warnings_are_unthrottled_sanitized_and_visible_when_quiet(quiet):
    output = io.StringIO()
    progress = ConsoleProgress(output, quiet=quiet)
    with patch("loom.model.fetch_console.time.monotonic", return_value=0):
        progress(
            ProgressEvent("warning", path="one\n.data", message="copy\x1b instead")
        )
        progress(ProgressEvent("warning", path="two.data", message="copy instead"))
    assert output.getvalue().splitlines() == [
        "loom-fetch  Warning   one\\u000a.data: copy\\u001b instead",
        "loom-fetch  Warning   two.data: copy instead",
    ]


def test_byte_observations_are_bounded_but_completion_is_immediate():
    events = []
    counter = ByteProgress(events.append, "download")
    with patch("loom.model.fetch_progress.time.monotonic", return_value=0):
        for count in range(1001):
            counter(count, 1000)
        counter(1000, 1000)
    assert [event.completed_bytes for event in events] == [0, 1, 1000]


def test_unknown_total_and_periodic_observations_preserve_actual_counts():
    events = []
    counter = ByteProgress(events.append, "transfer")
    with patch("loom.model.fetch_progress.time.monotonic", return_value=0):
        counter(0, None)
        counter(1024, None)
        counter(2048, None)
    with patch("loom.model.fetch_progress.time.monotonic", return_value=1):
        counter(4096, None)
    assert [event.completed_bytes for event in events] == [0, 1024, 4096]
    assert all(event.total_bytes is None for event in events)


def test_console_throttles_lines_without_hiding_new_phases_or_final_state():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    with patch("loom.model.fetch_console.time.monotonic", return_value=10):
        progress(ProgressEvent("acquire", completed_files=0, total_files=2))
        progress(ProgressEvent("download", path="weight\x1b\n.data", completed_bytes=0))
        for count in range(1, 1001):
            progress(ProgressEvent("download", completed_bytes=count, total_bytes=1000))
        progress(ProgressEvent("verify", completed_bytes=1000, total_bytes=1000))
        progress(ProgressEvent("acquire", completed_files=1, total_files=2))
    with patch("loom.model.fetch_console.time.monotonic", return_value=40):
        progress(ProgressEvent("download", path="second.data", completed_bytes=7))
        progress(ProgressEvent("complete", completed_files=2, total_files=2))
    lines = output.getvalue().splitlines()
    assert len(lines) == 5
    assert "weight\\u001b\\u000a.data" in lines[1]
    assert "1/2 files" in lines[3]
    assert lines[4] == "loom-fetch  Complete  2/2 files"
    assert "\x1b" not in output.getvalue()


def test_console_renders_known_empty_and_unknown_extents():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    with patch("loom.model.fetch_console.time.monotonic", return_value=0):
        progress(ProgressEvent("acquire", completed_files=1, total_files=3))
        progress(
            ProgressEvent(
                "download",
                path="weights.data",
                completed_bytes=1024**3,
                total_bytes=4 * 1024**3,
            )
        )
    with patch("loom.model.fetch_console.time.monotonic", return_value=30):
        progress(
            ProgressEvent(
                "transfer",
                path="weights.data",
                completed_bytes=384 * 1024**2,
            )
        )
        progress(
            ProgressEvent(
                "verify",
                path="empty.data",
                completed_bytes=0,
                total_bytes=0,
            )
        )
    assert output.getvalue().splitlines() == [
        "loom-fetch  Fetch     1/3 files",
        "loom-fetch  Download  [###.........]  25%  1.0 GiB / 4.0 GiB"
        "  ETA estimating  weights.data  1/3 files",
        "loom-fetch  Download  [###.........]  25%  1.0 GiB / 4.0 GiB"
        "  ETA estimating  received 384.0 MiB  weights.data  1/3 files",
        "loom-fetch  Verify    [############] 100%  0 bytes / 0 bytes"
        "  empty.data  1/3 files",
    ]


def _observe(progress, now, phase, completed, total=None, path="weights.data"):
    with patch("loom.model.fetch_console.time.monotonic", return_value=now):
        progress(ProgressEvent(phase, path, completed, total))


def test_wire_heartbeat_retains_file_total_and_estimates_file_completion():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    gib = 1024**3
    _observe(progress, 0, "download", 0, 4 * gib)
    _observe(progress, 1, "transfer", 0)
    _observe(progress, 15, "download", gib, 4 * gib)
    _observe(progress, 30, "transfer", gib // 2)
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[-1] == (
        "loom-fetch  Download  [###.........]  25%  1.0 GiB / 4.0 GiB"
        "  34.1 MiB/s  ETA 1m 30s  received 512.0 MiB (17.7 MiB/s)"
        "  weights.data"
    )


def test_resume_restart_and_concurrent_files_have_independent_rate_baselines():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    mib = 1024**2
    _observe(progress, 0, "download", 5 * mib, 10 * mib)
    _observe(progress, 0, "download", 0, 100 * mib, "other.data")
    _observe(progress, 30, "download", 6 * mib, 10 * mib)
    assert "34.1 KiB/s  ETA 2m 00s" in output.getvalue().splitlines()[-1]
    _observe(progress, 40, "download", 50 * mib, 100 * mib, "other.data")
    _observe(progress, 60, "transfer", 10 * mib, path="other.data")
    line = output.getvalue().splitlines()[-1]
    assert "50.0 MiB / 100.0 MiB  853.3 KiB/s  ETA 1m 00s" in line
    assert line.endswith("other.data")
    _observe(progress, 70, "transfer", 5 * mib)
    _observe(progress, 90, "download", 2 * mib, 10 * mib)
    line = output.getvalue().splitlines()[-1]
    assert "ETA estimating" in line
    assert "/s" not in line and "received" not in line
    _observe(progress, 120, "download", 4 * mib, 10 * mib)
    assert "68.3 KiB/s  ETA 1m 30s" in output.getvalue().splitlines()[-1]


def test_unknown_size_reports_rate_without_inventing_total_or_eta():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    _observe(progress, 0, "download", 0)
    _observe(progress, 30, "download", 150)
    line = output.getvalue().splitlines()[-1]
    assert "150 bytes (total unknown)  5 bytes/s" in line
    assert "ETA" not in line


def test_late_file_extent_keeps_already_observed_network_bytes():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    _observe(progress, 0, "transfer", 100)
    _observe(progress, 1, "download", 0, 500)
    assert (
        "0 bytes / 500 bytes  ETA estimating  received 100 bytes"
        in (output.getvalue().splitlines()[-1])
    )
    _observe(progress, 31, "download", 500, 500)
    line = output.getvalue().splitlines()[-1]
    assert "100%  500 bytes / 500 bytes" in line
    assert "ETA" not in line


def test_long_eta_keeps_hours_and_seconds():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    _observe(progress, 0, "download", 0, 10000)
    _observe(progress, 30, "download", 30, 10000)
    assert "1 bytes/s  ETA 2h 46m 10s" in output.getvalue().splitlines()[-1]


def test_download_logs_are_bounded_under_continuous_interleaved_updates():
    output = io.StringIO()
    progress = ConsoleProgress(output)
    _observe(progress, 0, "download", 0, 10000)
    for second in range(1, 601):
        _observe(progress, second, "download", second, 10000)
        _observe(progress, second, "transfer", second // 2)
    assert len(output.getvalue().splitlines()) == 21
