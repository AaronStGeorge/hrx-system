# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from unittest.mock import patch

from loom.model.fetch_console import ConsoleProgress
from loom.model.fetch_progress import ByteProgress, ProgressEvent


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
    with patch("loom.model.fetch_console.time.monotonic", return_value=15):
        progress(ProgressEvent("download", path="second.data", completed_bytes=7))
        progress(ProgressEvent("complete", completed_files=2, total_files=2))
    lines = output.getvalue().splitlines()
    assert len(lines) == 5
    assert "weight\\u001b\\u000a.data" in lines[1]
    assert "1/2 files complete" in lines[3]
    assert "Complete (2/2 files complete)" in lines[4]
    assert "\x1b" not in output.getvalue()
