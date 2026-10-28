# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from loom.model.fetch_filesystem import (
    SharingUnavailable,
    copy_file,
    publish_file,
    share_file,
)
from loom.model.fetch_source import FetchError


def test_publish_needs_no_links_and_never_replaces_existing_content(tmp_path):
    source, destination = tmp_path / "staged", tmp_path / "published"
    source.write_bytes(b"complete payload")
    with patch("os.link", side_effect=NotImplementedError("no hardlinks")):
        publish_file(source, destination)
        assert destination.read_bytes() == b"complete payload"
        source.write_bytes(b"conflicting bytes")
        with pytest.raises(FileExistsError):
            publish_file(source, destination)
    assert destination.read_bytes() == b"complete payload"
    assert source.read_bytes() == b"conflicting bytes"


def test_publish_preserves_a_dangling_destination_link(tmp_path):
    source, destination = tmp_path / "staged", tmp_path / "published"
    source.write_bytes(b"content")
    destination.symlink_to("absent")
    with pytest.raises(FileExistsError):
        publish_file(source, destination)
    assert destination.readlink() == Path("absent")
    assert source.read_bytes() == b"content"


def test_concurrent_publication_has_one_complete_winner(tmp_path):
    ready = threading.Barrier(2)
    destination = tmp_path / "published"

    def publish(index):
        source = tmp_path / str(index)
        content = bytes([index]) * 1024
        source.write_bytes(content)
        ready.wait()
        try:
            publish_file(source, destination)
        except FileExistsError:
            return False, content
        return True, content

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, [1, 2]))
    winners = [content for won, content in results if won]
    assert winners == [destination.read_bytes()]


def test_sharing_resolves_snapshot_without_optional_link_arguments(tmp_path):
    source = tmp_path / "blob"
    source.write_bytes(b"weights")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights").symlink_to(Path("..") / "blob")
    output = tmp_path / "output"
    output.mkdir()
    link = os.link

    # Python 3.14 on Windows rejects explicit follow_symlinks=True. The real
    # primitive must receive the resolved file and no optional keywords.
    def link_resolved(source, destination):
        assert not source.is_symlink()
        link(source, destination)

    with patch("os.link", side_effect=link_resolved):
        assert share_file(snapshot / "weights", output / "weights") == "hardlink"
    assert (output / "weights").read_bytes() == b"weights"
    assert (output / "weights").samefile(source)
    assert not (output / "weights").is_symlink()


@pytest.mark.parametrize(
    "error", [OSError(errno.EXDEV, "cross-volume link"), NotImplementedError("link")]
)
def test_sharing_falls_back_to_a_readable_absolute_symlink(tmp_path, error):
    source, destination = tmp_path / "source", tmp_path / "shared"
    source.write_bytes(b"weights")
    with patch("os.link", side_effect=error):
        assert share_file(source, destination) == "symlink"
    target = destination.readlink()
    assert target.is_absolute() and target.samefile(source)
    assert destination.read_bytes() == b"weights"


def test_unavailable_links_report_both_reasons_without_a_copy(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "shared"
    source.write_bytes(b"weights")
    with (
        patch("os.link", side_effect=OSError(errno.EXDEV, "cross-volume link")),
        patch("os.symlink", side_effect=PermissionError(errno.EACCES, "policy")),
        pytest.raises(
            SharingUnavailable, match=r"hardlink:.*cross-volume.*symlink:.*policy"
        ),
    ):
        share_file(source, destination)
    assert not destination.exists()
    assert source.read_bytes() == b"weights"


def test_created_but_unreadable_symlink_is_removed(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "shared"
    source.write_bytes(b"weights")
    open_file = Path.open

    def open_with_link_policy(path, *args, **kwargs):
        if path == destination:
            assert path.is_symlink()
            raise PermissionError(errno.EACCES, "symlink evaluation disabled")
        return open_file(path, *args, **kwargs)

    with (
        patch("os.link", side_effect=OSError(errno.EXDEV, "cross-volume link")),
        patch.object(Path, "open", open_with_link_policy),
        pytest.raises(SharingUnavailable, match="symlink evaluation disabled"),
    ):
        share_file(source, destination)
    assert not destination.is_symlink()
    assert source.read_bytes() == b"weights"


@pytest.mark.parametrize("primitive", ["link", "symlink"])
@pytest.mark.parametrize(
    "code", [errno.EEXIST, errno.ENOENT, errno.ENOSPC, errno.EIO, errno.EROFS]
)
def test_storage_errors_are_not_sharing_capabilities(tmp_path, primitive, code):
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    error = OSError(code, "storage failure")
    with (
        patch(
            "os.link",
            side_effect=error
            if primitive == "link"
            else OSError(errno.EXDEV, "cross-volume"),
        ),
        patch("os.symlink", side_effect=error) as symlink,
        pytest.raises(OSError, match="storage failure") as caught,
    ):
        share_file(source, tmp_path / "output")
    assert caught.value is error
    if primitive == "link":
        symlink.assert_not_called()


def test_symlink_fallback_never_replaces_an_existing_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "shared"
    source.write_bytes(b"weights")
    destination.write_bytes(b"keep me")
    with (
        patch("os.link", side_effect=OSError(errno.EXDEV, "cross-volume link")),
        pytest.raises(FileExistsError),
    ):
        share_file(source, destination)
    assert destination.read_bytes() == b"keep me"


@pytest.mark.parametrize("winerror", [1, 50, 1314, 1463])
def test_windows_link_capability_codes_are_recognized(tmp_path, winerror):
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    error = OSError(errno.EINVAL, "Windows link capability")
    error.winerror = winerror
    with (
        patch("os.link", side_effect=error),
        patch("os.symlink", side_effect=error),
        pytest.raises(SharingUnavailable, match="Windows link capability"),
    ):
        share_file(source, tmp_path / "shared")


def test_copy_stages_beside_destination_and_reports_its_full_extent(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    mounted = tmp_path / "output/mounted"
    mounted.mkdir(parents=True)
    destination = mounted / "weights"
    events = []

    def publish_same_directory(staged, target):
        assert staged.parent == target.parent
        publish_file(staged, target)

    with patch("loom.model.fetch_filesystem.publish_file", publish_same_directory):
        copy_file(
            source,
            destination,
            source.stat().st_size,
            cancelled=threading.Event(),
            progress=events.append,
        )
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.samefile(source)
    assert list(mounted.iterdir()) == [destination]
    assert events[0].completed_bytes == 0
    assert events[-1].completed_bytes == source.stat().st_size
    assert all(event.total_bytes == source.stat().st_size for event in events)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EFBIG, errno.EROFS, errno.EIO])
def test_copy_flush_failure_keeps_destination_absent_and_removes_staging(
    tmp_path, code
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"payload")
    with (
        patch("os.fsync", side_effect=OSError(code, "storage failure")),
        pytest.raises(OSError, match="storage failure"),
    ):
        copy_file(source, destination, 7, cancelled=threading.Event(), progress=None)
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("cancel_after", [0, 1024 * 1024])
def test_copy_cancellation_never_publishes_partial_content(tmp_path, cancel_after):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    cancelled = threading.Event()

    def observe(event):
        if event.completed_bytes == cancel_after:
            cancelled.set()

    with pytest.raises(FetchError, match="cancelled"):
        copy_file(
            source,
            destination,
            source.stat().st_size,
            cancelled=cancelled,
            progress=observe,
        )
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("size", [2, 5])
def test_copy_rejects_source_size_changes(tmp_path, size):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"code")
    with pytest.raises(FetchError, match="source changed size"):
        copy_file(source, destination, size, cancelled=threading.Event(), progress=None)
    assert list(tmp_path.iterdir()) == [source]
