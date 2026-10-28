# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exclusive file publication and capability-checked storage sharing."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
import tempfile
from functools import cache
from pathlib import Path

from .fetch_progress import ByteProgress
from .fetch_source import FetchError


@cache
def _exclusive_rename():
    """Resolves the POSIX extension once; Python exposes it directly on Windows."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        function = getattr(library, "renameat2", None)
        if function is not None:
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            # AT_FDCWD, RENAME_NOREPLACE.
            return lambda source, destination: function(
                -100, source, -100, destination, 1
            )
    elif sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is not None:
            function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            function.restype = ctypes.c_int
            # RENAME_EXCL.
            return lambda source, destination: function(source, destination, 4)
    return None


def publish_file(source: Path, destination: Path) -> None:
    """Publishes complete same-filesystem staging without replacing any entry.

    Success may consume source. The caller owns its staging directory and removes
    any remaining source. Neither branch exposes a partially copied destination.
    """
    if os.name == "nt":
        # Unlike POSIX rename, Windows rename fails when destination exists.
        os.rename(source, destination)
        return
    rename = _exclusive_rename()
    if rename is not None:
        if rename(os.fsencode(source), os.fsencode(destination)) == 0:
            return
        code = ctypes.get_errno()
        if code not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(code, os.strerror(code), str(source), None, str(destination))
    # Older kernels and filesystems may lack exclusive rename. A hardlink has
    # the same complete-file, no-replacement contract; ordinary rename does not.
    try:
        os.link(source, destination)
    except (NotImplementedError, OSError) as error:
        if isinstance(error, OSError) and error.errno not in (
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EPERM,
        ):
            raise
        raise FetchError(
            f"cannot publish {destination}: this filesystem supports neither "
            "exclusive rename nor hardlinks; choose another cache/output filesystem"
        ) from error


class SharingUnavailable(FetchError):
    """Both link strategies are unavailable for this source/destination pair."""


def _sharing_unavailable(error: BaseException) -> bool:
    if isinstance(error, NotImplementedError):
        return True
    return isinstance(error, OSError) and (
        error.errno
        in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EMLINK,
        )
        # INVALID_FUNCTION, NOT_SUPPORTED, PRIVILEGE_NOT_HELD,
        # SYMLINK_CLASS_DISABLED. Some map only to EINVAL in Python.
        or getattr(error, "winerror", None) in (1, 50, 1314, 1463)
    )


def share_file(source: Path, destination: Path) -> str:
    """Creates a readable shared view, or fails without copying or replacing."""
    # HF snapshots may use relative symlinks. Resolve their blob before changing
    # directory layout; os.link's optional follow_symlinks flag is not portable.
    try:
        source = source.resolve(strict=True)
    except RuntimeError as error:
        # Python before 3.13 reports symlink loops as RuntimeError, not ELOOP.
        raise FetchError(f"cannot resolve shared source {source}: {error}") from error
    failures = []
    for kind, create in (("hardlink", os.link), ("symlink", os.symlink)):
        try:
            create(source, destination)
        except (NotImplementedError, OSError) as error:
            if not _sharing_unavailable(error):
                raise
            failures.append(f"{kind}: {error}")
            continue
        try:
            # Link creation can succeed while opening through this destination
            # is denied, especially by Windows network/symlink evaluation policy.
            with destination.open("rb") as stream:
                stream.read(1)
            return kind
        except BaseException as error:
            destination.unlink()
            if not _sharing_unavailable(error):
                raise
            failures.append(f"{kind}: {error}")
    raise SharingUnavailable(
        f"cannot share {source} with {destination}: {'; '.join(failures)}. "
        "Use a filesystem/path that permits links or select --materialize=copy"
    )


def copy_file(
    source: Path, destination: Path, size: int, *, cancelled, progress
) -> None:
    """Copies in bounded chunks beside the destination, then publishes exclusively."""
    descriptor, name = tempfile.mkstemp(prefix=".loom-fetch-", dir=destination.parent)
    temporary = Path(name)
    counter = ByteProgress(progress, "copy")
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            count = 0
            counter(count, size)
            while True:
                if cancelled.is_set():
                    raise FetchError("model acquisition cancelled")
                chunk = input_stream.read(min(1024 * 1024, size - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise FetchError(f"source changed size while copying {source}")
                output.write(chunk)
                counter(count, size)
            if count != size:
                raise FetchError(f"source changed size while copying {source}")
            output.flush()
            os.fsync(output.fileno())
        if cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        publish_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
