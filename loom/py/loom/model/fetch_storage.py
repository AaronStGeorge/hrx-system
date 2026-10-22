# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verified content storage and exclusive installation publication."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .fetch_source import Document, FetchError, Source, document_source, open_source
from .lock import FileIdentity, file_digest
from .manifest import ManifestError

# Document limits protect the public metadata boundary; payloads stream instead.
_DOCUMENT_LIMIT = 16 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024


def atomic_write(path: Path, content: bytes) -> None:
    """Replaces one caller-owned metadata file after its complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".loom-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def share_file(source: Path, destination: Path) -> str:
    """Creates an explicitly shared view; never silently allocates a full copy."""
    # Repository snapshots often contain relative symlinks. Some platforms link
    # the symlink inode even with follow_symlinks=True; sharing the resolved blob
    # preserves its bytes across a different destination directory layout.
    source = source.resolve(strict=True)
    try:
        os.link(source, destination, follow_symlinks=True)
        return "hardlink"
    except OSError as error:
        if error.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        ):
            raise
    try:
        os.symlink(source.absolute(), destination)
        return "symlink"
    except OSError as error:
        raise FetchError(
            "this filesystem cannot share the selected file; use an output on "
            "the cache/source volume or explicitly select --materialize=copy"
        ) from error


@contextmanager
def installation_lock(directory: Path):
    """Holds an OS lock through publication, released automatically on process exit."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".loom-fetch.lock"
    if path.is_symlink():
        raise FetchError("installation lock must not be a symlink")
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if not stream.seek(0, os.SEEK_END):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise FetchError(
                    "another fetch owns this installation directory"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FetchError(
                    "another fetch owns this installation directory"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class AcquiredFile:
    # Verified bytes, owned by the caller's local source or the shared cache.
    path: Path
    # Observed identity including its original logical source coordinate.
    identity: FileIdentity


class ContentCache:
    """Whole files are shared by verified content, not by model/configuration name."""

    def __init__(self, root: Path):
        self.root = root.absolute()

    def blob_path(self, digest: str) -> Path:
        return self.root / "blob" / "sha256" / digest

    def find(self, identity: FileIdentity) -> Path | None:
        path = self.blob_path(identity.sha256)
        if not path.exists():
            return None
        identity.verify(path)
        return path

    def read_document(self, location: str, *, offline: bool) -> Document:
        source = document_source(location)
        remote = source.transport["kind"] == "url"
        key = hashlib.sha256(location.encode("utf-8")).hexdigest()
        cached = self.root / "document" / key
        if remote and offline:
            if not cached.is_file():
                raise FetchError(
                    "--offline document is not cached; provide a local card/lock"
                )
            source = document_source(str(cached))
        with open_source(source, offline=offline) as (stream, size):
            if size is not None and size > _DOCUMENT_LIMIT:
                raise FetchError(
                    "model/lock document exceeds the 16 MiB metadata limit"
                )
            content = stream.read(_DOCUMENT_LIMIT + 1)
            if len(content) > _DOCUMENT_LIMIT:
                raise FetchError(
                    "model/lock document exceeds the 16 MiB metadata limit"
                )
            if size is not None and len(content) != size:
                raise FetchError("document transfer ended before its declared length")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ManifestError("model/lock documents must use UTF-8") from error
        if remote and not offline:
            atomic_write(cached, content)
        return Document(location, text)

    def acquire(
        self, source: Source, identity: FileIdentity | None, *, offline: bool
    ) -> AcquiredFile:
        if source.transport["kind"] == "local":
            path = Path(source.transport["path"])
            size, digest = file_digest(path)
            observed = FileIdentity(source.origin, size, digest, source.resolved)
            if identity is not None and (size, digest) != (
                identity.size,
                identity.sha256,
            ):
                raise ManifestError(f"integrity mismatch for local source {path}")
            return AcquiredFile(path, observed)
        if identity is not None:
            cached = self.find(identity)
            if cached is not None:
                return AcquiredFile(cached, identity)
        temporary_directory = self.root / "temporary"
        temporary_directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with open_source(
                source, offline=offline, size=identity.size if identity else None
            ) as (input_stream, transport_size):
                expected_size = identity.size if identity else transport_size
                digest = hashlib.sha256()
                size = 0
                with tempfile.NamedTemporaryFile(
                    dir=temporary_directory, delete=False
                ) as output:
                    temporary = Path(output.name)
                    while True:
                        limit = (
                            _CHUNK_SIZE
                            if expected_size is None
                            else min(_CHUNK_SIZE, expected_size - size + 1)
                        )
                        chunk = input_stream.read(limit)
                        if not chunk:
                            break
                        size += len(chunk)
                        if expected_size is not None and size > expected_size:
                            raise FetchError("source exceeded its declared byte count")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            if expected_size is not None and size != expected_size:
                raise FetchError("source transfer ended before its declared byte count")
            digest = digest.hexdigest()
            if identity is not None and digest != identity.sha256:
                raise ManifestError("source SHA-256 does not match its lock")
            observed = FileIdentity(source.origin, size, digest, source.resolved)
            path = self.blob_path(digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            # A concurrent publisher can create the same immutable content.
            # Replacing its directory entry does not rewrite an existing inode.
            os.replace(temporary, path)
            return AcquiredFile(path, observed)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
