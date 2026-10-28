# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Acquisition observations, independent of transport logs and presentation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ProgressEvent:
    # Operation stage; byte counts describe this stage, not overall completion.
    phase: str
    # Authored destination path, never a transport URL or provider description.
    path: str | None = None
    # Bytes processed in this stage; absent for stages without a byte counter.
    completed_bytes: int | None = None
    # Stage byte extent, absent until known (and for Xet network transfers).
    total_bytes: int | None = None
    # Completed unique acquisitions or installed destinations for this phase.
    completed_files: int | None = None
    # Selected unique acquisitions or installed destinations for this phase.
    total_files: int | None = None
    # Diagnostic for a warning; contains no provider URLs or credentials.
    message: str | None = None


class Progress:
    """Serializes caller observations and binds per-file display identities."""

    def __init__(
        self,
        callback: Callable[[ProgressEvent], None] | None = None,
        *,
        path: str | None = None,
        lock=None,
    ):
        # Optional synchronous caller; exceptions fail the owning operation.
        self.callback = callback
        # Destination selected by the planner, separate from source coordinates.
        self.path = path
        # Shared across file scopes so callbacks never overlap across workers.
        self.lock = lock if lock is not None else threading.RLock()

    def file(self, path: str) -> Progress:
        return Progress(self.callback, path=path, lock=self.lock)

    def __bool__(self) -> bool:
        return self.callback is not None

    def __call__(self, event: ProgressEvent) -> None:
        if self.callback is not None:
            with self.lock:
                self.callback(replace(event, path=self.path) if self.path else event)

    def emit(self, phase: str, **counts) -> None:
        if self.callback is not None:
            self(ProgressEvent(phase, **counts))


class ByteProgress:
    """Bounds counter traffic while retaining initial and completed extents."""

    def __init__(self, callback, phase: str):
        # Synchronous observer, which may be the owned worker's result pipe.
        self.callback = callback
        # Meaning of the counter, such as download, verification, or source reads.
        self.phase = phase
        # Last emitted monotonic timestamp; initial progress is always observable.
        self.last_time = None
        # Last emitted extent suppresses duplicate completion notifications.
        self.last_count = None

    def __call__(self, completed: int, total: int | None) -> None:
        if not self.callback:
            return
        now = time.monotonic()
        counts = (completed, total)
        if counts == self.last_count:
            return
        if (
            self.last_time is None
            or (self.last_count[0] == 0 and completed > 0)
            or (total is not None and completed == total)
            or now - self.last_time >= 0.1
        ):
            self.callback(
                ProgressEvent(self.phase, completed_bytes=completed, total_bytes=total)
            )
            self.last_time = now
            self.last_count = counts
