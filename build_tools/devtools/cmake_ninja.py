# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selected-root builds for CMake's Ninja generators."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class NinjaBuildError(ValueError):
    """The selected roots cannot be represented in a Ninja manifest."""


def _escape_path(path: str) -> str:
    if "\n" in path or "\r" in path:
        raise NinjaBuildError(f"Ninja build target contains a newline: {path!r}")
    return path.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


@contextmanager
def selected_build_arguments(
    build_dir: Path,
    build_targets: tuple[str, ...],
    *,
    build_config: str | None = None,
    multi_config: bool = False,
) -> Iterator[list[str]]:
    """Adds one aggregate over selected roots in an already-refreshed graph.

    A supplementary manifest includes CMake's generated graph and contributes
    only one temporary phony root. The ordinary CMake build driver still owns
    Ninja selection, configuration, parallelism, and error propagation.
    """
    if multi_config:
        if not build_config:
            raise NinjaBuildError(
                "Ninja Multi-Config selected builds require a build configuration"
            )
        included_manifest = f"build-{build_config}.ninja"
    else:
        included_manifest = "build.ninja"

    manifest = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="iree-ctest-selected-",
        suffix=".ninja",
        dir=build_dir,
        delete=False,
    )
    manifest_path = Path(manifest.name)
    aggregate_target = manifest_path.stem
    try:
        manifest.write(f"include {_escape_path(included_manifest)}\n")
        manifest.write(f"build {aggregate_target}: phony")
        for target in build_targets:
            manifest.write(f" $\n  {_escape_path(target)}")
        manifest.write("\n")
        manifest.close()
        yield ["--target", aggregate_target, "--", "-f", str(manifest_path)]
    finally:
        manifest.close()
        manifest_path.unlink(missing_ok=True)
