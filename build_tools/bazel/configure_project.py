# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository-owned contributions to the generic Bazel configure workflow."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_contributor(relative_path: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load project configure contributor {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capability_presets = _load_contributor(
    "loom/py/loom/target/capability_presets.py",
    "loom_target_capability_presets",
)

LOOM_ARCHITECTURES = capability_presets.ARCHITECTURES
LOOM_ARTIFACT_FORMATS = capability_presets.ARTIFACT_FORMATS
LOOM_EXECUTION_SUBSTRATES = capability_presets.EXECUTION_SUBSTRATES

_CONTRIBUTORS = (capability_presets,)


def initialize_request(request: Any) -> None:
    for contributor in _CONTRIBUTORS:
        contributor.initialize_request(request)


def apply_define(
    request: Any,
    name: str,
    value: str,
    parse_bool: Callable[[str, str], bool],
) -> bool:
    handlers = []
    for contributor in _CONTRIBUTORS:
        if contributor.apply_define(request, name, value, parse_bool):
            handlers.append(contributor.__name__)
    if len(handlers) > 1:
        raise RuntimeError(
            f"project configure option {name} has multiple owners: "
            + ", ".join(handlers)
        )
    return bool(handlers)


def apply_native_bazel_arg(request: Any, arg: str) -> bool:
    handlers = []
    for contributor in _CONTRIBUTORS:
        if contributor.apply_native_bazel_arg(request, arg):
            handlers.append(contributor.__name__)
    if len(handlers) > 1:
        raise RuntimeError(
            f"Bazel configure argument {arg} has multiple owners: "
            + ", ".join(handlers)
        )
    return bool(handlers)


def finalize_request(request: Any) -> None:
    for contributor in _CONTRIBUTORS:
        contributor.finalize_request(request)


def request_summary(request: Any) -> tuple[str, ...]:
    return tuple(
        line
        for contributor in _CONTRIBUTORS
        for line in contributor.request_summary(request)
    )
