# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> target-low descriptor C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.low_descriptors import write_descriptor_set_to_paths  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_BUILDERS,
    build_amdgpu_core_descriptor_set,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-low descriptor C tables from vendor ISA XML.")
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(AMDGPU_DESCRIPTOR_SET_BUILDERS),
        help="AMDGPU descriptor target shard to generate.",
    )
    parser.add_argument(
        "--xml",
        required=True,
        type=Path,
        help="Path to the AMD machine-readable ISA XML file for the target family.",
    )
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated descriptor header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated descriptor source path.",
    )
    args = parser.parse_args(argv)

    descriptor_set = build_amdgpu_core_descriptor_set(args.target, args.xml)
    write_descriptor_set_to_paths(
        descriptor_set,
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
