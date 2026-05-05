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
from dataclasses import replace
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.low_descriptors import (  # noqa: E402
    generate_descriptor_set,
    generate_descriptor_set_shared_source,
    write_descriptor_set_to_paths,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_BUILDERS,
    build_amdgpu_core_descriptor_set,
)


def _parse_view_headers(values: Sequence[str]) -> dict[str, Path]:
    view_headers: dict[str, Path] = {}
    for value in values:
        target, separator, path = value.partition("=")
        if not separator or not target or not path:
            raise ValueError("AMDGPU descriptor --view-header must have form <target>=<path>")
        if target in view_headers:
            raise ValueError(f"duplicate AMDGPU descriptor view header for {target}")
        view_headers[target] = Path(path)
    return view_headers


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
    parser.add_argument(
        "--view-header",
        action="append",
        default=[],
        help="Generated descriptor view header as <target>=<path>.",
    )
    args = parser.parse_args(argv)

    view_headers = _parse_view_headers(args.view_header)
    descriptor_set = build_amdgpu_core_descriptor_set(args.target, args.xml)
    if args.target == "rdna4":
        unknown_view_headers = set(view_headers) - {"rdna4_gfx125x"}
        if unknown_view_headers:
            unknown_targets = ", ".join(sorted(unknown_view_headers))
            raise ValueError(f"RDNA4 descriptor generation cannot emit view headers for: {unknown_targets}")
        gfx125x_descriptor_set = build_amdgpu_core_descriptor_set("rdna4_gfx125x", args.xml)
        storage_descriptor_set = replace(
            descriptor_set,
            descriptors=gfx125x_descriptor_set.descriptors,
            resources=gfx125x_descriptor_set.resources,
            schedule_classes=gfx125x_descriptor_set.schedule_classes,
        )
        generated = generate_descriptor_set(descriptor_set, format_output=False)
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(generated.header, encoding="utf-8")
        args.source.write_text(
            generate_descriptor_set_shared_source(
                storage_descriptor_set,
                (descriptor_set, gfx125x_descriptor_set),
                format_output=False,
            ),
            encoding="utf-8",
        )
        gfx125x_header_path = view_headers.get("rdna4_gfx125x")
        if gfx125x_header_path is not None:
            gfx125x_generated = generate_descriptor_set(
                gfx125x_descriptor_set,
                format_output=False,
            )
            gfx125x_header_path.parent.mkdir(parents=True, exist_ok=True)
            gfx125x_header_path.write_text(
                gfx125x_generated.header,
                encoding="utf-8",
            )
        return 0

    if view_headers:
        unknown_targets = ", ".join(sorted(view_headers))
        raise ValueError(f"AMDGPU descriptor target {args.target} cannot emit view headers for: {unknown_targets}")

    if args.target == "rdna4_gfx125x":
        generated = generate_descriptor_set(descriptor_set, format_output=False)
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(generated.header, encoding="utf-8")
        args.source.write_text(
            "\n".join(
                [
                    "// Copyright 2026 The IREE Authors",
                    "//",
                    "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
                    "// See https://llvm.org/LICENSE.txt for license information.",
                    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
                    "",
                    f'#include "{descriptor_set.public_header}"',
                    "",
                    "// The gfx125x descriptor-set provider and backing storage are",
                    "// emitted with the shared RDNA4 descriptor source.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return 0

    write_descriptor_set_to_paths(
        descriptor_set,
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
