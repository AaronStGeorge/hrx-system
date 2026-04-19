# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> compact target-low fact snapshots."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.target.arch.amdgpu.isa_snapshot import (
    AmdgpuIsaSnapshotAllowlist,
    AmdgpuIsaSnapshotError,
    build_amdgpu_isa_snapshot,
    format_amdgpu_isa_snapshot_json,
    format_amdgpu_isa_snapshot_report_json,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaXmlError,
    parse_amdgpu_isa_xml_path,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a compact AMDGPU ISA fact snapshot from vendor XML.")
    parser.add_argument("--xml", required=True, type=Path, help="Input ISA XML path.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--target-family-key",
        required=True,
        help="Stable Loom target family key, such as amdgpu.gfx11.",
    )
    parser.add_argument(
        "--snapshot-name",
        help="Stable snapshot name. Defaults to --target-family-key.",
    )
    parser.add_argument(
        "--instruction",
        action="append",
        required=True,
        help="Instruction or alias to retain. Repeat for multiple instructions.",
    )
    parser.add_argument(
        "--encoding",
        action="append",
        required=True,
        help="Encoding bit-layout name to retain. Repeat for multiple encodings.",
    )
    parser.add_argument(
        "--instruction-encoding",
        action="append",
        help=(
            "Instruction encoding selector INSTRUCTION/ENCODING/CONDITION to retain. Repeat for multiple encodings. When omitted, all selected encoding names on selected instructions are retained."
        ),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional JSON report path for selected and dropped source facts.",
    )
    args = parser.parse_args(argv)

    try:
        spec = parse_amdgpu_isa_xml_path(args.xml)
        result = build_amdgpu_isa_snapshot(
            spec,
            target_family_key=args.target_family_key,
            snapshot_name=args.snapshot_name,
            allowlist=AmdgpuIsaSnapshotAllowlist(
                instruction_names=tuple(args.instruction),
                encoding_names=tuple(args.encoding),
                instruction_encoding_names=tuple(args.instruction_encoding or ()),
            ),
        )
    except (AmdgpuIsaSnapshotError, AmdgpuIsaXmlError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.write_text(format_amdgpu_isa_snapshot_json(result.snapshot), encoding="utf-8")
    if args.manifest_out is not None:
        args.manifest_out.write_text(format_amdgpu_isa_snapshot_report_json(result), encoding="utf-8")
    print(f"Generated AMDGPU ISA snapshot {result.snapshot.snapshot_name}: {len(result.snapshot.instructions)} instructions, {len(result.snapshot.encodings)} encodings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
