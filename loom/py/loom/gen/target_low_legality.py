# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: target-low legality policy -> compact C lookup table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.dsl import Dialect, Op
from loom.gen.generated_file import line_comment_header
from loom.target.low_legality import (
    TargetLowLegality,
    target_low_legality_by_name,
)

COPYRIGHT = """// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

REGENERATE = "python3 loom/py/loom/gen/run.py c_tables"


@dataclass(frozen=True, slots=True)
class GeneratedTargetLowLegalityTable:
    header: str
    source: str


@dataclass(frozen=True, slots=True)
class _ResolvedOp:
    name: str
    dialect_name: str
    dialect_id: int
    op_index: int
    legality: TargetLowLegality


def _c_enum_name(name: str) -> str:
    return "LOOM_OP_" + name.replace(".", "_").upper()


def _collect_production_ops(
    dialects: Sequence[tuple[Dialect, Sequence[Op]]],
) -> dict[str, tuple[Dialect, int]]:
    result: dict[str, tuple[Dialect, int]] = {}
    for dialect, ops in dialects:
        for op_index, op in enumerate(ops):
            if op.name in result:
                raise ValueError(f"duplicate op declaration '{op.name}'")
            result[op.name] = (dialect, op_index)
    return result


def _resolve_ops(
    dialects: Sequence[tuple[Dialect, Sequence[Op]]],
) -> list[_ResolvedOp]:
    production_ops = _collect_production_ops(dialects)
    legality_by_name = target_low_legality_by_name()
    resolved: list[_ResolvedOp] = []
    for name, legality in legality_by_name.items():
        dialect_and_index = production_ops.get(name)
        if dialect_and_index is None:
            raise ValueError(f"unknown target-low legality op '{name}'")
        dialect, op_index = dialect_and_index
        resolved.append(
            _ResolvedOp(
                name=name,
                dialect_name=dialect.name,
                dialect_id=dialect.dialect_id,
                op_index=op_index,
                legality=legality,
            )
        )
    resolved.sort(key=lambda op: (op.dialect_id, op.op_index))
    return resolved


def _generate_header() -> str:
    lines: list[str] = [COPYRIGHT.rstrip(), ""]
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.target_low_legality",
            regenerate=REGENERATE,
        )
    )
    lines.extend(
        [
            "// clang-format off",
            "#ifndef LOOM_TARGET_LOW_LEGALITY_TABLE_H_",
            "#define LOOM_TARGET_LOW_LEGALITY_TABLE_H_",
            "",
            "#include <stdint.h>",
            "",
            '#include "iree/base/api.h"',
            '#include "loom/ir/ir.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "typedef uint8_t loom_target_low_legality_t;",
            "",
            "typedef enum loom_target_low_legality_e {",
            "  LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED = 0,",
            "  LOOM_TARGET_LOW_LEGALITY_CORE = 1,",
            "  LOOM_TARGET_LOW_LEGALITY_PROVIDER = 2,",
            "  LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY = 3,",
            "  LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA = 4,",
            "} loom_target_low_legality_e;",
            "",
            "// Returns the target-low legality class for |kind|.",
            "loom_target_low_legality_t loom_target_low_legality_class(",
            "    loom_op_kind_t kind);",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif  // LOOM_TARGET_LOW_LEGALITY_TABLE_H_",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_source(resolved_ops: Sequence[_ResolvedOp]) -> str:
    included_dialects = sorted({op.dialect_name for op in resolved_ops})
    lines: list[str] = [COPYRIGHT.rstrip(), ""]
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.target_low_legality",
            regenerate=REGENERATE,
        )
    )
    lines.extend(
        [
            "// clang-format off",
            '#include "loom/target/low_legality_table.h"',
            "",
        ]
    )
    lines.extend(f'#include "loom/ops/{dialect}/ops.h"' for dialect in included_dialects)
    lines.extend(
        [
            "",
            "typedef struct loom_target_low_legality_entry_t {",
            "  loom_op_kind_t kind;",
            "  loom_target_low_legality_t legality;",
            "} loom_target_low_legality_entry_t;",
            "",
            "static const loom_target_low_legality_entry_t",
            "    loom_target_low_legality_entries[] = {",
        ]
    )
    lines.extend(f"        {{{_c_enum_name(op.name)}, {op.legality.value}}}," for op in resolved_ops)
    lines.extend(
        [
            "};",
            "",
            "loom_target_low_legality_t loom_target_low_legality_class(",
            "    loom_op_kind_t kind) {",
            "  iree_host_size_t low = 0;",
            "  iree_host_size_t high = IREE_ARRAYSIZE(loom_target_low_legality_entries);",
            "  while (low < high) {",
            "    iree_host_size_t mid = low + (high - low) / 2;",
            "    const loom_target_low_legality_entry_t* entry =",
            "        &loom_target_low_legality_entries[mid];",
            "    if (kind == entry->kind) {",
            "      return entry->legality;",
            "    } else if (kind < entry->kind) {",
            "      high = mid;",
            "    } else {",
            "      low = mid + 1;",
            "    }",
            "  }",
            "  return LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_target_low_legality_table(
    dialects: Sequence[tuple[Dialect, Sequence[Op]]],
) -> GeneratedTargetLowLegalityTable:
    resolved_ops = _resolve_ops(dialects)
    return GeneratedTargetLowLegalityTable(
        header=_generate_header(),
        source=_generate_source(resolved_ops),
    )
