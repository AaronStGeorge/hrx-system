# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V cooperative property facts -> compact C tables."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.spirv.cooperative_matrix import (  # noqa: E402
    COOPERATIVE_MATRIX_CASES,
    CooperativeMatrixCase,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET  # noqa: E402
from loom.target.low_descriptors import descriptor_set_relative_name  # noqa: E402


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "empty"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.lower()


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _descriptor_ref_constant_name(descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(
        SPIRV_LOGICAL_CORE_DESCRIPTOR_SET,
        descriptor_key,
    )
    descriptor_suffix = _c_identifier(descriptor_name).upper()
    return f"{SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.c_enum_prefix}_DESCRIPTOR_REF_{descriptor_suffix}"


def _mul_add_descriptor_key(case: CooperativeMatrixCase) -> str:
    return case.descriptor_key(
        "op_cooperative_matrix_mul_add_khr",
        include_operand_mode=True,
    )


def _matrix_property_row(case: CooperativeMatrixCase) -> list[str]:
    return [
        "    {",
        f'        .name = IREE_SVL("{_c_string_literal(case.property_name)}"),',
        f"        .required_feature_bits = {case.feature_bits_c_expression},",
        f"        .m_size = {case.m_size},",
        f"        .n_size = {case.n_size},",
        f"        .k_size = {case.k_size},",
        f"        .lhs_type = {case.lhs_scalar.scalar_enum},",
        f"        .rhs_type = {case.rhs_scalar.scalar_enum},",
        f"        .accumulator_type = {case.accumulator_scalar.scalar_enum},",
        f"        .result_type = {case.result_scalar.scalar_enum},",
        "        .scope = LOOM_SPIRV_SCOPE_SUBGROUP,",
        "        .layout_flags = MATRIX_LAYOUT_ANY,",
        f"        .storage_class_flags = {case.property_storage_flags},",
        f"        .operand_flags = {case.property_operand_flags},",
        (f"        .mul_add_descriptor_ref = {_descriptor_ref_constant_name(_mul_add_descriptor_key(case))},"),
        "    },",
    ]


def _shape_spans(
    cases: Sequence[CooperativeMatrixCase],
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    previous_shape_key: int | None = None
    for index, case in enumerate(cases):
        if case.shape_key != previous_shape_key:
            spans.append((case.shape_key, index, 1))
            previous_shape_key = case.shape_key
        else:
            shape_key, start, count = spans[-1]
            spans[-1] = (shape_key, start, count + 1)
    return spans


def _shape_span_row(shape_key: int, start: int, count: int) -> str:
    return f"    {{.shape_key = UINT64_C(0x{shape_key:012x}), .start = {start}, .count = {count}}},"


def _validate_matrix_cases(cases: Sequence[CooperativeMatrixCase]) -> None:
    descriptor_keys = {descriptor.key for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors}
    names: set[str] = set()
    previous_shape_key = 0
    for index, case in enumerate(cases):
        if case.property_name in names:
            raise ValueError(f"duplicate SPIR-V cooperative matrix property {case.property_name!r}")
        names.add(case.property_name)
        if index != 0 and case.shape_key < previous_shape_key:
            raise ValueError("SPIR-V cooperative matrix property rows must be sorted")
        descriptor_key = _mul_add_descriptor_key(case)
        if descriptor_key not in descriptor_keys:
            raise ValueError(f"SPIR-V cooperative matrix property references missing descriptor: {descriptor_key}")
        previous_shape_key = case.shape_key


def generate_tables() -> str:
    _validate_matrix_cases(COOPERATIVE_MATRIX_CASES)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.spirv_cooperative_properties"),
        "",
        ("static const loom_spirv_cooperative_matrix_property_t kCooperativeMatrixProperties[] = {"),
    ]
    for case in COOPERATIVE_MATRIX_CASES:
        lines.extend(_matrix_property_row(case))
    lines.extend(
        [
            "};",
            "",
            ("static const loom_spirv_cooperative_property_span_t kCooperativeMatrixShapeSpans[] = {"),
        ]
    )
    for shape_key, start, count in _shape_spans(COOPERATIVE_MATRIX_CASES):
        lines.append(_shape_span_row(shape_key, start, count))
    lines.extend(
        [
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args(argv)
    args.tables.write_text(generate_tables(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
