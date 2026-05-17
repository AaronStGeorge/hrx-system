# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V target-low descriptors -> compact packet emission rows."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[2]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.spirv.descriptors import (  # noqa: E402
    SPIRV_LOGICAL_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.spirv.scalar_memory import (  # noqa: E402
    STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
)
from loom.target.low_descriptors import descriptor_set_relative_name  # noqa: E402


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "empty"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.lower()


def _descriptor_ref_constant_name(descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, descriptor_key)
    return f"{SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.c_enum_prefix}_DESCRIPTOR_REF_{_c_identifier(descriptor_name).upper()}"


def _value_type(
    value_class: str,
    scalar_enum: str = "LOOM_SPIRV_SCALAR_TYPE_UNKNOWN",
) -> str:
    return f"{{.value_class = {value_class}, .scalar_type = {scalar_enum}}}"


def _scalar_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", scalar.scalar_enum)


def _storage_buffer_address_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS")


def _physical_storage_buffer_pointer_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER", scalar.scalar_enum)


def _offset64_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_OFFSET64", "LOOM_SPIRV_SCALAR_TYPE_U64")


def _unknown_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_UNKNOWN")


def _row(
    descriptor_key: str,
    *,
    opcode: str,
    form: str,
    result_type: str | None = None,
    operand_types: tuple[str, ...] = (),
    result_count: int = 0,
    immediate_index: int | None = None,
    literal_word_count: int = 0,
    memory_alignment: int = 0,
) -> str:
    lines = [
        f"    [{_descriptor_ref_constant_name(descriptor_key)}] =",
        "        {",
        f"            .opcode = {opcode},",
        f"            .form = {form},",
        f"            .result_type = {result_type or _unknown_value()},",
    ]
    if operand_types:
        lines.append("            .operand_types =")
        lines.append("                {")
        lines.extend(f"                    {operand_type}," for operand_type in operand_types)
        lines.append("                },")
    lines.extend(
        [
            f"            .result_count = {result_count},",
            f"            .operand_count = {len(operand_types)},",
            "            .immediate_index = " + ("LOOM_SPIRV_PACKET_IMMEDIATE_NONE" if immediate_index is None else str(immediate_index)) + ",",
        ]
    )
    if literal_word_count:
        lines.append(f"            .literal_word_count = {literal_word_count},")
    if memory_alignment:
        lines.append(f"            .memory_alignment = {memory_alignment},")
    lines.extend(
        [
            "        },",
        ]
    )
    return "\n".join(lines)


def _storage_buffer_rows() -> list[str]:
    rows: list[str] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rows.append(
            _row(
                f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset",
                opcode="LOOM_SPIRV_OP_PTR_ACCESS_CHAIN",
                form="LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN",
                result_type=_physical_storage_buffer_pointer_value(scalar),
                operand_types=(
                    _storage_buffer_address_value(),
                    _offset64_value(),
                ),
                result_count=1,
            )
        )
        rows.append(
            _row(
                f"spirv.op_load.storage_buffer.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_LOAD",
                form="LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED",
                result_type=_scalar_value(scalar),
                operand_types=(_physical_storage_buffer_pointer_value(scalar),),
                result_count=1,
                memory_alignment=scalar.byte_width,
            )
        )
        rows.append(
            _row(
                f"spirv.op_store.storage_buffer.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_STORE",
                form="LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED",
                operand_types=(
                    _physical_storage_buffer_pointer_value(scalar),
                    _scalar_value(scalar),
                ),
                memory_alignment=scalar.byte_width,
            )
        )
    return rows


def _binary_same_type_rows() -> list[str]:
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    offset64_value = _offset64_value()
    rows = [
        _row(
            "spirv.op_iadd.i32",
            opcode="LOOM_SPIRV_OP_I_ADD",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=i32_value,
            operand_types=(i32_value, i32_value),
            result_count=1,
        ),
        _row(
            "spirv.op_isub.i32",
            opcode="LOOM_SPIRV_OP_I_SUB",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=i32_value,
            operand_types=(i32_value, i32_value),
            result_count=1,
        ),
        _row(
            "spirv.op_imul.i32",
            opcode="LOOM_SPIRV_OP_I_MUL",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=i32_value,
            operand_types=(i32_value, i32_value),
            result_count=1,
        ),
        _row(
            "spirv.op_iadd.offset64",
            opcode="LOOM_SPIRV_OP_I_ADD",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
        _row(
            "spirv.op_isub.offset64",
            opcode="LOOM_SPIRV_OP_I_SUB",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
    ]
    return rows


def _mul_add_rows() -> list[str]:
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    return [
        _row(
            "spirv.op_imul_add.i32",
            opcode="LOOM_SPIRV_OP_I_MUL",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD",
            result_type=i32_value,
            operand_types=(i32_value, i32_value, i32_value),
            result_count=1,
        ),
    ]


def _validate_rows() -> None:
    descriptor_keys = {descriptor.key for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors}
    row_keys = {
        "spirv.op_constant.i32",
        "spirv.op_constant.offset64",
        "spirv.op_iadd.i32",
        "spirv.op_isub.i32",
        "spirv.op_imul.i32",
        "spirv.op_imul_add.i32",
        "spirv.op_iadd.offset64",
        "spirv.op_isub.offset64",
    }
    for scalar in STORAGE_BUFFER_SCALARS:
        row_keys.add(f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset")
        row_keys.add(f"spirv.op_load.storage_buffer.{scalar.suffix}")
        row_keys.add(f"spirv.op_store.storage_buffer.{scalar.suffix}")
    missing = sorted(row_keys - descriptor_keys)
    if missing:
        raise ValueError("SPIR-V packet rows reference missing descriptors: " + ", ".join(missing))
    suffixes = [scalar.suffix for scalar in STORAGE_BUFFER_SCALARS]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("SPIR-V storage-buffer scalar suffixes must be unique")


def generate_tables() -> str:
    _validate_rows()
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    rows = [
        _row(
            "spirv.op_constant.i32",
            opcode="LOOM_SPIRV_OP_CONSTANT",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT",
            result_type=i32_value,
            result_count=1,
            immediate_index=0,
            literal_word_count=1,
        ),
        _row(
            "spirv.op_constant.offset64",
            opcode="LOOM_SPIRV_OP_CONSTANT",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT",
            result_type=_offset64_value(),
            result_count=1,
            immediate_index=0,
            literal_word_count=2,
        ),
        *_binary_same_type_rows(),
        *_mul_add_rows(),
        *_storage_buffer_rows(),
    ]
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.spirv_packet_rows"),
        "",
        "static const loom_spirv_packet_row_t kSpirvLogicalCorePacketRows[] = {",
        *rows,
        "};",
        "",
    ]
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    tables = generate_tables()
    if args.check:
        return 0
    if args.tables is None:
        sys.stdout.write(tables)
    else:
        args.tables.write_text(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
