# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V feature catalog -> C feature API and rodata tables."""

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

from loom.gen.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.spirv.features import (  # noqa: E402
    FEATURE_ATOMS,
    FEATURE_PROFILES,
    feature_atom_enum,
    feature_bit_constant,
    feature_bit_constants,
    feature_bits_expression,
    feature_row_capacity,
    parse_isa_symbols,
    validate_feature_catalog,
)


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _c_string_view(value: str) -> str:
    return f'IREE_SVL("{_c_string_literal(value)}")'


def _c_identifier_suffix(key: str) -> str:
    return "".join(part.capitalize() for part in key.split("_"))


def _emit_row_array(
    lines: list[str],
    *,
    c_type: str,
    symbol_name: str,
    values: tuple[str, ...],
    string_values: bool = False,
) -> None:
    if not values:
        return
    lines.append(f"static const {c_type} {symbol_name}[] = {{")
    for value in values:
        row = _c_string_view(value) if string_values else value
        lines.append(f"    {row},")
    lines.append("};")
    lines.append("")


def _emit_descriptor_field(
    lines: list[str],
    *,
    field_name: str,
    count_name: str,
    array_symbol: str,
    values: tuple[str, ...],
) -> None:
    if not values:
        return
    lines.append(f"            .{field_name} = {array_symbol},")
    lines.append(f"            .{count_name} = IREE_ARRAYSIZE({array_symbol}),")


def _emit_enum_bitset(
    lines: list[str],
    *,
    name: str,
    parts: Sequence[str],
) -> None:
    if not parts:
        lines.append(f"  {name} = 0,")
        return
    if len(parts) == 1:
        lines.append(f"  {name} = {parts[0]},")
        return
    lines.append(f"  {name} =")
    for index, part in enumerate(parts):
        suffix = " |" if index + 1 < len(parts) else ","
        lines.append(f"      {part}{suffix}")


def generate_header() -> str:
    validate_feature_catalog(atoms=FEATURE_ATOMS, profiles=FEATURE_PROFILES)

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.spirv_features"),
        "",
        "// SPIR-V feature atoms and prepared feature-set views.",
        "//",
        "// Feature atoms are target-local units of SPIR-V extension composition.",
        "// They own the capability/extension state needed by binary emission and",
        "// the direct numeric bits used by target-local legality and lowering.",
        "// Core Loom code should see only generic target records and must not",
        "// depend on these tables.",
        "",
        "#ifndef LOOM_TARGET_ARCH_SPIRV_FEATURES_H_",
        "#define LOOM_TARGET_ARCH_SPIRV_FEATURES_H_",
        "",
        '#include "iree/base/api.h"',
        '#include "loom/target/arch/spirv/isa.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "enum {",
        "  // Feature atom does not impose an addressing model.",
        "  LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED = UINT32_MAX,",
        "  // Feature atom does not impose a memory model.",
        "  LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED = UINT32_MAX,",
        "};",
        "",
        "typedef enum loom_spirv_feature_atom_e {",
        "  // Unknown or uninitialized feature atom.",
        "  LOOM_SPIRV_FEATURE_ATOM_UNKNOWN = 0,",
    ]
    for index, atom in enumerate(FEATURE_ATOMS, start=1):
        lines.append(f"  // {atom.doc}")
        lines.append(f"  {feature_atom_enum(atom)} = {index},")
    lines.extend(
        [
            "  // Number of feature atom enum slots.",
            f"  LOOM_SPIRV_FEATURE_ATOM_COUNT = {len(FEATURE_ATOMS) + 1},",
            "} loom_spirv_feature_atom_t;",
            "",
            "// Bitset of loom_spirv_feature_atom_t values.",
            "typedef uint64_t loom_spirv_feature_bits_t;",
            "",
        ]
    )
    known_bits_parts: list[str] = []
    lines.extend(
        [
            "typedef enum loom_spirv_feature_bit_e {",
        ]
    )
    for atom in FEATURE_ATOMS:
        bit_constant = feature_bit_constant(atom)
        known_bits_parts.append(bit_constant)
        lines.append(f"  // Target enables {atom.doc.removesuffix('.')}.")
        lines.append(f"  {bit_constant} =")
        lines.append(f"      UINT64_C(1) << {feature_atom_enum(atom)},")
    lines.extend(
        [
            "",
            "  // Feature bits known by the generated SPIR-V target package.",
        ]
    )
    _emit_enum_bitset(lines, name="LOOM_SPIRV_FEATURE_KNOWN_BITS", parts=known_bits_parts)
    for profile in FEATURE_PROFILES:
        lines.append(f"  // {profile.doc}")
        _emit_enum_bitset(
            lines,
            name=f"LOOM_SPIRV_FEATURE_PROFILE_{profile.c_suffix}",
            parts=feature_bit_constants(profile.atoms),
        )
    lines.extend(
        [
            "} loom_spirv_feature_bit_t;",
            "",
            "// Maximum number of OpExtension rows emitted by all modeled atoms.",
            f"#define LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT {feature_row_capacity('extensions')}",
            "// Maximum number of OpCapability rows emitted by all modeled atoms.",
            f"#define LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT {feature_row_capacity('capabilities')}",
            "// Maximum number of opcode rows exposed by all modeled atoms.",
            f"#define LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT {feature_row_capacity('opcodes')}",
            "// Maximum number of storage-class rows exposed by all modeled atoms.",
            f"#define LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT {feature_row_capacity('storage_classes')}",
            "// Maximum number of decoration rows exposed by all modeled atoms.",
            f"#define LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT {feature_row_capacity('decorations')}",
            "",
            "typedef struct loom_spirv_feature_atom_descriptor_t {",
            "  // Stable feature atom selected by target profile bits.",
            "  loom_spirv_feature_atom_t atom;",
            "  // Stable feature atom name for diagnostics.",
            "  iree_string_view_t name;",
            "  // Atom bits that must also be selected.",
            "  loom_spirv_feature_bits_t required_atom_bits;",
            "  // Minimum SPIR-V binary version required by this atom.",
            "  uint32_t minimum_spirv_version;",
            "  // Addressing model after this atom is selected.",
            "  loom_spirv_addressing_model_t addressing_model;",
            "  // Memory model after this atom is selected.",
            "  loom_spirv_memory_model_t memory_model;",
            "  // OpExtension string rows owned by this atom.",
            "  const iree_string_view_t* extension_names;",
            "  // Number of entries in |extension_names|.",
            "  uint8_t extension_count;",
            "  // OpCapability numeric rows owned by this atom.",
            "  const uint32_t* capabilities;",
            "  // Number of entries in |capabilities|.",
            "  uint8_t capability_count;",
            "  // Opcode rows owned by this atom.",
            "  const uint32_t* opcodes;",
            "  // Number of entries in |opcodes|.",
            "  uint8_t opcode_count;",
            "  // Storage-class rows owned by this atom.",
            "  const uint32_t* storage_classes;",
            "  // Number of entries in |storage_classes|.",
            "  uint8_t storage_class_count;",
            "  // Decoration rows owned by this atom.",
            "  const uint32_t* decorations;",
            "  // Number of entries in |decorations|.",
            "  uint8_t decoration_count;",
            "} loom_spirv_feature_atom_descriptor_t;",
            "",
            "typedef struct loom_spirv_feature_set_t {",
            "  // Selected feature atom bits.",
            "  loom_spirv_feature_bits_t atom_bits;",
            "  // Minimum SPIR-V binary version required by selected atoms.",
            "  uint32_t minimum_spirv_version;",
            "  // Selected module addressing model.",
            "  loom_spirv_addressing_model_t addressing_model;",
            "  // Selected module memory model.",
            "  loom_spirv_memory_model_t memory_model;",
            "  // Deterministic OpExtension rows.",
            "  iree_string_view_t extension_names[LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT];",
            "  // Number of entries in |extension_names|.",
            "  uint8_t extension_count;",
            "  // Deterministic OpCapability rows.",
            "  uint32_t capabilities[LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT];",
            "  // Number of entries in |capabilities|.",
            "  uint8_t capability_count;",
            "  // Deterministic opcode rows.",
            "  uint32_t opcodes[LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT];",
            "  // Number of entries in |opcodes|.",
            "  uint8_t opcode_count;",
            "  // Deterministic storage-class rows.",
            "  uint32_t storage_classes[LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT];",
            "  // Number of entries in |storage_classes|.",
            "  uint8_t storage_class_count;",
            "  // Deterministic decoration rows.",
            "  uint32_t decorations[LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT];",
            "  // Number of entries in |decorations|.",
            "  uint8_t decoration_count;",
            "} loom_spirv_feature_set_t;",
            "",
            "// Returns the direct feature bit for |atom|, or zero for unknown atoms.",
            "loom_spirv_feature_bits_t loom_spirv_feature_atom_bit(",
            "    loom_spirv_feature_atom_t atom);",
            "",
            "// Returns the descriptor for |atom|, or NULL when |atom| is unknown.",
            "const loom_spirv_feature_atom_descriptor_t* loom_spirv_feature_atom_descriptor(",
            "    loom_spirv_feature_atom_t atom);",
            "",
            "// Returns the stable diagnostic spelling for |atom|.",
            "iree_string_view_t loom_spirv_feature_atom_name(",
            "    loom_spirv_feature_atom_t atom);",
            "",
            "// Returns the feature atoms currently modeled by this target package.",
            "loom_spirv_feature_bits_t loom_spirv_known_feature_bits(void);",
            "",
            "// Returns true when |feature_set| contains |atom|.",
            "bool loom_spirv_feature_set_has_atom(",
            "    const loom_spirv_feature_set_t* feature_set,",
            "    loom_spirv_feature_atom_t atom);",
            "",
            "// Prepares a deterministic feature-set view from selected target feature bits.",
            "iree_status_t loom_spirv_feature_set_prepare(",
            "    iree_string_view_t target_name,",
            "    loom_spirv_feature_bits_t requested_atom_bits,",
            "    loom_spirv_feature_set_t* out_feature_set);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            "#endif  // LOOM_TARGET_ARCH_SPIRV_FEATURES_H_",
            "",
        ]
    )
    return "\n".join(lines)


def generate_tables(isa_header: str | None = None) -> str:
    isa_symbols = parse_isa_symbols(isa_header) if isa_header is not None else None
    validate_feature_catalog(
        atoms=FEATURE_ATOMS,
        profiles=FEATURE_PROFILES,
        isa_symbols=isa_symbols,
    )
    lines = [
        *line_comment_header("//", generator="loom.gen.spirv_features"),
        "// clang-format off",
        "",
    ]
    for atom in FEATURE_ATOMS:
        suffix = _c_identifier_suffix(atom.key)
        _emit_row_array(
            lines,
            c_type="iree_string_view_t",
            symbol_name=f"kSpirv{suffix}Extensions",
            values=atom.extensions,
            string_values=True,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Capabilities",
            values=atom.capabilities,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Opcodes",
            values=atom.opcodes,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}StorageClasses",
            values=atom.storage_classes,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Decorations",
            values=atom.decorations,
        )

    lines.extend(
        [
            "static const loom_spirv_feature_atom_descriptor_t kSpirvFeatureAtoms[] = {",
            "    [LOOM_SPIRV_FEATURE_ATOM_UNKNOWN] = {0},",
        ]
    )
    for atom in FEATURE_ATOMS:
        suffix = _c_identifier_suffix(atom.key)
        lines.extend(
            [
                f"    [{feature_atom_enum(atom)}] =",
                "        {",
                f"            .atom = {feature_atom_enum(atom)},",
                f"            .name = {_c_string_view(atom.name)},",
                f"            .required_atom_bits = {feature_bits_expression(atom.required)},",
                f"            .minimum_spirv_version = UINT32_C(0x{atom.minimum_spirv_version:08x}),",
                f"            .addressing_model = {atom.addressing_model},",
                f"            .memory_model = {atom.memory_model},",
            ]
        )
        _emit_descriptor_field(
            lines,
            field_name="extension_names",
            count_name="extension_count",
            array_symbol=f"kSpirv{suffix}Extensions",
            values=atom.extensions,
        )
        _emit_descriptor_field(
            lines,
            field_name="capabilities",
            count_name="capability_count",
            array_symbol=f"kSpirv{suffix}Capabilities",
            values=atom.capabilities,
        )
        _emit_descriptor_field(
            lines,
            field_name="opcodes",
            count_name="opcode_count",
            array_symbol=f"kSpirv{suffix}Opcodes",
            values=atom.opcodes,
        )
        _emit_descriptor_field(
            lines,
            field_name="storage_classes",
            count_name="storage_class_count",
            array_symbol=f"kSpirv{suffix}StorageClasses",
            values=atom.storage_classes,
        )
        _emit_descriptor_field(
            lines,
            field_name="decorations",
            count_name="decoration_count",
            array_symbol=f"kSpirv{suffix}Decorations",
            values=atom.decorations,
        )
        lines.extend(["        },"])
    lines.extend(["};", "", "// clang-format on", ""])
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates SPIR-V feature API and table snippets.")
    parser.add_argument(
        "--header",
        type=Path,
        help="Path to write the generated public features.h header.",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        help="Path to write the generated private features_tables.inl include.",
    )
    parser.add_argument(
        "--isa-header",
        type=Path,
        help="Path to isa.h for validating referenced SPIR-V wire symbols.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation inputs without writing output files.",
    )
    args = parser.parse_args(argv)
    if args.check and (args.header is not None or args.tables is not None):
        parser.error("--check cannot be combined with output flags")
    if not args.check and args.header is None and args.tables is None:
        parser.error("at least one output flag is required unless --check is used")
    return args


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _read_optional_text(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8") if path is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    isa_header = _read_optional_text(args.isa_header)
    validate_feature_catalog(
        atoms=FEATURE_ATOMS,
        profiles=FEATURE_PROFILES,
        isa_symbols=parse_isa_symbols(isa_header) if isa_header is not None else None,
    )
    if args.check:
        return 0
    if args.header is not None:
        _write_text(args.header, generate_header())
    if args.tables is not None:
        _write_text(args.tables, generate_tables(isa_header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
