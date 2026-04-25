# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target planning dialect op definitions.

The target dialect owns durable module records that select and freeze code
generation facts before backend-specific lowering. ``target.profile`` is the
compact authoring form: it names a provider-owned preset and sparse overrides,
then symbol facts resolve it into dense target structs for compiler use.
"""

from loom.assembly import GLUE, LPAREN, RPAREN, Attr, AttrDict, SymbolRef, kw
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_STRING,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    SymbolReference,
)

# ============================================================================
# Dialect
# ============================================================================

target_ops = Dialect(
    "target",
    dialect_id=0x13,
    doc="Target planning records: profiles and artifacts.",
    default_phase=OpPhase.MODULE_METADATA,
)

# ============================================================================
# Shared enums
# ============================================================================

SnapshotCodegenFormat = EnumDef(
    "SnapshotCodegenFormat",
    [
        EnumCase("unknown", 0, doc="No codegen format selected."),
        EnumCase("llvmir", 1, doc="LLVM IR emission target."),
        EnumCase("spirv", 2, doc="SPIR-V emission target."),
        EnumCase("vm", 3, doc="IREE VM bytecode target."),
        EnumCase("low_native", 4, doc="Native low dialect code emission target."),
        EnumCase("wasm", 5, doc="WebAssembly module emission target."),
    ],
    doc="Primary codegen representation emitted for a target snapshot.",
)

_ARTIFACT_FORMAT_CASES = [
    EnumCase("unknown", 0, doc="No artifact format selected."),
    EnumCase("elf", 1, doc="ELF object or shared object artifact."),
    EnumCase("coff", 2, doc="COFF object artifact."),
    EnumCase("macho", 3, doc="Mach-O object artifact."),
    EnumCase("spirv_binary", 4, doc="SPIR-V binary artifact."),
    EnumCase("vm_bytecode", 5, doc="IREE VM bytecode artifact."),
    EnumCase("wasm_binary", 6, doc="WebAssembly binary module artifact."),
]

ArtifactFormatAttr = EnumDef(
    "ArtifactFormatAttr",
    _ARTIFACT_FORMAT_CASES,
    doc="Linkable or loadable artifact format produced for a snapshot.",
)

ArtifactRecordFormatAttr = EnumDef(
    "ArtifactRecordFormatAttr",
    _ARTIFACT_FORMAT_CASES,
    doc="Linkable or loadable artifact format produced for an artifact.",
)

ArtifactAbiKind = EnumDef(
    "ArtifactAbiKind",
    [
        EnumCase("unknown", 0, doc="No artifact packaging ABI selected."),
        EnumCase("object_file", 1, doc="Native object file packaging ABI."),
        EnumCase("hal_executable", 2, doc="IREE HAL executable packaging ABI."),
        EnumCase("vm_module", 3, doc="IREE VM module archive packaging ABI."),
        EnumCase("wasm_module", 4, doc="WebAssembly module packaging ABI."),
        EnumCase("spirv_module", 5, doc="SPIR-V module packaging ABI."),
    ],
    doc="Runtime or linker packaging ABI used by a target artifact.",
)

ExportAbiKind = EnumDef(
    "ExportAbiKind",
    [
        EnumCase("unknown", 0, doc="No ABI selected."),
        EnumCase("object_function", 1, doc="Native object function ABI."),
        EnumCase("hal_kernel", 2, doc="IREE HAL dispatch kernel ABI."),
        EnumCase("vm_module_function", 3, doc="IREE VM module function ABI."),
        EnumCase("shader_entry_point", 4, doc="Graphics shader entry point ABI."),
        EnumCase("wasm_function", 5, doc="WebAssembly module function ABI."),
    ],
    doc="Callable or package ABI used by an export plan.",
)

ExportLinkage = EnumDef(
    "ExportLinkage",
    [
        EnumCase("default", 0, doc="Default target linkage."),
        EnumCase("dso_local", 1, doc="DSO-local object linkage."),
    ],
    doc="ABI-required linkage for exported object functions or entry points.",
)

# ============================================================================
# target.profile
# ============================================================================

target_profile = Op(
    "target.profile",
    group=target_ops,
    doc=(
        "Compact reusable target environment profile. Providers own preset "
        "tables; the optional override dictionary is resolved once into dense "
        "symbol facts so target queries do not walk attr dictionaries."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target profile",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_profile_symbol_fact_domain",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("preset", ATTR_TYPE_STRING),
        AttrDef("overrides", "dict", optional=True),
    ],
    verify="loom_target_profile_verify",
    format=[
        SymbolRef("symbol"),
        kw("preset"),
        GLUE,
        LPAREN,
        GLUE,
        Attr("preset"),
        GLUE,
        RPAREN,
        AttrDict("overrides"),
    ],
    examples=[
        'target.profile @vm preset("iree-vm")',
        'target.profile @gfx1100 preset("amdgpu.gfx1100") {target_cpu = "gfx1100"}',
    ],
)

# ============================================================================
# target.artifact
# ============================================================================

target_artifact = Op(
    "target.artifact",
    group=target_ops,
    doc=("Packaging or compile-unit record. Entry functions are derived from function export facts that reference this artifact; the artifact itself never lists functions."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target artifact",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_artifact_symbol_fact_domain",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "target",
            "symbol",
            symbol_ref=SymbolReference("target profile", ["record"]),
        ),
        AttrDef(
            "artifact_format",
            ATTR_TYPE_ENUM,
            enum_def=ArtifactRecordFormatAttr,
            open_enum=True,
            optional=True,
        ),
        AttrDef(
            "abi",
            ATTR_TYPE_ENUM,
            enum_def=ArtifactAbiKind,
            open_enum=True,
            optional=True,
        ),
    ],
    verify="loom_target_artifact_verify",
    format=[
        SymbolRef("symbol"),
        kw("target"),
        GLUE,
        LPAREN,
        GLUE,
        Attr("target"),
        GLUE,
        RPAREN,
        AttrDict(),
    ],
    examples=[
        "target.artifact @gfx11_kernels target(@gfx11) {artifact_format = elf, abi = hal_executable}",
        "target.artifact @wasm_module target(@wasm)",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_TARGET_OPS: tuple[Op, ...] = (
    target_artifact,
    target_profile,
)
