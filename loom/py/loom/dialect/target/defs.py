# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target planning dialect op definitions.

The target dialect owns durable module records that select and freeze code
generation facts before backend-specific lowering. ``target.profile`` is the
compact authoring form: it names a provider-owned preset and sparse overrides,
then symbol facts resolve it into dense target structs for compiler use. The
older explicit snapshot/export/config/bundle records remain while the lowering
pipeline migrates onto function and artifact facts.
"""

from loom.assembly import GLUE, LPAREN, RPAREN, Attr, AttrDict, SymbolRef, kw
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_STRING,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    SymbolDefinition,
    SymbolReference,
)

# ============================================================================
# Dialect
# ============================================================================

target_ops = Dialect(
    "target",
    dialect_id=0x13,
    doc="Target planning records: profiles, snapshots, exports, configs, and bundles.",
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

ArtifactFormatAttr = EnumDef(
    "ArtifactFormatAttr",
    [
        EnumCase("unknown", 0, doc="No artifact format selected."),
        EnumCase("elf", 1, doc="ELF object or shared object artifact."),
        EnumCase("coff", 2, doc="COFF object artifact."),
        EnumCase("macho", 3, doc="Mach-O object artifact."),
        EnumCase("spirv_binary", 4, doc="SPIR-V binary artifact."),
        EnumCase("vm_bytecode", 5, doc="IREE VM bytecode artifact."),
        EnumCase("wasm_binary", 6, doc="WebAssembly binary module artifact."),
    ],
    doc="Linkable or loadable artifact format produced for a snapshot.",
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
# target.snapshot
# ============================================================================

target_snapshot = Op(
    "target.snapshot",
    group=target_ops,
    doc=("Frozen codegen target facts consumed by legality, lowering, and emission. The snapshot is target-neutral: target-family descriptor tables and feature catalogs remain opt-in packages."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target snapshot",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "codegen_format",
            ATTR_TYPE_ENUM,
            enum_def=SnapshotCodegenFormat,
            open_enum=True,
        ),
        AttrDef("target_triple", ATTR_TYPE_STRING),
        AttrDef("data_layout", ATTR_TYPE_STRING),
        AttrDef(
            "artifact_format",
            ATTR_TYPE_ENUM,
            enum_def=ArtifactFormatAttr,
            open_enum=True,
        ),
        AttrDef("target_cpu", ATTR_TYPE_STRING),
        AttrDef("target_features", ATTR_TYPE_STRING),
        AttrDef("default_pointer_bitwidth", ATTR_TYPE_I64),
        AttrDef("index_bitwidth", ATTR_TYPE_I64),
        AttrDef("offset_bitwidth", ATTR_TYPE_I64),
        AttrDef("memory_space_generic", ATTR_TYPE_I64),
        AttrDef("memory_space_global", ATTR_TYPE_I64),
        AttrDef("memory_space_workgroup", ATTR_TYPE_I64),
        AttrDef("memory_space_constant", ATTR_TYPE_I64),
        AttrDef("memory_space_private", ATTR_TYPE_I64),
        AttrDef("memory_space_host", ATTR_TYPE_I64),
        AttrDef("memory_space_descriptor", ATTR_TYPE_I64),
    ],
    verify="loom_target_snapshot_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'target.snapshot @x86_64 {codegen_format = llvmir, target_triple = "x86_64-pc-linux-gnu", data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128", artifact_format = elf, target_cpu = "x86-64-v3", target_features = "+avx2,+fma", default_pointer_bitwidth = 64, index_bitwidth = 64, offset_bitwidth = 64, memory_space_generic = 0, memory_space_global = 0, memory_space_workgroup = 4294967295, memory_space_constant = 0, memory_space_private = 0, memory_space_host = 0, memory_space_descriptor = 4294967295}',
    ],
)

# ============================================================================
# target.export
# ============================================================================

target_export = Op(
    "target.export",
    group=target_ops,
    doc=(
        "Callable/package boundary plan. It names the source function when one "
        "is already materialized and records the ABI facts needed by launch or "
        "link planning without hiding them in backend flags."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target export plan",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "source",
            "symbol",
            optional=True,
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("export_symbol", ATTR_TYPE_STRING),
        AttrDef("abi", ATTR_TYPE_ENUM, enum_def=ExportAbiKind, open_enum=True),
        AttrDef(
            "linkage",
            ATTR_TYPE_ENUM,
            enum_def=ExportLinkage,
            open_enum=True,
        ),
        AttrDef("hal_binding_alignment", ATTR_TYPE_I64),
        AttrDef("hal_workgroup_size_x", ATTR_TYPE_I64),
        AttrDef("hal_workgroup_size_y", ATTR_TYPE_I64),
        AttrDef("hal_workgroup_size_z", ATTR_TYPE_I64),
        AttrDef("hal_flat_workgroup_size_min", ATTR_TYPE_I64),
        AttrDef("hal_flat_workgroup_size_max", ATTR_TYPE_I64),
        AttrDef("hal_buffer_resource_flags", ATTR_TYPE_I64),
    ],
    verify="loom_target_export_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'target.export @matmul_cpu {source = @matmul, export_symbol = "matmul", abi = object_function, linkage = dso_local, hal_binding_alignment = 0, hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}',
        'target.export @matmul_hal {source = @matmul, export_symbol = "matmul", abi = hal_kernel, linkage = default, hal_binding_alignment = 16, hal_workgroup_size_x = 16, hal_workgroup_size_y = 4, hal_workgroup_size_z = 1, hal_flat_workgroup_size_min = 64, hal_flat_workgroup_size_max = 64, hal_buffer_resource_flags = 0}',
    ],
)

# ============================================================================
# target.config
# ============================================================================

target_config = Op(
    "target.config",
    group=target_ops,
    doc=("Legalization and schedule selection record. Target-family packages interpret the contract key and feature bits; core Loom treats them as cache-key and diagnostic inputs."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target config",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("contract_set_key", ATTR_TYPE_STRING),
        AttrDef("contract_feature_bits", ATTR_TYPE_I64),
    ],
    verify="loom_target_config_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'target.config @generic {contract_set_key = "default", contract_feature_bits = 0}',
    ],
)

# ============================================================================
# target.preset
# ============================================================================

target_preset = Op(
    "target.preset",
    group=target_ops,
    doc=(
        "Compact production authoring record for command-line-style target "
        "selection. Presets are expanded early into concrete target records; "
        "late target consumers should see target.bundle, not target.preset."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target preset",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("key", ATTR_TYPE_STRING),
        AttrDef(
            "source",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
    ],
    verify="loom_target_preset_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'target.preset @vm_target {key = "iree-vm", source = @sched}',
    ],
)

# ============================================================================
# target.bundle
# ============================================================================

target_bundle = Op(
    "target.bundle",
    group=target_ops,
    doc=("Named frozen target profile binding one snapshot, export plan, and config. Backends consume bundles so hidden target state does not leak into scalar/vector lowering callbacks."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target bundle",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "snapshot",
            "symbol",
            symbol_ref=SymbolReference("record", ["record"]),
        ),
        AttrDef(
            "export_plan",
            "symbol",
            symbol_ref=SymbolReference("record", ["record"]),
        ),
        AttrDef(
            "config",
            "symbol",
            symbol_ref=SymbolReference("record", ["record"]),
        ),
    ],
    verify="loom_target_bundle_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "target.bundle @matmul_x86 {snapshot = @x86_64, export_plan = @matmul_cpu, config = @generic}",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_TARGET_OPS: tuple[Op, ...] = (
    target_snapshot,
    target_export,
    target_config,
    target_bundle,
    target_preset,
    target_profile,
)
