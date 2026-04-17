# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the target planning record dialect declarations."""

from loom.assembly import AttrDict, SymbolRef
from loom.dialect.target import (
    ALL_TARGET_OPS,
    ArtifactFormatAttr,
    ExportAbiKind,
    ExportLinkage,
    SnapshotCodegenFormat,
    target_bundle,
    target_config,
    target_export,
    target_ops,
    target_snapshot,
)
from loom.dsl import ATTR_TYPE_ENUM, ATTR_TYPE_I64, ATTR_TYPE_STRING, SYMBOL_DEFINE


class TestTargetDialect:
    def test_dialect_id(self) -> None:
        assert target_ops.dialect_id == 0x13

    def test_inventory(self) -> None:
        assert [op.name for op in ALL_TARGET_OPS] == [
            "target.snapshot",
            "target.export",
            "target.config",
            "target.bundle",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert target_snapshot in ALL_TARGET_OPS
        assert target_export in ALL_TARGET_OPS
        assert target_config in ALL_TARGET_OPS
        assert target_bundle in ALL_TARGET_OPS

    def test_enums_match_runtime_values(self) -> None:
        assert [(case.keyword, case.value) for case in SnapshotCodegenFormat.cases] == [
            ("unknown", 0),
            ("llvmir", 1),
            ("spirv", 2),
            ("vm", 3),
            ("low_native", 4),
        ]
        assert [(case.keyword, case.value) for case in ArtifactFormatAttr.cases] == [
            ("unknown", 0),
            ("elf", 1),
            ("coff", 2),
            ("macho", 3),
            ("spirv_binary", 4),
            ("vm_bytecode", 5),
        ]
        assert [(case.keyword, case.value) for case in ExportAbiKind.cases] == [
            ("unknown", 0),
            ("object_function", 1),
            ("hal_kernel", 2),
            ("vm_module_function", 3),
            ("shader_entry_point", 4),
        ]
        assert [(case.keyword, case.value) for case in ExportLinkage.cases] == [
            ("default", 0),
            ("dso_local", 1),
        ]

    def test_records_define_generic_record_symbols(self) -> None:
        for op in ALL_TARGET_OPS:
            assert SYMBOL_DEFINE in op.traits
            assert op.symbol_def is not None
            assert op.symbol_def.field == "symbol"
            assert op.symbol_def.interfaces == ("record",)
            assert op.symbol_def.bytecode_kind == "LOOM_SYMBOL_RECORD"
            assert isinstance(op.format[0], SymbolRef)
            assert isinstance(op.format[1], AttrDict)

    def test_snapshot_shape(self) -> None:
        op = target_snapshot
        attrs = {attr.name: attr for attr in op.attrs}
        assert attrs["codegen_format"].attr_type == ATTR_TYPE_ENUM
        assert attrs["codegen_format"].enum_def is SnapshotCodegenFormat
        assert attrs["artifact_format"].attr_type == ATTR_TYPE_ENUM
        assert attrs["artifact_format"].enum_def is ArtifactFormatAttr
        assert attrs["target_triple"].attr_type == ATTR_TYPE_STRING
        assert attrs["target_cpu"].attr_type == ATTR_TYPE_STRING
        assert attrs["default_pointer_bitwidth"].attr_type == ATTR_TYPE_I64
        assert attrs["memory_space_descriptor"].attr_type == ATTR_TYPE_I64
        assert op.verify == "loom_target_snapshot_verify"

    def test_export_shape(self) -> None:
        op = target_export
        attrs = {attr.name: attr for attr in op.attrs}
        assert attrs["source"].optional
        source_ref = attrs["source"].symbol_ref
        assert source_ref is not None
        assert source_ref.interfaces == ("func_like",)
        assert attrs["abi"].enum_def is ExportAbiKind
        assert attrs["linkage"].enum_def is ExportLinkage
        assert attrs["hal_binding_alignment"].attr_type == ATTR_TYPE_I64
        assert op.verify == "loom_target_export_verify"

    def test_bundle_refs_are_record_symbols(self) -> None:
        op = target_bundle
        attrs = {attr.name: attr for attr in op.attrs}
        for name in ("snapshot", "export_plan", "config"):
            symbol_ref = attrs[name].symbol_ref
            assert symbol_ref is not None
            assert symbol_ref.interfaces == ("record",)
        assert op.verify == "loom_target_bundle_verify"
