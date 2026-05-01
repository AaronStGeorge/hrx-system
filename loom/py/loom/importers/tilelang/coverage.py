# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR importer coverage manifest."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class CoverageState(Enum):
    """Importer coverage state for one TileLang/TIR construct."""

    SUPPORTED = "supported"
    NORMALIZED = "normalized"
    IGNORED_METADATA = "ignored_metadata"
    TARGET_ONLY = "target_only"
    OPAQUE_FOREIGN = "opaque_foreign"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class OpFamily(Enum):
    """Namespace family for one TileLang/TIR construct."""

    TIR_NODE = "tir_node"
    TIR_OP = "tir_op"
    TILELANG_OP = "tilelang_op"
    TILELANG_TILEOP = "tilelang_tileop"
    ATTRIBUTE = "attribute"


@dataclass(frozen=True, slots=True)
class OpCoverage:
    """Coverage row for one TileLang/TIR construct."""

    name: str
    family: OpFamily
    state: CoverageState
    note: str


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    """Difference between an external TileLang/TVM inventory and the manifest."""

    known: tuple[str, ...]
    missing: tuple[str, ...]


TILELANG_OP_COVERAGE: tuple[OpCoverage, ...] = (
    OpCoverage(
        "tir.PrimFunc",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Kernel/function container; imports to a Loom kernel definition.",
    ),
    OpCoverage(
        "tir.Block",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Structured block wrapper; region contents are imported directly.",
    ),
    OpCoverage(
        "tir.For",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Loop structure; serial and thread-bound forms map to scf/kernel ops.",
    ),
    OpCoverage(
        "tir.IfThenElse",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Structured control flow; imports to scf.if once values are mapped.",
    ),
    OpCoverage(
        "tir.SeqStmt",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Statement list; preserves source order.",
    ),
    OpCoverage(
        "tir.LetStmt",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar binding; imports as the producing expression plus SSA name.",
    ),
    OpCoverage(
        "tir.Evaluate",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Effect statement wrapper for calls and annotations.",
    ),
    OpCoverage(
        "tir.BufferLoad",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Buffer read; maps to view/vector loads depending on result type.",
    ),
    OpCoverage(
        "tir.BufferStore",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Buffer write; maps to view/vector stores depending on value type.",
    ),
    OpCoverage(
        "tir.Allocate",
        OpFamily.TIR_NODE,
        CoverageState.DEFERRED,
        "Local memory allocation requires storage-space and lifetime mapping.",
    ),
    OpCoverage(
        "tir.AttrStmt",
        OpFamily.TIR_NODE,
        CoverageState.NORMALIZED,
        "Thread extent and storage attributes are decoded into kernel facts.",
    ),
    OpCoverage(
        "tir.Var",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "SSA, loop, buffer, and symbolic dimension references.",
    ),
    OpCoverage(
        "tir.IntImm",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Integer constants.",
    ),
    OpCoverage(
        "tir.FloatImm",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Floating-point constants.",
    ),
    OpCoverage(
        "tir.Cast",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar cast.",
    ),
    OpCoverage(
        "tir.Add",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar arithmetic.",
    ),
    OpCoverage(
        "tir.Sub",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar arithmetic.",
    ),
    OpCoverage(
        "tir.Mul",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar arithmetic.",
    ),
    OpCoverage(
        "tir.Div",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar arithmetic; exact signedness semantics come from dtype.",
    ),
    OpCoverage(
        "tir.FloorDiv",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Index/integer floor division.",
    ),
    OpCoverage(
        "tir.FloorMod",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Index/integer floor remainder.",
    ),
    OpCoverage(
        "tir.Min",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar min expression.",
    ),
    OpCoverage(
        "tir.Max",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar max expression.",
    ),
    OpCoverage(
        "tir.EQ",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.NE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.LT",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.LE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.GT",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.GE",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar comparison.",
    ),
    OpCoverage(
        "tir.And",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar boolean conjunction.",
    ),
    OpCoverage(
        "tir.Or",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar boolean disjunction.",
    ),
    OpCoverage(
        "tir.Not",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar boolean negation.",
    ),
    OpCoverage(
        "tir.Select",
        OpFamily.TIR_NODE,
        CoverageState.SUPPORTED,
        "Scalar select expression.",
    ),
    OpCoverage(
        "tir.Call",
        OpFamily.TIR_NODE,
        CoverageState.DEFERRED,
        "Dispatches by intrinsic/operator name; unknown calls fail loud.",
    ),
    OpCoverage(
        "tir.call_extern",
        OpFamily.TIR_OP,
        CoverageState.OPAQUE_FOREIGN,
        "Opaque external call; requires an explicit foreign-op policy.",
    ),
    OpCoverage(
        "tir.call_packed",
        OpFamily.TIR_OP,
        CoverageState.REJECTED,
        "Host/runtime call is not a kernel-body operation.",
    ),
    OpCoverage(
        "tir.tvm_thread_allreduce",
        OpFamily.TIR_OP,
        CoverageState.DEFERRED,
        "Needs a Loom subgroup/workgroup reduction representation.",
    ),
    OpCoverage(
        "tir.tvm_access_ptr",
        OpFamily.TIR_OP,
        CoverageState.NORMALIZED,
        "TVM pointer view helper; recover buffer/view facts instead.",
    ),
    OpCoverage(
        "thread_extent",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Launch topology fact; imports to kernel workgroup/subgroup metadata.",
    ),
    OpCoverage(
        "tir.kernel_launch_params",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "Late launch ABI metadata; Loom derives launch metadata structurally.",
    ),
    OpCoverage(
        "global_symbol",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "Debug/import naming hint only; HAL symbol names are not semantic.",
    ),
    OpCoverage(
        "calling_conv",
        OpFamily.ATTRIBUTE,
        CoverageState.IGNORED_METADATA,
        "TVM lowering artifact; Loom kernels use HAL ABI rules.",
    ),
    OpCoverage(
        "tl.readonly_param_indices",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Access fact; should become a target-independent buffer access encoding.",
    ),
    OpCoverage(
        "tl.non_restrict_params",
        OpFamily.ATTRIBUTE,
        CoverageState.NORMALIZED,
        "Aliasing fact; should become a target-independent buffer access encoding.",
    ),
    OpCoverage(
        "dyn_shared_memory_buf",
        OpFamily.ATTRIBUTE,
        CoverageState.TARGET_ONLY,
        "Late dynamic shared-memory ABI hook; source allocation facts are preferred.",
    ),
    OpCoverage(
        "tl.copy",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Bulk copy; needs memory space, vectorization, and async policy mapping.",
    ),
    OpCoverage(
        "tl.gemm",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Matrix multiply tile op; maps through vector.mma and fragment encodings.",
    ),
    OpCoverage(
        "tl.alloc_shared",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Shared-memory allocation; depends on storage-space and layout encodings.",
    ),
    OpCoverage(
        "tl.alloc_fragment",
        OpFamily.TILELANG_OP,
        CoverageState.DEFERRED,
        "Fragment allocation; depends on vector.fragment encoding import.",
    ),
    OpCoverage(
        "tl.tileop.*",
        OpFamily.TILELANG_TILEOP,
        CoverageState.DEFERRED,
        "Tile-level op family; each op must map to tile/vector dialect semantics.",
    ),
)


def coverage_by_name() -> Mapping[str, OpCoverage]:
    coverage: dict[str, OpCoverage] = {}
    for row in TILELANG_OP_COVERAGE:
        if row.name in coverage:
            raise ValueError(f"duplicate TileLang coverage row {row.name!r}")
        coverage[row.name] = row
    return coverage


def audit_coverage(names: Iterable[str]) -> CoverageAudit:
    known_rows = coverage_by_name()
    known: list[str] = []
    missing: list[str] = []
    for name in sorted(set(names)):
        if name in known_rows or _matches_wildcard(name, known_rows):
            known.append(name)
        else:
            missing.append(name)
    return CoverageAudit(
        known=tuple(known),
        missing=tuple(missing),
    )


def _matches_wildcard(name: str, rows: Mapping[str, OpCoverage]) -> bool:
    for row_name in rows:
        if not row_name.endswith(".*"):
            continue
        if name.startswith(row_name[:-1]):
            return True
    return False
