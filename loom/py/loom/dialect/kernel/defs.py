# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel dialect type and op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    FormatElement,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    Scope,
    SymbolRef,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dialect.target.defs import ExportLinkage
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    I1,
    INDEX,
    INTEGER,
    ISOLATED_FROM_ABOVE,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    VECTOR,
    VIEW,
    AttrDef,
    ContractFamily,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    Op,
    Operand,
    OpPhase,
    Reads,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolReference,
    TypeDef,
    TypeSemantic,
    Writes,
)

# ============================================================================
# Dialect
# ============================================================================

kernel_ops = Dialect(
    "kernel",
    dialect_id=0x10,
    doc="Kernel execution and synchronization operations.",
)

# ============================================================================
# Types
# ============================================================================

kernel_async_token_type = TypeDef(
    name="kernel.async.token",
    doc=("Opaque token for one initiated asynchronous memory transfer. A token must be committed to exactly one kernel.async.group."),
    semantic=TypeSemantic.CONTROL_TOKEN,
    contracts=[ContractFamily.KERNEL_ASYNC],
)

kernel_async_group_type = TypeDef(
    name="kernel.async.group",
    doc=("Opaque handle for one ordered asynchronous copy group. A group must be waited before leaving the kernel async-copy stream."),
    semantic=TypeSemantic.CONTROL_TOKEN,
    contracts=[ContractFamily.KERNEL_ASYNC],
)

kernel_tensor_lds_descriptor_type = TypeDef(
    name="kernel.tensor.lds.descriptor",
    semantic=TypeSemantic.TARGET_CONTRACT_VALUE,
    contracts=[ContractFamily.TENSOR_MEMORY],
    doc=(
        "Opaque descriptor grouping AMDGPU tensor-memory dgroups for one "
        "global/LDS tensor transfer. The descriptor contains the low-level "
        "dgroup values; the consuming async op still carries the source and "
        "destination views so layout, extent, and memory-space facts remain "
        "visible to Loom analyses."
    ),
)

# ============================================================================
# Shared attrs
# ============================================================================

KernelScope = EnumDef(
    "KernelScope",
    [
        EnumCase("thread", 0, doc="Current invocation or thread."),
        EnumCase("subgroup", 1, doc="Current SIMD subgroup or wave."),
        EnumCase("workgroup", 2, doc="Current workgroup or block."),
        EnumCase("device", 3, doc="Current device."),
        EnumCase("system", 4, doc="Whole system."),
    ],
    doc="Target-independent synchronization scope.",
)

KernelMemorySpace = EnumDef(
    "KernelMemorySpace",
    [
        EnumCase("unknown", 0, doc="No target-independent memory space is known."),
        EnumCase("global", 1, doc="Device-visible global storage."),
        EnumCase("workgroup", 2, doc="Workgroup/shared storage."),
        EnumCase("private", 3, doc="Invocation-private storage."),
        EnumCase("constant", 4, doc="Read-only constant storage."),
        EnumCase("host", 5, doc="Host-visible storage."),
        EnumCase("descriptor", 6, doc="Descriptor-backed storage identity."),
        EnumCase(
            "generic",
            7,
            doc="Target-generic device storage fenced by a synchronization op.",
        ),
    ],
    doc="Target-independent memory space fenced by a kernel synchronization op.",
)

KernelOrdering = EnumDef(
    "KernelOrdering",
    [
        EnumCase("relaxed", 0, doc="Atomicity without inter-address synchronization."),
        EnumCase("acquire", 1, doc="Acquire ordering."),
        EnumCase("release", 2, doc="Release ordering."),
        EnumCase("acq_rel", 3, doc="Acquire and release ordering."),
        EnumCase("seq_cst", 4, doc="Sequentially consistent ordering."),
    ],
    doc="Target-independent memory ordering for kernel synchronization ops.",
)

KernelDimension = EnumDef(
    "KernelDimension",
    [
        EnumCase("x", 0, doc="X dimension."),
        EnumCase("y", 1, doc="Y dimension."),
        EnumCase("z", 2, doc="Z dimension."),
    ],
    doc="Three-dimensional kernel coordinate axis.",
)

KernelAsyncDirection = EnumDef(
    "KernelAsyncDirection",
    [
        EnumCase(
            "global_to_workgroup",
            0,
            doc="Copy from global-like memory into workgroup/shared memory.",
        ),
        EnumCase(
            "workgroup_to_global",
            1,
            doc="Copy from workgroup/shared memory back to global-like memory.",
        ),
    ],
    doc="Required async copy direction.",
)

# ============================================================================
# Shared format fragments
# ============================================================================

_ENTRY_TARGET_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_ENTRY_EXPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("export"), GLUE, LPAREN, Attr("export_symbol"), GLUE, RPAREN],
        anchor="export_symbol",
    ),
    OptionalGroup(
        [kw("artifact"), GLUE, LPAREN, SymbolRef("artifact"), GLUE, RPAREN],
        anchor="artifact",
    ),
    OptionalGroup(
        [kw("ordinal"), GLUE, LPAREN, Attr("export_ordinal"), GLUE, RPAREN],
        anchor="export_ordinal",
    ),
    OptionalGroup(
        [kw("linkage"), GLUE, LPAREN, Attr("export_linkage"), GLUE, RPAREN],
        anchor="export_linkage",
    ),
]

_ENTRY_WORKGROUP_SIZE_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("workgroup_size"),
            GLUE,
            LPAREN,
            Attr("workgroup_size_x"),
            COMMA,
            Attr("workgroup_size_y"),
            COMMA,
            Attr("workgroup_size_z"),
            GLUE,
            RPAREN,
        ],
        anchor="workgroup_size_x",
    ),
]

_ENTRY_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_ENTRY_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target profile", ["record"]),
    ),
    AttrDef("export_symbol", "string", optional=True),
    AttrDef(
        "artifact",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target artifact", ["record"]),
    ),
    AttrDef("export_ordinal", ATTR_TYPE_I64, optional=True),
    AttrDef("export_linkage", "enum", enum_def=ExportLinkage, optional=True),
    AttrDef("workgroup_size_x", ATTR_TYPE_I64, optional=True),
    AttrDef("workgroup_size_y", ATTR_TYPE_I64, optional=True),
    AttrDef("workgroup_size_z", ATTR_TYPE_I64, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]


kernel_def = Op(
    "kernel.def",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Dispatchable source-level kernel entry. Kernel entries own launch and export contracts; ordinary func.def bodies remain helper/callable code."),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_ENTRY_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    regions=[RegionDef("body", doc="Kernel body.", terminator="kernel.return")],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            target="target",
            export_symbol="export_symbol",
            predicates="predicates",
            body="body",
        )
    ],
    verify="loom_kernel_def_verify",
    format=[
        *_ENTRY_TARGET_FORMAT,
        *_ENTRY_EXPORT_FORMAT,
        *_ENTRY_WORKGROUP_SIZE_FORMAT,
        *_ENTRY_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "kernel.def @entry(%buffer: buffer) {\n  kernel.return\n}",
        'kernel.def target(@gfx1100) export("matmul") artifact(@gfx_hsaco) workgroup_size(16, 4, 1) @matmul(%lhs: buffer, %rhs: buffer, %out: buffer) {\n  kernel.return\n}',
    ],
)


kernel_return = Op(
    "kernel.return",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return from a dispatchable kernel entry.",
    traits=[TERMINATOR],
    examples=["kernel.return"],
)


def _async_cache_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "cache_scope",
            ATTR_TYPE_ENUM,
            enum_def=CacheScope,
            doc="Required cache/coherency scope for the transfer.",
        ),
        AttrDef(
            "cache_temporal",
            ATTR_TYPE_ENUM,
            enum_def=CacheTemporal,
            doc="Required temporal cache hint for the transfer.",
        ),
    ]


# ============================================================================
# kernel.workitem.id — current invocation coordinate within the workgroup
# ============================================================================

kernel_workitem_id = Op(
    name="kernel.workitem.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read one coordinate of the current invocation within its workgroup. "
        "The result is a logical index value, not a byte offset; target "
        "lowering decides whether the coordinate is carried in scalar, vector, "
        "or dedicated target registers."
    ),
    results=[
        Result("result", INDEX, doc="Current workitem coordinate in the selected dimension."),
    ],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE],
    facts="loom_kernel_workitem_id_facts",
    format=[
        TemplateParam("dimension"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%tid = kernel.workitem.id<x> : index",
    ],
)


# ============================================================================
# kernel.workgroup.id — current workgroup coordinate within the dispatch grid
# ============================================================================

kernel_workgroup_id = Op(
    name="kernel.workgroup.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read one coordinate of the current workgroup within the dispatch "
        "grid. The result is a logical index value; target lowering decides "
        "whether the coordinate is carried in scalar registers, ABI state, or "
        "target-specific builtin values."
    ),
    results=[
        Result("result", INDEX, doc="Current workgroup coordinate in the selected dimension."),
    ],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE],
    facts="loom_kernel_workgroup_id_facts",
    format=[
        TemplateParam("dimension"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%bid = kernel.workgroup.id<x> : index",
    ],
)


# ============================================================================
# kernel.tensor.lds.descriptor — AMDGPU tensor-memory dgroup bundle
# ============================================================================

kernel_tensor_lds_descriptor = Op(
    name="kernel.tensor.lds.descriptor",
    group=kernel_ops,
    contracts=[ContractFamily.TENSOR_MEMORY],
    doc=(
        "Bundle AMDGPU tensor-memory descriptor groups into one typed SSA "
        "value. The dgroups are the exact operands lowered to "
        "llvm.amdgcn.tensor.load.to.lds or "
        "llvm.amdgcn.tensor.store.from.lds: D0 is vector<4xi32>, D1 is "
        "vector<8xi32>, and optional D2/D3 are vector<4xi32>. Gfx1250 uses "
        "two-group and four-group descriptor forms; the LLVM intrinsic's fifth "
        "D4 operand is lowered as zero because gfx1250 ignores it. The op is "
        "pure and contains no memory endpoints; endpoint views are operands of "
        "the async tensor ops so fact propagation and alias analysis do not "
        "need to decode hardware bitfields."
    ),
    operands=[
        Operand("dgroups", VECTOR, variadic=True, doc="AMDGPU tensor-memory descriptor groups D0..D3."),
    ],
    results=[
        Result("descriptor", ANY, doc="Typed tensor LDS descriptor value."),
    ],
    verify="loom_kernel_tensor_lds_descriptor_verify",
    format=[
        kw("dgroups"),
        GLUE,
        LPAREN,
        Refs("dgroups"),
        RPAREN,
        COLON,
        TypesOf("dgroups"),
        ARROW,
        ResultType("descriptor"),
    ],
    examples=[
        "%desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor",
        "%desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1, %d2, %d3) : vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> -> kernel.tensor.lds.descriptor",
    ],
)

# ============================================================================
# kernel.barrier — workgroup execution barrier with an explicit memory fence
# ============================================================================

kernel_barrier = Op(
    name="kernel.barrier",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    doc=(
        "Synchronize invocations in an explicit execution scope and fence a "
        "named memory space with a required ordering. The supported kernel "
        "barrier is a workgroup execution barrier over workgroup memory with "
        "acquire-release ordering. Async-copy completion is modeled by "
        "kernel.async.wait; use kernel.barrier only when invocations must "
        "rendezvous before consuming shared memory."
    ),
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=KernelMemorySpace,
            doc="Memory space whose accesses are fenced by the barrier.",
        ),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=KernelOrdering,
            doc="Memory ordering applied to fenced accesses.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=KernelScope,
            doc="Execution scope synchronized by the barrier.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_barrier_verify",
    format=[AttrDict()],
    examples=[
        "kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}",
    ],
)

# ============================================================================
# kernel.async.copy — initiate a view-to-view asynchronous copy
# ============================================================================


def _async_copy_attrs() -> list[AttrDef]:
    return [
        *_async_cache_attrs(),
        AttrDef(
            "direction",
            ATTR_TYPE_ENUM,
            enum_def=KernelAsyncDirection,
            doc="Required memory-space direction for the transfer.",
        ),
    ]


kernel_async_copy = Op(
    name="kernel.async.copy",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate an asynchronous byte-for-byte transfer between two already "
        "originated views. The source and destination view types may use "
        "different logical element types or shapes, but they must describe the "
        "same static byte footprint. The direction attribute makes the "
        "required memory-space flow explicit. The returned token must be "
        "committed to exactly one kernel.async.group before the copied bytes "
        "are waited or consumed."
    ),
    operands=[
        Operand("source", VIEW, doc="Typed source view whose base is copied from."),
        Operand("dest", VIEW, doc="Typed destination view whose base is copied to."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the initiated copy."),
    ],
    attrs=_async_copy_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_copy_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.copy %src to %dst {cache_scope = cu, cache_temporal = regular, direction = global_to_workgroup} : view<16xi8> to view<16xi8> -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.copy.mask — predicated view-to-view asynchronous copy
# ============================================================================

kernel_async_copy_mask = Op(
    name="kernel.async.copy.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.copy. When predicate is true, the op "
        "initiates the same transfer as kernel.async.copy. When predicate is "
        "false, the op performs no memory access and produces an already "
        "complete token so grouping and waiting remain structurally uniform."
    ),
    operands=[
        Operand("source", VIEW, doc="Typed source view whose base is copied from."),
        Operand("dest", VIEW, doc="Typed destination view whose base is copied to."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's copy."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated copy."),
    ],
    attrs=_async_copy_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_copy_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.copy.mask %src to %dst, %in_bounds {cache_scope = cu, cache_temporal = non_temporal, direction = global_to_workgroup} : view<16xi8> to view<16xi8>, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.gather — subgroup gather into a workgroup destination tile
# ============================================================================

kernel_async_gather = Op(
    name="kernel.async.gather",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate a subgroup-collective asynchronous gather from each "
        "invocation's source view into a lane-contiguous workgroup destination "
        "view. The destination view has one leading subgroup-lane axis and "
        "a trailing lane slot with enough static bytes to hold one source "
        "payload. If the lane slot is larger than the source footprint, the "
        "extra destination bytes are padding bytes with unspecified contents. "
        "The destination denotes the subgroup-uniform base tile; the current "
        "subgroup lane is applied by the op semantics and must not be "
        "pre-applied by forming a lane subview. This directly represents AMDGPU "
        "global_load_lds-style staging, including padded narrow loads, without "
        "requiring a later pass to rediscover that a set of per-lane copies was "
        "really one subgroup LDS DMA operation."
    ),
    operands=[
        Operand("source", VIEW, doc="Per-invocation global-like source fragment."),
        Operand("dest", VIEW, doc="Subgroup-uniform workgroup destination tile with a leading subgroup-lane axis."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the subgroup gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_gather_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8> -> kernel.async.token",
        "%copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<12xi8> to view<64x16xi8> -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.gather.mask — predicated subgroup gather
# ============================================================================

kernel_async_gather_mask = Op(
    name="kernel.async.gather.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.gather. False predicates perform no "
        "source or destination access for the current invocation but still "
        "produce a completed token, preserving a uniform async group shape for "
        "tails and guarded tiles."
    ),
    operands=[
        Operand("source", VIEW, doc="Per-invocation global-like source fragment."),
        Operand("dest", VIEW, doc="Subgroup-uniform workgroup destination tile with a leading subgroup-lane axis."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's gather."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_gather_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.gather.mask %src_lane to %lds_tile, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8>, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.cluster.gather — AMDGPU cluster broadcast into LDS
# ============================================================================

kernel_async_cluster_gather = Op(
    name="kernel.async.cluster.gather",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate an AMDGPU gfx1250+ cluster asynchronous load from a "
        "global-like source view into a workgroup/LDS destination view. The "
        "required i32 cluster_mask is the hardware workgroup broadcast mask "
        "loaded through M0. Source and destination must have the same static "
        "byte footprint, and that footprint must be exactly 1, 4, 8, or 16 "
        "bytes; target lowering maps those widths to "
        "llvm.amdgcn.cluster.load.async.to.lds.b8/b32/b64/b128. The LLVM "
        "offset and cache-policy immediate operands are lowering choices "
        "derived from the view address and cache attributes, not separate Loom "
        "semantics. The returned token must be committed to exactly one "
        "kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like source fragment broadcast across the cluster."),
        Operand("dest", VIEW, doc="Workgroup/LDS destination fragment for this workgroup."),
        Operand("cluster_mask", INTEGER, doc="i32 workgroup-cluster broadcast mask consumed by the target M0 operand."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the cluster gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_cluster_gather_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("cluster_mask"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("cluster_mask"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.cluster.gather %src to %lds using %mask {cache_scope = se, cache_temporal = high_temporal} : view<16xi8> to view<16xi8>, i32 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.cluster.gather.mask — predicated AMDGPU cluster gather
# ============================================================================

kernel_async_cluster_gather_mask = Op(
    name="kernel.async.cluster.gather.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.cluster.gather. False predicates "
        "perform no source or destination access for the current invocation "
        "but still produce a completed token, preserving a uniform async group "
        "shape for tails and guarded tiles. The cluster_mask remains the "
        "target workgroup broadcast mask and is distinct from the scalar i1 "
        "predicate."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like source fragment broadcast across the cluster."),
        Operand("dest", VIEW, doc="Workgroup/LDS destination fragment for this workgroup."),
        Operand("cluster_mask", INTEGER, doc="i32 workgroup-cluster broadcast mask consumed by the target M0 operand."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's cluster gather."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated cluster gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_cluster_gather_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("cluster_mask"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("cluster_mask"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.cluster.gather.mask %src to %lds using %mask, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<4xi8>, i32, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.tensor.load.to.lds — AMDGPU tensor-memory global-to-LDS transfer
# ============================================================================

kernel_async_tensor_load_to_lds = Op(
    name="kernel.async.tensor.load.to.lds",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC, ContractFamily.TENSOR_MEMORY],
    doc=(
        "Initiate an AMDGPU gfx1250+ tensor-memory load from a global-like "
        "source view into a workgroup/LDS destination view using an explicit "
        "kernel.tensor.lds.descriptor. The descriptor supplies the exact "
        "hardware dgroups, while the source and destination views keep the "
        "logical rank, element type, layout, and memory-space facts visible. "
        "The endpoints must have the same rank in [1, 5], the same 1/2/4/8 "
        "byte element type, and memory spaces global/constant/descriptor to "
        "workgroup. "
        "The returned token must be committed to exactly one kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like tensor-memory source view."),
        Operand("dest", VIEW, doc="Workgroup/LDS tensor-memory destination view."),
        Operand("descriptor", ANY, doc="Tensor LDS descriptor supplying AMDGPU dgroups."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the tensor load."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_tensor_load_to_lds_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("descriptor"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("descriptor"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.tensor.load.to.lds %global_tile to %lds_tile using %desc {cache_scope = cu, cache_temporal = regular} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.tensor.store.from.lds — AMDGPU tensor-memory LDS-to-global transfer
# ============================================================================

kernel_async_tensor_store_from_lds = Op(
    name="kernel.async.tensor.store.from.lds",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC, ContractFamily.TENSOR_MEMORY],
    doc=(
        "Initiate an AMDGPU gfx1250+ tensor-memory store from a workgroup/LDS "
        "source view into a global-like destination view using an explicit "
        "kernel.tensor.lds.descriptor. The descriptor supplies the exact "
        "hardware dgroups, while the source and destination views keep the "
        "logical rank, element type, layout, and memory-space facts visible. "
        "The endpoints must have the same rank in [1, 5], the same 1/2/4/8 "
        "byte element type, and memory spaces workgroup to global/descriptor. "
        "The "
        "returned token must be committed to exactly one kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Workgroup/LDS tensor-memory source view."),
        Operand("dest", VIEW, doc="Global-like tensor-memory destination view."),
        Operand("descriptor", ANY, doc="Tensor LDS descriptor supplying AMDGPU dgroups."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the tensor store."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_tensor_store_from_lds_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("descriptor"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("descriptor"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.tensor.store.from.lds %lds_tile to %global_tile using %desc {cache_scope = device, cache_temporal = non_temporal_writeback} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.group — commit async-copy tokens into one ordered group
# ============================================================================

kernel_async_group = Op(
    name="kernel.async.group",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Commit zero or more async copy/gather/cluster/tensor tokens into the "
        "ordered async stream. Empty groups are valid pipeline markers. The "
        "resulting group completes after all committed transfers complete. "
        "Groups are ordered by program order; waiting a group also completes "
        "older groups in the same stream."
    ),
    operands=[
        Operand(
            "tokens",
            ANY,
            variadic=True,
            doc="Async copy, gather, cluster-gather, or tensor-memory tokens to commit into this group.",
        ),
    ],
    results=[
        Result("group", ANY, doc="Opaque async group token to wait."),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_async_group_verify",
    format=[
        OptionalGroup(
            [Refs("tokens"), COLON, TypesOf("tokens")],
            anchor="tokens",
        ),
        ARROW,
        ResultType("group"),
    ],
    examples=[
        "%empty = kernel.async.group -> kernel.async.group",
        "%group = kernel.async.group %copy0, %copy1 : kernel.async.token, kernel.async.token -> kernel.async.group",
    ],
)

# ============================================================================
# kernel.async.wait — wait for an ordered async-copy group
# ============================================================================

kernel_async_wait = Op(
    name="kernel.async.wait",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Wait until a committed async-copy group has completed. This completes "
        "the named group and all older groups in the same ordered async stream. "
        "The required newer_groups value states how many younger groups are "
        "allowed to remain outstanding after the wait, matching AMDGPU "
        "wait.asyncmark and NVVM wait_group count semantics without making "
        "lowering rediscover the software-pipeline distance from scratch. "
        "It is not a workgroup barrier; use kernel.barrier separately when "
        "other invocations must observe the copied destination data."
    ),
    operands=[
        Operand("group", ANY, doc="Async group token to wait."),
    ],
    attrs=[
        AttrDef(
            "newer_groups",
            ATTR_TYPE_I64,
            doc="Maximum number of younger async groups allowed to remain outstanding.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_async_wait_verify",
    format=[
        Ref("group"),
        AttrDict(),
        COLON,
        TypeOf("group"),
    ],
    examples=[
        "kernel.async.wait %group {newer_groups = 0} : kernel.async.group",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_KERNEL_TYPES: tuple[TypeDef, ...] = (
    kernel_async_group_type,
    kernel_async_token_type,
    kernel_tensor_lds_descriptor_type,
)

ALL_KERNEL_OPS: tuple[Op, ...] = (
    kernel_def,
    kernel_return,
    kernel_barrier,
    kernel_async_copy,
    kernel_async_copy_mask,
    kernel_async_gather,
    kernel_async_gather_mask,
    kernel_async_group,
    kernel_async_wait,
    kernel_tensor_lds_descriptor,
    kernel_async_tensor_load_to_lds,
    kernel_async_tensor_store_from_lds,
    kernel_async_cluster_gather,
    kernel_async_cluster_gather_mask,
    kernel_workitem_id,
    kernel_workgroup_id,
)
