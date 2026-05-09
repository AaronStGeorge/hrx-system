# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for AMDGPU target-low shards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuEncodingFieldAllOnes,
    AmdgpuFixedEncodingValue,
    AmdgpuIgnoredOperandOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    AmdgpuOperandPredefinedValueRef,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_SOP1,
    AMDGPU_ENCODING_FORMAT_SOP2,
    AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
    amdgpu_encoding_field_id,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_ordinal,
    validate_amdgpu_descriptor_set_isa_xml,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    AsmImmediate,
    CEnum,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorCategory,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EncodingFieldValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandForm,
    OperandFormImmediateAction,
    OperandFormMatch,
    OperandFormMatchKind,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    RegisterPart,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"
_REG_AGPR = "amdgpu.agpr"
_REG_M0 = "amdgpu.m0"
_REG_SCC = "amdgpu.scc"
_REG_EXEC = "amdgpu.exec"

_REG_PART_VGPR_LOW16 = "amdgpu.vgpr.low16"
_REG_PART_VGPR_HIGH16 = "amdgpu.vgpr.high16"
_REG_PART_VGPR_FULL32_MASK = 0x3

_RESOURCE_SALU = "amdgpu.salu"
_RESOURCE_VALU = "amdgpu.valu"
_RESOURCE_SMEM = "amdgpu.smem"
_RESOURCE_VMEM_LOAD = "amdgpu.vmem.load"
_RESOURCE_VMEM_STORE = "amdgpu.vmem.store"
_RESOURCE_LDS_LOAD = "amdgpu.lds.load"
_RESOURCE_LDS_STORE = "amdgpu.lds.store"
_RESOURCE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_WMMA = "amdgpu.wmma"
_RESOURCE_SWMMAC = "amdgpu.swmmac"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_LOAD_LDS = "amdgpu.vmem.load.lds"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_VMEM_ATOMIC_RETURN = "amdgpu.vmem.atomic.return"
_SCHEDULE_VMEM_ATOMIC_NO_RETURN = "amdgpu.vmem.atomic.no_return"
_SCHEDULE_LDS_LOAD = "amdgpu.lds.load"
_SCHEDULE_LDS_STORE = "amdgpu.lds.store"
_SCHEDULE_LDS_ATOMIC = "amdgpu.lds.atomic"
_SCHEDULE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_SCHEDULE_BARRIER = "amdgpu.barrier"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WMMA = "amdgpu.wmma"
_SCHEDULE_WMMA_SCALE = "amdgpu.wmma.scale"
_SCHEDULE_SWMMAC = "amdgpu.swmmac"
_SCHEDULE_CACHE_CONTROL = "amdgpu.cache.control"
_SCHEDULE_WAIT_MEMORY = "amdgpu.wait.memory"
_SCHEDULE_WAIT_VMEM_STORE = "amdgpu.wait.vmem.store"
_SCHEDULE_WAIT_LDS = "amdgpu.wait.lds"
_SCHEDULE_WAIT_SMEM = "amdgpu.wait.smem"
_SCHEDULE_WAIT_LOAD = "amdgpu.wait.load"
_SCHEDULE_WAIT_STORE = "amdgpu.wait.store"
_SCHEDULE_WAIT_ALU = "amdgpu.wait.alu"
_SCHEDULE_WAIT_IDLE = "amdgpu.wait.idle"

# AMDGPU vector, memory, and matrix packets observe EXEC even when the vendor XML
# does not expose EXEC as an implicit operand. Model that state read in Loom so
# scheduling cannot move the packet across divergent-control EXEC writes.
_EXECUTION_MASKED_SCHEDULE_CLASSES = frozenset(
    (
        _SCHEDULE_VALU,
        _SCHEDULE_VMEM_LOAD,
        _SCHEDULE_VMEM_LOAD_LDS,
        _SCHEDULE_VMEM_STORE,
        _SCHEDULE_VMEM_ATOMIC_RETURN,
        _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
        _SCHEDULE_LDS_LOAD,
        _SCHEDULE_LDS_STORE,
        _SCHEDULE_LDS_ATOMIC,
        _SCHEDULE_LDS_CROSSLANE,
        _SCHEDULE_MFMA,
        _SCHEDULE_WMMA,
        _SCHEDULE_WMMA_SCALE,
        _SCHEDULE_SWMMAC,
    )
)

AMDGPU_SCALAR_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "scalar",
    doc="Scalar ALU, scalar data movement, and SGPR descriptors.",
)
AMDGPU_VECTOR_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "vector",
    doc="Vector ALU, vector data movement, and VGPR descriptors.",
)
AMDGPU_CONVERT_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "convert",
    doc="Numeric conversion descriptors.",
)
AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "compare_select",
    doc="Comparison and predicate-controlled selection descriptors.",
)
AMDGPU_MEMORY_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "memory",
    doc="Non-atomic memory load, store, and transfer descriptors.",
)
AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "atomic",
    doc="Atomic memory operation descriptors.",
)
AMDGPU_MATRIX_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "matrix",
    doc="Matrix, dot, WMMA, MFMA, and SWMMAC descriptors.",
)
AMDGPU_CONTROL_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "control",
    doc="Control, wait, barrier, and special state descriptors.",
)
AMDGPU_CACHE_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "cache",
    doc="Cache control and prefetch descriptors.",
)
AMDGPU_MISC_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "misc",
    doc="Descriptors that do not fit another stable AMDGPU category.",
)

AMDGPU_DESCRIPTOR_CATEGORIES = (
    AMDGPU_SCALAR_DESCRIPTOR_CATEGORY,
    AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
    AMDGPU_CONVERT_DESCRIPTOR_CATEGORY,
    AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
    AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
    AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    AMDGPU_MATRIX_DESCRIPTOR_CATEGORY,
    AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    AMDGPU_CACHE_DESCRIPTOR_CATEGORY,
    AMDGPU_MISC_DESCRIPTOR_CATEGORY,
)

_AMDGPU_DESCRIPTOR_SOURCE_DIR = Path("loom/src/loom/target/arch/amdgpu")
_AMDGPU_DESCRIPTOR_PUBLIC_HEADER_DIR = "loom/target/arch/amdgpu"


def _amdgpu_descriptor_set_file_stem(key: str) -> str:
    key_prefix = "amdgpu."
    key_suffix = ".core"
    if not key.startswith(key_prefix) or not key.endswith(key_suffix):
        raise ValueError(
            f"AMDGPU core descriptor set key '{key}' must have form "
            "'amdgpu.<family>.core'"
        )
    return key.removeprefix(key_prefix).removesuffix(key_suffix).replace(".", "_")


def _amdgpu_camel_case(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in value.split("_") if part)


def _amdgpu_core_descriptor_set(
    *,
    key: str,
    reg_classes: tuple[RegClass, ...],
    resources: tuple[Resource, ...],
    schedule_classes: tuple[ScheduleClass, ...],
    descriptors: tuple[Descriptor, ...] = (),
    register_parts: tuple[RegisterPart, ...] = (),
    categories: tuple[DescriptorCategory, ...] = AMDGPU_DESCRIPTOR_CATEGORIES,
) -> DescriptorSet:
    file_stem = _amdgpu_descriptor_set_file_stem(key)
    c_suffix = _amdgpu_camel_case(file_stem)
    c_enum_stem = file_stem.upper()
    feature_stem = key.removesuffix(".core")
    return DescriptorSet(
        key=key,
        target_key="amdgpu",
        feature_key=f"{feature_stem}.v1",
        c_header_path=_AMDGPU_DESCRIPTOR_SOURCE_DIR / f"{file_stem}_descriptors.h",
        c_source_path=_AMDGPU_DESCRIPTOR_SOURCE_DIR / f"{file_stem}_descriptors.c",
        header_guard=f"LOOM_TARGET_ARCH_AMDGPU_{c_enum_stem}_DESCRIPTORS_H_",
        public_header=(
            f"{_AMDGPU_DESCRIPTOR_PUBLIC_HEADER_DIR}/{file_stem}_descriptors.h"
        ),
        function_name=f"loom_amdgpu_{file_stem}_core_descriptor_set",
        c_table_prefix=f"Amdgpu{c_suffix}Core",
        c_enum_prefix=f"AMDGPU_{c_enum_stem}_CORE",
        generator_version=1,
        reg_classes=reg_classes,
        register_parts=register_parts,
        resources=resources,
        schedule_classes=schedule_classes,
        descriptors=descriptors,
        descriptor_set_ordinal=amdgpu_descriptor_set_ordinal(key),
        categories=categories,
    )


_COUNTER_VMEM_LOAD = 1
_COUNTER_VMEM_STORE = 2
_COUNTER_LDS = 3
_COUNTER_SMEM = 4
_COUNTER_ALU = 5


def _predefined(
    value_name: str, operand_type: str | None = None
) -> AmdgpuOperandPredefinedValueRef:
    return AmdgpuOperandPredefinedValueRef(value_name, operand_type)


def _instruction_encoding_opcode(
    spec: AmdgpuIsaFactSource, instruction_name: str, encoding_name: str
) -> int:
    opcodes = set()
    for summary in spec.instruction_encoding_summaries(
        (instruction_name,), include_aliases=False
    ):
        if summary.encoding_name == encoding_name:
            opcodes.add(summary.opcode)
    if len(opcodes) != 1:
        raise ValueError(
            f"{spec.source_name}: expected one {encoding_name} opcode for "
            f"{instruction_name}, found {len(opcodes)}"
        )
    return next(iter(opcodes))


_MUBUF_SOFFSET_INLINE_ZERO = _predefined("0")

_VMEM_LOAD_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_LOAD
)
_VMEM_STORE_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_STORE
)
_LDS_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_LDS)
_SMEM_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_SMEM)
_ALU_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_ALU)
_GFX950_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_GFX11_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_VMEM_LOAD_WAIT_HAZARDS = (_VMEM_LOAD_COUNTER_HAZARD,)
_VMEM_STORE_WAIT_HAZARDS = (_VMEM_STORE_COUNTER_HAZARD,)
_LDS_WAIT_HAZARDS = (_LDS_COUNTER_HAZARD,)
_SMEM_WAIT_HAZARDS = (_SMEM_COUNTER_HAZARD,)
_ALU_WAIT_HAZARDS = (_ALU_COUNTER_HAZARD,)
_IDLE_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
    _ALU_COUNTER_HAZARD,
)

_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS = (
    ("IMM", 1),
    ("SOFFSET_EN", 1),
)
_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS = (
    ("IMM", 1),
    ("SOFFSET_EN", 0),
)

_ADDRESS_OFFSET_BYTE_ENCODING_ID = 1
_ADDRESS_OFFSET_DWORD_ENCODING_ID = 2
_ADDRESS_OFFSET_QWORD_ENCODING_ID = 3
_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID = 4
_ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID = 5
_ADDRESS_OFFSET_DS16_ENCODING_ID = 6
_SOURCE_INLINE_U32_ENCODING_ID = 7
_WAIT_COUNTER_VMEM_ENCODING_ID = 16
_WAIT_COUNTER_LGKM_ENCODING_ID = 17
_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID = 18
_WAIT_COUNTER_VMEM_STORE_ENCODING_ID = 19
_WAIT_COUNTER_LDS_ENCODING_ID = 20
_WAIT_COUNTER_SMEM_ENCODING_ID = 21
_WAIT_COUNTER_ALU_ENCODING_ID = 22
_GFX9_11_VECTOR_CACHE_FIELDS = (("GLC", 1), ("SLC", 1), ("DLC", 1))
_GFX950_VECTOR_CACHE_FIELDS = (("NT", 1), ("SC0", 1), ("SC1", 1))
_GFX12_VECTOR_CACHE_FIELDS = (("NV", 1), ("SCOPE", 2), ("TH", 3))
_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS = ("SCOPE", "TH")
# VGLOBAL atomic instructions use TH bit 0 to request returning the old value.
_GFX12_TH_ATOMIC_RETURN_VALUE = 0x1
_ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS = frozenset(
    (
        _ADDRESS_OFFSET_BYTE_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_DS16_ENCODING_ID,
    )
)
_ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES = frozenset(("offset", "offset0", "offset1"))
_GLOBAL_SADDR_OFF = _predefined("NULL")
_GLOBAL_GFX950_SADDR_OFF = AmdgpuEncodingFieldAllOnes()
_MUBUF_VADDR_OFFSET_ONLY_SIZE_REASON = "idxen-disabled-mubuf-vaddr-uses-one-offset-vgpr"
_GLOBAL_SADDR_OFFSET_ONLY_SIZE_REASON = (
    "saddr-enabled-global-address-uses-one-offset-vgpr"
)
_D16_PARTIAL_REGISTER_SIZE_REASON = "d16-instruction-uses-half-vgpr-lane"
_U24_SOURCE_SIZE_REASON = "u24-instruction-reads-low-24-bits-of-b32-source"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)
_SGPR_VGPR_ALT = (RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR))
_VGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_VGPR_AGPR_ALT = (RegClassAlt(_REG_VGPR), RegClassAlt(_REG_AGPR))
_VGPR_AGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(_REG_AGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_M0_ALT = (RegClassAlt(_REG_M0, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)
_SCC_ALT = (RegClassAlt(_REG_SCC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)
_EXEC_ALT = (RegClassAlt(_REG_EXEC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)

_VGPR_REGISTER_PARTS = (
    RegisterPart(_REG_PART_VGPR_LOW16, _REG_VGPR, 0x1),
    RegisterPart(_REG_PART_VGPR_HIGH16, _REG_VGPR, 0x2),
)


class AmdgpuAtomicMemorySpace(CEnum):
    WORKGROUP = "LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP"
    GLOBAL = "LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL"
    GENERIC = "LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC"


class AmdgpuMemoryAddressForm(CEnum):
    DEFAULT = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT"
    GLOBAL_SADDR = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR"
    FLAT = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT"


class AmdgpuAtomicOperationKind(CEnum):
    REDUCE = "LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE"
    RMW = "LOOM_AMDGPU_ATOMIC_OPERATION_RMW"
    CMPXCHG = "LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG"


class AmdgpuAtomicKind(CEnum):
    ADDI = "LOOM_ATOMIC_KIND_ADDI"
    SUBI = "LOOM_ATOMIC_KIND_SUBI"
    MINSI = "LOOM_ATOMIC_KIND_MINSI"
    MAXSI = "LOOM_ATOMIC_KIND_MAXSI"
    MINUI = "LOOM_ATOMIC_KIND_MINUI"
    MAXUI = "LOOM_ATOMIC_KIND_MAXUI"
    ANDI = "LOOM_ATOMIC_KIND_ANDI"
    ORI = "LOOM_ATOMIC_KIND_ORI"
    XORI = "LOOM_ATOMIC_KIND_XORI"
    XCHGI = "LOOM_ATOMIC_KIND_XCHGI"
    ADDF = "LOOM_ATOMIC_KIND_ADDF"
    MINNUMF = "LOOM_ATOMIC_KIND_MINNUMF"
    MAXNUMF = "LOOM_ATOMIC_KIND_MAXNUMF"
    NONE = "LOOM_AMDGPU_ATOMIC_KIND_NONE"


class AmdgpuAtomicValueKind(CEnum):
    I32 = "LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32"
    F32 = "LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32"


@dataclass(frozen=True, slots=True)
class AmdgpuAtomicDescriptorCandidate:
    memory_space: AmdgpuAtomicMemorySpace
    address_form: AmdgpuMemoryAddressForm
    operation_kind: AmdgpuAtomicOperationKind
    atomic_kind: AmdgpuAtomicKind
    value_kind: AmdgpuAtomicValueKind
    descriptor_key: str


def _matrix_hazards(resource: str) -> tuple[Hazard, ...]:
    return (
        _ALU_COUNTER_HAZARD,
        Hazard(
            HazardKind.MIN_DISTANCE,
            resource=resource,
            producer_stage=0,
            consumer_stage=0,
            distance=2,
        ),
    )


def _common_scalar_vector_memory_resources() -> tuple[Resource, ...]:
    return (
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_LDS_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_CROSSLANE, capacity_per_cycle=1, kind=ResourceKind.LOAD),
    )


def _common_scalar_vector_memory_schedule_classes(
    *,
    smem_load_hazards: tuple[Hazard, ...],
    vmem_load_hazards: tuple[Hazard, ...],
    vmem_store_hazards: tuple[Hazard, ...],
    lds_load_hazards: tuple[Hazard, ...],
    lds_store_hazards: tuple[Hazard, ...],
    lds_atomic_hazards: tuple[Hazard, ...],
    lds_crosslane_hazards: tuple[Hazard, ...],
) -> tuple[ScheduleClass, ...]:
    return (
        ScheduleClass(
            _SCHEDULE_SALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SALU, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_SMEM, cycles=1, units=1),),
            hazards=smem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),),
            hazards=lds_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),),
            hazards=lds_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_ATOMIC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(
                IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=lds_atomic_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_CROSSLANE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_CROSSLANE, cycles=1, units=1),),
            hazards=lds_crosslane_hazards,
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_BARRIER,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CACHE_CONTROL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    )


def _asm(
    *,
    mnemonic: str | None = None,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    named_immediates: bool = False,
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(
                AsmImmediate(field_name, name=field_name if named_immediates else None)
                for field_name in immediates
            ),
        ),
    )


def _sgpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _SGPR_ALT, unit_count=units)


def _sgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SGPR_ALT, unit_count=units)


def _sgpr_predicate(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _SGPR_ALT, unit_count=units)


def _sgpr_resource(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _SGPR_ALT, unit_count=units)


def _m0_implicit_resource(field_name: str = "m0") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESOURCE,
        _M0_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _m0_result(field_name: str = "dst") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _M0_ALT,
        flags=(OperandFlag.STATE_WRITE,),
        unit_count=1,
    )


def _scc_result(field_name: str = "scc") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _scc_clobber(field_name: str = "scc") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _scc_state_read(field_name: str = "scc_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _scc_predicate(field_name: str) -> Operand:
    return Operand(
        field_name,
        OperandRole.PREDICATE,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _exec_clobber(field_name: str = "exec") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _EXEC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _exec_state_read(field_name: str = "exec_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _EXEC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _is_exec_state_read(operand: Operand) -> bool:
    return (
        OperandFlag.STATE_READ in operand.flags
        and len(operand.reg_alts) == 1
        and operand.reg_alts[0].reg_class == _REG_EXEC
    )


def _with_execution_mask_state_read(descriptor: Descriptor) -> Descriptor:
    if descriptor.schedule_class not in _EXECUTION_MASKED_SCHEDULE_CLASSES:
        return descriptor
    if any(_is_exec_state_read(operand) for operand in descriptor.operands):
        return descriptor
    return replace(descriptor, operands=(*descriptor.operands, _exec_state_read()))


def _with_execution_mask_state_reads(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        _with_execution_mask_state_read(descriptor) for descriptor in descriptors
    )


def _vgpr_result(
    field_name: str = "dst", *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _VGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _vgpr_operand(
    field_name: str, *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.OPERAND,
        _VGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _mubuf_vaddr_operand(field_name: str = "vaddr") -> AmdgpuOperandOverlay:
    return AmdgpuOperandOverlay(
        "VADDR",
        _vgpr_operand(field_name),
        size_exception_reason=_MUBUF_VADDR_OFFSET_ONLY_SIZE_REASON,
    )


def _global_addr_operand(
    xml_field_name: str, *, units: int, has_saddr: bool
) -> AmdgpuOperandOverlay:
    return AmdgpuOperandOverlay(
        xml_field_name,
        _vgpr_operand("addr", units=units),
        size_exception_reason=(
            _GLOBAL_SADDR_OFFSET_ONLY_SIZE_REASON if has_saddr and units == 1 else None
        ),
    )


def _sgpr_vgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SGPR_VGPR_ALT, unit_count=units)


def _vgpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_CONST_ALT, unit_count=units)


def _vgpr_agpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(
        field_name, OperandRole.OPERAND, _VGPR_AGPR_CONST_ALT, unit_count=units
    )


def _u32_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=(2**32) - 1,
    )


def _source_inline_u32_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=64,
        encoding_id=_SOURCE_INLINE_U32_ENCODING_ID,
    )


_U32_IMMEDIATE = _u32_immediate()

_SOURCE_INLINE_U32_IMMEDIATE = _source_inline_u32_immediate()

_LITERAL_U32_IMMEDIATE = replace(
    _U32_IMMEDIATE, encoding_field_id=amdgpu_encoding_field_id("LITERAL")
)

_HAL_BUFFER_DESCRIPTOR_EXTENT_IMMEDIATE = Immediate(
    "extent",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE = Immediate(
    "cache_swizzle_stride",
    ImmediateKind.UNSIGNED,
    bit_width=14,
    unsigned_max=(2**14) - 1,
)


_MANUAL_SCALAR_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_mov_b32_m0",
    "amdgpu.s_mov_b32_m0.imm",
    "amdgpu.s_mov_b64_exec",
    "amdgpu.s_mov_b64_exec_read",
    "amdgpu.s_xor_b64_exec",
)


def _s_mov_b32_contract_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mov_b32",
        instruction_name="S_MOV_B32",
        mnemonic="s_mov_b32",
        encoding_name="ENC_SOP1",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(AmdgpuOperandOverlay("SDST", _sgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediates=(_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _manual_scalar_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    s_mov_b32_opcode = _instruction_encoding_opcode(spec, "S_MOV_B32", "ENC_SOP1")
    s_mov_b64_opcode = _instruction_encoding_opcode(spec, "S_MOV_B64", "ENC_SOP1")
    s_xor_b64_opcode = _instruction_encoding_opcode(spec, "S_XOR_B64", "ENC_SOP2")
    return (
        Descriptor(
            key="amdgpu.s_mov_b32",
            mnemonic="s_mov_b32",
            semantic_tag="integer.const.u32",
            operands=(_sgpr_result(),),
            immediates=(_U32_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b32_m0",
            mnemonic="s_mov_b32",
            semantic_tag="special.m0.move.u32",
            operands=(
                _m0_result(),
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                ),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_M0", "M0"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b32_m0", results=("dst",), operands=("src",)
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b32_m0.imm",
            mnemonic="s_mov_b32",
            semantic_tag="special.m0.const.u32",
            operands=(_m0_result(),),
            immediates=(_U32_IMMEDIATE,),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_M0", "M0"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b32_m0_imm", results=("dst",), immediates=("imm32",)
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b64_exec",
            mnemonic="s_mov_b64",
            semantic_tag="control.exec.restore",
            operands=(
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                    unit_count=2,
                ),
                _exec_clobber(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_EXEC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(mnemonic="s_mov_b64_exec", operands=("src",)),
            effects=(_CONVERGENT_EFFECT,),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b64_opcode,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b64_exec_read",
            mnemonic="s_mov_b64",
            semantic_tag="control.exec.read",
            operands=(
                Operand(
                    "dst",
                    OperandRole.RESULT,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SDST"),
                    unit_count=2,
                ),
                _exec_state_read(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SSRC0"),
                    spec.operand_predefined_value("OPR_SSRC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(mnemonic="s_mov_b64_exec_read", results=("dst",)),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b64_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_xor_b64_exec",
            mnemonic="s_xor_b64",
            semantic_tag="control.exec.xor",
            operands=(
                _scc_result("active"),
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                    unit_count=2,
                ),
                _exec_clobber("exec_out"),
                _exec_state_read(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_EXEC", "EXEC_LO"),
                ),
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SSRC1"),
                    spec.operand_predefined_value("OPR_SSRC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_xor_b64_exec", results=("active",), operands=("src",)
            ),
            effects=(_CONVERGENT_EFFECT,),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2,
            encoding_id=s_xor_b64_opcode,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    )


_VMCNT_IMMEDIATE = Immediate(
    "vmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
    default_value=(2**6) - 1,
)

_LGKMCNT_4BIT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=4,
    encoding_id=_WAIT_COUNTER_LGKM_ENCODING_ID,
    unsigned_max=(2**4) - 1,
    default_value=(2**4) - 1,
)

_LGKMCNT_6BIT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LGKM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
    default_value=(2**6) - 1,
)

_LOADCNT_IMMEDIATE = Immediate(
    "loadcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_STORECNT_IMMEDIATE = Immediate(
    "storecnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_VSCNT_IMMEDIATE = Immediate(
    "vscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_DSCNT_IMMEDIATE = Immediate(
    "dscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LDS_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_KMCNT_IMMEDIATE = Immediate(
    "kmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    encoding_id=_WAIT_COUNTER_SMEM_ENCODING_ID,
    unsigned_max=(2**5) - 1,
)

_DEPCTR_IMMEDIATE = Immediate(
    "depctr",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    encoding_id=_WAIT_COUNTER_ALU_ENCODING_ID,
    unsigned_max=(2**16) - 1,
)

_PREFETCH_COUNT_IMMEDIATE = Immediate(
    "count",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    unsigned_max=(2**5) - 1,
)

_PREFETCH_DISTANCE_IMMEDIATE = Immediate(
    "distance",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    unsigned_max=(2**16) - 1,
)

_MATRIX_A_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_IMMEDIATE = Immediate(
    "matrix_a_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_IMMEDIATE = Immediate(
    "matrix_b_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_REUSE_IMMEDIATE = Immediate(
    "matrix_a_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_MATRIX_B_REUSE_IMMEDIATE = Immediate(
    "matrix_b_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_INDEX_KEY_16_IMMEDIATE = Immediate(
    "index_key_16bit",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_GLOBAL_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_LOAD_B16_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=16,
)

_GLOBAL_LOAD_B64_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_LOAD_B128_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_GLOBAL_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_STORE_B16_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=16,
)

_GLOBAL_STORE_B64_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_STORE_B128_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)


def _workgroup_memory_effect(kind: EffectKind, width_bits: int) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.WORKGROUP,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


_WORKGROUP_BARRIER_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_CACHE_CONTROL_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_GLOBAL_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
)

_INSTRUCTION_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
)

_CONVERGENT_EFFECT = Effect(
    EffectKind.CONVERGENT,
    flags=(EffectFlag.ORDERED,),
)


def _ds_crosslane_effects(width_bits: int) -> tuple[Effect, Effect]:
    # DS cross-lane instructions retire through LGKM even though they do not
    # access LDS memory. Their SSA results must still wait on the LDS counter.
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_LDS,
            width_bits=width_bits,
        ),
        _CONVERGENT_EFFECT,
    )


_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_VMEM_LOAD_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_LOAD,
)

_VMEM_STORE_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_STORE,
)

_LDS_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_LDS,
)

_SMEM_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_SMEM,
)

_ALU_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_ALU,
)

_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
    Constraint(ConstraintKind.EARLY_CLOBBER, 0),
)
_BUFFER_ATOMIC_VDATA_INPUT_REASON = "xml-models-buffer-atomic-vdata-as-output-only"
_SMFMAC_VDST_ACCUMULATOR_REASON = "xml-models-smfmac-accumulator-as-vdst"
_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)
_PSEUDO_DEAD_REMOVABLE_FLAGS = (DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.PSEUDO)


def _hal_buffer_descriptor_pseudos() -> tuple[Descriptor, ...]:
    return (
        Descriptor(
            key="amdgpu.hal.buffer_descriptor",
            mnemonic=None,
            semantic_tag="memory.hal.buffer_descriptor",
            operands=(
                _sgpr_result("descriptor", units=4),
                _sgpr_operand("binding", units=2),
            ),
            immediates=(
                _HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE,
                _HAL_BUFFER_DESCRIPTOR_EXTENT_IMMEDIATE,
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.hal.buffer_descriptor.extent",
            mnemonic=None,
            semantic_tag="memory.hal.buffer_descriptor",
            operands=(
                _sgpr_result("descriptor", units=4),
                _sgpr_operand("binding", units=2),
                _sgpr_operand("extent"),
            ),
            immediates=(_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE,),
            schedule_class=_SCHEDULE_SALU,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
    )


def _offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return _named_offset_immediate("offset", bit_width, encoding_id=encoding_id)


def _named_offset_immediate(
    field_name: str,
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        encoding_id=encoding_id,
        unsigned_max=(2**bit_width) - 1,
    )


def _ds_offset_immediate() -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=16,
        encoding_id=_ADDRESS_OFFSET_DS16_ENCODING_ID,
        encoding_slices=(
            ImmediateEncodingSlice(amdgpu_encoding_field_id("OFFSET0"), 0, 8),
            ImmediateEncodingSlice(amdgpu_encoding_field_id("OFFSET1"), 8, 8),
        ),
        unsigned_max=(2**16) - 1,
    )


def _ds_fixed_fields_without_offset1(
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[tuple[str, AmdgpuFixedEncodingValue], ...]:
    return tuple(
        fixed_field
        for fixed_field in fixed_encoding_fields
        if fixed_field[0] != "OFFSET1"
    )


def _signed_offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.SIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        encoding_id=encoding_id,
        signed_min=-(2 ** (bit_width - 1)),
        unsigned_max=(2 ** (bit_width - 1)) - 1,
    )


def _cache_immediate(
    field_name: str, bit_width: int, *, default_value: int = 0
) -> Immediate:
    return Immediate(
        field_name.lower(),
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        unsigned_max=(2**bit_width) - 1,
        default_value=default_value,
    )


def _cache_field_names(cache_fields: tuple[tuple[str, int], ...]) -> tuple[str, ...]:
    return tuple(field_name for field_name, _bit_width in cache_fields)


def _cache_immediates(
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(field_name, bit_width)
        for field_name, bit_width in cache_fields
    )


def _cache_immediates_with_defaults(
    cache_fields: tuple[tuple[str, int], ...],
    defaults: dict[str, int],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(
            field_name, bit_width, default_value=defaults.get(field_name, 0)
        )
        for field_name, bit_width in cache_fields
    )


def _scc_output(descriptor_operand: Operand) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SSRC_SPECIAL_SCC",
        descriptor_operand=descriptor_operand,
        data_format_name="FMT_NUM_B1",
        size_bits=1,
        is_input=False,
        is_output=True,
    )


def _scc_input(descriptor_operand: Operand) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SSRC_SPECIAL_SCC",
        descriptor_operand=descriptor_operand,
        data_format_name="FMT_NUM_B1",
        size_bits=1,
        is_input=True,
        is_output=False,
    )


_SCC_CLOBBER_OUTPUT = _scc_output(_scc_clobber())

_IGNORE_GLOBAL_READ_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B16",
    size_bits=16,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_U16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_U16",
    size_bits=16,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B16",
    size_bits=16,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)


def _ignore_workgroup_memory(
    *, width_bits: int, is_input: bool, data_format_name: str | None = None
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_DSMEM",
        data_format_name=data_format_name or f"FMT_NUM_B{width_bits}",
        size_bits=width_bits,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-workgroup-read-effect"
            if is_input
            else "modeled-by-workgroup-write-effect"
        ),
    )


def _ignore_global_read_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 16:
            return _IGNORE_GLOBAL_READ_MEMORY_B16
        case 32:
            return _IGNORE_GLOBAL_READ_MEMORY
        case 64:
            return _IGNORE_GLOBAL_READ_MEMORY_B64
        case 128:
            return _IGNORE_GLOBAL_READ_MEMORY_B128
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _ignore_global_write_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 16:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B16
        case 32:
            return _IGNORE_GLOBAL_WRITE_MEMORY
        case 64:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B64
        case 128:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B128
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _global_read_effect(width_bits: int) -> Effect:
    match width_bits:
        case 16:
            return _GLOBAL_LOAD_B16_EFFECT
        case 32:
            return _GLOBAL_LOAD_EFFECT
        case 64:
            return _GLOBAL_LOAD_B64_EFFECT
        case 128:
            return _GLOBAL_LOAD_B128_EFFECT
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _global_write_effect(width_bits: int) -> Effect:
    match width_bits:
        case 16:
            return _GLOBAL_STORE_B16_EFFECT
        case 32:
            return _GLOBAL_STORE_EFFECT
        case 64:
            return _GLOBAL_STORE_B64_EFFECT
        case 128:
            return _GLOBAL_STORE_B128_EFFECT
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _ignore_global_atomic_memory(
    *, data_format_name: str, is_input: bool
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name,
        size_bits=32,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-global-atomic-read-effect"
            if is_input
            else "modeled-by-global-atomic-write-effect"
        ),
    )


def _ignore_generic_atomic_memory(
    *, data_format_name: str, is_input: bool
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name,
        size_bits=32,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-generic-atomic-read-effect"
            if is_input
            else "modeled-by-generic-atomic-write-effect"
        ),
    )


_IGNORE_FLAT_SCRATCH_INPUT = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_FLAT_SCRATCH",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-flat-address-state",
)


def _atomic_effects(
    memory_space: MemorySpace, width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
    )


def _global_atomic_effects(
    width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return _atomic_effects(MemorySpace.GLOBAL, width_bits, counter_id=counter_id)


def _generic_atomic_effects(
    width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return _atomic_effects(MemorySpace.GENERIC, width_bits, counter_id=counter_id)


def _global_to_lds_effects(width_bits: int) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=width_bits,
        ),
    )


def _implicit_m0_input() -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SDST_M0",
        descriptor_operand=_m0_implicit_resource(),
        data_format_name="FMT_NUM_B32",
        size_bits=32,
        is_input=True,
        is_output=False,
    )


def _s_add_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_add_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_add_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_add_u32.rhs_inline",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        semantic_tag="integer.add.u32",
    )


def _s_addc_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_clobber("carry")),
            _scc_input(_scc_state_read("carry_in")),
        ),
        asm_forms=_asm(results=("sum",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_sub_u32",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_sub_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_sub_u32.rhs_inline",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        semantic_tag="integer.sub.u32",
    )


def _s_mul_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_i32",
        instruction_name="S_MUL_I32",
        mnemonic="s_mul_i32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.lo.i32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_hi_u32",
        instruction_name="S_MUL_HI_U32",
        mnemonic="s_mul_hi_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_inline_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if rhs_inline_descriptor_key is not None:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor=rhs_inline_descriptor_key,
                source_operand="rhs",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_rhs_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_rhs_inline",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_i32",
        instruction_name="S_MIN_I32",
        mnemonic="s_min_i32",
        semantic_tag="integer.min.i32",
    )


def _s_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_i32",
        instruction_name="S_MAX_I32",
        mnemonic="s_max_i32",
        semantic_tag="integer.max.i32",
    )


def _s_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_u32",
        instruction_name="S_MIN_U32",
        mnemonic="s_min_u32",
        semantic_tag="integer.min.u32",
    )


def _s_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_u32",
        instruction_name="S_MAX_U32",
        mnemonic="s_max_u32",
        semantic_tag="integer.max.u32",
    )


def _s_cselect_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_cselect_b32",
        instruction_name="S_CSELECT_B32",
        mnemonic="s_cselect_b32",
        encoding_name="ENC_SOP2",
        semantic_tag="control.select.b32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("true_value")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("false_value")),
        ),
        implicit_operands=(_scc_input(_scc_predicate("condition")),),
        asm_forms=_asm(
            results=("dst",),
            operands=("true_value", "false_value", "condition"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_shift_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("shift")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add_u32.lit",
                source_operand="lhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _literal_operand_form(
    *,
    replacement_descriptor: str,
    source_operand: str,
    immediate_field: str = "imm32",
) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand=source_operand,
                match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64,
            ),
        ),
        immediate_action=OperandFormImmediateAction.SET_MATCHED_I64,
        immediate_field=immediate_field,
        immediate_source_operand=source_operand,
    )


def _soffset_zero_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
        ),
    )


def _soffset_offset_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64,
            ),
        ),
        immediate_action=OperandFormImmediateAction.ADD_MATCHED_I64,
        immediate_field="offset",
        immediate_source_operand="soffset",
    )


def _buffer_off_zero_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="vaddr",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
        ),
    )


def _v_binary_literal_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_lit",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_literal_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_u32.lit",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
    )


def _v_add_co_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_u32",
        instruction_name="V_ADD_CO_U32",
        mnemonic="v_add_co_u32",
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_co_ci_u32_overlay(
    *, instruction_name: str = "V_ADD_CO_CI_U32", mnemonic: str = "v_add_co_ci_u32"
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_ci_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _sgpr_operand("carry_in", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_u32_overlay(instruction_name: str, mnemonic: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    literal_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand(lhs_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        operand_forms=()
        if literal_descriptor_key is None
        else (
            _literal_operand_form(
                replacement_descriptor=literal_descriptor_key,
                source_operand=lhs_name,
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_lo_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_lo_u32",
        instruction_name="V_MUL_LO_U32",
        mnemonic="v_mul_lo_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.lo.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_hi_u32",
        instruction_name="V_MUL_HI_U32",
        mnemonic="v_mul_hi_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_u32_u24_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mul_u32_u24_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.lit",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mad_u32_u24_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mad_u32_u24",
        instruction_name="V_MAD_U32_U24",
        mnemonic="v_mad_u32_u24",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("a"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _sgpr_vgpr_operand("b"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src0_lit",
                source_operand="a",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src1_lit",
                source_operand="b",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src2_lit",
                source_operand="addend",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mad_u32_u24_literal_overlay(literal_source: str) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "a", _sgpr_vgpr_operand("a"), _U24_SOURCE_SIZE_REASON),
        "src1": ("SRC1", "b", _sgpr_vgpr_operand("b"), _U24_SOURCE_SIZE_REASON),
        "src2": ("SRC2", "addend", _sgpr_vgpr_operand("addend"), None),
    }
    literal_field = source_fields[literal_source][0]
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result())]
    asm_operands = []
    for source_name, (
        xml_field,
        field_name,
        operand,
        size_reason,
    ) in source_fields.items():
        if source_name == literal_source:
            continue
        asm_operands.append(field_name)
        operands.append(
            AmdgpuOperandOverlay(
                xml_field,
                operand,
                size_exception_reason=size_reason,
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_mad_u32_u24.{literal_source}_lit",
        instruction_name="V_MAD_U32_U24",
        mnemonic=f"v_mad_u32_u24_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        asm_forms=_asm(
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_minmax_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _v_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_i32",
        instruction_name="V_MIN_I32",
        mnemonic="v_min_i32",
        semantic_tag="integer.min.i32",
    )


def _v_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_i32",
        instruction_name="V_MAX_I32",
        mnemonic="v_max_i32",
        semantic_tag="integer.max.i32",
    )


def _v_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_u32",
        instruction_name="V_MIN_U32",
        mnemonic="v_min_u32",
        semantic_tag="integer.min.u32",
    )


def _v_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_u32",
        instruction_name="V_MAX_U32",
        mnemonic="v_max_u32",
        semantic_tag="integer.max.u32",
    )


def _v_readfirstlane_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_readfirstlane_b32",
        instruction_name="V_READFIRSTLANE_B32",
        mnemonic="v_readfirstlane_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="lane.readfirst.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("value")),
        ),
        effects=(_CONVERGENT_EFFECT,),
    )


def _s_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_and_b32",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_or_b32",
        instruction_name="S_OR_B32",
        mnemonic="s_or_b32",
        semantic_tag="integer.or.u32",
    )


def _s_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_xor_b32",
        instruction_name="S_XOR_B32",
        mnemonic="s_xor_b32",
        semantic_tag="integer.xor.u32",
    )


def _s_and_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_and_b64",
        instruction_name="S_AND_B64",
        mnemonic="s_and_b64",
        semantic_tag="integer.and.u64",
    )


def _s_or_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_or_b64",
        instruction_name="S_OR_B64",
        mnemonic="s_or_b64",
        semantic_tag="integer.or.u64",
    )


def _s_xor_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_xor_b64",
        instruction_name="S_XOR_B64",
        mnemonic="s_xor_b64",
        semantic_tag="integer.xor.u64",
    )


def _s_cmp_i32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_eq_i32",
            instruction_name="S_CMP_EQ_I32",
            mnemonic="s_cmp_eq_i32",
            semantic_tag="integer.compare.eq.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lg_i32",
            instruction_name="S_CMP_LG_I32",
            mnemonic="s_cmp_lg_i32",
            semantic_tag="integer.compare.ne.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_i32",
            instruction_name="S_CMP_LT_I32",
            mnemonic="s_cmp_lt_i32",
            semantic_tag="integer.compare.slt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_i32",
            instruction_name="S_CMP_LE_I32",
            mnemonic="s_cmp_le_i32",
            semantic_tag="integer.compare.sle.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_i32",
            instruction_name="S_CMP_GT_I32",
            mnemonic="s_cmp_gt_i32",
            semantic_tag="integer.compare.sgt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_i32",
            instruction_name="S_CMP_GE_I32",
            mnemonic="s_cmp_ge_i32",
            semantic_tag="integer.compare.sge.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_u32",
            instruction_name="S_CMP_LT_U32",
            mnemonic="s_cmp_lt_u32",
            semantic_tag="integer.compare.ult.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_u32",
            instruction_name="S_CMP_LE_U32",
            mnemonic="s_cmp_le_u32",
            semantic_tag="integer.compare.ule.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_u32",
            instruction_name="S_CMP_GT_U32",
            mnemonic="s_cmp_gt_u32",
            semantic_tag="integer.compare.ugt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_u32",
            instruction_name="S_CMP_GE_U32",
            mnemonic="s_cmp_ge_u32",
            semantic_tag="integer.compare.uge.i32",
        ),
    )


def _s_cmp_u64_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_eq_u64",
            instruction_name="S_CMP_EQ_U64",
            mnemonic="s_cmp_eq_u64",
            semantic_tag="integer.compare.eq.u64",
        ),
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_lg_u64",
            instruction_name="S_CMP_LG_U64",
            mnemonic="s_cmp_lg_u64",
            semantic_tag="integer.compare.ne.u64",
        ),
    )


def _s_and_saveexec_b64_overlay(
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_and_saveexec_b64",
        instruction_name="S_AND_SAVEEXEC_B64",
        mnemonic="s_and_saveexec_b64",
        encoding_name="ENC_SOP1",
        encoding_condition=encoding_condition,
        semantic_tag="control.exec.and_save",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("saved_exec", units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("mask", units=2)),
        ),
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_clobber("exec_out"),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=False,
                is_output=True,
            ),
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_state_read(),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=True,
                is_output=False,
            ),
            _scc_output(_scc_result("active")),
        ),
        asm_forms=_asm(results=("saved_exec", "active"), operands=("mask",)),
        effects=(_CONVERGENT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_lshl_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshl_b32",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
    )


def _s_lshl_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
    )


def _s_lshl_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshl_b64",
        instruction_name="S_LSHL_B64",
        mnemonic="s_lshl_b64",
        semantic_tag="integer.shl.u64",
    )


def _s_lshr_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshr_b32",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
    )


def _s_lshr_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
    )


def _s_lshr_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshr_b64",
        instruction_name="S_LSHR_B64",
        mnemonic="s_lshr_b64",
        semantic_tag="integer.shr.u64",
    )


def _s_ashr_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_ashr_i32",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
        rhs_inline_descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
    )


def _s_ashr_i32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
    )


def _v_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_and_b32",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        literal_descriptor_key="amdgpu.v_and_b32.lit",
    )


def _v_and_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_and_b32.lit",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
    )


def _v_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_or_b32",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        literal_descriptor_key="amdgpu.v_or_b32.lit",
    )


def _v_or_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_or_b32.lit",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
    )


def _v_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_xor_b32",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        literal_descriptor_key="amdgpu.v_xor_b32.lit",
    )


def _v_xor_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_xor_b32.lit",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
    )


def _v_lshlrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        lhs_name="shift",
        rhs_name="value",
        literal_descriptor_key="amdgpu.v_lshlrev_b32.lit",
    )


def _v_lshlrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.lit",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
    )


def _v_lshlrev_b32_vop3_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("value")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshlrev_b32_vop3_imm",
            results=("dst",),
            operands=("value",),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshl_add_u32_shift_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshl_add_u32.shift_imm",
        instruction_name="V_LSHL_ADD_U32",
        mnemonic="v_lshl_add_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshl_add_u32_shift_imm",
            results=("dst",),
            operands=("value", "addend"),
            immediates=("shift",),
        ),
        immediate_fields=("SRC1",),
        immediates=(_source_inline_u32_immediate("shift"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshrrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        lhs_name="shift",
        rhs_name="value",
        literal_descriptor_key="amdgpu.v_lshrrev_b32.lit",
    )


def _v_lshrrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.lit",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
    )


def _v_ashrrev_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        lhs_name="shift",
        rhs_name="value",
        literal_descriptor_key="amdgpu.v_ashrrev_i32.lit",
    )


def _v_ashrrev_i32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.lit",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
    )


def _integer_bitwise_shift_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_and_b32_overlay(),
        _s_or_b32_overlay(),
        _s_xor_b32_overlay(),
        _s_and_b64_overlay(),
        _s_or_b64_overlay(),
        _s_xor_b64_overlay(),
        _s_lshl_b32_overlay(),
        _s_lshl_b32_rhs_inline_overlay(),
        _s_lshl_b64_overlay(),
        _s_lshr_b32_overlay(),
        _s_lshr_b32_rhs_inline_overlay(),
        _s_lshr_b64_overlay(),
        _s_ashr_i32_overlay(),
        _s_ashr_i32_rhs_inline_overlay(),
        _v_and_b32_overlay(),
        _v_and_b32_literal_overlay(),
        _v_or_b32_overlay(),
        _v_or_b32_literal_overlay(),
        _v_xor_b32_overlay(),
        _v_xor_b32_literal_overlay(),
        _v_lshlrev_b32_overlay(),
        _v_lshlrev_b32_literal_overlay(),
        _v_lshlrev_b32_vop3_immediate_overlay(),
        _v_lshl_add_u32_shift_immediate_overlay(),
        _v_lshrrev_b32_overlay(),
        _v_lshrrev_b32_literal_overlay(),
        _v_ashrrev_i32_overlay(),
        _v_ashrrev_i32_literal_overlay(),
    )


def _v_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_f32",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.add.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_f32.lit",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_sub_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_f32",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.sub.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_sub_f32.lit",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_f32",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.mul.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_f32.lit",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_min_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_min_f32",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.minnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_min_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_min_f32.lit",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_max_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_max_f32",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.maxnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_max_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_max_f32.lit",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_fma_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f32",
        instruction_name="V_FMA_F32",
        mnemonic="v_fma_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_unary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP1",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_exp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_exp_f32",
        instruction_name="V_EXP_F32",
        mnemonic="v_exp_f32",
        semantic_tag="float.exp2.f32",
    )


def _v_sqrt_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_sqrt_f32",
        instruction_name="V_SQRT_F32",
        mnemonic="v_sqrt_f32",
        semantic_tag="float.sqrt.f32",
    )


def _v_rsq_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_rsq_f32",
        instruction_name="V_RSQ_F32",
        mnemonic="v_rsq_f32",
        semantic_tag="float.rsqrt.f32",
    )


def _v_rcp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_rcp_f32",
        instruction_name="V_RCP_F32",
        mnemonic="v_rcp_f32",
        semantic_tag="float.reciprocal.f32",
    )


def _v_cvt_f32_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_i32",
        instruction_name="V_CVT_F32_I32",
        mnemonic="v_cvt_f32_i32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.signed.i32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_f16",
        instruction_name="V_CVT_F32_F16",
        mnemonic="v_cvt_f32_f16",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f16.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0", _vgpr_operand("input", register_part=_REG_PART_VGPR_LOW16)
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f16_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f16_f32",
        instruction_name="V_CVT_F16_F32",
        mnemonic="v_cvt_f16_f32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f32.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST", _vgpr_result(register_part=_REG_PART_VGPR_LOW16)
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_u32",
        instruction_name="V_CVT_F32_U32",
        mnemonic="v_cvt_f32_u32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.unsigned.u32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_i32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_u32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_U32",
        mnemonic=f"v_cmp_{instruction_predicate}_u32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.u32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_inline_operand_forms(descriptor_key: str) -> tuple[OperandForm, ...]:
    return (
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src0_inline",
            source_operand="lhs",
            immediate_field="lhs",
        ),
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src1_inline",
            source_operand="rhs",
            immediate_field="rhs",
        ),
    )


def _v_cmp_32_source_overlay(
    *,
    predicate: str,
    instruction_suffix: str,
    semantic_suffix: str,
    type_suffix: str,
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "lhs", _vgpr_const_operand("lhs")),
        "src1": ("SRC1", "rhs", _vgpr_const_operand("rhs")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    descriptor_key = f"amdgpu.v_cmp_{predicate}_{type_suffix}.{literal_source}_inline"
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_{type_suffix.upper()}",
        mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.{type_suffix}.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}_{literal_source}_inline",
            results=("mask",),
            operands=tuple(field_name for _, field_name, _ in remaining_operands),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(_source_inline_u32_immediate(literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_32_source_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="i32",
            literal_source=literal_source,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_u32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_32_source_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="u32",
            literal_source=literal_source,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_f32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_f32",
        instruction_name=f"V_CMP_{instruction_suffix}_F32",
        mnemonic=f"v_cmp_{instruction_predicate}_f32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.f32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cmp_i32_overlay(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        _v_cmp_i32_overlay(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        _v_cmp_i32_overlay(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        _v_cmp_i32_overlay(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        _v_cmp_i32_overlay(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        _v_cmp_i32_overlay(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        _v_cmp_u32_overlay(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        _v_cmp_u32_overlay(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        _v_cmp_u32_overlay(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        _v_cmp_u32_overlay(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="oeq", instruction_suffix="EQ", semantic_suffix="oeq"
        ),
        _v_cmp_f32_overlay(
            predicate="ogt", instruction_suffix="GT", semantic_suffix="ogt"
        ),
        _v_cmp_f32_overlay(
            predicate="oge", instruction_suffix="GE", semantic_suffix="oge"
        ),
        _v_cmp_f32_overlay(
            predicate="olt", instruction_suffix="LT", semantic_suffix="olt"
        ),
        _v_cmp_f32_overlay(
            predicate="ole", instruction_suffix="LE", semantic_suffix="ole"
        ),
        _v_cmp_f32_overlay(
            predicate="one", instruction_suffix="LG", semantic_suffix="one"
        ),
        _v_cmp_f32_overlay(
            predicate="ord", instruction_suffix="O", semantic_suffix="ord"
        ),
        _v_cmp_f32_overlay(
            predicate="ueq", instruction_suffix="NLG", semantic_suffix="ueq"
        ),
        _v_cmp_f32_overlay(
            predicate="ugt", instruction_suffix="NLE", semantic_suffix="ugt"
        ),
        _v_cmp_f32_overlay(
            predicate="uge", instruction_suffix="NLT", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="ult", instruction_suffix="NGE", semantic_suffix="ult"
        ),
        _v_cmp_f32_overlay(
            predicate="ule", instruction_suffix="NGT", semantic_suffix="ule"
        ),
        _v_cmp_f32_overlay(
            predicate="une", instruction_suffix="NEQ", semantic_suffix="une"
        ),
        _v_cmp_f32_overlay(
            predicate="uno", instruction_suffix="U", semantic_suffix="uno"
        ),
    )


def _v_cndmask_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("false_value")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("true_value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src0_inline",
                source_operand="false_value",
                immediate_field="false_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src1_inline",
                source_operand="true_value",
                immediate_field="true_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src0_lit",
                source_operand="false_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src1_lit",
                source_operand="true_value",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_inline",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cndmask_b32_{literal_source}_inline",
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(_source_inline_u32_immediate(literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_literal_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, _, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_lit",
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_literal_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value"),
        "src1": ("SRC1", "true_value"),
    }
    inline_source = "src1" if literal_source == "src0" else "src0"
    literal_field, _ = source_fields[literal_source]
    inline_field, inline_operand = source_fields[inline_source]
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cndmask_b32.{literal_source}_lit_{inline_source}_inline"
        ),
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit_{inline_source}_inline",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("mask",),
            immediates=("imm32", inline_operand),
            named_immediates=True,
        ),
        immediate_fields=(inline_field,),
        immediates=(
            _LITERAL_U32_IMMEDIATE,
            replace(
                _source_inline_u32_immediate(inline_operand),
                encoding_field_id=amdgpu_encoding_field_id(inline_field),
            ),
        ),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cndmask_b32_overlay(),
        _v_cndmask_b32_source_inline_overlay("src0"),
        _v_cndmask_b32_source_inline_overlay("src1"),
        _v_cndmask_b32_source_literal_overlay("src0"),
        _v_cndmask_b32_source_literal_overlay("src1"),
        _v_cndmask_b32_literal_inline_overlay("src0"),
        _v_cndmask_b32_literal_inline_overlay("src1"),
    )


def _v_mov_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="VOP1_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_copy_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32_copy",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="register.copy.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src")),
        ),
        asm_forms=_asm(mnemonic="v_mov_b32_copy", results=("dst",), operands=("src",)),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_buffer_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_buffer_load_dword",
        instruction_name="S_BUFFER_LOAD_DWORD",
        mnemonic="s_buffer_load_dword",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result()),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b64",
    instruction_name: str = "S_BUFFER_LOAD_B64",
    mnemonic: str = "s_buffer_load_b64",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_load_dwordx2_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx2",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx2_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx2_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX2",
        mnemonic="s_load_dwordx2",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _s_load_dwordx4_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx4",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result(units=4)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx4_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dwordx4_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX4",
        mnemonic="s_load_dwordx4",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u128",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B128,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_B128_EFFECT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _s_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dword",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result()),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor="amdgpu.s_load_dword_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor="amdgpu.s_load_dword_offset_only"
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORD",
        mnemonic="s_load_dword",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_GLOBAL_LOAD_EFFECT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=() if fixed_soffset is not None else None,
    )


def _buffer_load_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_dword_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_load_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_load_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key="amdgpu.buffer_load_dword_off_zero",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        semantic_tag="memory.load.u32",
        payload_units=1,
        memory_effect=_GLOBAL_LOAD_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_b16_d16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_b16_d16",
        instruction_name="BUFFER_LOAD_SHORT_D16",
        mnemonic="buffer_load_short_d16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_read_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_u16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_u16",
        instruction_name="BUFFER_LOAD_USHORT",
        mnemonic="buffer_load_u16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.zero_extend",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_U16,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b64_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_load_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64_off_zero",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u64",
        payload_units=2,
        memory_effect=_GLOBAL_LOAD_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b128_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.load.u128",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=4)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B128,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_LOAD_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_load_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128_off_zero",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u128",
        payload_units=4,
        memory_effect=_GLOBAL_LOAD_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_dword_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_dword",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u32",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_store_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_store_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key="amdgpu.buffer_store_dword_off_zero",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        semantic_tag="memory.store.u32",
        payload_units=1,
        memory_effect=_GLOBAL_STORE_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_b16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b16",
        instruction_name="BUFFER_STORE_SHORT",
        mnemonic="buffer_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b64_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.store.u64",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=2)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY_B64,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_STORE_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_store_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64_off_zero",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u64",
        payload_units=2,
        memory_effect=_GLOBAL_STORE_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b128_off_zero",
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if off_zero_descriptor_key is not None:
        operand_forms = (
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.store.u128",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=4)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY_B128,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_STORE_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=operand_forms,
    )


def _buffer_store_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128_off_zero",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u128",
        payload_units=4,
        memory_effect=_GLOBAL_STORE_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_b16_memory_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _buffer_load_u16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
        _buffer_load_b16_d16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
        _buffer_store_b16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
        ),
    )


def _global_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    global_read_memory: AmdgpuImplicitOperandOverlay | None = None,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        global_read_memory or _ignore_global_read_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_load_b16_d16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_read_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_SHORT_D16",
        mnemonic="global_load_short_d16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_load_u16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _global_load_overlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_USHORT",
        mnemonic="global_load_u16",
        encoding_name=encoding_name,
        address_field_name=address_field_name,
        data_field_name=data_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        saddr_off=saddr_off,
        width_bits=16,
        units=1,
        address_units=address_units,
        implicit_m0=implicit_m0,
        global_read_memory=_IGNORE_GLOBAL_READ_MEMORY_U16,
        cache_fields=cache_fields,
    )


def _global_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_store_b16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_STORE_SHORT",
        mnemonic="global_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_load_lds_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 13,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand("ADDR", units=address_units, has_saddr=saddr_off is None),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_FLAT_GLBL",
        semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=operands,
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="legacy-lds-dma-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(_implicit_m0_input(),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_to_lds_effects(width_bits),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


_GLOBAL_LOAD_LDS_DWORD_VARIANTS = (("DWORD", "dword", 32),)

_GLOBAL_LOAD_LDS_GFX950_VARIANTS = (
    ("DWORD", "dword", 32),
    ("DWORDX3", "dwordx3", 96),
    ("DWORDX4", "dwordx4", 128),
)


def _global_load_lds_overlays(
    *,
    descriptor_key_suffix: str = "",
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    cache_fields: tuple[tuple[str, int], ...] = (),
    variants: tuple[tuple[str, str, int], ...] = _GLOBAL_LOAD_LDS_GFX950_VARIANTS,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _global_load_lds_overlay(
            descriptor_key=(
                f"amdgpu.global_load_lds_{mnemonic_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"GLOBAL_LOAD_LDS_{instruction_suffix}",
            mnemonic=f"global_load_lds_{mnemonic_suffix}",
            width_bits=width_bits,
            address_units=address_units,
            saddr_off=saddr_off,
            cache_fields=cache_fields,
        )
        for instruction_suffix, mnemonic_suffix, width_bits in variants
    )


def _global_b16_memory_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _global_load_u16_overlay(
            descriptor_key=f"amdgpu.global_load_u16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_load_b16_d16_overlay(
            descriptor_key=f"amdgpu.global_load_b16_d16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_store_b16_overlay(
            descriptor_key=f"amdgpu.global_store_b16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=store_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
    )


def _global_memory_overlays(
    *,
    instruction_suffixes: tuple[str, str, str],
    mnemonic_suffixes: tuple[str, str, str],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    widths = ((32, 1), (64, 2), (128, 4))
    return (
        *(
            _global_load_overlay(
                descriptor_key=(
                    f"amdgpu.global_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_LOAD_{instruction_suffix}",
                mnemonic=f"global_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                widths, instruction_suffixes, mnemonic_suffixes, strict=True
            )
        ),
        *(
            _global_store_overlay(
                descriptor_key=(
                    f"amdgpu.global_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_STORE_{instruction_suffix}",
                mnemonic=f"global_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                widths, instruction_suffixes, mnemonic_suffixes, strict=True
            )
        ),
    )


def _global_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            _global_addr_operand(
                address_field_name, units=address_units, has_saddr=saddr_off is None
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
    else:
        operands = (
            _global_addr_operand(
                address_field_name, units=address_units, has_saddr=saddr_off is None
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
        ignored_operands = (
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="no-return-global-atomic-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        )

    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        ignored_operands=ignored_operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=False
            ),
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=True
            ),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=counter_id),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _global_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str,
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("VDST", _vgpr_result()),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=2)),
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=(f"amdgpu.global_atomic_cmpswap_b32_rtn{descriptor_key_suffix}"),
        instruction_name="GLOBAL_ATOMIC_CMPSWAP",
        mnemonic="global_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name="FMT_NUM_U32", is_input=False
            ),
            _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _global_atomic_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    include_packed_half_add: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = [
        ("add_u32", "GLOBAL_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
        ("sub_u32", "GLOBAL_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
        ("min_i32", "GLOBAL_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
        ("max_i32", "GLOBAL_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
        ("min_u32", "GLOBAL_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
        ("max_u32", "GLOBAL_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
        ("and_b32", "GLOBAL_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
        ("or_b32", "GLOBAL_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
        ("xor_b32", "GLOBAL_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
        (
            "swap_b32",
            "GLOBAL_ATOMIC_SWAP_B32",
            "exchange.b32",
            "FMT_NUM_B32",
            False,
        ),
        ("add_f32", "GLOBAL_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
        ("min_f32", "GLOBAL_ATOMIC_MIN_F32", "minnum.f32", "FMT_NUM_F32", True),
        ("max_f32", "GLOBAL_ATOMIC_MAX_F32", "maxnum.f32", "FMT_NUM_F32", True),
    ]
    if include_packed_half_add:
        rows.extend(
            (
                (
                    "pk_add_f16",
                    "GLOBAL_ATOMIC_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "pk_add_bf16",
                    "GLOBAL_ATOMIC_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _global_atomic_overlay(
                    descriptor_key=(
                        f"amdgpu.global_atomic_{mnemonic_suffix}{descriptor_key_suffix}"
                    ),
                    instruction_name=instruction_name,
                    mnemonic=f"global_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    address_field_name=address_field_name,
                    data_field_name=data_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                    saddr_off=saddr_off,
                    address_units=address_units,
                )
            )
        overlays.append(
            _global_atomic_overlay(
                descriptor_key=(
                    f"amdgpu.global_atomic_{mnemonic_suffix}_rtn{descriptor_key_suffix}"
                ),
                instruction_name=instruction_name,
                mnemonic=f"global_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
                saddr_off=saddr_off,
                address_units=address_units,
            )
        )
    overlays.append(
        _global_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
            saddr_off=saddr_off,
            address_units=address_units,
            descriptor_key_suffix=descriptor_key_suffix,
        )
    )
    return tuple(overlays)


_FLAT_ATOMIC_GFX11_ROWS = (
    ("add_u32", "add_u32", "FLAT_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
    ("sub_u32", "sub_u32", "FLAT_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
    ("min_i32", "min_i32", "FLAT_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
    ("max_i32", "max_i32", "FLAT_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
    ("min_u32", "min_u32", "FLAT_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
    ("max_u32", "max_u32", "FLAT_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
    ("and_b32", "and_b32", "FLAT_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
    ("or_b32", "or_b32", "FLAT_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
    ("xor_b32", "xor_b32", "FLAT_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
    (
        "swap_b32",
        "swap_b32",
        "FLAT_ATOMIC_SWAP_B32",
        "exchange.b32",
        "FMT_NUM_B32",
        False,
    ),
    ("add_f32", "add_f32", "FLAT_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    (
        "min_f32",
        "min_f32",
        "FLAT_ATOMIC_MIN_F32",
        "minnum.f32",
        "FMT_NUM_F32",
        True,
    ),
    (
        "max_f32",
        "max_f32",
        "FLAT_ATOMIC_MAX_F32",
        "maxnum.f32",
        "FMT_NUM_F32",
        True,
    ),
)

_FLAT_ATOMIC_GFX12_ROWS = (
    *_FLAT_ATOMIC_GFX11_ROWS[:-2],
    (
        "pk_add_f16",
        "pk_add_f16",
        "FLAT_ATOMIC_PK_ADD_F16",
        "add.pk2.f16",
        "FMT_NUM_PK2_F16",
        True,
    ),
    (
        "pk_add_bf16",
        "pk_add_bf16",
        "FLAT_ATOMIC_PK_ADD_BF16",
        "add.pk2.bf16",
        "FMT_NUM_PK2_BF16",
        True,
    ),
    (
        "min_f32",
        "min_f32",
        "FLAT_ATOMIC_MIN_NUM_F32",
        "minnum.f32",
        "FMT_NUM_F32",
        True,
    ),
    (
        "max_f32",
        "max_f32",
        "FLAT_ATOMIC_MAX_NUM_F32",
        "maxnum.f32",
        "FMT_NUM_F32",
        True,
    ),
)

_FLAT_ATOMIC_GFX950_ROWS = (
    ("add_u32", "add", "FLAT_ATOMIC_ADD", "add.u32", "FMT_NUM_U32", True),
    ("sub_u32", "sub", "FLAT_ATOMIC_SUB", "sub.u32", "FMT_NUM_U32", True),
    ("min_i32", "smin", "FLAT_ATOMIC_SMIN", "min.i32", "FMT_NUM_I32", True),
    ("max_i32", "smax", "FLAT_ATOMIC_SMAX", "max.i32", "FMT_NUM_I32", True),
    ("min_u32", "umin", "FLAT_ATOMIC_UMIN", "min.u32", "FMT_NUM_U32", True),
    ("max_u32", "umax", "FLAT_ATOMIC_UMAX", "max.u32", "FMT_NUM_U32", True),
    ("and_b32", "and", "FLAT_ATOMIC_AND", "and.b32", "FMT_NUM_B32", True),
    ("or_b32", "or", "FLAT_ATOMIC_OR", "or.b32", "FMT_NUM_B32", True),
    ("xor_b32", "xor", "FLAT_ATOMIC_XOR", "xor.b32", "FMT_NUM_B32", True),
    ("swap_b32", "swap", "FLAT_ATOMIC_SWAP", "exchange.b32", "FMT_NUM_B32", False),
    ("add_f32", "add_f32", "FLAT_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    (
        "pk_add_f16",
        "pk_add_f16",
        "FLAT_ATOMIC_PK_ADD_F16",
        "add.pk2.f16",
        "FMT_NUM_PK2_F16",
        True,
    ),
    (
        "pk_add_bf16",
        "pk_add_bf16",
        "FLAT_ATOMIC_PK_ADD_BF16",
        "add.pk2.bf16",
        "FMT_NUM_PK2_BF16",
        True,
    ),
)


def _flat_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    allow_accumulator_operands: bool,
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    result_operand = (
        _vgpr_agpr_result() if allow_accumulator_operands else _vgpr_result()
    )
    data_operand = (
        _vgpr_agpr_operand("value")
        if allow_accumulator_operands
        else _vgpr_operand("value")
    )
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDST", result_operand),
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        )
    else:
        operands = (
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        )
        ignored_operands = (
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="no-return-flat-atomic-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        )

    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_generic_atomic_memory(
            data_format_name=data_format_name, is_input=False
        ),
    )
    if implicit_flat_scratch:
        implicit_operands += (_IGNORE_FLAT_SCRATCH_INPUT,)
    implicit_operands += (
        _ignore_generic_atomic_memory(data_format_name=data_format_name, is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    offset_immediate = (
        _signed_offset_immediate(offset_bit_width)
        if offset_signed
        else _offset_immediate(offset_bit_width)
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        ignored_operands=ignored_operands,
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            offset_immediate,
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_generic_atomic_effects(32, counter_id=counter_id),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _flat_atomic_cmpswap_overlay(
    *,
    instruction_name: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    allow_accumulator_operands: bool,
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    result_operand = (
        _vgpr_agpr_result() if allow_accumulator_operands else _vgpr_result()
    )
    data_operand = (
        _vgpr_agpr_operand("value", units=2)
        if allow_accumulator_operands
        else _vgpr_operand("value", units=2)
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_generic_atomic_memory(data_format_name="FMT_NUM_U32", is_input=False),
    )
    if implicit_flat_scratch:
        implicit_operands += (_IGNORE_FLAT_SCRATCH_INPUT,)
    implicit_operands += (
        _ignore_generic_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    offset_immediate = (
        _signed_offset_immediate(offset_bit_width)
        if offset_signed
        else _offset_immediate(offset_bit_width)
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.flat_atomic_cmpswap_b32_rtn",
        instruction_name=instruction_name,
        mnemonic="flat_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.generic.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=(
            AmdgpuOperandOverlay("VDST", result_operand),
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        ),
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            offset_immediate,
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_generic_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _flat_atomic_overlays(
    *,
    rows: tuple[tuple[str, str, str, str, str, bool], ...],
    cmpswap_instruction_name: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    implicit_flat_scratch: bool,
    implicit_m0: bool = False,
    allow_accumulator_operands: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        descriptor_suffix,
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _flat_atomic_overlay(
                    descriptor_key=f"amdgpu.flat_atomic_{descriptor_suffix}",
                    instruction_name=instruction_name,
                    mnemonic=f"flat_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.generic.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    address_field_name=address_field_name,
                    data_field_name=data_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    offset_signed=offset_signed,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                    implicit_flat_scratch=implicit_flat_scratch,
                    implicit_m0=implicit_m0,
                    allow_accumulator_operands=allow_accumulator_operands,
                )
            )
        overlays.append(
            _flat_atomic_overlay(
                descriptor_key=f"amdgpu.flat_atomic_{descriptor_suffix}_rtn",
                instruction_name=instruction_name,
                mnemonic=f"flat_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.generic.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                allow_accumulator_operands=allow_accumulator_operands,
            )
        )
    overlays.append(
        _flat_atomic_cmpswap_overlay(
            instruction_name=cmpswap_instruction_name,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            offset_signed=offset_signed,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
            implicit_flat_scratch=implicit_flat_scratch,
            implicit_m0=implicit_m0,
            allow_accumulator_operands=allow_accumulator_operands,
        )
    )
    return tuple(overlays)


def _buffer_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    constraints: tuple[Constraint, ...]
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = _DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS
    else:
        operands = (
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = ()

    operands += (
        AmdgpuOperandOverlay(resource_field_name, _sgpr_resource("resource", units=4)),
        _mubuf_vaddr_operand(),
        AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
    )
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=False
            ),
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=True
            ),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=counter_id),
        constraints=constraints,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_atomic_cmpswap_b32_rtn",
        instruction_name="BUFFER_ATOMIC_CMPSWAP",
        mnemonic="buffer_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", units=2),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name="FMT_NUM_U32", is_input=False
            ),
            _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        constraints=_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_atomic_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    include_packed_half_add: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = [
        ("add_u32", "BUFFER_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
        ("sub_u32", "BUFFER_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
        ("min_i32", "BUFFER_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
        ("max_i32", "BUFFER_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
        ("min_u32", "BUFFER_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
        ("max_u32", "BUFFER_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
        ("and_b32", "BUFFER_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
        ("or_b32", "BUFFER_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
        ("xor_b32", "BUFFER_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
        (
            "swap_b32",
            "BUFFER_ATOMIC_SWAP_B32",
            "exchange.b32",
            "FMT_NUM_B32",
            False,
        ),
        ("add_f32", "BUFFER_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
        ("min_f32", "BUFFER_ATOMIC_MIN_F32", "minnum.f32", "FMT_NUM_F32", True),
        ("max_f32", "BUFFER_ATOMIC_MAX_F32", "maxnum.f32", "FMT_NUM_F32", True),
    ]
    if include_packed_half_add:
        rows.extend(
            (
                (
                    "pk_add_f16",
                    "BUFFER_ATOMIC_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "pk_add_bf16",
                    "BUFFER_ATOMIC_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _buffer_atomic_overlay(
                    descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}",
                    instruction_name=instruction_name,
                    mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                )
            )
        overlays.append(
            _buffer_atomic_overlay(
                descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}_rtn",
                instruction_name=instruction_name,
                mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
            )
        )
    overlays.append(
        _buffer_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
        )
    )
    return tuple(overlays)


def _ds_read_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read_{suffix}",
        instruction_name=f"DS_READ_{suffix.upper()}",
        mnemonic=f"ds_read_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read_u16_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_u16",
        instruction_name="DS_READ_U16",
        mnemonic="ds_read_u16",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.u16",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_load_u16_d16_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.ds_load_u16_d16",
            instruction_name="DS_LOAD_U16_D16",
            mnemonic="ds_load_u16_d16",
            encoding_name=encoding_name,
            semantic_tag="memory.workgroup.load.u16.d16.low",
            schedule_class=_SCHEDULE_LDS_LOAD,
            operands=(
                AmdgpuOperandOverlay(
                    "VDST",
                    _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            ),
            implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
            immediates=(_ds_offset_immediate(),),
            fixed_encoding_fields=_ds_fixed_fields_without_offset1(
                fixed_encoding_fields
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("addr",),
                immediates=("offset",),
                named_immediates=True,
            ),
            effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.ds_load_u16_d16_hi",
            instruction_name="DS_LOAD_U16_D16_HI",
            mnemonic="ds_load_u16_d16_hi",
            encoding_name=encoding_name,
            semantic_tag="memory.workgroup.load.u16.d16.high",
            schedule_class=_SCHEDULE_LDS_LOAD,
            operands=(
                AmdgpuOperandOverlay(
                    "VDST",
                    _vgpr_result(register_part=_REG_PART_VGPR_HIGH16),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay(
                    "VDST",
                    Operand(
                        "src",
                        OperandRole.OPERAND,
                        _VGPR_ALT,
                        flags=(OperandFlag.IMPLICIT,),
                        register_part=_REG_PART_VGPR_LOW16,
                    ),
                    role_exception_reason=(
                        "the encoded destination register is also the tied "
                        "source value carrying the low 16 bits"
                    ),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            ),
            implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
            immediates=(_ds_offset_immediate(),),
            fixed_encoding_fields=_ds_fixed_fields_without_offset1(
                fixed_encoding_fields
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("src", "addr"),
                immediates=("offset",),
                named_immediates=True,
            ),
            effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
            constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    )


def _ds_write_b16_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_b16",
        instruction_name="DS_WRITE_B16",
        mnemonic="ds_write_b16",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.u16.low",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay(
                "DATA0",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
            ),
        ),
        implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=False),),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, 16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write_{suffix}",
        instruction_name=f"DS_WRITE_{suffix.upper()}",
        mnemonic=f"ds_write_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value", units=units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=False),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read2_{suffix}",
        instruction_name=f"DS_READ2_{suffix.upper()}",
        mnemonic=f"ds_read2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=value_units * 2)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=True),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, element_width_bits * 2),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write2_{suffix}",
        instruction_name=f"DS_WRITE2_{suffix.upper()}",
        mnemonic=f"ds_write2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value0", units=value_units)),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("value1", units=value_units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=False),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.WRITE, element_width_bits * 2),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...]
    if returns_old_value:
        operands = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    else:
        operands = (
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=operands,
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name=data_format_name
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name=data_format_name
            ),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_cmpstore_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_cmpst_rtn_b32",
        instruction_name="DS_CMPST_RTN_B32",
        mnemonic="ds_cmpst_rtn_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("expected")),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("replacement")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name="FMT_NUM_B32"
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name="FMT_NUM_B32"
            ),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
    include_packed_half_add: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = [
        ("ds_add_u32", "DS_ADD_U32", "add.u32", "FMT_NUM_U32", False),
        ("ds_sub_u32", "DS_SUB_U32", "sub.u32", "FMT_NUM_U32", False),
        ("ds_min_i32", "DS_MIN_I32", "min.i32", "FMT_NUM_I32", False),
        ("ds_max_i32", "DS_MAX_I32", "max.i32", "FMT_NUM_I32", False),
        ("ds_min_u32", "DS_MIN_U32", "min.u32", "FMT_NUM_U32", False),
        ("ds_max_u32", "DS_MAX_U32", "max.u32", "FMT_NUM_U32", False),
        ("ds_and_b32", "DS_AND_B32", "and.b32", "FMT_NUM_B32", False),
        ("ds_or_b32", "DS_OR_B32", "or.b32", "FMT_NUM_B32", False),
        ("ds_xor_b32", "DS_XOR_B32", "xor.b32", "FMT_NUM_B32", False),
        ("ds_add_f32", "DS_ADD_F32", "add.f32", "FMT_NUM_F32", False),
        ("ds_min_f32", "DS_MIN_F32", "minnum.f32", "FMT_NUM_F32", False),
        ("ds_max_f32", "DS_MAX_F32", "maxnum.f32", "FMT_NUM_F32", False),
        ("ds_add_rtn_u32", "DS_ADD_RTN_U32", "add.return.u32", "FMT_NUM_U32", True),
        ("ds_sub_rtn_u32", "DS_SUB_RTN_U32", "sub.return.u32", "FMT_NUM_U32", True),
        ("ds_min_rtn_i32", "DS_MIN_RTN_I32", "min.return.i32", "FMT_NUM_I32", True),
        ("ds_max_rtn_i32", "DS_MAX_RTN_I32", "max.return.i32", "FMT_NUM_I32", True),
        ("ds_min_rtn_u32", "DS_MIN_RTN_U32", "min.return.u32", "FMT_NUM_U32", True),
        ("ds_max_rtn_u32", "DS_MAX_RTN_U32", "max.return.u32", "FMT_NUM_U32", True),
        ("ds_and_rtn_b32", "DS_AND_RTN_B32", "and.return.b32", "FMT_NUM_B32", True),
        ("ds_or_rtn_b32", "DS_OR_RTN_B32", "or.return.b32", "FMT_NUM_B32", True),
        ("ds_xor_rtn_b32", "DS_XOR_RTN_B32", "xor.return.b32", "FMT_NUM_B32", True),
        (
            "ds_wrxchg_rtn_b32",
            "DS_WRXCHG_RTN_B32",
            "exchange.return.b32",
            "FMT_NUM_B32",
            True,
        ),
        ("ds_add_rtn_f32", "DS_ADD_RTN_F32", "add.return.f32", "FMT_NUM_F32", True),
        (
            "ds_min_rtn_f32",
            "DS_MIN_RTN_F32",
            "minnum.return.f32",
            "FMT_NUM_F32",
            True,
        ),
        (
            "ds_max_rtn_f32",
            "DS_MAX_RTN_F32",
            "maxnum.return.f32",
            "FMT_NUM_F32",
            True,
        ),
    ]
    if include_packed_half_add:
        rows.extend(
            (
                (
                    "ds_pk_add_f16",
                    "DS_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    False,
                ),
                (
                    "ds_pk_add_bf16",
                    "DS_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    False,
                ),
                (
                    "ds_pk_add_rtn_f16",
                    "DS_PK_ADD_RTN_F16",
                    "add.return.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "ds_pk_add_rtn_bf16",
                    "DS_PK_ADD_RTN_BF16",
                    "add.return.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays = [
        _ds_atomic_overlay(
            descriptor_key=f"amdgpu.{mnemonic}",
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            semantic_tag=f"memory.workgroup.atomic.{semantic_suffix}",
            data_format_name=data_format_name,
            returns_old_value=returns_old_value,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            mnemonic,
            instruction_name,
            semantic_suffix,
            data_format_name,
            returns_old_value,
        ) in rows
    ]
    overlays.append(
        _ds_atomic_cmpstore_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
    )
    return tuple(overlays)


def _ds_stride64_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_read2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_read2st64_{suffix}",
        instruction_name=f"DS_READ2ST64_{suffix.upper()}",
        mnemonic=f"ds_read2st64_{suffix}",
        semantic_tag=f"memory.workgroup.load2.stride64.u{element_width_bits}",
    )


def _ds_stride64_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_write2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_write2st64_{suffix}",
        instruction_name=f"DS_WRITE2ST64_{suffix.upper()}",
        mnemonic=f"ds_write2st64_{suffix}",
        semantic_tag=f"memory.workgroup.store2.stride64.u{element_width_bits}",
    )


def _ds_read_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_addtid_b32",
        instruction_name="DS_READ_ADDTID_B32",
        mnemonic="ds_read_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.addtid.u32",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=True),
            _implicit_m0_input(),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_addtid_b32",
        instruction_name="DS_WRITE_ADDTID_B32",
        mnemonic="ds_write_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.addtid.u32",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=False),
            _implicit_m0_input(),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_crosslane_offset_immediate() -> Immediate:
    return _ds_offset_immediate()


def _ds_swizzle_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_swizzle_b32",
        instruction_name="DS_SWIZZLE_B32",
        mnemonic="ds_swizzle_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.crosslane.swizzle.u32",
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_ds_crosslane_effects(32),
    )


def _ds_permute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_permute_b32",
    instruction_name: str = "DS_PERMUTE_B32",
    mnemonic: str = "ds_permute_b32",
    semantic_tag: str = "memory.crosslane.permute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_ds_crosslane_effects(32),
    )


def _ds_bpermute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_bpermute_b32",
    instruction_name: str = "DS_BPERMUTE_B32",
    mnemonic: str = "ds_bpermute_b32",
    semantic_tag: str = "memory.crosslane.bpermute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return _ds_permute_b32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_bpermute_fi_b32_overlay(
    *,
    encoding_name: str = "ENC_VDS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _ds_bpermute_b32_overlay(
        descriptor_key="amdgpu.ds_bpermute_fi_b32",
        instruction_name="DS_BPERMUTE_FI_B32",
        mnemonic="ds_bpermute_fi_b32",
        semantic_tag="memory.crosslane.bpermute.fi.u32",
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_crosslane_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
    include_fetch_invalid: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = (
        _ds_swizzle_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_permute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_bpermute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )
    if not include_fetch_invalid:
        return overlays
    return (
        *overlays,
        _ds_bpermute_fi_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _ds_transpose_read_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx950_ds_transpose_read_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b4",
            instruction_name="DS_READ_B64_TR_B4",
            mnemonic="ds_read_b64_tr_b4",
            semantic_tag="memory.workgroup.transpose.load.b4.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b96_tr_b6",
            instruction_name="DS_READ_B96_TR_B6",
            mnemonic="ds_read_b96_tr_b6",
            semantic_tag="memory.workgroup.transpose.load.b6.u96",
            width_bits=96,
            units=3,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b8",
            instruction_name="DS_READ_B64_TR_B8",
            mnemonic="ds_read_b64_tr_b8",
            semantic_tag="memory.workgroup.transpose.load.b8.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b16",
            instruction_name="DS_READ_B64_TR_B16",
            mnemonic="ds_read_b64_tr_b16",
            semantic_tag="memory.workgroup.transpose.load.b16.u64",
            width_bits=64,
            units=2,
        ),
    )


def _ds_memory_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
    include_packed_half_atomic_add: bool = False,
    include_u16_d16_loads: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    widths = ((32, 1), (64, 2), (96, 3), (128, 4))
    two_addr_widths = ((32, 1), (64, 2))
    two_addr_fixed_encoding_fields = _ds_fixed_fields_without_offset1(
        fixed_encoding_fields
    )
    return (
        _ds_read_u16_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            _ds_read_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        _ds_write_b16_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            _ds_write_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        *_ds_atomic_overlays(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
            include_packed_half_add=include_packed_half_atomic_add,
        ),
        *(
            _ds_load_u16_d16_overlays(
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            if include_u16_d16_loads
            else ()
        ),
        *(
            _ds_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        _ds_read_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_write_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _v_wmma_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    input_units: int,
    accumulator_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=input_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=input_units)),
            AmdgpuOperandOverlay(
                "SRC2", _vgpr_const_operand("acc", units=accumulator_units)
            ),
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 3),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _with_zero_accumulator_form(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[AmdgpuDescriptorOverlay, AmdgpuDescriptorOverlay]:
    if overlay.mnemonic is None:
        raise ValueError(
            f"WMMA descriptor '{overlay.descriptor_key}' needs a mnemonic for "
            "its zero-accumulator asm form"
        )
    zero_descriptor_key = f"{overlay.descriptor_key}.acc_zero"
    source_overlay = replace(
        overlay,
        operand_forms=(
            *overlay.operand_forms,
            OperandForm(
                replacement_descriptor=zero_descriptor_key,
                matches=(
                    OperandFormMatch(
                        source_operand="acc",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )

    zero_operands = tuple(
        operand_overlay
        for operand_overlay in overlay.operands
        if operand_overlay.descriptor_operand.field_name != "acc"
    )
    zero_result_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role is OperandRole.RESULT
    )
    zero_operand_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role
        in (OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE)
        and OperandFlag.IMPLICIT not in operand_overlay.descriptor_operand.flags
    )
    zero_overlay = replace(
        overlay,
        descriptor_key=zero_descriptor_key,
        operands=zero_operands,
        fixed_encoding_fields=(
            *overlay.fixed_encoding_fields,
            ("SRC2", _predefined("0")),
        ),
        constraints=(),
        operand_forms=(),
        asm_forms=_asm(
            mnemonic=f"{overlay.mnemonic}_acc_zero",
            results=zero_result_names,
            operands=zero_operand_names,
        ),
    )
    return source_overlay, zero_overlay


def _v_wmma_f32_16x16x16_f16_overlay(
    *, input_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_f16",
        instruction_name="V_WMMA_F32_16X16X16_F16",
        mnemonic="v_wmma_f32_16x16x16_f16",
        semantic_tag="matrix.wmma.f32.16x16x16.f16",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_f32_16x16x16_bf16_overlay(
    *, input_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_bf16",
        instruction_name="V_WMMA_F32_16X16X16_BF16",
        mnemonic="v_wmma_f32_16x16x16_bf16",
        semantic_tag="matrix.wmma.f32.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_f16_16x16x16_f16_overlay(
    *, input_units: int = 4, accumulator_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f16_16x16x16_f16",
        instruction_name="V_WMMA_F16_16X16X16_F16",
        mnemonic="v_wmma_f16_16x16x16_f16",
        semantic_tag="matrix.wmma.f16.16x16x16.f16",
        input_units=input_units,
        accumulator_units=accumulator_units,
    )


def _v_wmma_bf16_16x16x16_bf16_overlay(
    *, input_units: int = 4, accumulator_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_bf16_16x16x16_bf16",
        instruction_name="V_WMMA_BF16_16X16X16_BF16",
        mnemonic="v_wmma_bf16_16x16x16_bf16",
        semantic_tag="matrix.wmma.bf16.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=accumulator_units,
    )


def _v_wmma_f32_16x16x16_packed8_overlay(
    *, lhs_type: str, rhs_type: str, input_units: int = 2
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}",
        instruction_name=f"V_WMMA_F32_16X16X16_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}",
        semantic_tag=f"matrix.wmma.f32.16x16x16.{lhs_type}.{rhs_type}",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_i32_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    operand_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=8)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=operand_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=operand_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc", units=8)),
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 3),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_wmma_i32_16x16x16_iu8_overlay(
    *, operand_units: int = 2
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu8",
        instruction_name="V_WMMA_I32_16X16X16_IU8",
        mnemonic="v_wmma_i32_16x16x16_iu8",
        semantic_tag="matrix.wmma.i32.16x16x16.iu8",
        operand_units=operand_units,
    )


def _v_wmma_i32_16x16x16_iu4_overlay(
    *, operand_units: int = 1
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu4",
        instruction_name="V_WMMA_I32_16X16X16_IU4",
        mnemonic="v_wmma_i32_16x16x16_iu4",
        semantic_tag="matrix.wmma.i32.16x16x16.iu4",
        operand_units=operand_units,
    )


def _v_wmma_i32_16x16x32_iu4_overlay() -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x32_iu4",
        instruction_name="V_WMMA_I32_16X16X32_IU4",
        mnemonic="v_wmma_i32_16x16x32_iu4",
        semantic_tag="matrix.wmma.i32.16x16x32.iu4",
        operand_units=2,
    )


def _v_mfma_f32_16x16x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x16_f16",
        instruction_name="V_MFMA_F32_16X16X16_F16",
        mnemonic="v_mfma_f32_16x16x16_f16",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x16.f16",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mfma_f32_16x16x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x16_bf16",
        instruction_name="V_MFMA_F32_16X16X16_BF16",
        mnemonic="v_mfma_f32_16x16x16_bf16",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x16.bf16.1k",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_smfmac_f32_16x16x32_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_bf16",
        instruction_name="V_SMFMAC_F32_16X16X32_BF16",
        mnemonic="v_smfmac_f32_16x16x32_bf16",
        semantic_tag="matrix.smfmac.f32.16x16x32.bf16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_16x16x32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_f16",
        instruction_name="V_SMFMAC_F32_16X16X32_F16",
        mnemonic="v_smfmac_f32_16x16x32_f16",
        semantic_tag="matrix.smfmac.f32.16x16x32.f16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_bf16",
        instruction_name="V_SMFMAC_F32_32X32X16_BF16",
        mnemonic="v_smfmac_f32_32x32x16_bf16",
        semantic_tag="matrix.smfmac.f32.32x32x16.bf16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_f16",
        instruction_name="V_SMFMAC_F32_32X32X16_F16",
        mnemonic="v_smfmac_f32_32x32x16_f16",
        semantic_tag="matrix.smfmac.f32.32x32x16.f16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3P_MFMA",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_agpr_operand("acc", units=accumulator_units),
                role_exception_reason=_SMFMAC_VDST_ACCUMULATOR_REASON,
            ),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=lhs_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=rhs_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_operand("index")),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _vop3p_packed_dot_fixed_fields(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
    lhs_signed: bool = False,
    rhs_signed: bool = False,
) -> tuple[tuple[str, int], ...]:
    # Packed VOP3P dot instructions use high-half source selection as the
    # canonical no-op modifier spelling. IU dot instructions additionally use
    # NEG low bits as integer signedness selectors for src0/src1.
    neg = (1 if lhs_signed else 0) | (2 if rhs_signed else 0)
    fields = [(op_sel_hi_field, 0x7)]
    if neg != 0:
        fields.append(("NEG", neg))
    return tuple(fields)


def _v_dot2_f32_packed_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot2_f32_f16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_f16",
        instruction_name="V_DOT2_F32_F16",
        mnemonic="v_dot2_f32_f16",
        semantic_tag="dot.f16f16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot2_f32_bf16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_bf16",
        instruction_name="V_DOT2_F32_BF16",
        mnemonic="v_dot2_f32_bf16",
        semantic_tag="dot.bf16bf16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot4_i32_i8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT4_I32_IU8" if signedness_modifiers else "V_DOT4_I32_I8"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_i32_i8",
        instruction_name=instruction_name,
        mnemonic="v_dot4_i32_i8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s8s8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_i32_iu8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s8" if lhs_signed else "u8"
    rhs_tag = "s8" if rhs_signed else "u8"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_i32_iu8.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT4_I32_IU8",
        mnemonic="v_dot4_i32_iu8",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot4_u32_u8_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_u32_u8",
        instruction_name="V_DOT4_U32_U8",
        mnemonic="v_dot4_u32_u8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u8u8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_f32_packed8_overlay(
    *, lhs_type: str, rhs_type: str
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_f32_{lhs_type}_{rhs_type}",
        instruction_name=f"V_DOT4_F32_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_dot4_f32_{lhs_type}_{rhs_type}",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_type}{rhs_type}.f32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_i4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT8_I32_IU4" if signedness_modifiers else "V_DOT8_I32_I4"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_i32_i4",
        instruction_name=instruction_name,
        mnemonic="v_dot8_i32_i4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s4s4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_iu4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s4" if lhs_signed else "u4"
    rhs_tag = "s4" if rhs_signed else "u4"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot8_i32_iu4.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT8_I32_IU4",
        mnemonic="v_dot8_i32_iu4",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot8_u32_u4_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_u32_u4",
        instruction_name="V_DOT8_U32_U4",
        mnemonic="v_dot8_u32_u4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u4u4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_waitcnt_overlay(
    *,
    effects: tuple[Effect, ...],
    lgkmcnt_immediate: Immediate = _LGKMCNT_6BIT_IMMEDIATE,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt",
        instruction_name="S_WAITCNT",
        mnemonic="s_waitcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt",
        schedule_class=_SCHEDULE_WAIT_MEMORY,
        operands=(),
        immediate_fields=("VM", "LGKM"),
        immediates=(_VMCNT_IMMEDIATE, lgkmcnt_immediate),
        fixed_encoding_fields=(("EXP", AmdgpuEncodingFieldAllOnes()),),
        effects=effects,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_depctr_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_depctr",
        instruction_name="S_WAITCNT_DEPCTR",
        mnemonic="s_waitcnt_depctr",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_vscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_vscnt",
        instruction_name="S_WAITCNT_VSCNT",
        mnemonic="s_waitcnt_vscnt",
        encoding_name="ENC_SOPK",
        semantic_tag="control.waitcnt.vmem_store",
        schedule_class=_SCHEDULE_WAIT_VMEM_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_VSCNT_IMMEDIATE,),
        fixed_encoding_fields=(("SDST", _predefined("NULL", "OPR_SDST")),),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_loadcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_loadcnt",
        instruction_name="S_WAIT_LOADCNT",
        mnemonic="s_wait_loadcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.load",
        schedule_class=_SCHEDULE_WAIT_LOAD,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_LOADCNT_IMMEDIATE,),
        effects=(_VMEM_LOAD_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_storecnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_storecnt",
        instruction_name="S_WAIT_STORECNT",
        mnemonic="s_wait_storecnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.store",
        schedule_class=_SCHEDULE_WAIT_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_STORECNT_IMMEDIATE,),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_dscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_dscnt",
        instruction_name="S_WAIT_DSCNT",
        mnemonic="s_wait_dscnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.lds",
        schedule_class=_SCHEDULE_WAIT_LDS,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DSCNT_IMMEDIATE,),
        effects=(_LDS_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_kmcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_kmcnt",
        instruction_name="S_WAIT_KMCNT",
        mnemonic="s_wait_kmcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.smem",
        schedule_class=_SCHEDULE_WAIT_SMEM,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_KMCNT_IMMEDIATE,),
        effects=(_SMEM_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_alu_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_alu",
        instruction_name="S_WAIT_ALU",
        mnemonic="s_wait_alu",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_idle_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_idle",
        instruction_name="S_WAIT_IDLE",
        mnemonic="s_wait_idle",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.idle",
        schedule_class=_SCHEDULE_WAIT_IDLE,
        operands=(),
        effects=(_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_barrier_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_barrier",
        instruction_name="S_BARRIER",
        mnemonic="s_barrier",
        encoding_name="ENC_SOPP",
        semantic_tag="control.barrier.workgroup",
        schedule_class=_SCHEDULE_BARRIER,
        operands=(),
        effects=(_WORKGROUP_BARRIER_EFFECT, _CONVERGENT_EFFECT),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _cache_control_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    semantic_tag: str,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=_cache_field_names(cache_fields),
        immediates=_cache_immediates(cache_fields),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_set_inst_prefetch_distance_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_set_inst_prefetch_distance",
        instruction_name="S_SET_INST_PREFETCH_DISTANCE",
        mnemonic="s_set_inst_prefetch_distance",
        encoding_name="ENC_SOPP",
        semantic_tag="memory.cache.prefetch.instruction.distance",
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_PREFETCH_DISTANCE_IMMEDIATE,),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_dcache_discard_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(
            AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_prefetch_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    base_operand: Operand | None,
    effect: Effect,
) -> AmdgpuDescriptorOverlay:
    operands = []
    if base_operand is not None:
        operands.append(AmdgpuOperandOverlay("SBASE", base_operand))
    operands.append(AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")))
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=tuple(operands),
        immediate_fields=("IOFFSET", "SDATA"),
        immediates=(
            _offset_immediate(24),
            _PREFETCH_COUNT_IMMEDIATE,
        ),
        effects=(effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx12_prefetch_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst",
            instruction_name="S_PREFETCH_INST",
            mnemonic="s_prefetch_inst",
            semantic_tag="memory.cache.prefetch.instruction",
            base_operand=_sgpr_operand("base", units=2),
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst_pc_rel",
            instruction_name="S_PREFETCH_INST_PC_REL",
            mnemonic="s_prefetch_inst_pc_rel",
            semantic_tag="memory.cache.prefetch.instruction.pc_relative",
            base_operand=None,
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data",
            instruction_name="S_PREFETCH_DATA",
            mnemonic="s_prefetch_data",
            semantic_tag="memory.cache.prefetch.data",
            base_operand=_sgpr_operand("base", units=2),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_buffer_prefetch_data",
            instruction_name="S_BUFFER_PREFETCH_DATA",
            mnemonic="s_buffer_prefetch_data",
            semantic_tag="memory.cache.prefetch.data.buffer",
            base_operand=_sgpr_resource("resource", units=4),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data_pc_rel",
            instruction_name="S_PREFETCH_DATA_PC_REL",
            mnemonic="s_prefetch_data_pc_rel",
            semantic_tag="memory.cache.prefetch.data.pc_relative",
            base_operand=None,
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
    )


def _gfx950_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_inv",
            instruction_name="BUFFER_INV",
            mnemonic="buffer_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_wbl2",
            instruction_name="BUFFER_WBL2",
            mnemonic="buffer_wbl2",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.writeback.buffer.l2",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb",
            instruction_name="S_DCACHE_WB",
            mnemonic="s_dcache_wb",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv_vol",
            instruction_name="S_DCACHE_INV_VOL",
            mnemonic="s_dcache_inv_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb_vol",
            instruction_name="S_DCACHE_WB_VOL",
            mnemonic="s_dcache_wb_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard",
            instruction_name="S_DCACHE_DISCARD",
            mnemonic="s_dcache_discard",
            semantic_tag="memory.cache.discard.data",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard_x2",
            instruction_name="S_DCACHE_DISCARD_X2",
            mnemonic="s_dcache_discard_x2",
            semantic_tag="memory.cache.discard.data.x2",
        ),
    )


def _gfx11_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl0_inv",
            instruction_name="BUFFER_GL0_INV",
            mnemonic="buffer_gl0_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl0",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl1_inv",
            instruction_name="BUFFER_GL1_INV",
            mnemonic="buffer_gl1_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_gl1_inv",
            instruction_name="S_GL1_INV",
            mnemonic="s_gl1_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.global.l1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_set_inst_prefetch_distance_overlay(),
    )


def _gfx12_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.global_inv",
            instruction_name="GLOBAL_INV",
            mnemonic="global_inv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wb",
            instruction_name="GLOBAL_WB",
            mnemonic="global_wb",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wbinv",
            instruction_name="GLOBAL_WBINV",
            mnemonic="global_wbinv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback_invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
    )


def _cdna_core_overlays(
    *,
    global_load_lds_variants: tuple[tuple[str, str, int], ...],
    include_v_dot2_f32_bf16: bool,
    include_ds_transpose_reads: bool,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("default"),
        _v_add_u32_overlay("V_ADD_U32"),
        _v_add_u32_literal_overlay("V_ADD_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(
            instruction_name="V_ADDC_CO_U32", mnemonic="v_addc_co_u32"
        ),
        _v_sub_u32_overlay("V_SUB_U32", "v_sub_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_fma_f32_overlay(),
        _v_exp_f32_overlay(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        _s_load_dword_overlay(fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS),
        _s_load_dwordx2_overlay(fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS),
        _s_load_dwordx4_overlay(fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS),
        _s_load_dword_overlay(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            descriptor_key="amdgpu.s_load_dword_offset_only",
            fixed_soffset=0,
        ),
        _s_load_dwordx2_overlay(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            descriptor_key="amdgpu.s_load_dwordx2_offset_only",
            fixed_soffset=0,
        ),
        _s_load_dwordx4_overlay(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            descriptor_key="amdgpu.s_load_dwordx4_offset_only",
            fixed_soffset=0,
        ),
        _s_buffer_load_dword_overlay(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        _s_buffer_load_64_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx2",
            instruction_name="S_BUFFER_LOAD_DWORDX2",
            mnemonic="s_buffer_load_dwordx2",
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS,
        ),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
        ),
        _buffer_load_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
        ),
        _buffer_load_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
        ),
        _buffer_store_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
        ),
        _buffer_store_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("DWORD", "DWORDX2", "DWORDX4"),
            mnemonic_suffixes=("dword", "dwordx2", "dwordx4"),
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("DWORD", "DWORDX2", "DWORDX4"),
            mnemonic_suffixes=("dword", "dwordx2", "dwordx4"),
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_load_lds_overlays(
            address_units=2,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_global_load_lds_overlays(
            address_units=1,
            saddr_off=None,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX950_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            offset_signed=False,
            return_field_name="SC0",
            return_field_value=1,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
            implicit_m0=True,
            allow_accumulator_operands=True,
        ),
        *_ds_memory_overlays(include_packed_half_atomic_add=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        *((_v_dot2_f32_bf16_overlay(),) if include_v_dot2_f32_bf16 else ()),
        _v_dot4_i32_i8_overlay(signedness_modifiers=False),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=False),
        _v_dot8_u32_u4_overlay(),
        *(_gfx950_ds_transpose_read_overlays() if include_ds_transpose_reads else ()),
        _v_mfma_f32_16x16x16_f16_overlay(),
        _v_mfma_f32_16x16x16_bf16_overlay(),
        _v_smfmac_f32_16x16x32_f16_overlay(),
        _v_smfmac_f32_16x16x32_bf16_overlay(),
        _v_smfmac_f32_32x32x16_f16_overlay(),
        _v_smfmac_f32_32x32x16_bf16_overlay(),
        _s_barrier_overlay(),
        *_gfx950_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _VMEM_STORE_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            ),
            lgkmcnt_immediate=_LGKMCNT_4BIT_IMMEDIATE,
        ),
    )


def _gfx940_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_DWORD_VARIANTS,
        include_v_dot2_f32_bf16=False,
        include_ds_transpose_reads=False,
    )


def _gfx950_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_GFX950_VARIANTS,
        include_v_dot2_f32_bf16=True,
        include_ds_transpose_reads=True,
    )


def _gfx940_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx940_core_overlays())
    )


def _gfx950_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx950_core_overlays())
    )


def _gfx11_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_fma_f32_overlay(),
        _v_exp_f32_overlay(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        _s_load_dword_overlay(),
        _s_load_dwordx2_overlay(),
        _s_load_dwordx4_overlay(),
        _s_load_dword_overlay(
            descriptor_key="amdgpu.s_load_dword_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx2_overlay(
            descriptor_key="amdgpu.s_load_dwordx2_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx4_overlay(
            descriptor_key="amdgpu.s_load_dwordx4_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_buffer_load_dword_overlay(),
        _s_buffer_load_64_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX11_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
        ),
        *_ds_memory_overlays(include_u16_d16_loads=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        _v_dot2_f32_bf16_overlay(),
        _v_dot4_i32_i8_overlay(signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot4_i32_iu8_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot8_i32_iu4_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot8_u32_u4_overlay(),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay(input_units=8)),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay(input_units=8)),
        *_with_zero_accumulator_form(
            _v_wmma_f16_16x16x16_f16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(
            _v_wmma_bf16_16x16x16_bf16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay(operand_units=4)),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay(operand_units=2)),
        _s_barrier_overlay(),
        *_gfx11_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            )
        ),
        _s_waitcnt_vscnt_overlay(),
        _s_waitcnt_depctr_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx11_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx11_core_overlays())
    )


def _gfx12_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_fma_f32_overlay(),
        _v_exp_f32_overlay(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        _s_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dwordx4_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dword_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dword_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx2_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dwordx2_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx4_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dwordx4_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            include_packed_half_add=True,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            include_packed_half_add=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX12_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            implicit_flat_scratch=False,
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
            include_packed_half_atomic_add=True,
            include_u16_d16_loads=True,
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot2_f32_f16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot2_f32_bf16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="bf8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="bf8"),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f16_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_bf16_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay()),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x32_iu4_overlay()),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx12_core_overlays())
    )


def _gfx1250_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_fma_f32_overlay(),
        _v_exp_f32_overlay(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        _s_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dwordx4_overlay("IOFFSET", offset_bit_width=24),
        _s_load_dword_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dword_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx2_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dwordx2_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_load_dwordx4_overlay(
            "IOFFSET",
            offset_bit_width=24,
            descriptor_key="amdgpu.s_load_dwordx4_offset_only",
            fixed_soffset=_predefined("NULL"),
        ),
        _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key=None,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            include_packed_half_add=True,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            include_packed_half_add=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX12_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            implicit_flat_scratch=False,
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
            include_packed_half_atomic_add=True,
            include_u16_d16_loads=True,
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot2_f32_f16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot2_f32_bf16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="bf8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="bf8"),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f16_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_bf16_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay()),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x32_iu4_overlay()),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx1250_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx1250_core_overlays())
    )


_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MFMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_MFMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_MFMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX950_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)


_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna3.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=102,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=512,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    register_parts=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
)


_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna3.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX11_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)

_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    register_parts=_VGPR_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_LDS_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_SMEM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_SMEM_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)

_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.rdna4.gfx125x.core",
    reg_classes=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.reg_classes,
    register_parts=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
    ),
    schedule_classes=(
        *_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
        ScheduleClass(
            _SCHEDULE_WMMA_SCALE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_SWMMAC),
            model_quality=ModelQuality.ESTIMATED,
        ),
    ),
    descriptors=(
        Descriptor(
            key="amdgpu.v_wmma_f32_16x16x32_f16",
            mnemonic="v_wmma_f32_16x16x32_f16",
            semantic_tag="matrix.wmma.f32.16x16x32.f16",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=8),
                _vgpr_operand("b", units=8),
                _vgpr_const_operand("acc", units=8),
            ),
            asm_forms=_asm(results=("dst",), operands=("a", "b", "acc")),
            schedule_class=_SCHEDULE_WMMA,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8",
            mnemonic="v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8",
            semantic_tag="matrix.wmma.scale.f32.16x16x128.f8f6f4.f8.f8",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=16),
                _vgpr_operand("b", units=16),
                _vgpr_const_operand("acc", units=8),
                _vgpr_operand("scale_src0", units=1),
                _vgpr_operand("scale_src1", units=1),
            ),
            immediates=(
                _MATRIX_A_FORMAT_IMMEDIATE,
                _MATRIX_B_FORMAT_IMMEDIATE,
                _MATRIX_A_SCALE_IMMEDIATE,
                _MATRIX_B_SCALE_IMMEDIATE,
                _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_A_REUSE_IMMEDIATE,
                _MATRIX_B_REUSE_IMMEDIATE,
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("a", "b", "acc", "scale_src0", "scale_src1"),
                immediates=(
                    "matrix_a_fmt",
                    "matrix_b_fmt",
                    "matrix_a_scale",
                    "matrix_b_scale",
                    "matrix_a_scale_fmt",
                    "matrix_b_scale_fmt",
                    "matrix_a_reuse",
                    "matrix_b_reuse",
                ),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_WMMA_SCALE,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8",
            mnemonic="v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8",
            semantic_tag="matrix.wmma.scale16.f32.16x16x128.f8f6f4.f8.f8",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=16),
                _vgpr_operand("b", units=16),
                _vgpr_const_operand("acc", units=8),
                _vgpr_operand("scale_src0", units=2),
                _vgpr_operand("scale_src1", units=2),
            ),
            immediates=(
                _MATRIX_A_FORMAT_IMMEDIATE,
                _MATRIX_B_FORMAT_IMMEDIATE,
                _MATRIX_A_SCALE_IMMEDIATE,
                _MATRIX_B_SCALE_IMMEDIATE,
                _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_A_REUSE_IMMEDIATE,
                _MATRIX_B_REUSE_IMMEDIATE,
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("a", "b", "acc", "scale_src0", "scale_src1"),
                immediates=(
                    "matrix_a_fmt",
                    "matrix_b_fmt",
                    "matrix_a_scale",
                    "matrix_b_scale",
                    "matrix_a_scale_fmt",
                    "matrix_b_scale_fmt",
                    "matrix_a_reuse",
                    "matrix_b_reuse",
                ),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_WMMA_SCALE,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_swmmac_f32_16x16x64_f16",
            mnemonic="v_swmmac_f32_16x16x64_f16",
            semantic_tag="matrix.swmmac.f32.16x16x64.f16",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("acc", units=8),
                _vgpr_operand("a", units=8),
                _vgpr_operand("b", units=16),
                _vgpr_operand("index", units=1),
            ),
            immediates=(_INDEX_KEY_16_IMMEDIATE,),
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
            asm_forms=_asm(
                results=("dst",),
                operands=("acc", "a", "b", "index"),
                immediates=("index_key_16bit",),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_SWMMAC,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
    ),
)


def _amdgpu_core_descriptor_set_bases() -> tuple[DescriptorSet, ...]:
    return (
        _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
    )


def _amdgpu_descriptor_ref_key_set() -> set[str]:
    keys: set[str] = set()
    keys.update(_MANUAL_SCALAR_DESCRIPTOR_KEYS)
    keys.update(descriptor.key for descriptor in _hal_buffer_descriptor_pseudos())
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        keys.update(descriptor.key for descriptor in descriptor_set.descriptors)
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        keys.update(overlay.descriptor_key for overlay in overlays)
    return keys


_ATOMIC_MEMORY_SPACE = {
    "workgroup": AmdgpuAtomicMemorySpace.WORKGROUP,
    "global": AmdgpuAtomicMemorySpace.GLOBAL,
    "generic": AmdgpuAtomicMemorySpace.GENERIC,
}

_ATOMIC_KIND = {
    "add.u32": (AmdgpuAtomicKind.ADDI, AmdgpuAtomicValueKind.I32),
    "sub.u32": (AmdgpuAtomicKind.SUBI, AmdgpuAtomicValueKind.I32),
    "min.i32": (AmdgpuAtomicKind.MINSI, AmdgpuAtomicValueKind.I32),
    "max.i32": (AmdgpuAtomicKind.MAXSI, AmdgpuAtomicValueKind.I32),
    "min.u32": (AmdgpuAtomicKind.MINUI, AmdgpuAtomicValueKind.I32),
    "max.u32": (AmdgpuAtomicKind.MAXUI, AmdgpuAtomicValueKind.I32),
    "and.b32": (AmdgpuAtomicKind.ANDI, AmdgpuAtomicValueKind.I32),
    "or.b32": (AmdgpuAtomicKind.ORI, AmdgpuAtomicValueKind.I32),
    "xor.b32": (AmdgpuAtomicKind.XORI, AmdgpuAtomicValueKind.I32),
    "exchange.b32": (AmdgpuAtomicKind.XCHGI, AmdgpuAtomicValueKind.I32),
    "add.f32": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.F32),
    "minnum.f32": (AmdgpuAtomicKind.MINNUMF, AmdgpuAtomicValueKind.F32),
    "maxnum.f32": (AmdgpuAtomicKind.MAXNUMF, AmdgpuAtomicValueKind.F32),
}

_ATOMIC_ADDRESS_FORM = (
    ("amdgpu.ds_", AmdgpuMemoryAddressForm.DEFAULT),
    ("amdgpu.buffer_atomic_", AmdgpuMemoryAddressForm.DEFAULT),
    ("amdgpu.global_atomic_", AmdgpuMemoryAddressForm.GLOBAL_SADDR),
    ("amdgpu.flat_atomic_", AmdgpuMemoryAddressForm.FLAT),
)

_ATOMIC_CANDIDATE_MEMORY_ORDER = {
    (
        AmdgpuAtomicMemorySpace.WORKGROUP,
        AmdgpuMemoryAddressForm.DEFAULT,
    ): 0,
    (
        AmdgpuAtomicMemorySpace.GLOBAL,
        AmdgpuMemoryAddressForm.DEFAULT,
    ): 1,
    (
        AmdgpuAtomicMemorySpace.GLOBAL,
        AmdgpuMemoryAddressForm.GLOBAL_SADDR,
    ): 2,
    (
        AmdgpuAtomicMemorySpace.GENERIC,
        AmdgpuMemoryAddressForm.FLAT,
    ): 3,
}

_ATOMIC_CANDIDATE_OPERATION_ORDER = {
    AmdgpuAtomicOperationKind.REDUCE: 0,
    AmdgpuAtomicOperationKind.RMW: 1,
    AmdgpuAtomicOperationKind.CMPXCHG: 2,
}

_ATOMIC_CANDIDATE_KIND_ORDER = {
    AmdgpuAtomicKind.ADDI: 0,
    AmdgpuAtomicKind.SUBI: 1,
    AmdgpuAtomicKind.MINSI: 2,
    AmdgpuAtomicKind.MAXSI: 3,
    AmdgpuAtomicKind.MINUI: 4,
    AmdgpuAtomicKind.MAXUI: 5,
    AmdgpuAtomicKind.ANDI: 6,
    AmdgpuAtomicKind.ORI: 7,
    AmdgpuAtomicKind.XORI: 8,
    AmdgpuAtomicKind.XCHGI: 9,
    AmdgpuAtomicKind.ADDF: 10,
    AmdgpuAtomicKind.MINNUMF: 11,
    AmdgpuAtomicKind.MAXNUMF: 12,
    AmdgpuAtomicKind.NONE: 13,
}


def _atomic_address_form(descriptor_key: str) -> AmdgpuMemoryAddressForm | None:
    for prefix, address_form in _ATOMIC_ADDRESS_FORM:
        if descriptor_key.startswith(prefix):
            return address_form
    return None


def _amdgpu_atomic_candidate_from_overlay(
    overlay: AmdgpuDescriptorOverlay,
) -> AmdgpuAtomicDescriptorCandidate | None:
    if ".atomic." not in overlay.semantic_tag:
        return None
    address_form = _atomic_address_form(overlay.descriptor_key)
    if address_form is None:
        return None
    if overlay.descriptor_key.startswith("amdgpu.global_atomic_") and not (
        overlay.descriptor_key.endswith("_saddr")
    ):
        return None

    tag_parts = overlay.semantic_tag.split(".")
    if len(tag_parts) < 4 or tag_parts[0] != "memory" or tag_parts[2] != "atomic":
        return None
    memory_space = _ATOMIC_MEMORY_SPACE.get(tag_parts[1])
    if memory_space is None:
        return None

    semantic_parts = tuple(part for part in tag_parts[3:] if part != "return")
    returns_old_value = "return" in tag_parts[3:]
    if semantic_parts == ("compare_exchange", "b32"):
        return AmdgpuAtomicDescriptorCandidate(
            memory_space=memory_space,
            address_form=address_form,
            operation_kind=AmdgpuAtomicOperationKind.CMPXCHG,
            atomic_kind=AmdgpuAtomicKind.NONE,
            value_kind=AmdgpuAtomicValueKind.I32,
            descriptor_key=overlay.descriptor_key,
        )

    atomic_kind = ".".join(semantic_parts)
    try:
        atomic_kind_enum, value_kind = _ATOMIC_KIND[atomic_kind]
    except KeyError:
        return None
    return AmdgpuAtomicDescriptorCandidate(
        memory_space=memory_space,
        address_form=address_form,
        operation_kind=(
            AmdgpuAtomicOperationKind.RMW
            if returns_old_value
            else AmdgpuAtomicOperationKind.REDUCE
        ),
        atomic_kind=atomic_kind_enum,
        value_kind=value_kind,
        descriptor_key=overlay.descriptor_key,
    )


def _amdgpu_atomic_candidate_sort_key(
    candidate: AmdgpuAtomicDescriptorCandidate,
) -> tuple[int, int, int, str]:
    return (
        _ATOMIC_CANDIDATE_MEMORY_ORDER[
            (candidate.memory_space, candidate.address_form)
        ],
        _ATOMIC_CANDIDATE_OPERATION_ORDER[candidate.operation_kind],
        _ATOMIC_CANDIDATE_KIND_ORDER[candidate.atomic_kind],
        candidate.descriptor_key,
    )


def amdgpu_atomic_descriptor_candidates() -> tuple[
    AmdgpuAtomicDescriptorCandidate, ...
]:
    """Returns source-to-low atomic descriptor candidates from AMDGPU metadata."""

    candidates_by_key: dict[str, AmdgpuAtomicDescriptorCandidate] = {}
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        for overlay in overlays:
            candidate = _amdgpu_atomic_candidate_from_overlay(overlay)
            if candidate is None:
                continue
            candidates_by_key.setdefault(candidate.descriptor_key, candidate)
    return tuple(
        sorted(candidates_by_key.values(), key=_amdgpu_atomic_candidate_sort_key)
    )


def _contract_overlay_builders_from_overlays(
    overlays: Sequence[AmdgpuDescriptorOverlay],
) -> dict[str, Callable[[], AmdgpuDescriptorOverlay]]:
    def _overlay_builder(
        overlay: AmdgpuDescriptorOverlay,
    ) -> Callable[[], AmdgpuDescriptorOverlay]:
        def build() -> AmdgpuDescriptorOverlay:
            return overlay

        return build

    return {overlay.descriptor_key: _overlay_builder(overlay) for overlay in overlays}


_AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS: dict[
    str,
    Callable[[], AmdgpuDescriptorOverlay],
] = {
    "amdgpu.s_mov_b32": _s_mov_b32_contract_overlay,
    "amdgpu.s_add_u32": _s_add_u32_overlay,
    "amdgpu.s_sub_u32": _s_sub_u32_overlay,
    "amdgpu.s_mul_i32": _s_mul_i32_overlay,
    "amdgpu.s_mul_hi_u32": _s_mul_hi_u32_overlay,
    "amdgpu.s_min_i32": _s_min_i32_overlay,
    "amdgpu.s_max_i32": _s_max_i32_overlay,
    "amdgpu.s_min_u32": _s_min_u32_overlay,
    "amdgpu.s_max_u32": _s_max_u32_overlay,
    "amdgpu.s_cselect_b32": _s_cselect_b32_overlay,
    "amdgpu.v_mov_b32": _v_mov_b32_literal_overlay,
    "amdgpu.v_add_u32": lambda: _v_add_u32_overlay("V_ADD_NC_U32"),
    "amdgpu.v_add_u32.lit": lambda: _v_add_u32_literal_overlay("V_ADD_NC_U32"),
    "amdgpu.v_sub_u32": lambda: _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
    "amdgpu.v_mul_lo_u32": _v_mul_lo_u32_overlay,
    "amdgpu.v_mul_hi_u32": _v_mul_hi_u32_overlay,
    "amdgpu.v_mul_u32_u24": _v_mul_u32_u24_overlay,
    "amdgpu.v_mul_u32_u24.lit": _v_mul_u32_u24_literal_overlay,
    "amdgpu.v_mad_u32_u24": _v_mad_u32_u24_overlay,
    "amdgpu.v_mad_u32_u24.src0_lit": lambda: _v_mad_u32_u24_literal_overlay("src0"),
    "amdgpu.v_mad_u32_u24.src1_lit": lambda: _v_mad_u32_u24_literal_overlay("src1"),
    "amdgpu.v_mad_u32_u24.src2_lit": lambda: _v_mad_u32_u24_literal_overlay("src2"),
    "amdgpu.v_min_i32": _v_min_i32_overlay,
    "amdgpu.v_max_i32": _v_max_i32_overlay,
    "amdgpu.v_min_u32": _v_min_u32_overlay,
    "amdgpu.v_max_u32": _v_max_u32_overlay,
    "amdgpu.v_add_f32": _v_add_f32_overlay,
    "amdgpu.v_add_f32.lit": _v_add_f32_literal_overlay,
    "amdgpu.v_sub_f32": _v_sub_f32_overlay,
    "amdgpu.v_sub_f32.lit": _v_sub_f32_literal_overlay,
    "amdgpu.v_mul_f32": _v_mul_f32_overlay,
    "amdgpu.v_mul_f32.lit": _v_mul_f32_literal_overlay,
    "amdgpu.v_min_f32": _v_min_f32_overlay,
    "amdgpu.v_min_f32.lit": _v_min_f32_literal_overlay,
    "amdgpu.v_max_f32": _v_max_f32_overlay,
    "amdgpu.v_max_f32.lit": _v_max_f32_literal_overlay,
    "amdgpu.v_fma_f32": _v_fma_f32_overlay,
    "amdgpu.v_exp_f32": _v_exp_f32_overlay,
    "amdgpu.v_sqrt_f32": _v_sqrt_f32_overlay,
    "amdgpu.v_rsq_f32": _v_rsq_f32_overlay,
    "amdgpu.v_rcp_f32": _v_rcp_f32_overlay,
    "amdgpu.v_cvt_f32_f16": _v_cvt_f32_f16_overlay,
    "amdgpu.v_cvt_f16_f32": _v_cvt_f16_f32_overlay,
    "amdgpu.v_cvt_f32_i32": _v_cvt_f32_i32_overlay,
    "amdgpu.v_cvt_f32_u32": _v_cvt_f32_u32_overlay,
    "amdgpu.v_dot2_f32_f16": _v_dot2_f32_f16_overlay,
    "amdgpu.v_dot2_f32_bf16": _v_dot2_f32_bf16_overlay,
    "amdgpu.v_dot4_i32_i8": lambda: _v_dot4_i32_i8_overlay(
        signedness_modifiers=False,
    ),
    "amdgpu.v_dot4_u32_u8": _v_dot4_u32_u8_overlay,
    "amdgpu.v_dot4_i32_iu8.u8s8": lambda: _v_dot4_i32_iu8_overlay(
        lhs_signed=False,
        rhs_signed=True,
    ),
    "amdgpu.v_dot4_i32_iu8.s8u8": lambda: _v_dot4_i32_iu8_overlay(
        lhs_signed=True,
        rhs_signed=False,
    ),
    "amdgpu.v_dot8_i32_i4": lambda: _v_dot8_i32_i4_overlay(
        signedness_modifiers=False,
    ),
    "amdgpu.v_dot8_u32_u4": _v_dot8_u32_u4_overlay,
    "amdgpu.v_dot8_i32_iu4.s4u4": lambda: _v_dot8_i32_iu4_overlay(
        lhs_signed=True,
        rhs_signed=False,
    ),
    "amdgpu.v_dot8_i32_iu4.u4s4": lambda: _v_dot8_i32_iu4_overlay(
        lhs_signed=False,
        rhs_signed=True,
    ),
    "amdgpu.v_dot4_f32_fp8_bf8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="fp8",
        rhs_type="bf8",
    ),
    "amdgpu.v_dot4_f32_bf8_fp8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="bf8",
        rhs_type="fp8",
    ),
    "amdgpu.v_dot4_f32_fp8_fp8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="fp8",
        rhs_type="fp8",
    ),
    "amdgpu.v_dot4_f32_bf8_bf8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="bf8",
        rhs_type="bf8",
    ),
    **_contract_overlay_builders_from_overlays(_integer_bitwise_shift_overlays()),
    **_contract_overlay_builders_from_overlays(_s_cmp_i32_overlays()),
    **_contract_overlay_builders_from_overlays(_v_cmp_overlays()),
}


def build_amdgpu_contract_descriptor_set(
    *,
    key: str,
    descriptor_keys: Sequence[str],
) -> DescriptorSet:
    """Builds an XML-free descriptor projection for source-to-low contracts."""

    descriptors = []
    for descriptor_key in descriptor_keys:
        try:
            overlay = _AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS[descriptor_key]()
        except KeyError as exc:
            raise ValueError(
                f"AMDGPU contract descriptor '{descriptor_key}' is not registered"
            ) from exc
        descriptors.append(_amdgpu_contract_descriptor_from_overlay(overlay))
    return replace(
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        key=key,
        feature_key=None,
        c_header_path=Path("loom/src/loom/target/arch/amdgpu/target_refs.h"),
        c_source_path=Path("loom/src/loom/target/arch/amdgpu/target_refs.c"),
        header_guard="LOOM_TARGET_ARCH_AMDGPU_TARGET_REFS_H_",
        public_header="loom/target/arch/amdgpu/target_refs.h",
        function_name="loom_amdgpu_contract_descriptor_set",
        c_table_prefix=_amdgpu_contract_fragment_prefix(key),
        c_enum_prefix="LOOM_AMDGPU",
        descriptors=_categorize_amdgpu_descriptors(tuple(descriptors)),
    )


def _amdgpu_contract_fragment_prefix(key: str) -> str:
    parts = tuple(
        part for part in key.replace("_", ".").replace("-", ".").split(".") if part
    )
    if not parts:
        raise ValueError("AMDGPU contract descriptor set key must be non-empty")
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _amdgpu_contract_descriptor_from_overlay(
    overlay: AmdgpuDescriptorOverlay,
) -> Descriptor:
    operands = tuple(
        operand_overlay.descriptor_operand for operand_overlay in overlay.operands
    ) + tuple(
        implicit_overlay.descriptor_operand
        for implicit_overlay in overlay.implicit_operands
        if implicit_overlay.descriptor_operand is not None
    )
    descriptor = Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            operand for operand in operands if operand.role is OperandRole.RESULT
        )
        + tuple(
            operand for operand in operands if operand.role is not OperandRole.RESULT
        ),
        immediates=overlay.immediates,
        effects=overlay.effects,
        constraints=overlay.constraints,
        operand_forms=overlay.operand_forms,
        feature_mask_words=overlay.feature_mask_words,
        schedule_class=overlay.schedule_class,
        flags=overlay.flags,
    )
    return _with_execution_mask_state_read(descriptor)


def _descriptor_has_memory_effect(descriptor: Descriptor) -> bool:
    return any(
        effect.kind in (EffectKind.READ, EffectKind.WRITE)
        and effect.memory_space in (MemorySpace.GLOBAL, MemorySpace.WORKGROUP)
        for effect in descriptor.effects
    )


def _descriptor_address_offset_immediates(
    descriptor: Descriptor,
) -> tuple[Immediate, ...]:
    return tuple(
        immediate
        for immediate in descriptor.immediates
        if immediate.field_name in _ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES
    )


def _validate_address_immediate_units(descriptor_set: DescriptorSet) -> None:
    for descriptor in descriptor_set.descriptors:
        if not _descriptor_has_memory_effect(descriptor):
            continue
        offset_immediates = _descriptor_address_offset_immediates(descriptor)
        if not offset_immediates:
            continue
        for immediate in offset_immediates:
            if immediate.encoding_id not in _ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' immediate "
                    f"'{immediate.field_name}' has no address-unit encoding"
                )
        split_offset_immediates = tuple(
            immediate
            for immediate in offset_immediates
            if immediate.field_name in ("offset0", "offset1")
        )
        if split_offset_immediates:
            if len(split_offset_immediates) != 2:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has an "
                    "incomplete split address offset"
                )
            first_encoding_id = split_offset_immediates[0].encoding_id
            if any(
                immediate.encoding_id != first_encoding_id
                for immediate in split_offset_immediates[1:]
            ):
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has "
                    "inconsistent split address offset units"
                )


def amdgpu_descriptor_ref_keys() -> tuple[str, ...]:
    """Returns descriptor keys known to the AMDGPU target family."""

    return tuple(sorted(_amdgpu_descriptor_ref_key_set()))


def amdgpu_descriptor_id_keys() -> tuple[str, ...]:
    """Returns descriptor keys that still need stable-ID compatibility refs."""

    return amdgpu_descriptor_ref_keys()


def amdgpu_immediate_encoding_id_items() -> tuple[tuple[str, int], ...]:
    """Returns target-owned immediate encoding IDs used by AMDGPU descriptors."""

    return (
        ("address_offset_byte", _ADDRESS_OFFSET_BYTE_ENCODING_ID),
        ("address_offset_dword", _ADDRESS_OFFSET_DWORD_ENCODING_ID),
        ("address_offset_qword", _ADDRESS_OFFSET_QWORD_ENCODING_ID),
        ("address_offset_dword_stride64", _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID),
        ("address_offset_qword_stride64", _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID),
        ("address_offset_ds16", _ADDRESS_OFFSET_DS16_ENCODING_ID),
        ("source_inline_u32", _SOURCE_INLINE_U32_ENCODING_ID),
        ("wait_counter_vmem", _WAIT_COUNTER_VMEM_ENCODING_ID),
        ("wait_counter_lgkm", _WAIT_COUNTER_LGKM_ENCODING_ID),
        ("wait_counter_vmem_load", _WAIT_COUNTER_VMEM_LOAD_ENCODING_ID),
        ("wait_counter_vmem_store", _WAIT_COUNTER_VMEM_STORE_ENCODING_ID),
        ("wait_counter_lds", _WAIT_COUNTER_LDS_ENCODING_ID),
        ("wait_counter_smem", _WAIT_COUNTER_SMEM_ENCODING_ID),
        ("wait_counter_alu", _WAIT_COUNTER_ALU_ENCODING_ID),
    )


def amdgpu_common_reg_class_ids() -> tuple[tuple[str, int], ...]:
    """Returns descriptor-set-local register-class IDs shared by all AMDGPU sets."""

    result: list[tuple[str, int]] = []
    for reg_class_name in (_REG_SGPR, _REG_VGPR, _REG_SCC, _REG_EXEC):
        expected_reg_class_id: int | None = None
        for descriptor_set in _amdgpu_core_descriptor_set_bases():
            reg_class_id = next(
                i
                for i, reg_class in enumerate(descriptor_set.reg_classes)
                if reg_class.name == reg_class_name
            )
            if expected_reg_class_id is None:
                expected_reg_class_id = reg_class_id
            elif expected_reg_class_id != reg_class_id:
                raise ValueError(
                    f"AMDGPU common register class '{reg_class_name}' has "
                    "inconsistent descriptor-set-local IDs"
                )
        if expected_reg_class_id is None:
            raise ValueError(
                f"AMDGPU common register class '{reg_class_name}' is missing"
            )
        result.append((reg_class_name, expected_reg_class_id))
    return tuple(result)


def _amdgpu_descriptor_category(descriptor: Descriptor) -> DescriptorCategory:
    semantic_tag = descriptor.semantic_tag or ""
    if semantic_tag.startswith(("matrix.", "dot.")):
        return AMDGPU_MATRIX_DESCRIPTOR_CATEGORY
    if ".atomic." in semantic_tag:
        return AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("memory.cache."):
        return AMDGPU_CACHE_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("memory."):
        return AMDGPU_MEMORY_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith(("control.", "special.")):
        return AMDGPU_CONTROL_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith(("cmp.", "integer.compare.", "float.compare.")):
        return AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("select."):
        return AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("convert."):
        return AMDGPU_CONVERT_DESCRIPTOR_CATEGORY
    if descriptor.key.startswith("amdgpu.s_"):
        return AMDGPU_SCALAR_DESCRIPTOR_CATEGORY
    if descriptor.key.startswith("amdgpu.v_"):
        return AMDGPU_VECTOR_DESCRIPTOR_CATEGORY
    return AMDGPU_MISC_DESCRIPTOR_CATEGORY


def _categorize_amdgpu_descriptors(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        replace(descriptor, category=_amdgpu_descriptor_category(descriptor))
        for descriptor in descriptors
    )


def amdgpu_descriptor_category_groups(
    descriptors: tuple[Descriptor, ...],
) -> tuple[tuple[DescriptorCategory, tuple[Descriptor, ...]], ...]:
    """Groups AMDGPU descriptors by stable category while preserving order."""

    grouped: dict[DescriptorCategory, list[Descriptor]] = {
        category: [] for category in AMDGPU_DESCRIPTOR_CATEGORIES
    }
    for descriptor in descriptors:
        category = descriptor.category or _amdgpu_descriptor_category(descriptor)
        grouped[category].append(descriptor)
    return tuple(
        (category, tuple(grouped[category]))
        for category in AMDGPU_DESCRIPTOR_CATEGORIES
        if grouped[category]
    )


def _with_overlay_descriptors(
    base: DescriptorSet,
    spec: AmdgpuIsaFactSource,
    overlay_descriptors: tuple[Descriptor, ...],
) -> DescriptorSet:
    manual_descriptors = _manual_scalar_descriptors(spec)
    descriptor_set = replace(
        base,
        descriptors=_categorize_amdgpu_descriptors(
            (
                manual_descriptors[0],
                *overlay_descriptors,
                *manual_descriptors[1:],
                *_hal_buffer_descriptor_pseudos(),
                *base.descriptors,
            )
        ),
    )
    _validate_address_immediate_units(descriptor_set)
    return descriptor_set


@dataclass(frozen=True, slots=True)
class _AmdgpuCoreDescriptorSetBuilder:
    base: DescriptorSet
    overlay_descriptors: Callable[[AmdgpuIsaFactSource], tuple[Descriptor, ...]]


_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS = {
    "cdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx940_core_overlay_descriptors,
    ),
    "cdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx950_core_overlay_descriptors,
    ),
    "rdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx11_core_overlay_descriptors,
    ),
    "rdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx12_core_overlay_descriptors,
    ),
    "rdna4_gfx125x": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx1250_core_overlay_descriptors,
    ),
}

AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS = tuple(
    sorted(_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
)


def build_amdgpu_core_descriptor_set_from_spec(
    target: str,
    spec: AmdgpuIsaFactSource,
) -> DescriptorSet:
    try:
        builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS)
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target(target), spec
    )
    return _with_overlay_descriptors(
        builder.base,
        spec,
        builder.overlay_descriptors(spec),
    )


def build_amdgpu_core_descriptor_set(
    target: str,
    xml_path: str | Path,
) -> DescriptorSet:
    return build_amdgpu_core_descriptor_set_from_spec(
        target, parse_amdgpu_isa_xml_path(xml_path)
    )
