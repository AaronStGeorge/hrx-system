# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python source schema for target-low descriptor sets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loom.stable_id import stable_id_from_string

LOW_DESCRIPTOR_SET_ABI_VERSION = 17
LOW_DESCRIPTOR_ENCODING_ID_NONE = (2**16) - 1


def descriptor_stable_id(key: str) -> int:
    """Returns the deterministic non-zero 63-bit identity for a descriptor key."""
    return stable_id_from_string(key)


class CEnum(Enum):
    @property
    def c_name(self) -> str:
        return str(self.value)


class OperandRole(CEnum):
    RESULT = "LOOM_LOW_OPERAND_ROLE_RESULT"
    OPERAND = "LOOM_LOW_OPERAND_ROLE_OPERAND"
    OPERAND_RESULT = "LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT"
    PREDICATE = "LOOM_LOW_OPERAND_ROLE_PREDICATE"
    RESOURCE = "LOOM_LOW_OPERAND_ROLE_RESOURCE"
    IMPLICIT = "LOOM_LOW_OPERAND_ROLE_IMPLICIT"


class OperandFlag(CEnum):
    IMPLICIT = "LOOM_LOW_OPERAND_FLAG_IMPLICIT"
    TIED = "LOOM_LOW_OPERAND_FLAG_TIED"
    EARLY_CLOBBER = "LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER"
    OPTIONAL = "LOOM_LOW_OPERAND_FLAG_OPTIONAL"
    STATE_READ = "LOOM_LOW_OPERAND_FLAG_STATE_READ"
    STATE_WRITE = "LOOM_LOW_OPERAND_FLAG_STATE_WRITE"


class RegClassAltFlag(CEnum):
    PREFERRED = "LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED"
    IMMEDIATE = "LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE"
    PHYSICAL_ONLY = "LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY"


class RegClassFlag(CEnum):
    VIRTUAL_ONLY = "LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY"
    PHYSICAL = "LOOM_LOW_REG_CLASS_FLAG_PHYSICAL"
    REFERENCE = "LOOM_LOW_REG_CLASS_FLAG_REFERENCE"
    UNSPILLABLE = "LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE"


class SpillSlotSpace(CEnum):
    STACK = "LOOM_LOW_SPILL_SLOT_SPACE_STACK"
    SCRATCH = "LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH"
    PRIVATE = "LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE"
    LDS = "LOOM_LOW_SPILL_SLOT_SPACE_LDS"


class ImmediateKind(CEnum):
    SIGNED = "LOOM_LOW_IMMEDIATE_KIND_SIGNED"
    UNSIGNED = "LOOM_LOW_IMMEDIATE_KIND_UNSIGNED"
    ORDINAL = "LOOM_LOW_IMMEDIATE_KIND_ORDINAL"
    ENUM = "LOOM_LOW_IMMEDIATE_KIND_ENUM"


class ImmediateFlag(CEnum):
    SYMBOLIC = "LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC"
    RELATIVE = "LOOM_LOW_IMMEDIATE_FLAG_RELATIVE"
    DEFAULT_VALUE = "LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE"


class EffectKind(CEnum):
    READ = "LOOM_LOW_EFFECT_KIND_READ"
    WRITE = "LOOM_LOW_EFFECT_KIND_WRITE"
    CALL = "LOOM_LOW_EFFECT_KIND_CALL"
    BARRIER = "LOOM_LOW_EFFECT_KIND_BARRIER"
    COUNTER = "LOOM_LOW_EFFECT_KIND_COUNTER"
    CONVERGENT = "LOOM_LOW_EFFECT_KIND_CONVERGENT"
    CONTROL = "LOOM_LOW_EFFECT_KIND_CONTROL"


class MemorySpace(CEnum):
    NONE = "LOOM_LOW_MEMORY_SPACE_NONE"
    GENERIC = "LOOM_LOW_MEMORY_SPACE_GENERIC"
    GLOBAL = "LOOM_LOW_MEMORY_SPACE_GLOBAL"
    WORKGROUP = "LOOM_LOW_MEMORY_SPACE_WORKGROUP"
    STACK = "LOOM_LOW_MEMORY_SPACE_STACK"
    VM_REF = "LOOM_LOW_MEMORY_SPACE_VM_REF"
    WASM_MEMORY = "LOOM_LOW_MEMORY_SPACE_WASM_MEMORY"


class EffectFlag(CEnum):
    ORDERED = "LOOM_LOW_EFFECT_FLAG_ORDERED"
    DEPENDENCY = "LOOM_LOW_EFFECT_FLAG_DEPENDENCY"


class ConstraintKind(CEnum):
    TIED = "LOOM_LOW_CONSTRAINT_KIND_TIED"
    COMMUTABLE = "LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE"
    DESTRUCTIVE = "LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE"
    EARLY_CLOBBER = "LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER"
    REMATERIALIZABLE = "LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE"
    FOLDABLE = "LOOM_LOW_CONSTRAINT_KIND_FOLDABLE"


class LatencyKind(CEnum):
    EXACT = "LOOM_LOW_LATENCY_KIND_EXACT"
    ESTIMATE = "LOOM_LOW_LATENCY_KIND_ESTIMATE"
    VARIABLE = "LOOM_LOW_LATENCY_KIND_VARIABLE"


class ModelQuality(CEnum):
    EXACT = "LOOM_LOW_MODEL_QUALITY_EXACT"
    CALIBRATED = "LOOM_LOW_MODEL_QUALITY_CALIBRATED"
    ESTIMATED = "LOOM_LOW_MODEL_QUALITY_ESTIMATED"
    FALLBACK = "LOOM_LOW_MODEL_QUALITY_FALLBACK"


class ScheduleClassFlag(CEnum):
    MAY_LOAD = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD"
    MAY_STORE = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE"
    MAY_CALL = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL"
    CONTROL = "LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL"


class ResourceKind(CEnum):
    SCALAR_ALU = "LOOM_LOW_RESOURCE_KIND_SCALAR_ALU"
    VECTOR_ALU = "LOOM_LOW_RESOURCE_KIND_VECTOR_ALU"
    MATRIX = "LOOM_LOW_RESOURCE_KIND_MATRIX"
    LOAD = "LOOM_LOW_RESOURCE_KIND_LOAD"
    STORE = "LOOM_LOW_RESOURCE_KIND_STORE"
    CONTROL = "LOOM_LOW_RESOURCE_KIND_CONTROL"
    ADDRESS = "LOOM_LOW_RESOURCE_KIND_ADDRESS"


class HazardKind(CEnum):
    MIN_DISTANCE = "LOOM_LOW_HAZARD_KIND_MIN_DISTANCE"
    WAIT_COUNTER = "LOOM_LOW_HAZARD_KIND_WAIT_COUNTER"
    BYPASS = "LOOM_LOW_HAZARD_KIND_BYPASS"
    FUSION = "LOOM_LOW_HAZARD_KIND_FUSION"


class HazardReferenceKind(CEnum):
    RESOURCE = "LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE"
    COUNTER = "LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER"
    TARGET = "LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET"


class DescriptorFlag(CEnum):
    SIDE_EFFECTING = "LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING"
    TERMINATOR = "LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR"
    DEAD_REMOVABLE = "LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE"
    PSEUDO = "LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO"


@dataclass(frozen=True, slots=True)
class RegClass:
    name: str
    alloc_unit_bits: int
    spill_slot_space: SpillSlotSpace
    flags: tuple[RegClassFlag, ...] = ()
    target_bank_id: int = 0
    physical_count: int = 0
    alias_set_id: int = 0
    spill_class: str | None = None
    full_register_part_mask: int = 1


@dataclass(frozen=True, slots=True)
class RegisterPart:
    name: str
    reg_class: str
    mask: int


@dataclass(frozen=True, slots=True)
class RegClassAlt:
    reg_class: str | None
    flags: tuple[RegClassAltFlag, ...] = (RegClassAltFlag.PREFERRED,)


@dataclass(frozen=True, slots=True)
class Operand:
    field_name: str
    role: OperandRole
    reg_alts: tuple[RegClassAlt, ...]
    flags: tuple[OperandFlag, ...] = ()
    unit_count: int = 1
    encoding_field_id: int = 0
    data_format_id: int = 0
    register_part: str | None = None
    read_stage: int = 0
    ready_stage: int = 0


@dataclass(frozen=True, slots=True)
class ImmediateEncodingSlice:
    encoding_field_id: int
    source_bit_offset: int
    bit_count: int


@dataclass(frozen=True, slots=True)
class Immediate:
    field_name: str
    kind: ImmediateKind
    flags: tuple[ImmediateFlag, ...] = ()
    bit_width: int = 0
    encoding_field_id: int = 0
    encoding_slices: tuple[ImmediateEncodingSlice, ...] = ()
    enum_domain: str | None = None
    encoding_id: int = 0
    signed_min: int = 0
    unsigned_max: int = 0
    default_value: int = 0


@dataclass(frozen=True, slots=True)
class EncodingFieldValue:
    encoding_field_id: int
    value: int


@dataclass(frozen=True, slots=True)
class AsmImmediate:
    field_name: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class AsmForm:
    mnemonic: str | None = None
    results: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    immediates: tuple[AsmImmediate, ...] = ()


@dataclass(frozen=True, slots=True)
class EnumValue:
    token: str
    value: int


@dataclass(frozen=True, slots=True)
class EnumDomain:
    name: str
    values: tuple[EnumValue, ...]


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    memory_space: MemorySpace = MemorySpace.NONE
    scope_id: int = 0
    flags: tuple[EffectFlag, ...] = ()
    counter_id: int = 0
    width_bits: int = 0


@dataclass(frozen=True, slots=True)
class Constraint:
    kind: ConstraintKind
    lhs_operand_index: int
    rhs_operand_index: int | None = None
    flags: tuple[CEnum, ...] = ()


@dataclass(frozen=True, slots=True)
class IssueUse:
    resource: str
    cycles: int
    units: int
    stage: int = 0


@dataclass(frozen=True, slots=True)
class PressureDelta:
    reg_class: str
    delta: int


@dataclass(frozen=True, slots=True)
class Resource:
    name: str
    capacity_per_cycle: int
    kind: ResourceKind
    flags: tuple[CEnum, ...] = ()
    contention_group_id: int = 0


@dataclass(frozen=True, slots=True)
class Hazard:
    kind: HazardKind
    resource: str | None = None
    counter_id: int | None = None
    target_id: int | None = None
    producer_stage: int = 0
    consumer_stage: int = 0
    distance: int = 0
    flags: tuple[CEnum, ...] = ()


@dataclass(frozen=True, slots=True)
class ScheduleClass:
    name: str
    latency_kind: LatencyKind
    model_quality: ModelQuality
    latency_cycles: int = 0
    issue_uses: tuple[IssueUse, ...] = ()
    hazards: tuple[Hazard, ...] = ()
    flags: tuple[ScheduleClassFlag, ...] = ()
    pressure_deltas: tuple[PressureDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class Descriptor:
    key: str
    mnemonic: str | None
    semantic_tag: str | None
    operands: tuple[Operand, ...]
    schedule_class: str
    immediates: tuple[Immediate, ...] = ()
    encoding_field_values: tuple[EncodingFieldValue, ...] = ()
    asm_forms: tuple[AsmForm, ...] = ()
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    encoding_format_id: int = 0
    encoding_id: int = 0
    flags: tuple[DescriptorFlag, ...] = ()


@dataclass(frozen=True, slots=True)
class DescriptorSet:
    key: str
    target_key: str | None
    feature_key: str | None
    c_header_path: Path
    c_source_path: Path
    header_guard: str
    public_header: str
    function_name: str
    c_table_prefix: str
    c_enum_prefix: str
    generator_version: int
    reg_classes: tuple[RegClass, ...]
    resources: tuple[Resource, ...]
    schedule_classes: tuple[ScheduleClass, ...]
    descriptors: tuple[Descriptor, ...]
    register_parts: tuple[RegisterPart, ...] = ()
    enum_domains: tuple[EnumDomain, ...] = ()
