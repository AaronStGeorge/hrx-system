# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: target-low descriptor inputs -> dense C tables.

The generator consumes a rich, explicit Python schema and emits compact
runtime tables under loom/src/loom. The C build only sees dense .rodata
arrays; Python owns source readability, validation, allowlist closure, and
manifest emission.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from loom.gen.generated_file import line_comment_header
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    LOW_DESCRIPTOR_SET_ABI_VERSION,
    LOW_DESCRIPTOR_SET_ORDINAL_NONE,
    AsmForm,
    CEnum,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    Hazard,
    HazardReferenceKind,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandForm,
    OperandFormImmediateAction,
    OperandFormMatchKind,
    OperandRole,
    PressureDelta,
    RegClass,
    RegClassAltFlag,
    RegisterPart,
    Resource,
    ScheduleClass,
    descriptor_set_relative_name,
    descriptor_stable_id,
)


@dataclass(frozen=True, slots=True)
class DescriptorAllowlist:
    keys: tuple[str, ...] = ()
    semantic_tags: tuple[str, ...] = ()
    mnemonics: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not self.keys and not self.semantic_tags and not self.mnemonics


@dataclass(frozen=True, slots=True)
class GeneratedDescriptorSet:
    header: str
    source: str
    manifest_json: str


@dataclass(slots=True)
class _StringEntry:
    label: str
    value: str
    offset: int


@dataclass(slots=True)
class _StringPool:
    c_enum_prefix: str
    entries: list[_StringEntry] = field(default_factory=list)
    value_to_label: dict[str, str] = field(default_factory=dict)
    label_to_primary: dict[str, str] = field(default_factory=dict)
    next_offset: int = 0

    def intern(self, label: str, value: str) -> str:
        if len(value.encode()) > 255:
            raise ValueError(f"B-string '{value}' exceeds 255 bytes")
        label = _c_identifier(label)
        if label in self.label_to_primary:
            primary_label = self.label_to_primary[label]
            if self.entries_by_label[primary_label].value != value:
                raise ValueError(f"string label '{label}' was reused for different values")
            return primary_label
        if value in self.value_to_label:
            primary_label = self.value_to_label[value]
            self.label_to_primary[label] = primary_label
            return primary_label
        self.entries.append(_StringEntry(label, value, self.next_offset))
        self.value_to_label[value] = label
        self.label_to_primary[label] = label
        self.next_offset += len(value.encode()) + 1
        return label

    @property
    def entries_by_label(self) -> dict[str, _StringEntry]:
        return {entry.label: entry for entry in self.entries}

    def ref(self, label: str) -> str:
        primary_label = self.label_to_primary[_c_identifier(label)]
        return f"{self.c_enum_prefix}_STRING_{primary_label}"


@dataclass(slots=True)
class _CompiledDescriptorSet:
    spec: DescriptorSet
    descriptors: list[Descriptor]
    reg_classes: list[RegClass]
    register_parts: list[RegisterPart]
    resources: list[Resource]
    schedule_classes: list[ScheduleClass]
    enum_domains: list[EnumDomain]
    reg_class_ids: dict[str, int]
    register_part_ids: dict[str, int]
    resource_ids: dict[str, int]
    schedule_class_ids: dict[str, int]
    enum_domain_ids: dict[str, int]
    string_pool: _StringPool
    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]]
    operands: list[Operand]
    operand_alt_starts: list[int]
    immediates: list[Immediate]
    immediate_encoding_slices: list[ImmediateEncodingSlice]
    immediate_encoding_slice_starts: list[int]
    enum_values: list[EnumValue]
    immediate_enum_domain_ids: list[int | None]
    effects: list[Effect]
    constraints: list[Constraint]
    issue_uses: list[IssueUse]
    hazards: list[Hazard]
    pressure_deltas: list[PressureDelta]
    feature_mask_words: list[int]
    encoding_field_values: list[EncodingFieldValue]
    operand_forms: list[_CompiledOperandForm]
    operand_form_matches: list[_CompiledOperandFormMatch]
    operand_form_operand_indices: list[int]
    descriptor_rows: list[dict[str, int]]
    descriptor_refs: list[tuple[str, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[_CompiledAsmForm]
    asm_operand_indices: list[int]
    asm_immediates: list[_CompiledAsmImmediate]
    schedule_rows: list[dict[str, int]]
    enum_domain_rows: list[dict[str, int]]


@dataclass(frozen=True, slots=True)
class _DescriptorSetView:
    spec: DescriptorSet
    descriptor_ordinals: tuple[int, ...]
    descriptor_refs: list[tuple[str, int]]
    descriptor_rows: list[dict[str, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[_CompiledAsmForm]
    operand_forms: list[_CompiledOperandForm]
    uses_storage_descriptor_tables: bool
    uses_storage_asm_form_tables: bool
    uses_storage_operand_form_tables: bool

    @property
    def descriptor_count(self) -> int:
        return len(self.descriptor_ordinals)


@dataclass(frozen=True, slots=True)
class _CompiledAsmImmediate:
    immediate_index: int
    name_label: str | None
    name: str | None


@dataclass(slots=True)
class _CompiledAsmForm:
    descriptor_ordinal: int
    mnemonic_label: str
    mnemonic: str
    result_indices: tuple[int, ...]
    operand_indices: tuple[int, ...]
    immediates: tuple[_CompiledAsmImmediate, ...]
    result_index_start: int = 0
    operand_index_start: int = 0
    immediate_start: int = 0


@dataclass(frozen=True, slots=True)
class _CompiledOperandFormMatch:
    source_operand_index: int
    source_packet_operand_index: int
    match_kind: OperandFormMatchKind
    match_i64: int


@dataclass(slots=True)
class _CompiledOperandForm:
    replacement_descriptor_ordinal: int
    source_immediate_index: int
    replacement_immediate_index: int
    immediate_match_index: int
    immediate_action: OperandFormImmediateAction
    match_start: int
    match_count: int
    operand_map_start: int
    operand_map_count: int


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "empty"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.lower()


_T = TypeVar("_T")


def _dedupe_by_name(items: Sequence[_T], get_name: Callable[[_T], str]) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for item in items:
        name = get_name(item)
        if name in result:
            raise ValueError(f"duplicate low descriptor input name '{name}'")
        result[name] = item
    return result


def _flag_expr(flags: Iterable[CEnum]) -> str:
    names = [flag.c_name for flag in flags]
    return " | ".join(names) if names else "0"


def _optional_string_expr(string_pool: _StringPool, label: str | None) -> str:
    if label is None:
        return "LOOM_LOW_STRING_OFFSET_NONE"
    return string_pool.ref(label)


def _i64_literal(value: int) -> str:
    if value == -(1 << 63):
        return "INT64_MIN"
    if value < 0:
        return f"(-INT64_C({abs(value)}))"
    return f"INT64_C({value})"


def _u64_literal(value: int) -> str:
    return f"UINT64_C({value})"


def _hex_u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:x})"


def _u16_literal(value: int) -> str:
    return f"UINT16_C({value})"


def _descriptor_ref_constant_name(spec: DescriptorSet, descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(spec, descriptor_key)
    return f"{spec.c_enum_prefix}_DESCRIPTOR_REF_{_c_identifier(descriptor_name).upper()}"


def _descriptor_ref_define(spec: DescriptorSet, descriptor_key: str, descriptor_ordinal: int) -> str:
    return f"#define {_descriptor_ref_constant_name(spec, descriptor_key)} {descriptor_ordinal}u"


def _reg_class_id_constant_name(spec: DescriptorSet, reg_class_name: str) -> str:
    reg_class_name = descriptor_set_relative_name(spec, reg_class_name)
    return f"{spec.c_enum_prefix}_REG_CLASS_ID_{_c_identifier(reg_class_name).upper()}"


def _register_part_id_constant_name(spec: DescriptorSet, part_name: str) -> str:
    part_name = descriptor_set_relative_name(spec, part_name)
    return f"{spec.c_enum_prefix}_REGISTER_PART_ID_{_c_identifier(part_name).upper()}"


def _emit_id_enum(typedef_name: str, entries: Sequence[tuple[str, int]]) -> list[str]:
    if not entries:
        return []
    lines = [f"typedef enum {typedef_name}_e {{"]
    lines.extend(f"  {name} = {value}u," for name, value in entries)
    lines.append(f"}} {typedef_name}_t;")
    return lines


def _register_part_id_expr(compiled: _CompiledDescriptorSet, part_name: str | None) -> str:
    if part_name is None:
        return "LOOM_LOW_REGISTER_PART_NONE"
    return str(compiled.register_part_ids[part_name])


def _encoding_id_expr(value: int) -> str:
    return "LOOM_LOW_ID_NONE" if value == LOW_DESCRIPTOR_ENCODING_ID_NONE else str(value)


def _canonical_asm_form_ordinal_expr(value: int | None) -> str:
    if value is None:
        return "LOOM_LOW_ASM_FORM_ORDINAL_NONE"
    return str(value)


def _hazard_reference_kind(hazard: Hazard) -> HazardReferenceKind:
    if hazard.resource is not None:
        return HazardReferenceKind.RESOURCE
    if hazard.counter_id is not None:
        return HazardReferenceKind.COUNTER
    if hazard.target_id is not None:
        return HazardReferenceKind.TARGET
    raise ValueError("hazard has no reference")


def _hazard_reference_id(hazard: Hazard, resource_ids: dict[str, int]) -> int:
    if hazard.resource is not None:
        return resource_ids[hazard.resource]
    if hazard.counter_id is not None:
        return hazard.counter_id
    if hazard.target_id is not None:
        return hazard.target_id
    raise ValueError("hazard has no reference")


def _validate_u16(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"{description} does not fit u16")


def _validate_u16_table_count(count: int, description: str) -> None:
    if count > 0xFFFF:
        raise ValueError(f"{description} count does not fit u16 descriptor references")


def _validate_u64(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{description} does not fit u64")


def _validate_i64(value: int, description: str) -> None:
    if value < -(1 << 63) or value > (1 << 63) - 1:
        raise ValueError(f"{description} does not fit i64")


def _bit_mask(bit_count: int) -> int:
    if bit_count == 0:
        return 0
    return (1 << bit_count) - 1


def _validate_immediate_encoding(descriptor: Descriptor, immediate: Immediate) -> None:
    if immediate.encoding_field_id and immediate.encoding_slices:
        raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' uses both direct and sliced encoding fields")
    if not immediate.encoding_slices:
        return
    covered_bits = 0
    for slice_index, encoding_slice in enumerate(immediate.encoding_slices):
        slice_description = f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding slice {slice_index}"
        _validate_u16(
            encoding_slice.encoding_field_id,
            f"{slice_description} field id",
        )
        if encoding_slice.encoding_field_id == 0:
            raise ValueError(f"{slice_description} has field id zero")
        if encoding_slice.bit_count <= 0 or encoding_slice.bit_count > 64:
            raise ValueError(f"{slice_description} has invalid bit count {encoding_slice.bit_count}")
        if encoding_slice.source_bit_offset < 0 or encoding_slice.source_bit_offset > 255:
            raise ValueError(f"{slice_description} source bit offset does not fit u8")
        source_end = encoding_slice.source_bit_offset + encoding_slice.bit_count
        if source_end > immediate.bit_width:
            raise ValueError(f"{slice_description} source range [{encoding_slice.source_bit_offset}, {source_end}) exceeds {immediate.bit_width} bits")
        slice_bits = _bit_mask(encoding_slice.bit_count) << encoding_slice.source_bit_offset
        if covered_bits & slice_bits:
            raise ValueError(f"{slice_description} overlaps another slice")
        covered_bits |= slice_bits
    expected_bits = _bit_mask(immediate.bit_width)
    if covered_bits != expected_bits:
        raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding slices cover 0x{covered_bits:x} instead of 0x{expected_bits:x}")


def _hazard_reference_count(hazard: Hazard) -> int:
    return sum(reference is not None for reference in (hazard.resource, hazard.counter_id, hazard.target_id))


def _clang_format_source(source: str, assume_filename: Path) -> str:
    result = subprocess.run(
        ["clang-format", f"--assume-filename={assume_filename}"],
        input=source,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def _select_descriptors(spec: DescriptorSet, allowlist: DescriptorAllowlist | None) -> list[Descriptor]:
    key_map = {descriptor.key: descriptor for descriptor in spec.descriptors}
    semantic_map: dict[str, list[Descriptor]] = {}
    mnemonic_map: dict[str, list[Descriptor]] = {}
    for descriptor in spec.descriptors:
        if descriptor.semantic_tag is not None:
            semantic_map.setdefault(descriptor.semantic_tag, []).append(descriptor)
        if descriptor.mnemonic is not None:
            mnemonic_map.setdefault(descriptor.mnemonic, []).append(descriptor)

    if allowlist is None or allowlist.is_empty():
        return list(spec.descriptors)

    selected: dict[str, Descriptor] = {}
    for key in allowlist.keys:
        selected_descriptor = key_map.get(key)
        if selected_descriptor is None:
            raise ValueError(f"allowlist references unknown descriptor key '{key}'")
        selected[selected_descriptor.key] = selected_descriptor
    for semantic_tag in allowlist.semantic_tags:
        descriptors = semantic_map.get(semantic_tag)
        if descriptors is None:
            raise ValueError(f"allowlist references unknown semantic tag '{semantic_tag}'")
        for descriptor in descriptors:
            selected[descriptor.key] = descriptor
    for mnemonic in allowlist.mnemonics:
        descriptors = mnemonic_map.get(mnemonic)
        if descriptors is None:
            raise ValueError(f"allowlist references unknown mnemonic '{mnemonic}'")
        for descriptor in descriptors:
            selected[descriptor.key] = descriptor

    changed = True
    while changed:
        changed = False
        for descriptor in tuple(selected.values()):
            for operand_form in descriptor.operand_forms:
                replacement = key_map.get(operand_form.replacement_descriptor)
                if replacement is None:
                    raise ValueError(f"descriptor '{descriptor.key}' operand form references unknown replacement descriptor '{operand_form.replacement_descriptor}'")
                if replacement.key not in selected:
                    selected[replacement.key] = replacement
                    changed = True

    return [descriptor for descriptor in spec.descriptors if descriptor.key in selected]


def _validate_descriptor_operands(descriptor: Descriptor) -> int:
    result_count = 0
    seen_non_result = False
    for operand in descriptor.operands:
        if operand.role is OperandRole.OPERAND_RESULT:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint")
        is_result = operand.role is OperandRole.RESULT
        if is_result and seen_non_result:
            raise ValueError(f"descriptor '{descriptor.key}' has result operand '{operand.field_name}' after non-result operands")
        if is_result:
            result_count += 1
        else:
            seen_non_result = True
        if operand.role is OperandRole.IMPLICIT and OperandFlag.IMPLICIT not in operand.flags:
            raise ValueError(f"descriptor '{descriptor.key}' implicit operand '{operand.field_name}' must set the implicit flag")
        if not operand.reg_alts:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has no register-class alternatives")
        _validate_u16(
            operand.addressable_unit_count,
            f"descriptor '{descriptor.key}' operand '{operand.field_name}' addressable unit count",
        )
        is_explicit_packet_value = _operand_role_is_packet_input(operand.role) and OperandFlag.IMPLICIT not in operand.flags
        has_addressable_assignment = is_result or is_explicit_packet_value
        if operand.address_map_kind is OperandAddressMapKind.DIRECT:
            if operand.addressable_unit_count != 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' direct address map must not set an addressable unit count")
        elif operand.address_map_kind in (OperandAddressMapKind.LOW_SUBSET, OperandAddressMapKind.TARGET_STATE):
            if not has_addressable_assignment:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must apply to an explicit value operand")
            if operand.addressable_unit_count == 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must set an addressable unit count")
            if operand.addressable_unit_count < operand.unit_count:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map covers fewer units than the operand consumes")
            if not any(reg_alt.reg_class is not None for reg_alt in operand.reg_alts):
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map requires a concrete register-class alternative")
        else:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has unknown address map kind '{operand.address_map_kind}'")
        state_flags = {
            OperandFlag.STATE_READ,
            OperandFlag.STATE_WRITE,
        }.intersection(operand.flags)
        if state_flags:
            if len(operand.reg_alts) != 1:
                raise ValueError(f"descriptor '{descriptor.key}' state operand '{operand.field_name}' must name exactly one register-class alternative")
            if operand.reg_alts[0].reg_class is None:
                raise ValueError(f"descriptor '{descriptor.key}' state operand '{operand.field_name}' must name a concrete register class")
    return result_count


def _validate_register_part(part: RegisterPart) -> None:
    if part.mask == 0:
        raise ValueError(f"register part '{part.name}' has an empty mask")
    if part.mask < 0 or part.mask > (2**64) - 1:
        raise ValueError(f"register part '{part.name}' mask does not fit u64")


def _descriptor_has_tied_constraint(
    descriptor: Descriptor,
    result_index: int,
    operand_index: int,
) -> bool:
    for constraint in descriptor.constraints:
        if constraint.kind is not ConstraintKind.TIED:
            continue
        if constraint.lhs_operand_index != result_index:
            continue
        if constraint.rhs_operand_index != operand_index:
            continue
        return True
    return False


def _operand_role_is_packet_input(role: OperandRole) -> bool:
    return role in (
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _operands_may_share_encoding_field(
    descriptor: Descriptor,
    lhs_index: int,
    rhs_index: int,
) -> bool:
    lhs = descriptor.operands[lhs_index]
    rhs = descriptor.operands[rhs_index]
    if lhs.role is OperandRole.RESULT and _operand_role_is_packet_input(rhs.role):
        return _descriptor_has_tied_constraint(descriptor, lhs_index, rhs_index)
    if rhs.role is OperandRole.RESULT and _operand_role_is_packet_input(lhs.role):
        return _descriptor_has_tied_constraint(descriptor, rhs_index, lhs_index)
    return False


def _validate_descriptor_encoding_fields(descriptor: Descriptor) -> None:
    fixed_fields: set[int] = set()
    for field_value in descriptor.encoding_field_values:
        if field_value.encoding_field_id != 0:
            fixed_fields.add(field_value.encoding_field_id)
    for operand_index, operand in enumerate(descriptor.operands):
        if operand.encoding_field_id == 0:
            continue
        if operand.encoding_field_id in fixed_fields:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' shares fixed encoding field id {operand.encoding_field_id}")
        for previous_index, previous_operand in enumerate(descriptor.operands[:operand_index]):
            if previous_operand.encoding_field_id != operand.encoding_field_id:
                continue
            if _operands_may_share_encoding_field(
                descriptor,
                previous_index,
                operand_index,
            ):
                continue
            raise ValueError(
                f"descriptor '{descriptor.key}' operands '{previous_operand.field_name}' and '{operand.field_name}' share encoding field id {operand.encoding_field_id} without a tied constraint"
            )


def _validate_enum_domain(domain: EnumDomain) -> tuple[EnumValue, ...]:
    if not domain.values:
        raise ValueError(f"enum domain '{domain.name}' has no values")
    by_token: dict[str, EnumValue] = {}
    for value in domain.values:
        if value.token in by_token:
            raise ValueError(f"enum domain '{domain.name}' repeats token '{value.token}'")
        by_token[value.token] = value
        if value.value < -(1 << 63) or value.value > (1 << 63) - 1:
            raise ValueError(f"enum domain '{domain.name}' token '{value.token}' value does not fit i64")
    return tuple(sorted(domain.values, key=lambda value: value.token))


def _validate_immediate_default(descriptor: Descriptor, immediate: Immediate, enum_domains: dict[str, EnumDomain]) -> None:
    if ImmediateFlag.DEFAULT_VALUE not in immediate.flags:
        if immediate.default_value != 0:
            raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' has a default value without the default-value flag")
        return
    match immediate.kind:
        case ImmediateKind.SIGNED:
            maximum = min(immediate.unsigned_max, (1 << 63) - 1)
            if immediate.default_value < immediate.signed_min or immediate.default_value > maximum:
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is out of signed range")
        case ImmediateKind.UNSIGNED | ImmediateKind.ORDINAL:
            if immediate.default_value < 0 or immediate.default_value > immediate.unsigned_max:
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is out of unsigned range")
        case ImmediateKind.ENUM:
            assert immediate.enum_domain is not None
            domain = enum_domains[immediate.enum_domain]
            if all(value.value != immediate.default_value for value in domain.values):
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is not in enum domain '{domain.name}'")


def _asm_form_mnemonic(descriptor: Descriptor, asm_form: AsmForm) -> str:
    mnemonic = descriptor.mnemonic if asm_form.mnemonic is None else asm_form.mnemonic
    if mnemonic is None:
        raise ValueError(f"descriptor '{descriptor.key}' asm form must specify a mnemonic because the descriptor has no default mnemonic")
    if mnemonic == "":
        raise ValueError(f"descriptor '{descriptor.key}' asm form specifies an empty mnemonic")
    if len(mnemonic.encode()) > 255:
        raise ValueError(f"descriptor '{descriptor.key}' asm mnemonic '{mnemonic}' exceeds 255 bytes")
    return mnemonic


def _index_descriptor_fields(
    descriptor: Descriptor,
) -> tuple[dict[str, int], dict[str, int]]:
    operand_indices: dict[str, int] = {}
    for i, operand in enumerate(descriptor.operands):
        if operand.field_name in operand_indices:
            raise ValueError(f"descriptor '{descriptor.key}' operand field '{operand.field_name}' is duplicated and cannot be used by asm forms")
        operand_indices[operand.field_name] = i

    immediate_indices: dict[str, int] = {}
    for i, immediate in enumerate(descriptor.immediates):
        if immediate.field_name in immediate_indices:
            raise ValueError(f"descriptor '{descriptor.key}' immediate field '{immediate.field_name}' is duplicated and cannot be used by asm forms")
        immediate_indices[immediate.field_name] = i
    return operand_indices, immediate_indices


def _descriptor_packet_operand_indices(descriptor: Descriptor) -> tuple[int, ...]:
    return tuple(i for i, operand in enumerate(descriptor.operands) if _operand_role_is_packet_input(operand.role) and OperandFlag.IMPLICIT not in operand.flags)


def _immediate_accepts_i64_assignment(immediate: Immediate) -> bool:
    return immediate.kind in (
        ImmediateKind.SIGNED,
        ImmediateKind.UNSIGNED,
        ImmediateKind.ORDINAL,
        ImmediateKind.ENUM,
    )


def _immediate_accepts_i64_arithmetic(immediate: Immediate) -> bool:
    return immediate.kind in (
        ImmediateKind.SIGNED,
        ImmediateKind.UNSIGNED,
        ImmediateKind.ORDINAL,
    )


def _compile_operand_form(
    descriptor: Descriptor,
    descriptor_ordinals: dict[str, int],
    selected_descriptors: Sequence[Descriptor],
    operand_form: OperandForm,
    match_start: int,
    operand_map_start: int,
) -> tuple[
    _CompiledOperandForm,
    tuple[_CompiledOperandFormMatch, ...],
    tuple[int, ...],
]:
    operand_indices, immediate_indices = _index_descriptor_fields(descriptor)
    source_packet_indices = _descriptor_packet_operand_indices(descriptor)
    source_packet_index_by_operand_index = {operand_index: packet_index for packet_index, operand_index in enumerate(source_packet_indices)}
    compiled_matches: list[_CompiledOperandFormMatch] = []
    match_index_by_source_operand: dict[str, int] = {}
    for match in operand_form.matches:
        source_operand_index = operand_indices.get(match.source_operand)
        if source_operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form references unknown source operand '{match.source_operand}'")
        source_packet_operand_index = source_packet_index_by_operand_index.get(source_operand_index)
        if source_packet_operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form source '{match.source_operand}' is not a packet operand")
        _validate_i64(
            match.match_i64,
            f"descriptor '{descriptor.key}' operand form match value",
        )
        match_index_by_source_operand[match.source_operand] = len(compiled_matches)
        compiled_matches.append(
            _CompiledOperandFormMatch(
                source_operand_index=source_operand_index,
                source_packet_operand_index=source_packet_operand_index,
                match_kind=match.match_kind,
                match_i64=match.match_i64,
            )
        )

    replacement_ordinal = descriptor_ordinals[operand_form.replacement_descriptor]
    replacement = selected_descriptors[replacement_ordinal]
    _replacement_operand_indices, replacement_immediate_indices = _index_descriptor_fields(replacement)
    source_result_count = _validate_descriptor_operands(descriptor)
    replacement_result_count = _validate_descriptor_operands(replacement)
    if replacement_result_count != source_result_count:
        raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' must have the same result count")
    for i in range(source_result_count):
        source_result = descriptor.operands[i].field_name
        replacement_result = replacement.operands[i].field_name
        if replacement_result != source_result:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' result {i} is '{replacement_result}' instead of '{source_result}'")
    source_immediates = tuple(immediate.field_name for immediate in descriptor.immediates)
    replacement_immediates = tuple(immediate.field_name for immediate in replacement.immediates)
    source_immediate_indices = {field_name: i for i, field_name in enumerate(source_immediates)}
    replacement_immediate_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    source_immediate_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    immediate_match_index = LOW_DESCRIPTOR_SET_ORDINAL_NONE
    if operand_form.immediate_action is OperandFormImmediateAction.NONE:
        if operand_form.immediate_field is not None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form without an immediate action names immediate field '{operand_form.immediate_field}'")
        expected_replacement_immediates = source_immediates
    elif operand_form.immediate_action is OperandFormImmediateAction.SET_MATCHED_I64:
        if operand_form.immediate_field is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate field")
        if operand_form.immediate_source_operand is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate source")
        immediate_match_index = match_index_by_source_operand[operand_form.immediate_source_operand]
        replacement_immediate_index = replacement_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        if replacement_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has no immediate field '{operand_form.immediate_field}'")
        if not _immediate_accepts_i64_assignment(replacement.immediates[replacement_immediate_index]):
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' cannot hold an i64 value")
        if operand_form.immediate_field in immediate_indices:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement immediate '{operand_form.immediate_field}' already exists on the source descriptor")
        expected_replacement_immediates = tuple((*source_immediates, operand_form.immediate_field))
    elif operand_form.immediate_action is OperandFormImmediateAction.ADD_MATCHED_I64:
        if operand_form.immediate_field is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate field")
        if operand_form.immediate_source_operand is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form immediate action requires an immediate source")
        immediate_match_index = match_index_by_source_operand[operand_form.immediate_source_operand]
        source_immediate_index = source_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        replacement_immediate_index = replacement_immediate_indices.get(operand_form.immediate_field, LOW_DESCRIPTOR_SET_ORDINAL_NONE)
        if source_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form source has no immediate field '{operand_form.immediate_field}'")
        if replacement_immediate_index == LOW_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has no immediate field '{operand_form.immediate_field}'")
        if descriptor.immediates[source_immediate_index] != replacement.immediates[replacement_immediate_index]:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' has different metadata")
        if not _immediate_accepts_i64_arithmetic(replacement.immediates[replacement_immediate_index]):
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' immediate field '{operand_form.immediate_field}' cannot hold an i64 value")
        expected_replacement_immediates = source_immediates
    else:
        raise ValueError(f"descriptor '{descriptor.key}' operand form has unsupported immediate action {operand_form.immediate_action}")
    if replacement_immediates != expected_replacement_immediates:
        raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' has immediate fields {replacement_immediates!r}, expected {expected_replacement_immediates!r}")

    replacement_packet_indices = _descriptor_packet_operand_indices(replacement)
    source_packet_index_by_field = {descriptor.operands[operand_index].field_name: packet_index for packet_index, operand_index in enumerate(source_packet_indices)}
    operand_map: list[int] = []
    matched_source_operands = {match.source_operand for match in operand_form.matches}
    for replacement_operand_index in replacement_packet_indices:
        field_name = replacement.operands[replacement_operand_index].field_name
        if field_name in matched_source_operands:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' still consumes source operand '{field_name}'")
        source_packet_index = source_packet_index_by_field.get(field_name)
        if source_packet_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{replacement.key}' operand '{field_name}' does not exist in the source packet")
        operand_map.append(source_packet_index)

    return (
        _CompiledOperandForm(
            replacement_descriptor_ordinal=replacement_ordinal,
            source_immediate_index=source_immediate_index,
            replacement_immediate_index=replacement_immediate_index,
            immediate_match_index=immediate_match_index,
            immediate_action=operand_form.immediate_action,
            match_start=match_start,
            match_count=len(compiled_matches),
            operand_map_start=operand_map_start,
            operand_map_count=len(operand_map),
        ),
        tuple(compiled_matches),
        tuple(operand_map),
    )


def _validate_unique_asm_fields(descriptor: Descriptor, asm_form: AsmForm, mnemonic: str) -> None:
    seen_fields: set[str] = set()
    for field_name in (
        *asm_form.results,
        *asm_form.operands,
        *(immediate.field_name for immediate in asm_form.immediates),
    ):
        if field_name in seen_fields:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' references field '{field_name}' more than once")
        seen_fields.add(field_name)


def _compile_asm_form(
    string_pool: _StringPool,
    descriptor: Descriptor,
    descriptor_ordinal: int,
    asm_form: AsmForm,
    form_ordinal: int,
) -> _CompiledAsmForm:
    mnemonic = _asm_form_mnemonic(descriptor, asm_form)
    _validate_unique_asm_fields(descriptor, asm_form, mnemonic)
    operand_indices, immediate_indices = _index_descriptor_fields(descriptor)

    mnemonic_label = string_pool.intern(f"asm_mnemonic_{descriptor.key}_{form_ordinal}", mnemonic)

    result_indices = []
    for field_name in asm_form.results:
        operand_index = operand_indices.get(field_name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' result references unknown operand field '{field_name}'")
        operand = descriptor.operands[operand_index]
        if operand.role is not OperandRole.RESULT:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' result field '{field_name}' names a non-result operand")
        result_indices.append(operand_index)

    packet_operand_roles = {
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    }
    operand_order = []
    for field_name in asm_form.operands:
        operand_index = operand_indices.get(field_name)
        if operand_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand references unknown operand field '{field_name}'")
        operand = descriptor.operands[operand_index]
        if operand.role not in packet_operand_roles:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand field '{field_name}' does not name an explicit packet operand")
        operand_order.append(operand_index)

    immediate_order = []
    seen_immediate_names: set[str] = set()
    for immediate in asm_form.immediates:
        immediate_index = immediate_indices.get(immediate.field_name)
        if immediate_index is None:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate references unknown immediate field '{immediate.field_name}'")
        name_label = None
        if immediate.name is not None:
            if immediate.name == "":
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate '{immediate.field_name}' has an empty name")
            if len(immediate.name.encode()) > 255:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' immediate name '{immediate.name}' exceeds 255 bytes")
            if immediate.name in seen_immediate_names:
                raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' uses immediate name '{immediate.name}' more than once")
            seen_immediate_names.add(immediate.name)
            name_label = string_pool.intern(
                f"asm_immediate_{descriptor.key}_{form_ordinal}_{immediate.field_name}",
                immediate.name,
            )
        immediate_order.append(
            _CompiledAsmImmediate(
                immediate_index=immediate_index,
                name_label=name_label,
                name=immediate.name,
            )
        )

    return _CompiledAsmForm(
        descriptor_ordinal=descriptor_ordinal,
        mnemonic_label=mnemonic_label,
        mnemonic=mnemonic,
        result_indices=tuple(result_indices),
        operand_indices=tuple(operand_order),
        immediates=tuple(immediate_order),
    )


def _compile_asm_forms(
    string_pool: _StringPool,
    descriptors: Sequence[Descriptor],
    *,
    allow_ambiguous_mnemonics: bool = False,
) -> list[_CompiledAsmForm]:
    compiled_forms: list[_CompiledAsmForm] = []
    seen_mnemonics: dict[str, str] = {}
    for descriptor_ordinal, descriptor in enumerate(descriptors):
        for form_ordinal, asm_form in enumerate(descriptor.asm_forms):
            compiled_form = _compile_asm_form(string_pool, descriptor, descriptor_ordinal, asm_form, form_ordinal)
            previous_descriptor_key = seen_mnemonics.get(compiled_form.mnemonic)
            if previous_descriptor_key is not None and not allow_ambiguous_mnemonics:
                raise ValueError(f"asm mnemonic '{compiled_form.mnemonic}' is ambiguous between descriptors '{previous_descriptor_key}' and '{descriptor.key}'")
            seen_mnemonics[compiled_form.mnemonic] = descriptor.key
            compiled_forms.append(compiled_form)
    return sorted(compiled_forms, key=lambda form: form.mnemonic)


def _compile_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
    *,
    allow_ambiguous_asm_mnemonics: bool = False,
) -> _CompiledDescriptorSet:
    if spec.generator_version == 0:
        raise ValueError(f"descriptor set '{spec.key}' has zero generator version")
    reg_class_inputs = _dedupe_by_name(spec.reg_classes, lambda item: item.name)
    register_part_inputs = _dedupe_by_name(spec.register_parts, lambda item: item.name)
    resource_inputs = _dedupe_by_name(spec.resources, lambda item: item.name)
    schedule_inputs = _dedupe_by_name(spec.schedule_classes, lambda item: item.name)
    enum_domain_inputs = _dedupe_by_name(spec.enum_domains, lambda item: item.name)
    _dedupe_by_name(spec.descriptors, lambda item: item.key)

    selected_descriptors = _select_descriptors(spec, allowlist)
    if not selected_descriptors:
        raise ValueError(f"descriptor set '{spec.key}' selected no descriptors")
    descriptor_ordinals = {descriptor.key: i for i, descriptor in enumerate(selected_descriptors)}
    for reg_class in spec.reg_classes:
        if reg_class.full_register_part_mask == 0:
            raise ValueError(f"register class '{reg_class.name}' has an empty full register-part mask")
        if reg_class.full_register_part_mask < 0 or reg_class.full_register_part_mask > (2**64) - 1:
            raise ValueError(f"register class '{reg_class.name}' full register-part mask does not fit u64")

    # Register classes are target vocabulary, not just descriptor closure:
    # low function signatures and allocation diagnostics may reference classes
    # that a tiny allowlisted descriptor slice does not happen to use.
    used_reg_class_names: set[str] = set(reg_class_inputs)
    used_register_part_names: set[str] = set()
    used_resource_names: set[str] = set()
    used_schedule_names: set[str] = set()
    used_enum_domain_names: set[str] = set()

    for descriptor in selected_descriptors:
        _validate_descriptor_operands(descriptor)
        _validate_descriptor_encoding_fields(descriptor)
        if descriptor.encoding_id < 0 or descriptor.encoding_id > LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' encoding id does not fit u16")
        if descriptor.encoding_id == LOW_DESCRIPTOR_ENCODING_ID_NONE and DescriptorFlag.PSEUDO not in descriptor.flags:
            raise ValueError(f"descriptor '{descriptor.key}' uses absent encoding id without the pseudo flag")
        if DescriptorFlag.PSEUDO in descriptor.flags and descriptor.encoding_id != LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"descriptor '{descriptor.key}' uses the pseudo flag with a target encoding id")
        if DescriptorFlag.PSEUDO in descriptor.flags and descriptor.encoding_format_id != 0:
            raise ValueError(f"descriptor '{descriptor.key}' uses the pseudo flag with a target encoding format id")
        if descriptor.schedule_class is None:
            raise ValueError(f"descriptor '{descriptor.key}' has no schedule class")
        if descriptor.schedule_class not in schedule_inputs:
            raise ValueError(f"descriptor '{descriptor.key}' references unknown schedule class '{descriptor.schedule_class}'")
        used_schedule_names.add(descriptor.schedule_class)
        for immediate in descriptor.immediates:
            _validate_u16(
                immediate.encoding_field_id,
                f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding field id",
            )
            _validate_u16(
                immediate.encoding_id,
                f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding id",
            )
            _validate_immediate_encoding(descriptor, immediate)
            if immediate.kind is ImmediateKind.ENUM:
                if immediate.enum_domain is None:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' has no enum domain")
                if immediate.enum_domain not in enum_domain_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' references unknown enum domain '{immediate.enum_domain}'")
                used_enum_domain_names.add(immediate.enum_domain)
            elif immediate.enum_domain is not None:
                raise ValueError(f"descriptor '{descriptor.key}' non-enum immediate '{immediate.field_name}' references enum domain '{immediate.enum_domain}'")
            _validate_immediate_default(descriptor, immediate, enum_domain_inputs)
        for operand in descriptor.operands:
            _validate_u16(
                operand.encoding_field_id,
                f"descriptor '{descriptor.key}' operand '{operand.field_name}' encoding field id",
            )
            concrete_reg_alt_count = 0
            for reg_alt in operand.reg_alts:
                if reg_alt.reg_class is None:
                    if RegClassAltFlag.IMMEDIATE not in reg_alt.flags:
                        raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has a classless alternative without the immediate flag")
                    continue
                if reg_alt.reg_class not in reg_class_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register class '{reg_alt.reg_class}'")
                used_reg_class_names.add(reg_alt.reg_class)
                concrete_reg_alt_count += 1
            if operand.register_part is not None:
                if operand.register_part not in register_part_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register part '{operand.register_part}'")
                if concrete_reg_alt_count != 1:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' with a register part must name exactly one concrete register class alternative")
                register_part = register_part_inputs[operand.register_part]
                _validate_register_part(register_part)
                if register_part.reg_class not in reg_class_inputs:
                    raise ValueError(f"register part '{register_part.name}' references unknown register class '{register_part.reg_class}'")
                if not any(reg_alt.reg_class == register_part.reg_class for reg_alt in operand.reg_alts):
                    raise ValueError(
                        f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses register part '{register_part.name}' for register class '{register_part.reg_class}' but the operand does not accept that class"
                    )
                used_register_part_names.add(operand.register_part)
                used_reg_class_names.add(register_part.reg_class)
        seen_fixed_encoding_fields: set[int] = set()
        for field_value in descriptor.encoding_field_values:
            _validate_u16(
                field_value.encoding_field_id,
                f"descriptor '{descriptor.key}' fixed encoding field id",
            )
            if field_value.encoding_field_id == 0:
                raise ValueError(f"descriptor '{descriptor.key}' has fixed encoding field id zero")
            if field_value.encoding_field_id in seen_fixed_encoding_fields:
                raise ValueError(f"descriptor '{descriptor.key}' repeats fixed encoding field id {field_value.encoding_field_id}")
            seen_fixed_encoding_fields.add(field_value.encoding_field_id)
            _validate_u64(
                field_value.value,
                f"descriptor '{descriptor.key}' fixed encoding field value",
            )

    for schedule_name in list(used_schedule_names):
        schedule_class = schedule_inputs[schedule_name]
        for issue_use in schedule_class.issue_uses:
            if issue_use.resource not in resource_inputs:
                raise ValueError(f"schedule class '{schedule_name}' references unknown resource '{issue_use.resource}'")
            used_resource_names.add(issue_use.resource)
        for hazard in schedule_class.hazards:
            if _hazard_reference_count(hazard) != 1:
                raise ValueError(f"schedule class '{schedule_name}' hazard must reference exactly one resource, counter, or target id")
            if hazard.resource is not None:
                if hazard.resource not in resource_inputs:
                    raise ValueError(f"schedule class '{schedule_name}' hazard references unknown resource '{hazard.resource}'")
                used_resource_names.add(hazard.resource)
            if hazard.counter_id is not None:
                _validate_u16(
                    hazard.counter_id,
                    f"schedule class '{schedule_name}' hazard counter id",
                )
            if hazard.target_id is not None:
                _validate_u16(
                    hazard.target_id,
                    f"schedule class '{schedule_name}' hazard target id",
                )
            _validate_u16(
                hazard.producer_stage,
                f"schedule class '{schedule_name}' hazard producer stage",
            )
            _validate_u16(
                hazard.consumer_stage,
                f"schedule class '{schedule_name}' hazard consumer stage",
            )
            _validate_u16(
                hazard.distance,
                f"schedule class '{schedule_name}' hazard distance",
            )
        for pressure_delta in schedule_class.pressure_deltas:
            if pressure_delta.reg_class not in reg_class_inputs:
                raise ValueError(f"schedule class '{schedule_name}' references unknown pressure register class '{pressure_delta.reg_class}'")
            used_reg_class_names.add(pressure_delta.reg_class)

    changed = True
    while changed:
        changed = False
        for reg_class_name in list(used_reg_class_names):
            spill_class = reg_class_inputs[reg_class_name].spill_class
            if spill_class is not None:
                if spill_class not in reg_class_inputs:
                    raise ValueError(f"register class '{reg_class_name}' references unknown spill class '{spill_class}'")
                if spill_class not in used_reg_class_names:
                    used_reg_class_names.add(spill_class)
                    changed = True

    for part_name in list(used_register_part_names):
        register_part = register_part_inputs[part_name]
        reg_class = reg_class_inputs[register_part.reg_class]
        if register_part.mask & ~reg_class.full_register_part_mask:
            raise ValueError(f"register part '{part_name}' mask 0x{register_part.mask:x} exceeds full mask 0x{reg_class.full_register_part_mask:x} for register class '{reg_class.name}'")

    reg_classes = [reg_class for reg_class in spec.reg_classes if reg_class.name in used_reg_class_names]
    register_parts = [part for part in spec.register_parts if part.name in used_register_part_names]
    resources = [resource for resource in spec.resources if resource.name in used_resource_names]
    schedule_classes = [schedule_class for schedule_class in spec.schedule_classes if schedule_class.name in used_schedule_names]
    enum_domains = [domain for domain in spec.enum_domains if domain.name in used_enum_domain_names]

    _validate_u16_table_count(len(reg_classes), f"descriptor set '{spec.key}' register class")
    _validate_u16_table_count(len(register_parts), f"descriptor set '{spec.key}' register part")
    reg_class_ids = {reg_class.name: i for i, reg_class in enumerate(reg_classes)}
    register_part_ids = {part.name: i for i, part in enumerate(register_parts)}
    resource_ids = {resource.name: i for i, resource in enumerate(resources)}
    schedule_class_ids = {schedule_class.name: i for i, schedule_class in enumerate(schedule_classes)}
    enum_domain_ids = {domain.name: i for i, domain in enumerate(enum_domains)}

    string_pool = _StringPool(spec.c_enum_prefix)
    string_pool.intern("empty", "")
    string_pool.intern("set_key", spec.key)
    if spec.target_key is not None:
        string_pool.intern("target_key", spec.target_key)
    if spec.feature_key is not None:
        string_pool.intern("feature_key", spec.feature_key)
    for reg_class in reg_classes:
        string_pool.intern(f"reg_{reg_class.name}", reg_class.name)
    for part in register_parts:
        string_pool.intern(f"register_part_{part.name}", part.name)
    for resource in resources:
        string_pool.intern(f"resource_{resource.name}", resource.name)
    for schedule_class in schedule_classes:
        string_pool.intern(f"schedule_{schedule_class.name}", schedule_class.name)
    for enum_domain in enum_domains:
        string_pool.intern(f"enum_domain_{enum_domain.name}", enum_domain.name)
        for enum_value in _validate_enum_domain(enum_domain):
            string_pool.intern(f"enum_value_{enum_domain.name}_{enum_value.token}", enum_value.token)
    for descriptor in selected_descriptors:
        string_pool.intern(f"descriptor_{descriptor.key}", descriptor.key)
        if descriptor.mnemonic is not None:
            string_pool.intern(f"mnemonic_{descriptor.key}", descriptor.mnemonic)
        if descriptor.semantic_tag is not None:
            string_pool.intern(f"semantic_{descriptor.key}", descriptor.semantic_tag)
        for operand in descriptor.operands:
            string_pool.intern(f"field_{operand.field_name}", operand.field_name)
        for immediate in descriptor.immediates:
            string_pool.intern(f"immediate_{immediate.field_name}", immediate.field_name)

    asm_forms = _compile_asm_forms(
        string_pool,
        selected_descriptors,
        allow_ambiguous_mnemonics=allow_ambiguous_asm_mnemonics,
    )
    canonical_asm_form_ordinals: list[int | None] = [None] * len(selected_descriptors)
    asm_form_counts_by_descriptor = [0] * len(selected_descriptors)
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        descriptor_ordinal = asm_form.descriptor_ordinal
        asm_form_counts_by_descriptor[descriptor_ordinal] += 1
        canonical_asm_form_ordinals[descriptor_ordinal] = asm_form_ordinal
    for descriptor_ordinal, form_count in enumerate(asm_form_counts_by_descriptor):
        if form_count != 1:
            canonical_asm_form_ordinals[descriptor_ordinal] = None

    asm_operand_indices: list[int] = []
    asm_immediates: list[_CompiledAsmImmediate] = []
    for asm_form in asm_forms:
        asm_form.result_index_start = len(asm_operand_indices)
        asm_operand_indices.extend(asm_form.result_indices)
        asm_form.operand_index_start = len(asm_operand_indices)
        asm_operand_indices.extend(asm_form.operand_indices)
        asm_form.immediate_start = len(asm_immediates)
        asm_immediates.extend(asm_form.immediates)

    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]] = []
    reg_alt_group_starts: dict[tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...], int] = {}
    immediate_encoding_slice_group_starts: dict[tuple[ImmediateEncodingSlice, ...], int] = {}
    effect_group_starts: dict[tuple[Effect, ...], int] = {}
    operands: list[Operand] = []
    operand_alt_starts: list[int] = []
    immediates: list[Immediate] = []
    immediate_encoding_slices: list[ImmediateEncodingSlice] = []
    immediate_encoding_slice_starts: list[int] = []
    enum_values: list[EnumValue] = []
    immediate_enum_domain_ids: list[int | None] = []
    effects: list[Effect] = []
    constraints: list[Constraint] = []
    issue_uses: list[IssueUse] = []
    hazards: list[Hazard] = []
    pressure_deltas: list[PressureDelta] = []
    feature_mask_words: list[int] = []
    encoding_field_values: list[EncodingFieldValue] = []
    operand_forms: list[_CompiledOperandForm] = []
    operand_form_matches: list[_CompiledOperandFormMatch] = []
    operand_form_operand_indices: list[int] = []
    descriptor_rows: list[dict[str, int]] = []
    schedule_rows: list[dict[str, int]] = []
    enum_domain_rows: list[dict[str, int]] = []

    for schedule_class in schedule_classes:
        issue_use_start = len(issue_uses)
        issue_uses.extend(schedule_class.issue_uses)
        hazard_start = len(hazards)
        hazards.extend(schedule_class.hazards)
        pressure_delta_start = len(pressure_deltas)
        pressure_deltas.extend(schedule_class.pressure_deltas)
        schedule_rows.append(
            {
                "issue_use_start": issue_use_start,
                "issue_use_count": len(schedule_class.issue_uses),
                "hazard_start": hazard_start,
                "hazard_count": len(schedule_class.hazards),
                "pressure_delta_start": pressure_delta_start,
                "pressure_delta_count": len(schedule_class.pressure_deltas),
            }
        )

    for enum_domain in enum_domains:
        value_start = len(enum_values)
        sorted_values = _validate_enum_domain(enum_domain)
        enum_values.extend(sorted_values)
        enum_domain_rows.append(
            {
                "value_start": value_start,
                "value_count": len(sorted_values),
            }
        )

    for descriptor in selected_descriptors:
        operand_start = len(operands)
        for operand in descriptor.operands:
            alt_group: tuple[tuple[int | None, tuple[RegClassAltFlag, ...]], ...] = tuple(
                (
                    None if reg_alt.reg_class is None else reg_class_ids[reg_alt.reg_class],
                    reg_alt.flags,
                )
                for reg_alt in operand.reg_alts
            )
            alt_start = reg_alt_group_starts.get(alt_group)
            if alt_start is None:
                alt_start = len(reg_class_alts)
                reg_alt_group_starts[alt_group] = alt_start
                reg_class_alts.extend(alt_group)
            operands.append(operand)
            operand_alt_starts.append(alt_start)
        immediate_start = len(immediates)
        for immediate in descriptor.immediates:
            slice_start = 0
            if immediate.encoding_slices:
                group_start = immediate_encoding_slice_group_starts.get(immediate.encoding_slices)
                if group_start is None:
                    group_start = len(immediate_encoding_slices)
                    immediate_encoding_slice_group_starts[immediate.encoding_slices] = group_start
                    immediate_encoding_slices.extend(immediate.encoding_slices)
                slice_start = group_start
            immediate_encoding_slice_starts.append(slice_start)
            immediates.append(immediate)
            immediate_enum_domain_ids.append(None if immediate.enum_domain is None else enum_domain_ids[immediate.enum_domain])
        if descriptor.effects:
            effect_start = effect_group_starts.get(descriptor.effects)
            if effect_start is None:
                effect_start = len(effects)
                effect_group_starts[descriptor.effects] = effect_start
                effects.extend(descriptor.effects)
        else:
            effect_start = 0
        constraint_start = len(constraints)
        constraints.extend(descriptor.constraints)
        feature_mask_word_start = len(feature_mask_words)
        feature_mask_words.extend(descriptor.feature_mask_words)
        encoding_field_value_start = len(encoding_field_values)
        encoding_field_values.extend(descriptor.encoding_field_values)
        operand_form_start = len(operand_forms)
        for operand_form in descriptor.operand_forms:
            if operand_form.replacement_descriptor not in descriptor_ordinals:
                raise ValueError(f"descriptor '{descriptor.key}' operand form replacement '{operand_form.replacement_descriptor}' was not selected")
            compiled_form, compiled_matches, operand_map = _compile_operand_form(
                descriptor,
                descriptor_ordinals,
                selected_descriptors,
                operand_form,
                len(operand_form_matches),
                len(operand_form_operand_indices),
            )
            operand_forms.append(compiled_form)
            operand_form_matches.extend(compiled_matches)
            operand_form_operand_indices.extend(operand_map)
        descriptor_rows.append(
            {
                "operand_start": operand_start,
                "operand_count": len(descriptor.operands),
                "result_count": _validate_descriptor_operands(descriptor),
                "immediate_start": immediate_start,
                "immediate_count": len(descriptor.immediates),
                "effect_start": effect_start,
                "effect_count": len(descriptor.effects),
                "constraint_start": constraint_start,
                "constraint_count": len(descriptor.constraints),
                "feature_mask_word_start": feature_mask_word_start,
                "feature_mask_word_count": len(descriptor.feature_mask_words),
                "encoding_field_value_start": encoding_field_value_start,
                "encoding_field_value_count": len(descriptor.encoding_field_values),
                "operand_form_start": operand_form_start,
                "operand_form_count": len(descriptor.operand_forms),
            }
        )

    descriptor_refs = sorted((descriptor.key, i) for i, descriptor in enumerate(selected_descriptors))
    seen_stable_ids: dict[int, str] = {}
    for descriptor in selected_descriptors:
        stable_id = descriptor_stable_id(descriptor.key)
        previous_key = seen_stable_ids.get(stable_id)
        if previous_key is not None:
            raise ValueError(f"descriptor '{descriptor.key}' stable ID collides with '{previous_key}'")
        seen_stable_ids[stable_id] = descriptor.key

    return _CompiledDescriptorSet(
        spec=spec,
        descriptors=selected_descriptors,
        reg_classes=reg_classes,
        register_parts=register_parts,
        resources=resources,
        schedule_classes=schedule_classes,
        enum_domains=enum_domains,
        reg_class_ids=reg_class_ids,
        register_part_ids=register_part_ids,
        resource_ids=resource_ids,
        schedule_class_ids=schedule_class_ids,
        enum_domain_ids=enum_domain_ids,
        string_pool=string_pool,
        reg_class_alts=reg_class_alts,
        operands=operands,
        operand_alt_starts=operand_alt_starts,
        immediates=immediates,
        immediate_encoding_slices=immediate_encoding_slices,
        immediate_encoding_slice_starts=immediate_encoding_slice_starts,
        enum_values=enum_values,
        immediate_enum_domain_ids=immediate_enum_domain_ids,
        effects=effects,
        constraints=constraints,
        issue_uses=issue_uses,
        hazards=hazards,
        pressure_deltas=pressure_deltas,
        feature_mask_words=feature_mask_words,
        encoding_field_values=encoding_field_values,
        operand_forms=operand_forms,
        operand_form_matches=operand_form_matches,
        operand_form_operand_indices=operand_form_operand_indices,
        descriptor_rows=descriptor_rows,
        descriptor_refs=descriptor_refs,
        canonical_asm_form_ordinals=canonical_asm_form_ordinals,
        asm_forms=asm_forms,
        asm_operand_indices=asm_operand_indices,
        asm_immediates=asm_immediates,
        schedule_rows=schedule_rows,
        enum_domain_rows=enum_domain_rows,
    )


def _emit_header_for_spec(
    compiled: _CompiledDescriptorSet,
    header_spec: DescriptorSet,
    *,
    format_output: bool,
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.low_descriptors"),
        "",
        f"#ifndef {header_spec.header_guard}",
        f"#define {header_spec.header_guard}",
        "",
        '#include "loom/codegen/low/descriptors.h"',
        "",
    ]
    lines.extend(_descriptor_ref_define(header_spec, descriptor.key, i) for i, descriptor in enumerate(header_spec.descriptors))
    lines.append(f"#define {header_spec.c_enum_prefix}_DESCRIPTOR_SET_ID UINT64_C(0x{descriptor_stable_id(header_spec.key):016x})")
    lines.append(
        f"#define {header_spec.c_enum_prefix}_DESCRIPTOR_SET_ORDINAL {_u16_literal(header_spec.descriptor_set_ordinal if header_spec.descriptor_set_ordinal is not None else LOW_DESCRIPTOR_SET_ORDINAL_NONE)}"
    )
    if header_spec.target_key is not None:
        lines.append(f"#define {header_spec.c_enum_prefix}_TARGET_ID UINT64_C(0x{descriptor_stable_id(header_spec.target_key):016x})")
    missing_reg_classes = [reg_class.name for reg_class in header_spec.reg_classes if reg_class.name not in compiled.reg_class_ids]
    if missing_reg_classes:
        raise ValueError(f"descriptor set header '{header_spec.key}' references reg classes missing from storage set '{compiled.spec.key}': {', '.join(missing_reg_classes)}")
    header_reg_class_ids = [(reg_class, compiled.reg_class_ids[reg_class.name]) for reg_class in header_spec.reg_classes]
    if header_reg_class_ids:
        lines.append("")
        lines.extend(
            _emit_id_enum(
                f"{header_spec.c_enum_prefix.lower()}_reg_class_id",
                [
                    (
                        _reg_class_id_constant_name(header_spec, reg_class.name),
                        reg_class_id,
                    )
                    for reg_class, reg_class_id in header_reg_class_ids
                ],
            )
        )
    header_register_part_ids = [(part, compiled.register_part_ids[part.name]) for part in header_spec.register_parts if part.name in compiled.register_part_ids]
    if header_register_part_ids:
        lines.append("")
        lines.extend(
            _emit_id_enum(
                f"{header_spec.c_enum_prefix.lower()}_register_part_id",
                [
                    (
                        _register_part_id_constant_name(header_spec, part.name),
                        part_id,
                    )
                    for part, part_id in header_register_part_ids
                ],
            )
        )
    lines.append("")
    lines.extend(
        [
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"const loom_low_descriptor_set_t* {header_spec.function_name}(void);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {header_spec.header_guard}",
        ]
    )
    source = "\n".join(lines) + "\n"
    if not format_output:
        return source
    return _clang_format_source(source, header_spec.c_header_path)


def _emit_header(compiled: _CompiledDescriptorSet, *, format_output: bool) -> str:
    return _emit_header_for_spec(
        compiled,
        compiled.spec,
        format_output=format_output,
    )


def _emit_string_table(compiled: _CompiledDescriptorSet, lines: list[str]) -> None:
    spec = compiled.spec
    pool = compiled.string_pool
    lines.extend(
        [
            "// clang-format off",
            f"static const uint8_t k{spec.c_table_prefix}StringData[] =",
        ]
    )
    for entry in pool.entries:
        length = len(entry.value.encode())
        escaped = _c_string_literal(entry.value)
        lines.append(f'    LOOM_BSTRING_LITERAL({length}, "{escaped}")')
    lines[-1] += ";"
    lines.append("// clang-format on")
    lines.append("")
    lines.append("enum {")
    previous_label: str | None = None
    entries_by_label = pool.entries_by_label
    for entry in pool.entries:
        enum_name = f"{pool.c_enum_prefix}_STRING_{entry.label}"
        if previous_label is None:
            lines.append(f"  {enum_name} = 0,")
        else:
            previous_entry = entries_by_label[previous_label]
            previous_enum_name = f"{pool.c_enum_prefix}_STRING_{previous_label}"
            lines.append(f'  {enum_name} = {previous_enum_name} + sizeof("{_c_string_literal(previous_entry.value)}"),')
        previous_label = entry.label
    if previous_label is None:
        lines.append(f"  {pool.c_enum_prefix}_STRING_END = 0,")
    else:
        previous_entry = entries_by_label[previous_label]
        previous_enum_name = f"{pool.c_enum_prefix}_STRING_{previous_label}"
        lines.append(f'  {pool.c_enum_prefix}_STRING_END = {previous_enum_name} + sizeof("{_c_string_literal(previous_entry.value)}"),')
    lines.append("};")
    lines.append("")
    lines.append(f'static_assert({pool.c_enum_prefix}_STRING_END == sizeof(k{spec.c_table_prefix}StringData) - 1, "descriptor string offsets must cover the table payload");')
    lines.append("")


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _emit_array(
    lines: list[str],
    c_type: str,
    table_prefix: str,
    name: str,
    row_lines: Sequence[list[str]],
) -> None:
    if not row_lines:
        return
    lines.append(f"static const {c_type} k{table_prefix}{name}[] = {{")
    for row in row_lines:
        lines.append("    {")
        lines.extend(f"        {line}" for line in row)
        lines.append("    },")
    lines.append("};")
    lines.append("")


def _metadata_string_label(storage_spec: DescriptorSet, view_spec: DescriptorSet, field_name: str) -> str:
    if view_spec.key == storage_spec.key and view_spec.target_key == storage_spec.target_key and view_spec.feature_key == storage_spec.feature_key:
        return field_name
    return f"{field_name}_{_c_identifier(view_spec.key)}"


def _descriptor_refs_for_ordinals(
    descriptors: Sequence[Descriptor],
    descriptor_ordinals: Sequence[int],
) -> list[tuple[str, int]]:
    return sorted((descriptors[descriptor_ordinal].key, view_ordinal) for view_ordinal, descriptor_ordinal in enumerate(descriptor_ordinals))


def _clone_asm_form_for_view(
    asm_form: _CompiledAsmForm,
    descriptor_ordinal: int,
) -> _CompiledAsmForm:
    return _CompiledAsmForm(
        descriptor_ordinal=descriptor_ordinal,
        mnemonic_label=asm_form.mnemonic_label,
        mnemonic=asm_form.mnemonic,
        result_indices=asm_form.result_indices,
        operand_indices=asm_form.operand_indices,
        immediates=asm_form.immediates,
        result_index_start=asm_form.result_index_start,
        operand_index_start=asm_form.operand_index_start,
        immediate_start=asm_form.immediate_start,
    )


def _clone_operand_form_for_view(
    operand_form: _CompiledOperandForm,
    replacement_descriptor_ordinal: int,
) -> _CompiledOperandForm:
    return _CompiledOperandForm(
        replacement_descriptor_ordinal=replacement_descriptor_ordinal,
        source_immediate_index=operand_form.source_immediate_index,
        replacement_immediate_index=operand_form.replacement_immediate_index,
        immediate_match_index=operand_form.immediate_match_index,
        immediate_action=operand_form.immediate_action,
        match_start=operand_form.match_start,
        match_count=operand_form.match_count,
        operand_map_start=operand_form.operand_map_start,
        operand_map_count=operand_form.operand_map_count,
    )


def _asm_forms_have_duplicate_mnemonics(
    asm_forms: Sequence[_CompiledAsmForm],
) -> bool:
    seen_mnemonics: set[str] = set()
    for asm_form in asm_forms:
        if asm_form.mnemonic in seen_mnemonics:
            return True
        seen_mnemonics.add(asm_form.mnemonic)
    return False


def _validate_view_asm_forms_unique(
    view_spec: DescriptorSet,
    asm_forms: Sequence[_CompiledAsmForm],
) -> None:
    seen_mnemonics: dict[str, int] = {}
    for asm_form_ordinal, asm_form in enumerate(asm_forms):
        previous_ordinal = seen_mnemonics.get(asm_form.mnemonic)
        if previous_ordinal is not None:
            previous_descriptor = view_spec.descriptors[asm_forms[previous_ordinal].descriptor_ordinal]
            descriptor = view_spec.descriptors[asm_form.descriptor_ordinal]
            raise ValueError(f"descriptor set view '{view_spec.key}' has ambiguous asm mnemonic '{asm_form.mnemonic}' between descriptors '{previous_descriptor.key}' and '{descriptor.key}'")
        seen_mnemonics[asm_form.mnemonic] = asm_form_ordinal


def _validate_view_operand_forms_closed(
    compiled: _CompiledDescriptorSet,
    view_spec: DescriptorSet,
    descriptor_ordinals: Sequence[int],
) -> None:
    selected_descriptor_ordinals = set(descriptor_ordinals)
    for storage_descriptor_ordinal in descriptor_ordinals:
        storage_descriptor = compiled.descriptors[storage_descriptor_ordinal]
        storage_row = compiled.descriptor_rows[storage_descriptor_ordinal]
        operand_form_start = storage_row["operand_form_start"]
        operand_form_count = storage_row["operand_form_count"]
        for form_ordinal in range(operand_form_count):
            operand_form = compiled.operand_forms[operand_form_start + form_ordinal]
            if operand_form.replacement_descriptor_ordinal in selected_descriptor_ordinals:
                continue
            replacement = compiled.descriptors[operand_form.replacement_descriptor_ordinal]
            raise ValueError(f"descriptor set view '{view_spec.key}' selects descriptor '{storage_descriptor.key}' without operand-form replacement descriptor '{replacement.key}'")


def _descriptor_set_view_for_spec(
    compiled: _CompiledDescriptorSet,
    view_spec: DescriptorSet,
) -> _DescriptorSetView:
    if not view_spec.descriptors:
        raise ValueError(f"descriptor set view '{view_spec.key}' selects no descriptors")

    storage_ordinals_by_key = {descriptor.key: i for i, descriptor in enumerate(compiled.descriptors)}
    descriptor_ordinals = []
    for descriptor in view_spec.descriptors:
        descriptor_ordinal = storage_ordinals_by_key.get(descriptor.key)
        if descriptor_ordinal is None:
            raise ValueError(f"descriptor set view '{view_spec.key}' selects descriptor '{descriptor.key}' that is not in storage set '{compiled.spec.key}'")
        descriptor_ordinals.append(descriptor_ordinal)
    descriptor_ordinal_tuple = tuple(descriptor_ordinals)
    _validate_view_operand_forms_closed(
        compiled,
        view_spec,
        descriptor_ordinal_tuple,
    )
    if descriptor_ordinal_tuple == tuple(range(len(descriptor_ordinal_tuple))) and not _asm_forms_have_duplicate_mnemonics(compiled.asm_forms):
        descriptor_count = len(descriptor_ordinal_tuple)
        return _DescriptorSetView(
            spec=view_spec,
            descriptor_ordinals=descriptor_ordinal_tuple,
            descriptor_refs=_descriptor_refs_for_ordinals(
                compiled.descriptors,
                descriptor_ordinal_tuple,
            ),
            descriptor_rows=compiled.descriptor_rows[:descriptor_count],
            canonical_asm_form_ordinals=compiled.canonical_asm_form_ordinals[:descriptor_count],
            asm_forms=compiled.asm_forms,
            operand_forms=compiled.operand_forms,
            uses_storage_descriptor_tables=True,
            uses_storage_asm_form_tables=True,
            uses_storage_operand_form_tables=True,
        )

    storage_to_view_ordinals = {descriptor_ordinal: view_ordinal for view_ordinal, descriptor_ordinal in enumerate(descriptor_ordinal_tuple)}
    asm_form_ordinals_by_storage_ordinal: dict[int, int] = {}
    asm_forms: list[_CompiledAsmForm] = []
    for storage_asm_form_ordinal, asm_form in enumerate(compiled.asm_forms):
        view_descriptor_ordinal = storage_to_view_ordinals.get(asm_form.descriptor_ordinal)
        if view_descriptor_ordinal is None:
            continue
        asm_form_ordinals_by_storage_ordinal[storage_asm_form_ordinal] = len(asm_forms)
        asm_forms.append(_clone_asm_form_for_view(asm_form, view_descriptor_ordinal))
    _validate_view_asm_forms_unique(view_spec, asm_forms)

    descriptor_rows = []
    canonical_asm_form_ordinals: list[int | None] = []
    operand_forms: list[_CompiledOperandForm] = []
    for storage_descriptor_ordinal in descriptor_ordinal_tuple:
        storage_row = compiled.descriptor_rows[storage_descriptor_ordinal]
        descriptor_row = dict(storage_row)
        descriptor_row["operand_form_start"] = len(operand_forms)
        operand_form_start = storage_row["operand_form_start"]
        operand_form_count = storage_row["operand_form_count"]
        for form_ordinal in range(operand_form_count):
            operand_form = compiled.operand_forms[operand_form_start + form_ordinal]
            operand_forms.append(
                _clone_operand_form_for_view(
                    operand_form,
                    storage_to_view_ordinals[operand_form.replacement_descriptor_ordinal],
                )
            )
        descriptor_rows.append(descriptor_row)

        canonical_storage_asm_form_ordinal = compiled.canonical_asm_form_ordinals[storage_descriptor_ordinal]
        if canonical_storage_asm_form_ordinal is None:
            canonical_asm_form_ordinals.append(None)
        else:
            canonical_asm_form_ordinals.append(asm_form_ordinals_by_storage_ordinal.get(canonical_storage_asm_form_ordinal))

    return _DescriptorSetView(
        spec=view_spec,
        descriptor_ordinals=descriptor_ordinal_tuple,
        descriptor_refs=_descriptor_refs_for_ordinals(
            compiled.descriptors,
            descriptor_ordinal_tuple,
        ),
        descriptor_rows=descriptor_rows,
        canonical_asm_form_ordinals=canonical_asm_form_ordinals,
        asm_forms=asm_forms,
        operand_forms=operand_forms,
        uses_storage_descriptor_tables=False,
        uses_storage_asm_form_tables=False,
        uses_storage_operand_form_tables=False,
    )


def _intern_descriptor_set_view_metadata(compiled: _CompiledDescriptorSet, view_spec: DescriptorSet) -> None:
    pool = compiled.string_pool
    storage_spec = compiled.spec
    pool.intern(_metadata_string_label(storage_spec, view_spec, "set_key"), view_spec.key)
    if view_spec.target_key is not None:
        pool.intern(
            _metadata_string_label(storage_spec, view_spec, "target_key"),
            view_spec.target_key,
        )
    if view_spec.feature_key is not None:
        pool.intern(
            _metadata_string_label(storage_spec, view_spec, "feature_key"),
            view_spec.feature_key,
        )


def _descriptor_row_lines(
    compiled: _CompiledDescriptorSet,
    descriptors: Sequence[Descriptor],
    descriptor_rows: Sequence[dict[str, int]],
    canonical_asm_form_ordinals: Sequence[int | None],
) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".key_string_offset = {pool.ref(f'descriptor_{descriptor.key}')},",
            f".stable_id = {_hex_u64_literal(descriptor_stable_id(descriptor.key))},",
            f".mnemonic_string_offset = {_optional_string_expr(pool, f'mnemonic_{descriptor.key}' if descriptor.mnemonic is not None else None)},",
            f".semantic_tag_string_offset = {_optional_string_expr(pool, f'semantic_{descriptor.key}' if descriptor.semantic_tag is not None else None)},",
            f".feature_mask_word_start = {descriptor_rows[i]['feature_mask_word_start']},",
            f".feature_mask_word_count = {descriptor_rows[i]['feature_mask_word_count']},",
            f".encoding_field_value_start = {descriptor_rows[i]['encoding_field_value_start']},",
            f".encoding_field_value_count = {descriptor_rows[i]['encoding_field_value_count']},",
            f".encoding_format_id = {descriptor.encoding_format_id},",
            f".encoding_id = {_encoding_id_expr(descriptor.encoding_id)},",
            f".operand_start = {descriptor_rows[i]['operand_start']},",
            f".operand_count = {descriptor_rows[i]['operand_count']},",
            f".result_count = {descriptor_rows[i]['result_count']},",
            f".immediate_start = {descriptor_rows[i]['immediate_start']},",
            f".immediate_count = {descriptor_rows[i]['immediate_count']},",
            f".effect_start = {descriptor_rows[i]['effect_start']},",
            f".effect_count = {descriptor_rows[i]['effect_count']},",
            f".constraint_start = {descriptor_rows[i]['constraint_start']},",
            f".constraint_count = {descriptor_rows[i]['constraint_count']},",
            f".operand_form_start = {descriptor_rows[i]['operand_form_start']},",
            f".operand_form_count = {descriptor_rows[i]['operand_form_count']},",
            f".schedule_class_id = {compiled.schedule_class_ids[descriptor.schedule_class]},",
            f".flags = {_flag_expr(descriptor.flags)},",
            f".canonical_asm_form_ordinal = {_canonical_asm_form_ordinal_expr(canonical_asm_form_ordinals[i])},",
        ]
        for i, descriptor in enumerate(descriptors)
    ]


def _operand_form_row_lines(
    operand_forms: Sequence[_CompiledOperandForm],
) -> list[list[str]]:
    return [
        [
            f".replacement_descriptor_ordinal = {operand_form.replacement_descriptor_ordinal},",
            f".operand_map_start = {operand_form.operand_map_start},",
            f".match_start = {operand_form.match_start},",
            f".source_immediate_index = {operand_form.source_immediate_index},",
            f".replacement_immediate_index = {operand_form.replacement_immediate_index},",
            f".immediate_match_index = {operand_form.immediate_match_index},",
            f".operand_map_count = {operand_form.operand_map_count},",
            f".immediate_action = {operand_form.immediate_action.c_name},",
            f".match_count = {operand_form.match_count},",
        ]
        for operand_form in operand_forms
    ]


def _asm_form_row_lines(
    compiled: _CompiledDescriptorSet,
    asm_forms: Sequence[_CompiledAsmForm],
) -> list[list[str]]:
    pool = compiled.string_pool
    return [
        [
            f".mnemonic_string_offset = {pool.ref(asm_form.mnemonic_label)},",
            f".descriptor_ordinal = {asm_form.descriptor_ordinal},",
            f".result_operand_index_start = {asm_form.result_index_start},",
            f".result_operand_index_count = {len(asm_form.result_indices)},",
            f".operand_index_start = {asm_form.operand_index_start},",
            f".operand_index_count = {len(asm_form.operand_indices)},",
            f".immediate_start = {asm_form.immediate_start},",
            f".immediate_count = {len(asm_form.immediates)},",
        ]
        for asm_form in asm_forms
    ]


def _emit_source_for_views(
    compiled: _CompiledDescriptorSet,
    *,
    views: Sequence[_DescriptorSetView],
    format_output: bool,
) -> str:
    spec = compiled.spec
    pool = compiled.string_pool
    for view in views:
        _intern_descriptor_set_view_metadata(compiled, view.spec)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.low_descriptors"),
        "",
        f'#include "{spec.public_header}"',
        "",
        "#include <stdint.h>",
        "",
    ]
    for view in views:
        if view.spec is spec:
            continue
        lines.append(f"const loom_low_descriptor_set_t* {view.spec.function_name}(void);")
    if len(views) > 1:
        lines.append("")
    _emit_string_table(compiled, lines)

    _emit_array(
        lines,
        "loom_low_reg_class_t",
        spec.c_table_prefix,
        "RegClasses",
        [
            [
                f".name_string_offset = {pool.ref(f'reg_{reg_class.name}')},",
                f".target_bank_id = {reg_class.target_bank_id},",
                f".flags = {_flag_expr(reg_class.flags)},",
                f".alloc_unit_bits = {reg_class.alloc_unit_bits},",
                f".allocatable_count = {reg_class.allocatable_count},",
                f".alias_set_id = {reg_class.alias_set_id},",
                ".spill_class_id = " + ("LOOM_LOW_REG_CLASS_NONE" if reg_class.spill_class is None else str(compiled.reg_class_ids[reg_class.spill_class])) + ",",
                f".full_register_part_mask = {_hex_u64_literal(reg_class.full_register_part_mask)},",
                f".spill_slot_space = {reg_class.spill_slot_space.c_name},",
            ]
            for reg_class in compiled.reg_classes
        ],
    )
    _emit_array(
        lines,
        "loom_low_register_part_t",
        spec.c_table_prefix,
        "RegisterParts",
        [
            [
                f".name_string_offset = {pool.ref(f'register_part_{part.name}')},",
                f".reg_class_id = {compiled.reg_class_ids[part.reg_class]},",
                ".reserved = 0,",
                f".mask = {_hex_u64_literal(part.mask)},",
            ]
            for part in compiled.register_parts
        ],
    )
    _emit_array(
        lines,
        "loom_low_reg_class_alt_t",
        spec.c_table_prefix,
        "RegClassAlts",
        [
            [
                ".reg_class_id = " + ("LOOM_LOW_REG_CLASS_NONE" if reg_class_id is None else str(reg_class_id)) + ",",
                f".flags = {_flag_expr(flags)},",
            ]
            for reg_class_id, flags in compiled.reg_class_alts
        ],
    )
    _emit_array(
        lines,
        "loom_low_operand_t",
        spec.c_table_prefix,
        "Operands",
        [
            [
                f".field_name_string_offset = {pool.ref(f'field_{operand.field_name}')},",
                f".role = {operand.role.c_name},",
                f".flags = {_flag_expr(operand.flags)},",
                f".reg_class_alt_start = {compiled.operand_alt_starts[i]},",
                f".reg_class_alt_count = {len(operand.reg_alts)},",
                f".unit_count = {operand.unit_count},",
                f".address_map_kind = {operand.address_map_kind.c_name},",
                f".addressable_unit_count = {operand.addressable_unit_count},",
                f".encoding_field_id = {operand.encoding_field_id},",
                f".data_format_id = {operand.data_format_id},",
                f".register_part_id = {_register_part_id_expr(compiled, operand.register_part)},",
                f".read_stage = {operand.read_stage},",
                f".ready_stage = {operand.ready_stage},",
            ]
            for i, operand in enumerate(compiled.operands)
        ],
    )
    _emit_array(
        lines,
        "loom_low_immediate_t",
        spec.c_table_prefix,
        "Immediates",
        [
            [
                f".field_name_string_offset = {pool.ref(f'immediate_{immediate.field_name}')},",
                f".encoding_slice_start = {compiled.immediate_encoding_slice_starts[i]},",
                f".kind = {immediate.kind.c_name},",
                f".flags = {_flag_expr(immediate.flags)},",
                f".bit_width = {immediate.bit_width},",
                f".encoding_field_id = {immediate.encoding_field_id},",
                f".encoding_slice_count = {len(immediate.encoding_slices)},",
                ".enum_domain_id = " + ("LOOM_LOW_ENUM_DOMAIN_NONE" if compiled.immediate_enum_domain_ids[i] is None else str(compiled.immediate_enum_domain_ids[i])) + ",",
                f".encoding_id = {immediate.encoding_id},",
                f".signed_min = {_i64_literal(immediate.signed_min)},",
                f".unsigned_max = {_u64_literal(immediate.unsigned_max)},",
                f".default_value = {_i64_literal(immediate.default_value)},",
            ]
            for i, immediate in enumerate(compiled.immediates)
        ],
    )
    _emit_array(
        lines,
        "loom_low_immediate_encoding_slice_t",
        spec.c_table_prefix,
        "ImmediateEncodingSlices",
        [
            [
                f".encoding_field_id = {encoding_slice.encoding_field_id},",
                f".source_bit_offset = {encoding_slice.source_bit_offset},",
                f".bit_count = {encoding_slice.bit_count},",
            ]
            for encoding_slice in compiled.immediate_encoding_slices
        ],
    )
    _emit_array(
        lines,
        "loom_low_enum_domain_t",
        spec.c_table_prefix,
        "EnumDomains",
        [
            [
                f".name_string_offset = {pool.ref(f'enum_domain_{domain.name}')},",
                f".value_start = {compiled.enum_domain_rows[i]['value_start']},",
                f".value_count = {compiled.enum_domain_rows[i]['value_count']},",
            ]
            for i, domain in enumerate(compiled.enum_domains)
        ],
    )
    _emit_array(
        lines,
        "loom_low_enum_value_t",
        spec.c_table_prefix,
        "EnumValues",
        [
            [
                f".token_string_offset = {pool.ref(f'enum_value_{domain.name}_{value.token}')},",
                f".value = {_i64_literal(value.value)},",
            ]
            for domain in compiled.enum_domains
            for value in _validate_enum_domain(domain)
        ],
    )
    _emit_array(
        lines,
        "loom_low_effect_t",
        spec.c_table_prefix,
        "Effects",
        [
            [
                f".kind = {effect.kind.c_name},",
                f".memory_space = {effect.memory_space.c_name},",
                f".scope_id = {effect.scope_id},",
                f".flags = {_flag_expr(effect.flags)},",
                f".counter_id = {effect.counter_id},",
                f".width_bits = {effect.width_bits},",
            ]
            for effect in compiled.effects
        ],
    )
    _emit_array(
        lines,
        "loom_low_constraint_t",
        spec.c_table_prefix,
        "Constraints",
        [
            [
                f".kind = {constraint.kind.c_name},",
                f".lhs_operand_index = {constraint.lhs_operand_index},",
                ".rhs_operand_index = " + ("LOOM_LOW_ID_NONE" if constraint.rhs_operand_index is None else str(constraint.rhs_operand_index)) + ",",
                f".flags = {_flag_expr(constraint.flags)},",
            ]
            for constraint in compiled.constraints
        ],
    )
    _emit_array(
        lines,
        "loom_low_resource_t",
        spec.c_table_prefix,
        "Resources",
        [
            [
                f".name_string_offset = {pool.ref(f'resource_{resource.name}')},",
                f".capacity_per_cycle = {resource.capacity_per_cycle},",
                f".flags = {_flag_expr(resource.flags)},",
                f".kind = {resource.kind.c_name},",
                f".contention_group_id = {resource.contention_group_id},",
            ]
            for resource in compiled.resources
        ],
    )
    _emit_array(
        lines,
        "loom_low_issue_use_t",
        spec.c_table_prefix,
        "IssueUses",
        [
            [
                f".resource_id = {compiled.resource_ids[issue_use.resource]},",
                f".cycles = {issue_use.cycles},",
                f".units = {issue_use.units},",
                f".stage = {issue_use.stage},",
            ]
            for issue_use in compiled.issue_uses
        ],
    )
    _emit_array(
        lines,
        "loom_low_hazard_t",
        spec.c_table_prefix,
        "Hazards",
        [
            [
                f".kind = {hazard.kind.c_name},",
                f".reference_kind = {_hazard_reference_kind(hazard).c_name},",
                f".reference_id = {_hazard_reference_id(hazard, compiled.resource_ids)},",
                f".producer_stage = {hazard.producer_stage},",
                f".consumer_stage = {hazard.consumer_stage},",
                f".distance = {hazard.distance},",
                f".flags = {_flag_expr(hazard.flags)},",
            ]
            for hazard in compiled.hazards
        ],
    )
    _emit_array(
        lines,
        "loom_low_pressure_delta_t",
        spec.c_table_prefix,
        "PressureDeltas",
        [
            [
                f".reg_class_id = {compiled.reg_class_ids[pressure.reg_class]},",
                f".delta = {pressure.delta},",
            ]
            for pressure in compiled.pressure_deltas
        ],
    )
    _emit_array(
        lines,
        "loom_low_schedule_class_t",
        spec.c_table_prefix,
        "ScheduleClasses",
        [
            [
                f".name_string_offset = {pool.ref(f'schedule_{schedule_class.name}')},",
                f".latency_cycles = {schedule_class.latency_cycles},",
                f".latency_kind = {schedule_class.latency_kind.c_name},",
                f".issue_use_start = {compiled.schedule_rows[i]['issue_use_start']},",
                f".issue_use_count = {compiled.schedule_rows[i]['issue_use_count']},",
                f".hazard_start = {compiled.schedule_rows[i]['hazard_start']},",
                f".hazard_count = {compiled.schedule_rows[i]['hazard_count']},",
                f".flags = {_flag_expr(schedule_class.flags)},",
                f".model_quality = {schedule_class.model_quality.c_name},",
                f".pressure_delta_start = {compiled.schedule_rows[i]['pressure_delta_start']},",
                f".pressure_delta_count = {compiled.schedule_rows[i]['pressure_delta_count']},",
            ]
            for i, schedule_class in enumerate(compiled.schedule_classes)
        ],
    )
    if compiled.feature_mask_words:
        lines.append(f"static const uint64_t k{spec.c_table_prefix}FeatureMaskWords[] = {{")
        lines.extend(f"    {_hex_u64_literal(word)}," for word in compiled.feature_mask_words)
        lines.append("};")
        lines.append("")
    _emit_array(
        lines,
        "loom_low_encoding_field_value_t",
        spec.c_table_prefix,
        "EncodingFieldValues",
        [
            [
                f".encoding_field_id = {field_value.encoding_field_id},",
                ".reserved = 0,",
                f".value = {_u64_literal(field_value.value)},",
            ]
            for field_value in compiled.encoding_field_values
        ],
    )
    _emit_array(
        lines,
        "loom_low_operand_form_t",
        spec.c_table_prefix,
        "OperandForms",
        _operand_form_row_lines(compiled.operand_forms),
    )
    _emit_array(
        lines,
        "loom_low_operand_form_match_t",
        spec.c_table_prefix,
        "OperandFormMatches",
        [
            [
                f".source_operand_index = {match.source_operand_index},",
                f".source_packet_operand_index = {match.source_packet_operand_index},",
                f".match_kind = {match.match_kind.c_name},",
                f".match_i64 = {_i64_literal(match.match_i64)},",
            ]
            for match in compiled.operand_form_matches
        ],
    )
    if compiled.operand_form_operand_indices:
        lines.append(f"static const uint16_t k{spec.c_table_prefix}OperandFormOperandIndices[] = {{")
        lines.extend(f"    {operand_index}," for operand_index in compiled.operand_form_operand_indices)
        lines.append("};")
        lines.append("")
    _emit_array(
        lines,
        "loom_low_descriptor_t",
        spec.c_table_prefix,
        "Descriptors",
        _descriptor_row_lines(
            compiled,
            compiled.descriptors,
            compiled.descriptor_rows,
            compiled.canonical_asm_form_ordinals,
        ),
    )
    for view in views:
        if view.uses_storage_descriptor_tables:
            continue
        view_descriptors = [compiled.descriptors[descriptor_ordinal] for descriptor_ordinal in view.descriptor_ordinals]
        _emit_array(
            lines,
            "loom_low_descriptor_t",
            view.spec.c_table_prefix,
            "Descriptors",
            _descriptor_row_lines(
                compiled,
                view_descriptors,
                view.descriptor_rows,
                view.canonical_asm_form_ordinals,
            ),
        )
        _emit_array(
            lines,
            "loom_low_operand_form_t",
            view.spec.c_table_prefix,
            "OperandForms",
            _operand_form_row_lines(view.operand_forms),
        )
    for view in views:
        _emit_array(
            lines,
            "loom_low_descriptor_ref_t",
            view.spec.c_table_prefix,
            "DescriptorRefs",
            [
                [
                    f".key_string_offset = {pool.ref(f'descriptor_{descriptor_key}')},",
                    f".descriptor_ordinal = {descriptor_ordinal},",
                ]
                for descriptor_key, descriptor_ordinal in view.descriptor_refs
            ],
        )
    if compiled.asm_operand_indices:
        lines.append(f"static const uint16_t k{spec.c_table_prefix}AsmOperandIndices[] = {{")
        lines.extend(f"    {operand_index}," for operand_index in compiled.asm_operand_indices)
        lines.append("};")
        lines.append("")
    _emit_array(
        lines,
        "loom_low_asm_immediate_t",
        spec.c_table_prefix,
        "AsmImmediates",
        [
            [
                f".immediate_index = {immediate.immediate_index},",
                f".name_string_offset = {_optional_string_expr(pool, immediate.name_label)},",
            ]
            for immediate in compiled.asm_immediates
        ],
    )
    _emit_array(
        lines,
        "loom_low_asm_form_t",
        spec.c_table_prefix,
        "AsmForms",
        _asm_form_row_lines(compiled, compiled.asm_forms),
    )
    for view in views:
        if view.uses_storage_asm_form_tables:
            continue
        _emit_array(
            lines,
            "loom_low_asm_form_t",
            view.spec.c_table_prefix,
            "AsmForms",
            _asm_form_row_lines(compiled, view.asm_forms),
        )

    table_count_fields = {
        "operands": "operand_count",
        "immediates": "immediate_count",
        "immediate_encoding_slices": "immediate_encoding_slice_count",
        "enum_domains": "enum_domain_count",
        "enum_values": "enum_value_count",
        "effects": "effect_count",
        "constraints": "constraint_count",
        "reg_classes": "reg_class_count",
        "register_parts": "register_part_count",
        "reg_class_alts": "reg_class_alt_count",
        "schedule_classes": "schedule_class_count",
        "issue_uses": "issue_use_count",
        "resources": "resource_count",
        "hazards": "hazard_count",
        "pressure_deltas": "pressure_delta_count",
        "asm_forms": "asm_form_count",
        "asm_operand_indices": "asm_operand_index_count",
        "asm_immediates": "asm_immediate_count",
        "encoding_field_values": "encoding_field_value_count",
        "operand_forms": "operand_form_count",
        "operand_form_matches": "operand_form_match_count",
        "operand_form_operand_indices": "operand_form_operand_index_count",
    }

    def append_optional_table(field_name: str, table_name: str, view_lines: list[str]) -> None:
        rows = getattr(compiled, field_name)
        if rows:
            view_lines.append(f"    .{field_name} = k{spec.c_table_prefix}{table_name},")
            view_lines.append(f"    .{table_count_fields[field_name]} = IREE_ARRAYSIZE(k{spec.c_table_prefix}{table_name}),")

    for view in views:
        view_spec = view.spec
        descriptor_table_prefix = spec.c_table_prefix if view.uses_storage_descriptor_tables else view_spec.c_table_prefix
        asm_form_table_prefix = spec.c_table_prefix if view.uses_storage_asm_form_tables else view_spec.c_table_prefix
        operand_form_table_prefix = spec.c_table_prefix if view.uses_storage_operand_form_tables else view_spec.c_table_prefix
        view_lines = [
            f"static const loom_low_descriptor_set_t k{view_spec.c_table_prefix}Set = {{",
            "    .abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION,",
            f"    .generator_version = {view_spec.generator_version},",
            f"    .stable_id = UINT64_C(0x{descriptor_stable_id(view_spec.key):016x}),",
            f"    .target_stable_id = {_hex_u64_literal(descriptor_stable_id(view_spec.target_key)) if view_spec.target_key is not None else 'LOOM_LOW_STABLE_ID_NONE'},",
            f"    .descriptor_set_ordinal = {_u16_literal(view_spec.descriptor_set_ordinal if view_spec.descriptor_set_ordinal is not None else LOW_DESCRIPTOR_SET_ORDINAL_NONE)},",
            f"    .key_string_offset = {pool.ref(_metadata_string_label(spec, view_spec, 'set_key'))},",
            f"    .target_key_string_offset = {_optional_string_expr(pool, _metadata_string_label(spec, view_spec, 'target_key') if view_spec.target_key is not None else None)},",
            f"    .feature_key_string_offset = {_optional_string_expr(pool, _metadata_string_label(spec, view_spec, 'feature_key') if view_spec.feature_key is not None else None)},",
            "    .string_table =",
            "        {",
            f"            .data = k{spec.c_table_prefix}StringData,",
            f"            .data_length = sizeof(k{spec.c_table_prefix}StringData) - 1,",
            "        },",
            f"    .descriptors = k{descriptor_table_prefix}Descriptors,",
            f"    .descriptor_count = {view.descriptor_count},",
            f"    .descriptor_refs = k{view_spec.c_table_prefix}DescriptorRefs,",
            f"    .descriptor_ref_count = IREE_ARRAYSIZE(k{view_spec.c_table_prefix}DescriptorRefs),",
        ]

        append_optional_table("operands", "Operands", view_lines)
        append_optional_table("immediates", "Immediates", view_lines)
        append_optional_table("immediate_encoding_slices", "ImmediateEncodingSlices", view_lines)
        append_optional_table("enum_domains", "EnumDomains", view_lines)
        append_optional_table("enum_values", "EnumValues", view_lines)
        append_optional_table("effects", "Effects", view_lines)
        append_optional_table("constraints", "Constraints", view_lines)
        append_optional_table("reg_classes", "RegClasses", view_lines)
        append_optional_table("register_parts", "RegisterParts", view_lines)
        append_optional_table("reg_class_alts", "RegClassAlts", view_lines)
        append_optional_table("schedule_classes", "ScheduleClasses", view_lines)
        append_optional_table("issue_uses", "IssueUses", view_lines)
        append_optional_table("resources", "Resources", view_lines)
        append_optional_table("hazards", "Hazards", view_lines)
        append_optional_table("pressure_deltas", "PressureDeltas", view_lines)
        if view.asm_forms:
            view_lines.append(f"    .asm_forms = k{asm_form_table_prefix}AsmForms,")
            view_lines.append(f"    .asm_form_count = IREE_ARRAYSIZE(k{asm_form_table_prefix}AsmForms),")
            append_optional_table("asm_operand_indices", "AsmOperandIndices", view_lines)
            append_optional_table("asm_immediates", "AsmImmediates", view_lines)
        append_optional_table("encoding_field_values", "EncodingFieldValues", view_lines)
        if view.operand_forms:
            view_lines.append(f"    .operand_forms = k{operand_form_table_prefix}OperandForms,")
            view_lines.append(f"    .operand_form_count = IREE_ARRAYSIZE(k{operand_form_table_prefix}OperandForms),")
        append_optional_table(
            "operand_form_matches",
            "OperandFormMatches",
            view_lines,
        )
        append_optional_table(
            "operand_form_operand_indices",
            "OperandFormOperandIndices",
            view_lines,
        )
        if compiled.feature_mask_words:
            view_lines.append(f"    .feature_mask_words = k{spec.c_table_prefix}FeatureMaskWords,")
            view_lines.append(f"    .feature_mask_word_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}FeatureMaskWords),")
        view_lines.append("};")
        view_lines.append("")
        view_lines.append(f"const loom_low_descriptor_set_t* {view_spec.function_name}(void) {{")
        view_lines.append(f"  return &k{view_spec.c_table_prefix}Set;")
        view_lines.append("}")
        lines.extend(view_lines)
        lines.append("")
    source = "\n".join(lines) + "\n"
    if not format_output:
        return source
    return _clang_format_source(source, spec.c_source_path)


def _emit_source(compiled: _CompiledDescriptorSet, *, format_output: bool) -> str:
    return _emit_source_for_views(
        compiled,
        views=[
            _DescriptorSetView(
                spec=compiled.spec,
                descriptor_ordinals=tuple(range(len(compiled.descriptors))),
                descriptor_refs=compiled.descriptor_refs,
                descriptor_rows=compiled.descriptor_rows,
                canonical_asm_form_ordinals=compiled.canonical_asm_form_ordinals,
                asm_forms=compiled.asm_forms,
                operand_forms=compiled.operand_forms,
                uses_storage_descriptor_tables=True,
                uses_storage_asm_form_tables=True,
                uses_storage_operand_form_tables=True,
            )
        ],
        format_output=format_output,
    )


def _emit_manifest_json(compiled: _CompiledDescriptorSet) -> str:
    spec = compiled.spec
    descriptors = []
    for i, descriptor in enumerate(compiled.descriptors):
        descriptors.append(
            {
                "ordinal": i,
                "stable_id": f"0x{descriptor_stable_id(descriptor.key):016x}",
                "key": descriptor.key,
                "mnemonic": descriptor.mnemonic or "",
                "semantic_tag": descriptor.semantic_tag or "",
                "encoding_format": descriptor.encoding_format_id,
                "encoding": descriptor.encoding_id,
                "schedule_class": compiled.schedule_class_ids[descriptor.schedule_class],
                "operands": compiled.descriptor_rows[i]["operand_count"],
                "results": compiled.descriptor_rows[i]["result_count"],
                "immediates": compiled.descriptor_rows[i]["immediate_count"],
                "fixed_encoding_fields": [
                    {
                        "encoding_field": field_value.encoding_field_id,
                        "value": field_value.value,
                    }
                    for field_value in descriptor.encoding_field_values
                ],
                "effects": compiled.descriptor_rows[i]["effect_count"],
                "operand_forms": compiled.descriptor_rows[i]["operand_form_count"],
                "asm_forms": sum(1 for asm_form in compiled.asm_forms if asm_form.descriptor_ordinal == i),
                "flags": [flag.c_name for flag in descriptor.flags],
            }
        )
    asm_forms = []
    for i, asm_form in enumerate(compiled.asm_forms):
        descriptor = compiled.descriptors[asm_form.descriptor_ordinal]
        asm_forms.append(
            {
                "ordinal": i,
                "mnemonic": asm_form.mnemonic,
                "descriptor": asm_form.descriptor_ordinal,
                "descriptor_key": descriptor.key,
                "results": [descriptor.operands[operand_index].field_name for operand_index in asm_form.result_indices],
                "operands": [descriptor.operands[operand_index].field_name for operand_index in asm_form.operand_indices],
                "immediates": [
                    {
                        "field": descriptor.immediates[immediate.immediate_index].field_name,
                        "name": immediate.name or "",
                    }
                    for immediate in asm_form.immediates
                ],
            }
        )
    manifest = {
        "key": spec.key,
        "target": spec.target_key or "",
        "feature_namespace": spec.feature_key or "",
        "abi_version": LOW_DESCRIPTOR_SET_ABI_VERSION,
        "generator_version": spec.generator_version,
        "stable_id": descriptor_stable_id(spec.key),
        "target_stable_id": descriptor_stable_id(spec.target_key) if spec.target_key is not None else 0,
        "table_counts": {
            "descriptors": len(compiled.descriptors),
            "descriptor_refs": len(compiled.descriptor_refs),
            "operands": len(compiled.operands),
            "immediates": len(compiled.immediates),
            "immediate_encoding_slices": len(compiled.immediate_encoding_slices),
            "enum_domains": len(compiled.enum_domains),
            "enum_values": len(compiled.enum_values),
            "effects": len(compiled.effects),
            "constraints": len(compiled.constraints),
            "reg_classes": len(compiled.reg_classes),
            "register_parts": len(compiled.register_parts),
            "reg_class_alts": len(compiled.reg_class_alts),
            "schedule_classes": len(compiled.schedule_classes),
            "issue_uses": len(compiled.issue_uses),
            "resources": len(compiled.resources),
            "hazards": len(compiled.hazards),
            "pressure_deltas": len(compiled.pressure_deltas),
            "feature_mask_words": len(compiled.feature_mask_words),
            "encoding_field_values": len(compiled.encoding_field_values),
            "operand_forms": len(compiled.operand_forms),
            "operand_form_matches": len(compiled.operand_form_matches),
            "operand_form_operand_indices": len(compiled.operand_form_operand_indices),
            "asm_forms": len(compiled.asm_forms),
            "asm_operand_indices": len(compiled.asm_operand_indices),
            "asm_immediates": len(compiled.asm_immediates),
        },
        "asm_forms": asm_forms,
        "descriptors": descriptors,
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def generate_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
    *,
    format_output: bool = True,
) -> GeneratedDescriptorSet:
    compiled = _compile_descriptor_set(spec, allowlist)
    return GeneratedDescriptorSet(
        header=_emit_header(compiled, format_output=format_output),
        source=_emit_source(compiled, format_output=format_output),
        manifest_json=_emit_manifest_json(compiled),
    )


def generate_descriptor_set_shared_source(
    storage_spec: DescriptorSet,
    view_specs: Sequence[DescriptorSet],
    *,
    format_output: bool = True,
) -> str:
    """Generates one C source containing shared storage and multiple set views.

    Each view must be a descriptor prefix of |storage_spec|. The emitted
    descriptor-set wrapper keeps the view's own identity and descriptor-ref
    lookup table while pointing at the storage spec's dense backing arrays.
    Supporting tables are shared as a storage superset, so extension rows must
    only be reachable through descriptors that are hidden from smaller views.
    Hidden asm rows may exist after descriptor_count; lookup helpers keep those
    rows unreachable from smaller views.
    """

    compiled = _compile_descriptor_set(
        storage_spec,
        allowlist=None,
        allow_ambiguous_asm_mnemonics=True,
    )
    views = tuple(_descriptor_set_view_for_spec(compiled, view_spec) for view_spec in view_specs)
    return _emit_source_for_views(compiled, views=views, format_output=format_output)


def generate_descriptor_set_shared_header(
    storage_spec: DescriptorSet,
    view_spec: DescriptorSet,
    *,
    format_output: bool = True,
) -> str:
    """Generates a public view header for a shared descriptor storage source."""

    compiled = _compile_descriptor_set(
        storage_spec,
        allowlist=None,
        allow_ambiguous_asm_mnemonics=True,
    )
    _descriptor_set_view_for_spec(compiled, view_spec)
    return _emit_header_for_spec(
        compiled,
        view_spec,
        format_output=format_output,
    )


def write_descriptor_set(spec: DescriptorSet, allowlist: DescriptorAllowlist | None = None) -> None:
    from loom.gen import bootstrap as _bootstrap

    generated = generate_descriptor_set(spec, allowlist)
    (_bootstrap.REPO_ROOT / spec.c_header_path).write_text(generated.header, encoding="utf-8")
    (_bootstrap.REPO_ROOT / spec.c_source_path).write_text(generated.source, encoding="utf-8")


def write_descriptor_set_to_paths(
    spec: DescriptorSet,
    *,
    header_path: Path,
    source_path: Path,
    allowlist: DescriptorAllowlist | None = None,
    format_output: bool = False,
) -> None:
    generated = generate_descriptor_set(spec, allowlist, format_output=format_output)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def main() -> None:
    from loom.gen.x86_packed_dot_contract import write_x86_packed_dot_contract_data
    from loom.target.descriptor_sets import iter_checked_in_c_descriptor_sets

    descriptor_sets = tuple(iter_checked_in_c_descriptor_sets())
    for descriptor_set in descriptor_sets:
        write_descriptor_set(descriptor_set)
    write_x86_packed_dot_contract_data()
    print(f"Generated {len(descriptor_sets)} low descriptor sets")


if __name__ == "__main__":
    main()
