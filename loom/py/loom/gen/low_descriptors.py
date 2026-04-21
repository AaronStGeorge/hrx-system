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

from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    LOW_DESCRIPTOR_SET_ABI_VERSION,
    AsmForm,
    CEnum,
    Constraint,
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
    ImmediateKind,
    IssueUse,
    Operand,
    OperandFlag,
    OperandRole,
    PressureDelta,
    RegClass,
    RegClassAltFlag,
    Resource,
    ScheduleClass,
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
    resources: list[Resource]
    schedule_classes: list[ScheduleClass]
    enum_domains: list[EnumDomain]
    reg_class_ids: dict[str, int]
    resource_ids: dict[str, int]
    schedule_class_ids: dict[str, int]
    enum_domain_ids: dict[str, int]
    string_pool: _StringPool
    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]]
    operands: list[Operand]
    operand_alt_starts: list[int]
    immediates: list[Immediate]
    enum_values: list[EnumValue]
    immediate_enum_domain_ids: list[int | None]
    effects: list[Effect]
    constraints: list[Constraint]
    issue_uses: list[IssueUse]
    hazards: list[Hazard]
    pressure_deltas: list[PressureDelta]
    feature_mask_words: list[int]
    encoding_field_values: list[EncodingFieldValue]
    descriptor_rows: list[dict[str, int]]
    descriptor_refs: list[tuple[str, int]]
    descriptor_id_refs: list[tuple[int, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[_CompiledAsmForm]
    asm_operand_indices: list[int]
    asm_immediates: list[_CompiledAsmImmediate]
    schedule_rows: list[dict[str, int]]
    enum_domain_rows: list[dict[str, int]]


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
    if value < 0:
        return f"(-INT64_C({abs(value)}))"
    return f"INT64_C({value})"


def _u64_literal(value: int) -> str:
    return f"UINT64_C({value})"


def _hex_u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:x})"


def _descriptor_stable_id(key: str) -> int:
    """Returns a deterministic non-zero 63-bit descriptor identity."""
    value = 0xCBF29CE484222325
    for byte in key.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    value &= 0x7FFFFFFFFFFFFFFF
    return value if value != 0 else 1


def _descriptor_id_constant_name(c_enum_prefix: str, descriptor_key: str) -> str:
    return f"{c_enum_prefix}_DESCRIPTOR_ID_{_c_identifier(descriptor_key).upper()}"


def _descriptor_id_define(c_enum_prefix: str, descriptor_key: str) -> str:
    return f"#define {_descriptor_id_constant_name(c_enum_prefix, descriptor_key)} {_hex_u64_literal(_descriptor_stable_id(descriptor_key))}"


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


def _validate_u64(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{description} does not fit u64")


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
    return result_count


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
        if OperandFlag.IMPLICIT in operand.flags:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' operand field '{field_name}' is marked implicit")
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


def _compile_asm_forms(string_pool: _StringPool, descriptors: Sequence[Descriptor]) -> list[_CompiledAsmForm]:
    compiled_forms: list[_CompiledAsmForm] = []
    seen_mnemonics: dict[str, str] = {}
    for descriptor_ordinal, descriptor in enumerate(descriptors):
        for form_ordinal, asm_form in enumerate(descriptor.asm_forms):
            compiled_form = _compile_asm_form(string_pool, descriptor, descriptor_ordinal, asm_form, form_ordinal)
            previous_descriptor_key = seen_mnemonics.get(compiled_form.mnemonic)
            if previous_descriptor_key is not None:
                raise ValueError(f"asm mnemonic '{compiled_form.mnemonic}' is ambiguous between descriptors '{previous_descriptor_key}' and '{descriptor.key}'")
            seen_mnemonics[compiled_form.mnemonic] = descriptor.key
            compiled_forms.append(compiled_form)
    return sorted(compiled_forms, key=lambda form: form.mnemonic)


def _compile_descriptor_set(spec: DescriptorSet, allowlist: DescriptorAllowlist | None = None) -> _CompiledDescriptorSet:
    if spec.generator_version == 0:
        raise ValueError(f"descriptor set '{spec.key}' has zero generator version")
    reg_class_inputs = _dedupe_by_name(spec.reg_classes, lambda item: item.name)
    resource_inputs = _dedupe_by_name(spec.resources, lambda item: item.name)
    schedule_inputs = _dedupe_by_name(spec.schedule_classes, lambda item: item.name)
    enum_domain_inputs = _dedupe_by_name(spec.enum_domains, lambda item: item.name)
    _dedupe_by_name(spec.descriptors, lambda item: item.key)

    selected_descriptors = _select_descriptors(spec, allowlist)
    if not selected_descriptors:
        raise ValueError(f"descriptor set '{spec.key}' selected no descriptors")

    # Register classes are target vocabulary, not just descriptor closure:
    # low function signatures and allocation diagnostics may reference classes
    # that a tiny allowlisted descriptor slice does not happen to use.
    used_reg_class_names: set[str] = set(reg_class_inputs)
    used_resource_names: set[str] = set()
    used_schedule_names: set[str] = set()
    used_enum_domain_names: set[str] = set()

    for descriptor in selected_descriptors:
        _validate_descriptor_operands(descriptor)
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
            if immediate.kind is ImmediateKind.ENUM:
                if immediate.enum_domain is None:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' has no enum domain")
                if immediate.enum_domain not in enum_domain_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' enum immediate '{immediate.field_name}' references unknown enum domain '{immediate.enum_domain}'")
                used_enum_domain_names.add(immediate.enum_domain)
            elif immediate.enum_domain is not None:
                raise ValueError(f"descriptor '{descriptor.key}' non-enum immediate '{immediate.field_name}' references enum domain '{immediate.enum_domain}'")
        for operand in descriptor.operands:
            _validate_u16(
                operand.encoding_field_id,
                f"descriptor '{descriptor.key}' operand '{operand.field_name}' encoding field id",
            )
            for reg_alt in operand.reg_alts:
                if reg_alt.reg_class is None:
                    if RegClassAltFlag.IMMEDIATE not in reg_alt.flags:
                        raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has a classless alternative without the immediate flag")
                    continue
                if reg_alt.reg_class not in reg_class_inputs:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register class '{reg_alt.reg_class}'")
                used_reg_class_names.add(reg_alt.reg_class)
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

    reg_classes = [reg_class for reg_class in spec.reg_classes if reg_class.name in used_reg_class_names]
    resources = [resource for resource in spec.resources if resource.name in used_resource_names]
    schedule_classes = [schedule_class for schedule_class in spec.schedule_classes if schedule_class.name in used_schedule_names]
    enum_domains = [domain for domain in spec.enum_domains if domain.name in used_enum_domain_names]

    reg_class_ids = {reg_class.name: i for i, reg_class in enumerate(reg_classes)}
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

    asm_forms = _compile_asm_forms(string_pool, selected_descriptors)
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
    effect_group_starts: dict[tuple[Effect, ...], int] = {}
    operands: list[Operand] = []
    operand_alt_starts: list[int] = []
    immediates: list[Immediate] = []
    enum_values: list[EnumValue] = []
    immediate_enum_domain_ids: list[int | None] = []
    effects: list[Effect] = []
    constraints: list[Constraint] = []
    issue_uses: list[IssueUse] = []
    hazards: list[Hazard] = []
    pressure_deltas: list[PressureDelta] = []
    feature_mask_words: list[int] = []
    encoding_field_values: list[EncodingFieldValue] = []
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
        immediates.extend(descriptor.immediates)
        immediate_enum_domain_ids.extend((None if immediate.enum_domain is None else enum_domain_ids[immediate.enum_domain]) for immediate in descriptor.immediates)
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
            }
        )

    descriptor_refs = sorted((descriptor.key, i) for i, descriptor in enumerate(selected_descriptors))
    seen_descriptor_ids: dict[int, str] = {}
    descriptor_id_refs: list[tuple[int, int]] = []
    for descriptor_ordinal, descriptor in enumerate(selected_descriptors):
        stable_id = _descriptor_stable_id(descriptor.key)
        previous_key = seen_descriptor_ids.get(stable_id)
        if previous_key is not None:
            raise ValueError(f"descriptor '{descriptor.key}' stable ID collides with '{previous_key}'")
        seen_descriptor_ids[stable_id] = descriptor.key
        descriptor_id_refs.append((stable_id, descriptor_ordinal))
    descriptor_id_refs.sort()

    return _CompiledDescriptorSet(
        spec=spec,
        descriptors=selected_descriptors,
        reg_classes=reg_classes,
        resources=resources,
        schedule_classes=schedule_classes,
        enum_domains=enum_domains,
        reg_class_ids=reg_class_ids,
        resource_ids=resource_ids,
        schedule_class_ids=schedule_class_ids,
        enum_domain_ids=enum_domain_ids,
        string_pool=string_pool,
        reg_class_alts=reg_class_alts,
        operands=operands,
        operand_alt_starts=operand_alt_starts,
        immediates=immediates,
        enum_values=enum_values,
        immediate_enum_domain_ids=immediate_enum_domain_ids,
        effects=effects,
        constraints=constraints,
        issue_uses=issue_uses,
        hazards=hazards,
        pressure_deltas=pressure_deltas,
        feature_mask_words=feature_mask_words,
        encoding_field_values=encoding_field_values,
        descriptor_rows=descriptor_rows,
        descriptor_refs=descriptor_refs,
        descriptor_id_refs=descriptor_id_refs,
        canonical_asm_form_ordinals=canonical_asm_form_ordinals,
        asm_forms=asm_forms,
        asm_operand_indices=asm_operand_indices,
        asm_immediates=asm_immediates,
        schedule_rows=schedule_rows,
        enum_domain_rows=enum_domain_rows,
    )


def _emit_header(compiled: _CompiledDescriptorSet, *, format_output: bool) -> str:
    spec = compiled.spec
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED by loom.gen.low_descriptors. Do not edit by hand.",
        "",
        f"#ifndef {spec.header_guard}",
        f"#define {spec.header_guard}",
        "",
        '#include "loom/codegen/low/descriptors.h"',
        "",
    ]
    lines.extend(_descriptor_id_define(spec.c_enum_prefix, descriptor.key) for descriptor in compiled.descriptors)
    lines.append("")
    lines.extend(
        [
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"const loom_low_descriptor_set_t* {spec.function_name}(void);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {spec.header_guard}",
        ]
    )
    source = "\n".join(lines) + "\n"
    if not format_output:
        return source
    return _clang_format_source(source, spec.c_header_path)


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
        lines.append(f'    LOOM_BSTRING_LITERAL("\\x{length:02x}", "{escaped}")')
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


def _emit_source(compiled: _CompiledDescriptorSet, *, format_output: bool) -> str:
    spec = compiled.spec
    pool = compiled.string_pool
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED by loom.gen.low_descriptors. Do not edit by hand.",
        "",
        f'#include "{spec.public_header}"',
        "",
        "#include <stdint.h>",
        "",
    ]
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
                f".physical_count = {reg_class.physical_count},",
                f".alias_set_id = {reg_class.alias_set_id},",
                ".spill_class_id = " + ("LOOM_LOW_REG_CLASS_NONE" if reg_class.spill_class is None else str(compiled.reg_class_ids[reg_class.spill_class])) + ",",
                f".spill_slot_space = {reg_class.spill_slot_space.c_name},",
            ]
            for reg_class in compiled.reg_classes
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
                f".encoding_field_id = {operand.encoding_field_id},",
                f".data_format_id = {operand.data_format_id},",
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
                f".kind = {immediate.kind.c_name},",
                f".flags = {_flag_expr(immediate.flags)},",
                f".bit_width = {immediate.bit_width},",
                f".encoding_field_id = {immediate.encoding_field_id},",
                ".enum_domain_id = " + ("LOOM_LOW_ENUM_DOMAIN_NONE" if compiled.immediate_enum_domain_ids[i] is None else str(compiled.immediate_enum_domain_ids[i])) + ",",
                f".encoding_id = {immediate.encoding_id},",
                f".signed_min = {_i64_literal(immediate.signed_min)},",
                f".unsigned_max = {_u64_literal(immediate.unsigned_max)},",
            ]
            for i, immediate in enumerate(compiled.immediates)
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
        "loom_low_descriptor_t",
        spec.c_table_prefix,
        "Descriptors",
        [
            [
                f".key_string_offset = {pool.ref(f'descriptor_{descriptor.key}')},",
                f".stable_id = {_descriptor_id_constant_name(spec.c_enum_prefix, descriptor.key)},",
                f".mnemonic_string_offset = {_optional_string_expr(pool, f'mnemonic_{descriptor.key}' if descriptor.mnemonic is not None else None)},",
                f".semantic_tag_string_offset = {_optional_string_expr(pool, f'semantic_{descriptor.key}' if descriptor.semantic_tag is not None else None)},",
                f".feature_mask_word_start = {compiled.descriptor_rows[i]['feature_mask_word_start']},",
                f".feature_mask_word_count = {compiled.descriptor_rows[i]['feature_mask_word_count']},",
                f".encoding_field_value_start = {compiled.descriptor_rows[i]['encoding_field_value_start']},",
                f".encoding_field_value_count = {compiled.descriptor_rows[i]['encoding_field_value_count']},",
                f".encoding_format_id = {descriptor.encoding_format_id},",
                f".encoding_id = {_encoding_id_expr(descriptor.encoding_id)},",
                f".operand_start = {compiled.descriptor_rows[i]['operand_start']},",
                f".operand_count = {compiled.descriptor_rows[i]['operand_count']},",
                f".result_count = {compiled.descriptor_rows[i]['result_count']},",
                f".immediate_start = {compiled.descriptor_rows[i]['immediate_start']},",
                f".immediate_count = {compiled.descriptor_rows[i]['immediate_count']},",
                f".effect_start = {compiled.descriptor_rows[i]['effect_start']},",
                f".effect_count = {compiled.descriptor_rows[i]['effect_count']},",
                f".constraint_start = {compiled.descriptor_rows[i]['constraint_start']},",
                f".constraint_count = {compiled.descriptor_rows[i]['constraint_count']},",
                f".schedule_class_id = {compiled.schedule_class_ids[descriptor.schedule_class]},",
                f".flags = {_flag_expr(descriptor.flags)},",
                f".canonical_asm_form_ordinal = {_canonical_asm_form_ordinal_expr(compiled.canonical_asm_form_ordinals[i])},",
            ]
            for i, descriptor in enumerate(compiled.descriptors)
        ],
    )
    _emit_array(
        lines,
        "loom_low_descriptor_ref_t",
        spec.c_table_prefix,
        "DescriptorRefs",
        [
            [
                f".key_string_offset = {pool.ref(f'descriptor_{descriptor_key}')},",
                f".descriptor_ordinal = {descriptor_ordinal},",
            ]
            for descriptor_key, descriptor_ordinal in compiled.descriptor_refs
        ],
    )
    _emit_array(
        lines,
        "loom_low_descriptor_id_ref_t",
        spec.c_table_prefix,
        "DescriptorIdRefs",
        [
            [
                f".stable_id = {_hex_u64_literal(stable_id)},",
                f".descriptor_ordinal = {descriptor_ordinal},",
            ]
            for stable_id, descriptor_ordinal in compiled.descriptor_id_refs
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
        [
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
            for asm_form in compiled.asm_forms
        ],
    )

    lines.append(f"static const loom_low_descriptor_set_t k{spec.c_table_prefix}Set = {{")
    lines.extend(
        [
            "    .abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION,",
            f"    .generator_version = {spec.generator_version},",
            f"    .key_string_offset = {pool.ref('set_key')},",
            f"    .target_key_string_offset = {_optional_string_expr(pool, 'target_key' if spec.target_key is not None else None)},",
            f"    .feature_key_string_offset = {_optional_string_expr(pool, 'feature_key' if spec.feature_key is not None else None)},",
            "    .string_table =",
            "        {",
            f"            .data = k{spec.c_table_prefix}StringData,",
            f"            .data_length = sizeof(k{spec.c_table_prefix}StringData) - 1,",
            "        },",
            f"    .descriptors = k{spec.c_table_prefix}Descriptors,",
            f"    .descriptor_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}Descriptors),",
            f"    .descriptor_refs = k{spec.c_table_prefix}DescriptorRefs,",
            f"    .descriptor_ref_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}DescriptorRefs),",
            f"    .descriptor_id_refs = k{spec.c_table_prefix}DescriptorIdRefs,",
            f"    .descriptor_id_ref_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}DescriptorIdRefs),",
        ]
    )

    table_count_fields = {
        "operands": "operand_count",
        "immediates": "immediate_count",
        "enum_domains": "enum_domain_count",
        "enum_values": "enum_value_count",
        "effects": "effect_count",
        "constraints": "constraint_count",
        "reg_classes": "reg_class_count",
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
    }

    def append_optional_table(field_name: str, table_name: str) -> None:
        rows = getattr(compiled, field_name)
        if rows:
            lines.append(f"    .{field_name} = k{spec.c_table_prefix}{table_name},")
            lines.append(f"    .{table_count_fields[field_name]} = IREE_ARRAYSIZE(k{spec.c_table_prefix}{table_name}),")

    append_optional_table("operands", "Operands")
    append_optional_table("immediates", "Immediates")
    append_optional_table("enum_domains", "EnumDomains")
    append_optional_table("enum_values", "EnumValues")
    append_optional_table("effects", "Effects")
    append_optional_table("constraints", "Constraints")
    append_optional_table("reg_classes", "RegClasses")
    append_optional_table("reg_class_alts", "RegClassAlts")
    append_optional_table("schedule_classes", "ScheduleClasses")
    append_optional_table("issue_uses", "IssueUses")
    append_optional_table("resources", "Resources")
    append_optional_table("hazards", "Hazards")
    append_optional_table("pressure_deltas", "PressureDeltas")
    append_optional_table("asm_forms", "AsmForms")
    append_optional_table("asm_operand_indices", "AsmOperandIndices")
    append_optional_table("asm_immediates", "AsmImmediates")
    append_optional_table("encoding_field_values", "EncodingFieldValues")
    if compiled.feature_mask_words:
        lines.append(f"    .feature_mask_words = k{spec.c_table_prefix}FeatureMaskWords,")
        lines.append(f"    .feature_mask_word_count = IREE_ARRAYSIZE(k{spec.c_table_prefix}FeatureMaskWords),")
    lines.append("};")
    lines.append("")
    lines.append(f"const loom_low_descriptor_set_t* {spec.function_name}(void) {{")
    lines.append(f"  return &k{spec.c_table_prefix}Set;")
    lines.append("}")
    source = "\n".join(lines) + "\n"
    if not format_output:
        return source
    return _clang_format_source(source, spec.c_source_path)


def _emit_manifest_json(compiled: _CompiledDescriptorSet) -> str:
    spec = compiled.spec
    descriptors = []
    for i, descriptor in enumerate(compiled.descriptors):
        descriptors.append(
            {
                "ordinal": i,
                "stable_id": f"0x{_descriptor_stable_id(descriptor.key):016x}",
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
        "table_counts": {
            "descriptors": len(compiled.descriptors),
            "descriptor_refs": len(compiled.descriptor_refs),
            "descriptor_id_refs": len(compiled.descriptor_id_refs),
            "operands": len(compiled.operands),
            "immediates": len(compiled.immediates),
            "enum_domains": len(compiled.enum_domains),
            "enum_values": len(compiled.enum_values),
            "effects": len(compiled.effects),
            "constraints": len(compiled.constraints),
            "reg_classes": len(compiled.reg_classes),
            "reg_class_alts": len(compiled.reg_class_alts),
            "schedule_classes": len(compiled.schedule_classes),
            "issue_uses": len(compiled.issue_uses),
            "resources": len(compiled.resources),
            "hazards": len(compiled.hazards),
            "pressure_deltas": len(compiled.pressure_deltas),
            "feature_mask_words": len(compiled.feature_mask_words),
            "encoding_field_values": len(compiled.encoding_field_values),
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
