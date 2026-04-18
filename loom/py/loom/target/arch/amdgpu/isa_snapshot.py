# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact AMDGPU ISA fact snapshots for target-low descriptor generation.

The vendor XML is intentionally too large and too broad to be a durable Loom
input. Snapshots keep only the architecture facts, selected encoding bit
layouts, selected instruction encodings, operand facts, aliases, and stable
integer/string identifiers that later descriptor overlays consume.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaBitRange,
    AmdgpuIsaEncoding,
    AmdgpuIsaEncodingField,
    AmdgpuIsaEncodingFieldSummary,
    AmdgpuIsaFunctionalGroup,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaInstructionEncodingSummary,
    AmdgpuIsaInstructionFlags,
    AmdgpuIsaOperand,
    AmdgpuIsaSpec,
    AmdgpuIsaXmlError,
)

AMDGPU_ISA_SNAPSHOT_SCHEMA_VERSION = 1


class AmdgpuIsaSnapshotError(ValueError):
    """Raised when an AMDGPU ISA snapshot or allowlist is malformed."""


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSnapshotAllowlist:
    instruction_names: tuple[str, ...]
    encoding_names: tuple[str, ...]
    include_instruction_aliases: bool = True


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSnapshot:
    snapshot_name: str
    target_family_key: str
    spec: AmdgpuIsaSpec

    @property
    def source_name(self) -> str:
        return self.snapshot_name

    @property
    def architecture_name(self) -> str:
        return self.spec.architecture_name

    @property
    def architecture_id(self) -> int:
        return self.spec.architecture_id

    @property
    def encodings(self) -> tuple[AmdgpuIsaEncoding, ...]:
        return self.spec.encodings

    @property
    def instructions(self) -> tuple[AmdgpuIsaInstruction, ...]:
        return self.spec.instructions

    def encoding_map(self) -> dict[str, AmdgpuIsaEncoding]:
        return self.spec.encoding_map()

    def select_encodings(self, names: Iterable[str]) -> tuple[AmdgpuIsaEncoding, ...]:
        return self.spec.select_encodings(names)

    def encoding_field_summaries(
        self, names: Iterable[str] | None = None
    ) -> tuple[AmdgpuIsaEncodingFieldSummary, ...]:
        return self.spec.encoding_field_summaries(names)

    def instruction_map(
        self, *, include_aliases: bool = False
    ) -> dict[str, AmdgpuIsaInstruction]:
        return self.spec.instruction_map(include_aliases=include_aliases)

    def select_instructions(
        self,
        names: Iterable[str],
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstruction, ...]:
        return self.spec.select_instructions(names, include_aliases=include_aliases)

    def instruction_encoding_summaries(
        self,
        names: Iterable[str] | None = None,
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstructionEncodingSummary, ...]:
        return self.spec.instruction_encoding_summaries(
            names, include_aliases=include_aliases
        )


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSnapshotReport:
    selected_instruction_names: tuple[str, ...]
    selected_encoding_names: tuple[str, ...]
    dropped_instruction_names: tuple[str, ...]
    dropped_encoding_names: tuple[str, ...]
    dropped_instruction_encoding_names: tuple[str, ...]
    unreferenced_encoding_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSnapshotBuildResult:
    snapshot: AmdgpuIsaSnapshot
    report: AmdgpuIsaSnapshotReport


def build_amdgpu_isa_snapshot(
    spec: AmdgpuIsaSpec,
    *,
    target_family_key: str,
    allowlist: AmdgpuIsaSnapshotAllowlist,
    snapshot_name: str | None = None,
) -> AmdgpuIsaSnapshotBuildResult:
    """Filters parsed AMDGPU XML facts into a compact snapshot."""

    target_family_key = target_family_key.strip()
    if not target_family_key:
        raise AmdgpuIsaSnapshotError("AMDGPU ISA snapshot target family key is empty")
    if not allowlist.instruction_names:
        raise AmdgpuIsaSnapshotError(
            "AMDGPU ISA snapshot instruction allowlist is empty"
        )
    if not allowlist.encoding_names:
        raise AmdgpuIsaSnapshotError("AMDGPU ISA snapshot encoding allowlist is empty")
    _validate_unique_names(allowlist.instruction_names, "instruction allowlist")
    _validate_unique_names(allowlist.encoding_names, "encoding allowlist")

    selected_encodings = _select_snapshot_encodings(spec, allowlist.encoding_names)
    selected_encoding_names = tuple(encoding.name for encoding in selected_encodings)
    selected_encoding_name_set = set(selected_encoding_names)
    selected_instructions = _select_snapshot_instructions(spec, allowlist)

    snapshot_instructions: list[AmdgpuIsaInstruction] = []
    dropped_instruction_encoding_names: list[str] = []
    referenced_encoding_names: set[str] = set()
    for instruction in selected_instructions:
        filtered_encodings: list[AmdgpuIsaInstructionEncoding] = []
        for encoding in instruction.encodings:
            if encoding.encoding_name in selected_encoding_name_set:
                filtered_encodings.append(encoding)
                referenced_encoding_names.add(encoding.encoding_name)
            else:
                dropped_instruction_encoding_names.append(
                    _instruction_encoding_report_name(instruction, encoding)
                )
        if not filtered_encodings:
            raise AmdgpuIsaSnapshotError(
                "AMDGPU ISA snapshot instruction "
                f"'{instruction.name}' has no encodings selected by allowlist"
            )
        snapshot_instructions.append(
            AmdgpuIsaInstruction(
                name=instruction.name,
                aliases=instruction.aliases,
                flags=instruction.flags,
                encodings=tuple(filtered_encodings),
                functional_groups=instruction.functional_groups,
            )
        )

    snapshot_name = (
        target_family_key if snapshot_name is None else snapshot_name.strip()
    )
    if not snapshot_name:
        raise AmdgpuIsaSnapshotError("AMDGPU ISA snapshot name is empty")

    snapshot_spec = AmdgpuIsaSpec(
        source_name=snapshot_name,
        architecture_name=spec.architecture_name,
        architecture_id=spec.architecture_id,
        encodings=selected_encodings,
        instructions=tuple(snapshot_instructions),
    )
    selected_instruction_names = tuple(
        instruction.name for instruction in snapshot_instructions
    )
    report = AmdgpuIsaSnapshotReport(
        selected_instruction_names=selected_instruction_names,
        selected_encoding_names=selected_encoding_names,
        dropped_instruction_names=tuple(
            instruction.name
            for instruction in spec.instructions
            if instruction.name not in selected_instruction_names
        ),
        dropped_encoding_names=tuple(
            encoding.name
            for encoding in spec.encodings
            if encoding.name not in selected_encoding_names
        ),
        dropped_instruction_encoding_names=tuple(dropped_instruction_encoding_names),
        unreferenced_encoding_names=tuple(
            encoding_name
            for encoding_name in selected_encoding_names
            if encoding_name not in referenced_encoding_names
        ),
    )
    return AmdgpuIsaSnapshotBuildResult(
        snapshot=AmdgpuIsaSnapshot(
            snapshot_name=snapshot_name,
            target_family_key=target_family_key,
            spec=snapshot_spec,
        ),
        report=report,
    )


def format_amdgpu_isa_snapshot_json(snapshot: AmdgpuIsaSnapshot) -> str:
    return json.dumps(_snapshot_to_json_object(snapshot), indent=2) + "\n"


def parse_amdgpu_isa_snapshot_json(
    json_text: str,
    *,
    source_name: str = "<snapshot>",
) -> AmdgpuIsaSnapshot:
    try:
        root = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise AmdgpuIsaSnapshotError(
            f"{source_name}: malformed AMDGPU ISA snapshot JSON: {exc}"
        ) from exc
    root_object = _required_object(root, source_name)
    schema_version = _required_int(root_object, "schema_version", source_name)
    if schema_version != AMDGPU_ISA_SNAPSHOT_SCHEMA_VERSION:
        raise AmdgpuIsaSnapshotError(
            f"{source_name}: unsupported AMDGPU ISA snapshot schema_version "
            f"{schema_version}"
        )

    snapshot_name = _required_string(root_object, "snapshot_name", source_name)
    target_family_key = _required_string(root_object, "target_family_key", source_name)
    architecture_object = _required_object(
        root_object.get("architecture"), f"{source_name}/architecture"
    )
    architecture_name = _required_string(
        architecture_object, "name", f"{source_name}/architecture"
    )
    architecture_id = _required_int(
        architecture_object, "id", f"{source_name}/architecture"
    )
    spec = AmdgpuIsaSpec(
        source_name=snapshot_name,
        architecture_name=architecture_name,
        architecture_id=architecture_id,
        encodings=tuple(
            _encoding_from_json_object(encoding_object, source_name)
            for encoding_object in _required_list(root_object, "encodings", source_name)
        ),
        instructions=tuple(
            _instruction_from_json_object(instruction_object, source_name)
            for instruction_object in _required_list(
                root_object, "instructions", source_name
            )
        ),
    )
    _validate_snapshot_spec(spec, source_name)
    return AmdgpuIsaSnapshot(
        snapshot_name=snapshot_name,
        target_family_key=target_family_key,
        spec=spec,
    )


def format_amdgpu_isa_snapshot_report_json(
    result: AmdgpuIsaSnapshotBuildResult,
) -> str:
    report = result.report
    manifest = {
        "schema_version": AMDGPU_ISA_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_name": result.snapshot.snapshot_name,
        "target_family_key": result.snapshot.target_family_key,
        "architecture": {
            "name": result.snapshot.architecture_name,
            "id": result.snapshot.architecture_id,
        },
        "selected": {
            "instructions": list(report.selected_instruction_names),
            "encodings": list(report.selected_encoding_names),
        },
        "dropped": {
            "instructions": list(report.dropped_instruction_names),
            "encodings": list(report.dropped_encoding_names),
            "instruction_encodings": list(report.dropped_instruction_encoding_names),
        },
        "unreferenced_selected_encodings": list(report.unreferenced_encoding_names),
    }
    return json.dumps(manifest, indent=2) + "\n"


def _select_snapshot_encodings(
    spec: AmdgpuIsaSpec, encoding_names: Iterable[str]
) -> tuple[AmdgpuIsaEncoding, ...]:
    try:
        return spec.select_encodings(encoding_names)
    except AmdgpuIsaXmlError as exc:
        raise AmdgpuIsaSnapshotError(str(exc)) from exc


def _select_snapshot_instructions(
    spec: AmdgpuIsaSpec, allowlist: AmdgpuIsaSnapshotAllowlist
) -> tuple[AmdgpuIsaInstruction, ...]:
    try:
        return spec.select_instructions(
            allowlist.instruction_names,
            include_aliases=allowlist.include_instruction_aliases,
        )
    except AmdgpuIsaXmlError as exc:
        raise AmdgpuIsaSnapshotError(str(exc)) from exc


def _snapshot_to_json_object(snapshot: AmdgpuIsaSnapshot) -> dict[str, Any]:
    return {
        "schema_version": AMDGPU_ISA_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_name": snapshot.snapshot_name,
        "target_family_key": snapshot.target_family_key,
        "architecture": {
            "name": snapshot.architecture_name,
            "id": snapshot.architecture_id,
        },
        "encodings": [
            _encoding_to_json_object(encoding) for encoding in snapshot.encodings
        ],
        "instructions": [
            _instruction_to_json_object(instruction)
            for instruction in snapshot.instructions
        ],
    }


def _encoding_to_json_object(encoding: AmdgpuIsaEncoding) -> dict[str, Any]:
    return {
        "name": encoding.name,
        "order": encoding.order,
        "bit_count": encoding.bit_count,
        "identifier_mask": encoding.identifier_mask,
        "identifier_values": list(encoding.identifier_values),
        "fields": [_encoding_field_to_json_object(field) for field in encoding.fields],
    }


def _encoding_from_json_object(value: Any, source_name: str) -> AmdgpuIsaEncoding:
    encoding_object = _required_object(value, f"{source_name}/encodings[]")
    return AmdgpuIsaEncoding(
        name=_required_string(encoding_object, "name", source_name),
        order=_required_int(encoding_object, "order", source_name),
        bit_count=_required_int(encoding_object, "bit_count", source_name),
        identifier_mask=_required_int(encoding_object, "identifier_mask", source_name),
        identifier_values=tuple(
            _required_int_value(value, f"{source_name}/identifier_values[]")
            for value in _required_list(
                encoding_object, "identifier_values", source_name
            )
        ),
        fields=tuple(
            _encoding_field_from_json_object(field_object, source_name)
            for field_object in _required_list(encoding_object, "fields", source_name)
        ),
    )


def _encoding_field_to_json_object(
    field: AmdgpuIsaEncodingField,
) -> dict[str, Any]:
    return {
        "name": field.name,
        "is_conditional": field.is_conditional,
        "ranges": [_bit_range_to_json_object(bit_range) for bit_range in field.ranges],
    }


def _encoding_field_from_json_object(
    value: Any, source_name: str
) -> AmdgpuIsaEncodingField:
    field_object = _required_object(value, f"{source_name}/fields[]")
    return AmdgpuIsaEncodingField(
        name=_required_string(field_object, "name", source_name),
        is_conditional=_required_bool(field_object, "is_conditional", source_name),
        ranges=tuple(
            _bit_range_from_json_object(bit_range_object, source_name)
            for bit_range_object in _required_list(field_object, "ranges", source_name)
        ),
    )


def _bit_range_to_json_object(bit_range: AmdgpuIsaBitRange) -> dict[str, int]:
    result = {
        "order": bit_range.order,
        "bit_count": bit_range.bit_count,
        "bit_offset": bit_range.bit_offset,
    }
    if bit_range.padding_bit_count:
        result["padding_bit_count"] = bit_range.padding_bit_count
        result["padding_value"] = bit_range.padding_value
    return result


def _bit_range_from_json_object(value: Any, source_name: str) -> AmdgpuIsaBitRange:
    bit_range_object = _required_object(value, f"{source_name}/ranges[]")
    return AmdgpuIsaBitRange(
        order=_required_int(bit_range_object, "order", source_name),
        bit_count=_required_int(bit_range_object, "bit_count", source_name),
        bit_offset=_required_int(bit_range_object, "bit_offset", source_name),
        padding_bit_count=_optional_int(
            bit_range_object, "padding_bit_count", source_name
        ),
        padding_value=_optional_int(bit_range_object, "padding_value", source_name),
    )


def _instruction_to_json_object(
    instruction: AmdgpuIsaInstruction,
) -> dict[str, Any]:
    return {
        "name": instruction.name,
        "aliases": list(instruction.aliases),
        "flags": _instruction_flags_to_json_object(instruction.flags),
        "encodings": [
            _instruction_encoding_to_json_object(encoding)
            for encoding in instruction.encodings
        ],
        "functional_groups": [
            _functional_group_to_json_object(group)
            for group in instruction.functional_groups
        ],
    }


def _instruction_from_json_object(value: Any, source_name: str) -> AmdgpuIsaInstruction:
    instruction_object = _required_object(value, f"{source_name}/instructions[]")
    return AmdgpuIsaInstruction(
        name=_required_string(instruction_object, "name", source_name),
        aliases=tuple(
            _required_string_value(alias, f"{source_name}/aliases[]")
            for alias in _required_list(instruction_object, "aliases", source_name)
        ),
        flags=_instruction_flags_from_json_object(
            _required_object(instruction_object.get("flags"), f"{source_name}/flags"),
            source_name,
        ),
        encodings=tuple(
            _instruction_encoding_from_json_object(encoding_object, source_name)
            for encoding_object in _required_list(
                instruction_object, "encodings", source_name
            )
        ),
        functional_groups=tuple(
            _functional_group_from_json_object(group_object, source_name)
            for group_object in _required_list(
                instruction_object, "functional_groups", source_name
            )
        ),
    )


def _instruction_flags_to_json_object(
    flags: AmdgpuIsaInstructionFlags,
) -> dict[str, bool]:
    return {
        "is_branch": flags.is_branch,
        "is_conditional_branch": flags.is_conditional_branch,
        "is_indirect_branch": flags.is_indirect_branch,
        "is_program_terminator": flags.is_program_terminator,
        "is_immediately_executed": flags.is_immediately_executed,
    }


def _instruction_flags_from_json_object(
    flags_object: dict[str, Any], source_name: str
) -> AmdgpuIsaInstructionFlags:
    return AmdgpuIsaInstructionFlags(
        is_branch=_required_bool(flags_object, "is_branch", source_name),
        is_conditional_branch=_required_bool(
            flags_object, "is_conditional_branch", source_name
        ),
        is_indirect_branch=_required_bool(
            flags_object, "is_indirect_branch", source_name
        ),
        is_program_terminator=_required_bool(
            flags_object, "is_program_terminator", source_name
        ),
        is_immediately_executed=_required_bool(
            flags_object, "is_immediately_executed", source_name
        ),
    )


def _instruction_encoding_to_json_object(
    encoding: AmdgpuIsaInstructionEncoding,
) -> dict[str, Any]:
    return {
        "encoding_name": encoding.encoding_name,
        "condition_name": encoding.condition_name,
        "opcode": encoding.opcode,
        "operands": [_operand_to_json_object(operand) for operand in encoding.operands],
    }


def _instruction_encoding_from_json_object(
    value: Any, source_name: str
) -> AmdgpuIsaInstructionEncoding:
    encoding_object = _required_object(value, f"{source_name}/instruction_encodings[]")
    return AmdgpuIsaInstructionEncoding(
        encoding_name=_required_string(encoding_object, "encoding_name", source_name),
        condition_name=_required_string(encoding_object, "condition_name", source_name),
        opcode=_required_int(encoding_object, "opcode", source_name),
        operands=tuple(
            _operand_from_json_object(operand_object, source_name)
            for operand_object in _required_list(
                encoding_object, "operands", source_name
            )
        ),
    )


def _operand_to_json_object(operand: AmdgpuIsaOperand) -> dict[str, Any]:
    result: dict[str, Any] = {
        "order": operand.order,
        "operand_type": operand.operand_type,
        "size_bits": operand.size_bits,
        "is_input": operand.is_input,
        "is_output": operand.is_output,
        "is_implicit": operand.is_implicit,
        "is_binary_microcode_required": operand.is_binary_microcode_required,
    }
    if operand.field_name is not None:
        result["field_name"] = operand.field_name
    if operand.data_format_name is not None:
        result["data_format_name"] = operand.data_format_name
    return result


def _operand_from_json_object(value: Any, source_name: str) -> AmdgpuIsaOperand:
    operand_object = _required_object(value, f"{source_name}/operands[]")
    return AmdgpuIsaOperand(
        order=_required_int(operand_object, "order", source_name),
        field_name=_optional_string(operand_object, "field_name", source_name),
        data_format_name=_optional_string(
            operand_object, "data_format_name", source_name
        ),
        operand_type=_required_string(operand_object, "operand_type", source_name),
        size_bits=_required_int(operand_object, "size_bits", source_name),
        is_input=_required_bool(operand_object, "is_input", source_name),
        is_output=_required_bool(operand_object, "is_output", source_name),
        is_implicit=_required_bool(operand_object, "is_implicit", source_name),
        is_binary_microcode_required=_required_bool(
            operand_object, "is_binary_microcode_required", source_name
        ),
    )


def _functional_group_to_json_object(
    group: AmdgpuIsaFunctionalGroup,
) -> dict[str, Any]:
    return {
        "name": group.name,
        "subgroups": list(group.subgroups),
    }


def _functional_group_from_json_object(
    value: Any, source_name: str
) -> AmdgpuIsaFunctionalGroup:
    group_object = _required_object(value, f"{source_name}/functional_groups[]")
    return AmdgpuIsaFunctionalGroup(
        name=_required_string(group_object, "name", source_name),
        subgroups=tuple(
            _required_string_value(subgroup, f"{source_name}/subgroups[]")
            for subgroup in _required_list(group_object, "subgroups", source_name)
        ),
    )


def _instruction_encoding_report_name(
    instruction: AmdgpuIsaInstruction, encoding: AmdgpuIsaInstructionEncoding
) -> str:
    return (
        f"{instruction.name}/{encoding.encoding_name}/"
        f"{encoding.condition_name}/opcode={encoding.opcode}"
    )


def _validate_unique_names(names: Iterable[str], noun: str) -> None:
    seen_names: set[str] = set()
    for name in names:
        if name in seen_names:
            raise AmdgpuIsaSnapshotError(f"AMDGPU ISA snapshot {noun} repeats '{name}'")
        seen_names.add(name)


def _validate_snapshot_spec(spec: AmdgpuIsaSpec, source_name: str) -> None:
    if not spec.encodings:
        raise AmdgpuIsaSnapshotError(f"{source_name}: snapshot has no encodings")
    if not spec.instructions:
        raise AmdgpuIsaSnapshotError(f"{source_name}: snapshot has no instructions")
    _validate_unique_names((encoding.name for encoding in spec.encodings), "encoding")
    _validate_unique_names(
        (instruction.name for instruction in spec.instructions), "instruction"
    )
    for encoding in spec.encodings:
        if not encoding.identifier_values:
            raise AmdgpuIsaSnapshotError(
                f"{source_name}: encoding '{encoding.name}' has no identifiers"
            )
        if not encoding.fields:
            raise AmdgpuIsaSnapshotError(
                f"{source_name}: encoding '{encoding.name}' has no fields"
            )
        _validate_unique_names(
            (field.name for field in encoding.fields),
            f"encoding '{encoding.name}' field",
        )
        for field in encoding.fields:
            if not field.ranges:
                raise AmdgpuIsaSnapshotError(
                    f"{source_name}: encoding '{encoding.name}' field "
                    f"'{field.name}' has no bit ranges"
                )
    for instruction in spec.instructions:
        if not instruction.encodings:
            raise AmdgpuIsaSnapshotError(
                f"{source_name}: instruction '{instruction.name}' has no encodings"
            )
        if not instruction.functional_groups:
            raise AmdgpuIsaSnapshotError(
                f"{source_name}: instruction '{instruction.name}' has no "
                "functional groups"
            )


def _required_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AmdgpuIsaSnapshotError(f"{context}: expected object")
    return value


def _required_list(mapping: dict[str, Any], key: str, source_name: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise AmdgpuIsaSnapshotError(f"{source_name}: expected list field '{key}'")
    return value


def _required_string(mapping: dict[str, Any], key: str, source_name: str) -> str:
    return _required_string_value(mapping.get(key), f"{source_name}/{key}")


def _required_string_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise AmdgpuIsaSnapshotError(f"{context}: expected non-empty string")
    return value


def _optional_string(mapping: dict[str, Any], key: str, source_name: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AmdgpuIsaSnapshotError(
            f"{source_name}/{key}: expected non-empty string or null"
        )
    return value


def _required_int(mapping: dict[str, Any], key: str, source_name: str) -> int:
    return _required_int_value(mapping.get(key), f"{source_name}/{key}")


def _required_int_value(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AmdgpuIsaSnapshotError(f"{context}: expected integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str, source_name: str) -> int:
    value = mapping.get(key)
    if value is None:
        return 0
    return _required_int_value(value, f"{source_name}/{key}")


def _required_bool(mapping: dict[str, Any], key: str, source_name: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise AmdgpuIsaSnapshotError(f"{source_name}/{key}: expected boolean")
    return value
