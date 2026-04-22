# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU ISA XML overlays for target-low descriptor materialization.

The ISA XML importer owns vendor facts. This module owns the Loom-specific
overlay that turns an allowlisted instruction encoding into a low descriptor.
Keeping those layers separate lets the vendor parser remain fact-only while
making semantic choices explicit, reviewable, and testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from loom.target.arch.amdgpu.encoding import (
    amdgpu_encoding_field_id,
    amdgpu_encoding_format_id,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaOperand,
    AmdgpuIsaXmlError,
)
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    Descriptor,
    DescriptorFlag,
    Effect,
    EncodingFieldValue,
    Immediate,
    Operand,
    OperandFlag,
    OperandRole,
)


class AmdgpuDescriptorOverlayError(ValueError):
    """Raised when a Loom overlay does not match parsed AMDGPU ISA facts."""


@dataclass(frozen=True, slots=True)
class AmdgpuOperandOverlay:
    xml_field_name: str
    descriptor_operand: Operand


@dataclass(frozen=True, slots=True)
class AmdgpuIgnoredOperandOverlay:
    xml_field_name: str
    ignore_reason: str
    fixed_encoding_value: int | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuImplicitOperandOverlay:
    operand_type: str
    descriptor_operand: Operand | None = None
    ignore_reason: str | None = None
    data_format_name: str | None = None
    size_bits: int | None = None
    is_input: bool | None = None
    is_output: bool | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorOverlay:
    descriptor_key: str
    instruction_name: str
    encoding_name: str
    semantic_tag: str
    schedule_class: str
    operands: tuple[AmdgpuOperandOverlay, ...]
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = ()
    encoding_condition: str = "default"
    mnemonic: str | None = None
    encoding_format_id: int | None = None
    encoding_id: int | None = None
    immediate_fields: tuple[str, ...] = ()
    immediates: tuple[Immediate, ...] = ()
    fixed_encoding_fields: tuple[tuple[str, int], ...] = ()
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    flags: tuple[DescriptorFlag, ...] = ()
    asm_forms: tuple[AsmForm, ...] | None = None


def _asm_forms_for_overlay(overlay: AmdgpuDescriptorOverlay) -> tuple[AsmForm, ...]:
    if overlay.asm_forms is not None:
        return overlay.asm_forms

    results = []
    operands = []
    for operand_overlay in overlay.operands:
        operand = operand_overlay.descriptor_operand
        if operand.role is OperandRole.RESULT:
            results.append(operand.field_name)
        elif OperandFlag.IMPLICIT in operand.flags:
            continue
        elif operand.role in (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        ):
            operands.append(operand.field_name)
    return (
        AsmForm(
            results=tuple(results),
            operands=tuple(operands),
            immediates=tuple(
                AsmImmediate(immediate.field_name, name=immediate.field_name)
                for immediate in overlay.immediates
            ),
        ),
    )


def materialize_amdgpu_descriptor_overlay(
    spec: AmdgpuIsaFactSource, overlay: AmdgpuDescriptorOverlay
) -> Descriptor:
    instruction = _select_instruction(spec, overlay)
    encoding = _select_instruction_encoding(instruction, overlay)
    _validate_operand_overlay(spec, overlay, instruction, encoding)
    return Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            _materialize_operand_overlay(operand_overlay)
            for operand_overlay in overlay.operands
        )
        + tuple(
            implicit_overlay.descriptor_operand
            for implicit_overlay in overlay.implicit_operands
            if implicit_overlay.descriptor_operand is not None
        ),
        immediates=_materialize_immediates(overlay),
        encoding_field_values=tuple(
            EncodingFieldValue(
                amdgpu_encoding_field_id(field_name),
                value,
            )
            for field_name, value in overlay.fixed_encoding_fields
        )
        + tuple(
            EncodingFieldValue(
                amdgpu_encoding_field_id(ignored_operand.xml_field_name),
                ignored_operand.fixed_encoding_value,
            )
            for ignored_operand in overlay.ignored_operands
            if ignored_operand.fixed_encoding_value is not None
        ),
        asm_forms=_asm_forms_for_overlay(overlay),
        effects=overlay.effects,
        constraints=overlay.constraints,
        feature_mask_words=overlay.feature_mask_words,
        encoding_format_id=_encoding_format_id(overlay),
        encoding_id=encoding.opcode
        if overlay.encoding_id is None
        else overlay.encoding_id,
        schedule_class=overlay.schedule_class,
        flags=overlay.flags,
    )


def _encoding_format_id(overlay: AmdgpuDescriptorOverlay) -> int:
    if overlay.encoding_format_id is not None:
        return overlay.encoding_format_id
    try:
        return amdgpu_encoding_format_id(overlay.encoding_name)
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references "
            f"unmapped AMDGPU encoding format '{overlay.encoding_name}'"
        ) from exc


def _materialize_operand_overlay(operand_overlay: AmdgpuOperandOverlay) -> Operand:
    try:
        return replace(
            operand_overlay.descriptor_operand,
            encoding_field_id=amdgpu_encoding_field_id(operand_overlay.xml_field_name),
        )
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"AMDGPU operand overlay references unmapped encoding field "
            f"'{operand_overlay.xml_field_name}'"
        ) from exc


def _materialize_immediates(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[Immediate, ...]:
    if len(overlay.immediate_fields) != len(overlay.immediates):
        return overlay.immediates
    try:
        return tuple(
            replace(immediate, encoding_field_id=amdgpu_encoding_field_id(field_name))
            for field_name, immediate in zip(
                overlay.immediate_fields, overlay.immediates, strict=False
            )
        )
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references unmapped "
            "immediate encoding field"
        ) from exc


def materialize_amdgpu_descriptor_overlays(
    spec: AmdgpuIsaFactSource, overlays: Iterable[AmdgpuDescriptorOverlay]
) -> tuple[Descriptor, ...]:
    return tuple(
        materialize_amdgpu_descriptor_overlay(spec, overlay) for overlay in overlays
    )


def _select_instruction(
    spec: AmdgpuIsaFactSource, overlay: AmdgpuDescriptorOverlay
) -> AmdgpuIsaInstruction:
    try:
        return spec.select_instructions(
            [overlay.instruction_name], include_aliases=True
        )[0]
    except AmdgpuIsaXmlError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references unknown "
            f"AMDGPU instruction '{overlay.instruction_name}'"
        ) from exc


def _select_instruction_encoding(
    instruction: AmdgpuIsaInstruction, overlay: AmdgpuDescriptorOverlay
) -> AmdgpuIsaInstructionEncoding:
    matches = tuple(
        encoding
        for encoding in instruction.encodings
        if encoding.encoding_name == overlay.encoding_name
        and encoding.condition_name == overlay.encoding_condition
    )
    if not matches:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' could not find "
            f"encoding '{overlay.encoding_name}' condition "
            f"'{overlay.encoding_condition}' on instruction '{instruction.name}'"
        )
    if len(matches) > 1:
        first_match = matches[0]
        if any(match != first_match for match in matches[1:]):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found "
                f"conflicting duplicate encoding '{overlay.encoding_name}' "
                f"condition '{overlay.encoding_condition}' on instruction "
                f"'{instruction.name}'"
            )
        return first_match
    return matches[0]


def _validate_operand_overlay(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> None:
    _validate_unique_overlay_fields(overlay)
    xml_operands = _explicit_xml_operands_by_field_name(encoding, overlay)
    microcode_encoding = spec.encoding_map().get(overlay.encoding_name)
    encoding_fields = (
        {field.name for field in microcode_encoding.fields}
        if microcode_encoding is not None
        else set(xml_operands)
    )
    covered_fields: set[str] = set()
    for operand_overlay in overlay.operands:
        xml_operand = xml_operands.get(operand_overlay.xml_field_name)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing XML operand field '{operand_overlay.xml_field_name}' "
                f"on instruction '{instruction.name}' encoding "
                f"'{encoding.encoding_name}'"
            )
        _validate_operand_role(overlay, operand_overlay, xml_operand)
        covered_fields.add(operand_overlay.xml_field_name)

    for ignored_operand in overlay.ignored_operands:
        xml_operand = xml_operands.get(ignored_operand.xml_field_name)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores missing "
                f"XML operand field '{ignored_operand.xml_field_name}' on "
                f"instruction '{instruction.name}' encoding '{encoding.encoding_name}'"
            )
        if not ignored_operand.ignore_reason:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores XML "
                f"operand field '{ignored_operand.xml_field_name}' without a "
                "named reason"
            )
        if (
            xml_operand.is_binary_microcode_required
            and ignored_operand.fixed_encoding_value is None
        ):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores binary "
                f"microcode field '{ignored_operand.xml_field_name}' without a "
                "fixed encoding value"
            )
        covered_fields.add(ignored_operand.xml_field_name)

    for immediate_field in overlay.immediate_fields:
        xml_operand = xml_operands.get(immediate_field)
        if xml_operand is not None:
            if not xml_operand.is_input or xml_operand.is_output:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' immediate field "
                    f"'{immediate_field}' does not describe an input operand"
                )
            covered_fields.add(immediate_field)
        elif immediate_field not in encoding_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing immediate encoding field '{immediate_field}' on instruction "
                f"'{instruction.name}' encoding '{encoding.encoding_name}'"
            )
    for field_name, _value in overlay.fixed_encoding_fields:
        xml_operand = xml_operands.get(field_name)
        if field_name not in encoding_fields and xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing fixed encoding field '{field_name}' on instruction "
                f"'{instruction.name}' encoding '{encoding.encoding_name}'"
            )
        if xml_operand is not None:
            if not xml_operand.is_input or xml_operand.is_output:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' fixed field "
                    f"'{field_name}' does not describe an input operand"
                )
            covered_fields.add(field_name)

    missing_fields = sorted(set(xml_operands) - covered_fields)
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' does not cover XML "
            f"operand field(s): {missing_text}"
        )
    _validate_implicit_operand_overlays(overlay, instruction, encoding)


def _validate_implicit_operand_overlays(
    overlay: AmdgpuDescriptorOverlay,
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> None:
    implicit_xml_operands = tuple(
        xml_operand for xml_operand in encoding.operands if xml_operand.is_implicit
    )
    if not implicit_xml_operands:
        if overlay.implicit_operands:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' has implicit "
                f"operand decision(s), but instruction '{instruction.name}' "
                f"encoding '{encoding.encoding_name}' has no implicit operands"
            )
        return

    covered_orders: set[int] = set()
    for implicit_overlay in overlay.implicit_operands:
        matches = tuple(
            xml_operand
            for xml_operand in implicit_xml_operands
            if _implicit_overlay_matches_xml_operand(implicit_overlay, xml_operand)
        )
        if not matches:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                "missing implicit XML operand "
                f"'{_format_implicit_overlay_key(implicit_overlay)}' on "
                f"instruction '{instruction.name}' encoding "
                f"'{encoding.encoding_name}'"
            )
        if len(matches) > 1:
            match_text = ", ".join(
                _format_implicit_xml_operand(xml_operand) for xml_operand in matches
            )
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' implicit "
                f"operand decision '{_format_implicit_overlay_key(implicit_overlay)}' "
                f"matches multiple XML operands: {match_text}"
            )
        xml_operand = matches[0]
        if xml_operand.order in covered_orders:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats "
                f"implicit XML operand {_format_implicit_xml_operand(xml_operand)}"
            )
        covered_orders.add(xml_operand.order)
        _validate_implicit_operand_decision(overlay, implicit_overlay, xml_operand)

    missing_operands = tuple(
        xml_operand
        for xml_operand in implicit_xml_operands
        if xml_operand.order not in covered_orders
    )
    if missing_operands:
        missing_text = ", ".join(
            _format_implicit_xml_operand(xml_operand)
            for xml_operand in missing_operands
        )
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' does not cover "
            f"implicit XML operand(s): {missing_text}"
        )


def _implicit_overlay_matches_xml_operand(
    implicit_overlay: AmdgpuImplicitOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> bool:
    if implicit_overlay.operand_type != xml_operand.operand_type:
        return False
    if (
        implicit_overlay.data_format_name is not None
        and implicit_overlay.data_format_name != xml_operand.data_format_name
    ):
        return False
    if (
        implicit_overlay.size_bits is not None
        and implicit_overlay.size_bits != xml_operand.size_bits
    ):
        return False
    if (
        implicit_overlay.is_input is not None
        and implicit_overlay.is_input != xml_operand.is_input
    ):
        return False
    if (
        implicit_overlay.is_output is not None
        and implicit_overlay.is_output != xml_operand.is_output
    ):
        return False
    return True


def _validate_implicit_operand_decision(
    overlay: AmdgpuDescriptorOverlay,
    implicit_overlay: AmdgpuImplicitOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> None:
    if implicit_overlay.descriptor_operand is None:
        if (
            not implicit_overlay.ignore_reason
            or not implicit_overlay.ignore_reason.strip()
        ):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores "
                f"implicit XML operand {_format_implicit_xml_operand(xml_operand)} "
                "without a named reason"
            )
        return

    if implicit_overlay.ignore_reason is not None:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} and also "
            "provides an ignore reason"
        )
    descriptor_operand = implicit_overlay.descriptor_operand
    if descriptor_operand.role is not OperandRole.IMPLICIT:
        explicit_low_roles = (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        )
        if descriptor_operand.role not in explicit_low_roles:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
                f"operand {_format_implicit_xml_operand(xml_operand)} to low operand "
                f"'{descriptor_operand.field_name}' with invalid low role"
            )
    if OperandFlag.IMPLICIT not in descriptor_operand.flags:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} to low operand "
            f"'{descriptor_operand.field_name}' without implicit flag"
        )


def _format_implicit_overlay_key(
    implicit_overlay: AmdgpuImplicitOperandOverlay,
) -> str:
    parts = [f"type={implicit_overlay.operand_type}"]
    if implicit_overlay.data_format_name is not None:
        parts.append(f"format={implicit_overlay.data_format_name}")
    if implicit_overlay.size_bits is not None:
        parts.append(f"bits={implicit_overlay.size_bits}")
    if implicit_overlay.is_input is not None:
        parts.append(f"input={implicit_overlay.is_input}")
    if implicit_overlay.is_output is not None:
        parts.append(f"output={implicit_overlay.is_output}")
    return ",".join(parts)


def _format_implicit_xml_operand(xml_operand: AmdgpuIsaOperand) -> str:
    format_name = (
        "<none>"
        if xml_operand.data_format_name is None
        else xml_operand.data_format_name
    )
    return (
        f"order={xml_operand.order},type={xml_operand.operand_type},"
        f"format={format_name},input={xml_operand.is_input},"
        f"output={xml_operand.is_output}"
    )


def _validate_unique_overlay_fields(overlay: AmdgpuDescriptorOverlay) -> None:
    covered_fields: set[str] = set()
    for operand_overlay in overlay.operands:
        if operand_overlay.xml_field_name in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{operand_overlay.xml_field_name}'"
            )
        covered_fields.add(operand_overlay.xml_field_name)
    for ignored_operand in overlay.ignored_operands:
        if ignored_operand.xml_field_name in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{ignored_operand.xml_field_name}' across operands and ignored "
                "operands"
            )
        covered_fields.add(ignored_operand.xml_field_name)
    for immediate_field in overlay.immediate_fields:
        if immediate_field in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{immediate_field}' across operands and immediates"
            )
        covered_fields.add(immediate_field)
    for fixed_field, _value in overlay.fixed_encoding_fields:
        if fixed_field in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{fixed_field}' across variable and fixed encoding fields"
            )
        covered_fields.add(fixed_field)


def _explicit_xml_operands_by_field_name(
    encoding: AmdgpuIsaInstructionEncoding,
    overlay: AmdgpuDescriptorOverlay,
) -> dict[str, AmdgpuIsaOperand]:
    operands: dict[str, AmdgpuIsaOperand] = {}
    for xml_operand in encoding.operands:
        if xml_operand.is_implicit:
            continue
        if xml_operand.field_name is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found explicit "
                "XML operand without a field name"
            )
        if xml_operand.field_name in operands:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found duplicate "
                f"XML operand field '{xml_operand.field_name}'"
            )
        operands[xml_operand.field_name] = xml_operand
    return operands


def _validate_operand_role(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> None:
    descriptor_operand = operand_overlay.descriptor_operand
    match descriptor_operand.role:
        case OperandRole.RESULT:
            if not xml_operand.is_output or xml_operand.is_input:
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "output")
        case OperandRole.OPERAND | OperandRole.RESOURCE | OperandRole.PREDICATE:
            if not xml_operand.is_input or xml_operand.is_output:
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "input")
        case OperandRole.IMPLICIT:
            if not xml_operand.is_implicit:
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "implicit")
        case OperandRole.OPERAND_RESULT:
            if not xml_operand.is_input or not xml_operand.is_output:
                _raise_role_mismatch(
                    overlay, operand_overlay, xml_operand, "input/output"
                )


def _raise_role_mismatch(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
    expected_role: str,
) -> None:
    raise AmdgpuDescriptorOverlayError(
        f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
        f"'{operand_overlay.xml_field_name}' to low operand "
        f"'{operand_overlay.descriptor_operand.field_name}' as {expected_role}, "
        f"but XML has input={xml_operand.is_input} output={xml_operand.is_output} "
        f"implicit={xml_operand.is_implicit}"
    )
