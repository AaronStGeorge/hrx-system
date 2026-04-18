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
from dataclasses import dataclass

from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaOperand,
    AmdgpuIsaSpec,
    AmdgpuIsaXmlError,
)
from loom.target.low_descriptors import (
    Constraint,
    Descriptor,
    DescriptorFlag,
    Effect,
    Immediate,
    Operand,
    OperandRole,
)


class AmdgpuDescriptorOverlayError(ValueError):
    """Raised when a Loom overlay does not match parsed AMDGPU ISA facts."""


@dataclass(frozen=True, slots=True)
class AmdgpuOperandOverlay:
    xml_field_name: str
    descriptor_operand: Operand


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorOverlay:
    descriptor_key: str
    instruction_name: str
    encoding_name: str
    semantic_tag: str
    schedule_class: str
    operands: tuple[AmdgpuOperandOverlay, ...]
    encoding_condition: str = "default"
    mnemonic: str | None = None
    encoding_id: int | None = None
    immediate_fields: tuple[str, ...] = ()
    immediates: tuple[Immediate, ...] = ()
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    flags: tuple[DescriptorFlag, ...] = ()


def materialize_amdgpu_descriptor_overlay(
    spec: AmdgpuIsaSpec, overlay: AmdgpuDescriptorOverlay
) -> Descriptor:
    instruction = _select_instruction(spec, overlay)
    encoding = _select_instruction_encoding(instruction, overlay)
    _validate_operand_overlay(overlay, instruction, encoding)
    return Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            operand_overlay.descriptor_operand for operand_overlay in overlay.operands
        ),
        immediates=overlay.immediates,
        effects=overlay.effects,
        constraints=overlay.constraints,
        feature_mask_words=overlay.feature_mask_words,
        encoding_id=encoding.opcode
        if overlay.encoding_id is None
        else overlay.encoding_id,
        schedule_class=overlay.schedule_class,
        flags=overlay.flags,
    )


def materialize_amdgpu_descriptor_overlays(
    spec: AmdgpuIsaSpec, overlays: Iterable[AmdgpuDescriptorOverlay]
) -> tuple[Descriptor, ...]:
    return tuple(
        materialize_amdgpu_descriptor_overlay(spec, overlay) for overlay in overlays
    )


def _select_instruction(
    spec: AmdgpuIsaSpec, overlay: AmdgpuDescriptorOverlay
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
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' found duplicate "
            f"encoding '{overlay.encoding_name}' condition "
            f"'{overlay.encoding_condition}' on instruction '{instruction.name}'"
        )
    return matches[0]


def _validate_operand_overlay(
    overlay: AmdgpuDescriptorOverlay,
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> None:
    _validate_unique_overlay_fields(overlay)
    xml_operands = _explicit_xml_operands_by_field_name(encoding, overlay)
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

    for immediate_field in overlay.immediate_fields:
        xml_operand = xml_operands.get(immediate_field)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing XML immediate field '{immediate_field}' on instruction "
                f"'{instruction.name}' encoding '{encoding.encoding_name}'"
            )
        if not xml_operand.is_input or xml_operand.is_output:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' immediate field "
                f"'{immediate_field}' does not describe an input operand"
            )
        covered_fields.add(immediate_field)

    missing_fields = sorted(set(xml_operands) - covered_fields)
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' does not cover XML "
            f"operand field(s): {missing_text}"
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
    for immediate_field in overlay.immediate_fields:
        if immediate_field in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{immediate_field}' across operands and immediates"
            )
        covered_fields.add(immediate_field)


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
