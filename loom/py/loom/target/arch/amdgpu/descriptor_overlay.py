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
    Immediate,
    Operand,
    OperandFlag,
    OperandRole,
)


class AmdgpuDescriptorOverlayError(ValueError):
    """Raised when a Loom overlay does not match parsed AMDGPU ISA facts."""


AMDGPU_ENCODING_FORMAT_NONE = 0
AMDGPU_ENCODING_FORMAT_SOP1 = 1
AMDGPU_ENCODING_FORMAT_SOP2 = 2
AMDGPU_ENCODING_FORMAT_SOPP = 3
AMDGPU_ENCODING_FORMAT_VOP2 = 4
AMDGPU_ENCODING_FORMAT_VOP2_LITERAL = 5
AMDGPU_ENCODING_FORMAT_VOP3 = 6
AMDGPU_ENCODING_FORMAT_VOP3P = 7
AMDGPU_ENCODING_FORMAT_SMEM = 8
AMDGPU_ENCODING_FORMAT_MUBUF = 9
AMDGPU_ENCODING_FORMAT_VBUFFER = 10

_AMDGPU_ENCODING_FORMATS_BY_XML_NAME = {
    "ENC_SOP1": AMDGPU_ENCODING_FORMAT_SOP1,
    "ENC_SOP2": AMDGPU_ENCODING_FORMAT_SOP2,
    "ENC_SOPP": AMDGPU_ENCODING_FORMAT_SOPP,
    "ENC_VOP2": AMDGPU_ENCODING_FORMAT_VOP2,
    "VOP2_INST_LITERAL": AMDGPU_ENCODING_FORMAT_VOP2_LITERAL,
    "ENC_VOP3": AMDGPU_ENCODING_FORMAT_VOP3,
    "ENC_VOP3P": AMDGPU_ENCODING_FORMAT_VOP3P,
    "VOP3P_MFMA": AMDGPU_ENCODING_FORMAT_VOP3P,
    "ENC_SMEM": AMDGPU_ENCODING_FORMAT_SMEM,
    "ENC_MUBUF": AMDGPU_ENCODING_FORMAT_MUBUF,
    "ENC_VBUFFER": AMDGPU_ENCODING_FORMAT_VBUFFER,
}


@dataclass(frozen=True, slots=True)
class AmdgpuOperandOverlay:
    xml_field_name: str
    descriptor_operand: Operand


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
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = ()
    encoding_condition: str = "default"
    mnemonic: str | None = None
    encoding_format_id: int | None = None
    encoding_id: int | None = None
    immediate_fields: tuple[str, ...] = ()
    immediates: tuple[Immediate, ...] = ()
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    flags: tuple[DescriptorFlag, ...] = ()


def _asm_forms_for_overlay(overlay: AmdgpuDescriptorOverlay) -> tuple[AsmForm, ...]:
    results = []
    operands = []
    for operand_overlay in overlay.operands:
        operand = operand_overlay.descriptor_operand
        if operand.role is OperandRole.RESULT:
            results.append(operand.field_name)
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
    _validate_operand_overlay(overlay, instruction, encoding)
    return Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            operand_overlay.descriptor_operand for operand_overlay in overlay.operands
        )
        + tuple(
            implicit_overlay.descriptor_operand
            for implicit_overlay in overlay.implicit_operands
            if implicit_overlay.descriptor_operand is not None
        ),
        immediates=overlay.immediates,
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
    encoding_format_id = _AMDGPU_ENCODING_FORMATS_BY_XML_NAME.get(overlay.encoding_name)
    if encoding_format_id is None:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references "
            f"unmapped AMDGPU encoding format '{overlay.encoding_name}'"
        )
    return encoding_format_id


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
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} to low operand "
            f"'{descriptor_operand.field_name}' without implicit role"
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
