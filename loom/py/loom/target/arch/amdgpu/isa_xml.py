# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU ISA XML importer for target-low descriptor source data.

The vendor XML is a fact source, not a Loom descriptor language. This module
parses architecture, encoding, instruction, operand, and functional-group facts
into a small normalized model. Target-low semantics, scheduling, register
allocation policy, and descriptor selection belong in separate Loom-owned
overlays keyed by the normalized instruction facts produced here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


class AmdgpuIsaXmlError(ValueError):
    """Raised when AMDGPU ISA XML is malformed or missing required facts."""


@dataclass(frozen=True, slots=True)
class AmdgpuIsaEncoding:
    name: str
    order: int
    bit_count: int
    identifier_mask: int
    identifier_values: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionFlags:
    is_branch: bool
    is_conditional_branch: bool
    is_indirect_branch: bool
    is_program_terminator: bool
    is_immediately_executed: bool


@dataclass(frozen=True, slots=True)
class AmdgpuIsaOperand:
    order: int
    field_name: str | None
    data_format_name: str | None
    operand_type: str
    size_bits: int
    is_input: bool
    is_output: bool
    is_implicit: bool
    is_binary_microcode_required: bool


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionEncoding:
    encoding_name: str
    condition_name: str
    opcode: int
    operands: tuple[AmdgpuIsaOperand, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaFunctionalGroup:
    name: str
    subgroups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstruction:
    name: str
    aliases: tuple[str, ...]
    flags: AmdgpuIsaInstructionFlags
    encodings: tuple[AmdgpuIsaInstructionEncoding, ...]
    functional_groups: tuple[AmdgpuIsaFunctionalGroup, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionEncodingSummary:
    instruction_name: str
    aliases: tuple[str, ...]
    encoding_name: str
    condition_name: str
    opcode: int
    operand_types: tuple[str, ...]
    data_format_names: tuple[str, ...]
    functional_groups: tuple[str, ...]
    functional_subgroups: tuple[str, ...]
    explicit_operand_count: int
    implicit_operand_count: int


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSpec:
    source_name: str
    architecture_name: str
    architecture_id: int
    encodings: tuple[AmdgpuIsaEncoding, ...]
    instructions: tuple[AmdgpuIsaInstruction, ...]

    def encoding_map(self) -> dict[str, AmdgpuIsaEncoding]:
        return {encoding.name: encoding for encoding in self.encodings}

    def instruction_map(
        self, *, include_aliases: bool = False
    ) -> dict[str, AmdgpuIsaInstruction]:
        instructions: dict[str, AmdgpuIsaInstruction] = {}
        for instruction in self.instructions:
            _insert_unique_instruction_name(
                instructions, instruction.name, instruction, self.source_name
            )
            if include_aliases:
                for alias in instruction.aliases:
                    _insert_unique_instruction_name(
                        instructions, alias, instruction, self.source_name
                    )
        return instructions

    def select_instructions(
        self,
        names: Iterable[str],
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstruction, ...]:
        requested_names = tuple(names)
        instructions_by_name = self.instruction_map(include_aliases=include_aliases)
        selected: dict[str, AmdgpuIsaInstruction] = {}
        missing_names: list[str] = []
        for name in requested_names:
            instruction = instructions_by_name.get(name)
            if instruction is None:
                missing_names.append(name)
            else:
                selected[instruction.name] = instruction
        if missing_names:
            missing_text = ", ".join(sorted(missing_names))
            raise AmdgpuIsaXmlError(
                f"{self.source_name}: unknown AMDGPU ISA instruction(s): {missing_text}"
            )
        return tuple(selected[name] for name in sorted(selected))

    def instruction_encoding_summaries(
        self,
        names: Iterable[str] | None = None,
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstructionEncodingSummary, ...]:
        if names is None:
            instructions = self.instructions
        else:
            instructions = self.select_instructions(
                names, include_aliases=include_aliases
            )

        summaries = [
            _summarize_instruction_encoding(instruction, encoding)
            for instruction in instructions
            for encoding in instruction.encodings
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda summary: (
                    summary.instruction_name,
                    summary.encoding_name,
                    summary.condition_name,
                    summary.opcode,
                ),
            )
        )


def parse_amdgpu_isa_xml_path(path: str | Path) -> AmdgpuIsaSpec:
    xml_path = Path(path)
    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as exc:
        raise AmdgpuIsaXmlError(f"{xml_path}: malformed AMDGPU ISA XML: {exc}") from exc
    return _parse_spec_root(tree.getroot(), str(xml_path))


def parse_amdgpu_isa_xml_text(
    xml_text: str,
    *,
    source_name: str = "<string>",
) -> AmdgpuIsaSpec:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise AmdgpuIsaXmlError(
            f"{source_name}: malformed AMDGPU ISA XML: {exc}"
        ) from exc
    return _parse_spec_root(root, source_name)


def _parse_spec_root(root: ElementTree.Element, source_name: str) -> AmdgpuIsaSpec:
    if root.tag != "Spec":
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected root element <Spec>, found <{root.tag}>"
        )

    isa_element = _required_child(root, "ISA", source_name)
    architecture_element = _required_child(isa_element, "Architecture", source_name)
    architecture_name = _required_text(
        architecture_element, "ArchitectureName", source_name
    )
    architecture_id = _required_integer(
        architecture_element, "ArchitectureId", source_name
    )

    encodings_element = _required_child(isa_element, "Encodings", source_name)
    encodings = tuple(
        sorted(
            (
                _parse_encoding(encoding_element, source_name)
                for encoding_element in encodings_element.findall("Encoding")
            ),
            key=lambda encoding: (encoding.order, encoding.name),
        )
    )
    if not encodings:
        raise AmdgpuIsaXmlError(f"{source_name}: <Encodings> has no <Encoding> rows")
    _ensure_unique_names(
        (encoding.name for encoding in encodings),
        "encoding",
        source_name,
    )

    instructions_element = _required_child(isa_element, "Instructions", source_name)
    instructions = tuple(
        sorted(
            (
                _parse_instruction(instruction_element, source_name)
                for instruction_element in instructions_element.findall("Instruction")
            ),
            key=lambda instruction: instruction.name,
        )
    )
    if not instructions:
        raise AmdgpuIsaXmlError(
            f"{source_name}: <Instructions> has no <Instruction> rows"
        )
    _ensure_unique_names(
        (instruction.name for instruction in instructions),
        "instruction",
        source_name,
    )

    return AmdgpuIsaSpec(
        source_name=source_name,
        architecture_name=architecture_name,
        architecture_id=architecture_id,
        encodings=encodings,
        instructions=instructions,
    )


def _parse_encoding(
    encoding_element: ElementTree.Element, source_name: str
) -> AmdgpuIsaEncoding:
    context = _context("Encoding", encoding_element.get("Order"))
    name = _required_text(encoding_element, "EncodingName", source_name)
    order = _required_integer_attribute(encoding_element, "Order", source_name, context)
    bit_count = _required_integer(encoding_element, "BitCount", source_name)
    identifier_mask_element = _required_child(
        encoding_element, "EncodingIdentifierMask", source_name
    )
    identifier_mask = _parse_integer_element(
        identifier_mask_element, source_name, f"{context}/EncodingIdentifierMask"
    )
    identifier_values_element = _required_child(
        encoding_element, "EncodingIdentifiers", source_name
    )
    identifier_values = tuple(
        _parse_integer_element(
            identifier_element,
            source_name,
            f"{context}/EncodingIdentifiers/EncodingIdentifier",
        )
        for identifier_element in identifier_values_element.findall(
            "EncodingIdentifier"
        )
    )
    if not identifier_values:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no identifiers")
    return AmdgpuIsaEncoding(
        name=name,
        order=order,
        bit_count=bit_count,
        identifier_mask=identifier_mask,
        identifier_values=identifier_values,
    )


def _parse_instruction(
    instruction_element: ElementTree.Element, source_name: str
) -> AmdgpuIsaInstruction:
    name = _required_text(instruction_element, "InstructionName", source_name)
    context = _context("Instruction", name)
    aliases_element = instruction_element.find("AliasedInstructionNames")
    aliases: tuple[str, ...]
    if aliases_element is None:
        aliases = ()
    else:
        aliases = tuple(
            _required_element_text(alias_element, source_name, f"{context}/alias")
            for alias_element in aliases_element.findall("InstructionName")
        )
    flags = _parse_instruction_flags(
        _required_child(instruction_element, "InstructionFlags", source_name),
        source_name,
        context,
    )
    instruction_encodings_element = _required_child(
        instruction_element, "InstructionEncodings", source_name
    )
    encodings = tuple(
        _parse_instruction_encoding(encoding_element, source_name, context)
        for encoding_element in instruction_encodings_element.findall(
            "InstructionEncoding"
        )
    )
    if not encodings:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no encodings")

    functional_groups = tuple(
        _parse_functional_group(functional_group_element, source_name, context)
        for functional_group_element in instruction_element.findall("FunctionalGroup")
    )
    if not functional_groups:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no functional group")

    return AmdgpuIsaInstruction(
        name=name,
        aliases=aliases,
        flags=flags,
        encodings=encodings,
        functional_groups=functional_groups,
    )


def _parse_instruction_flags(
    flags_element: ElementTree.Element,
    source_name: str,
    context: str,
) -> AmdgpuIsaInstructionFlags:
    return AmdgpuIsaInstructionFlags(
        is_branch=_required_boolean(flags_element, "IsBranch", source_name, context),
        is_conditional_branch=_required_boolean(
            flags_element, "IsConditionalBranch", source_name, context
        ),
        is_indirect_branch=_required_boolean(
            flags_element, "IsIndirectBranch", source_name, context
        ),
        is_program_terminator=_required_boolean(
            flags_element, "IsProgramTerminator", source_name, context
        ),
        is_immediately_executed=_required_boolean(
            flags_element, "IsImmediatelyExecuted", source_name, context
        ),
    )


def _parse_instruction_encoding(
    encoding_element: ElementTree.Element,
    source_name: str,
    instruction_context: str,
) -> AmdgpuIsaInstructionEncoding:
    encoding_name = _required_text(encoding_element, "EncodingName", source_name)
    context = f"{instruction_context}/InstructionEncoding({encoding_name})"
    condition_name = _required_text(encoding_element, "EncodingCondition", source_name)
    opcode = _required_integer(encoding_element, "Opcode", source_name)
    operands_element = _required_child(encoding_element, "Operands", source_name)
    operands = tuple(
        sorted(
            (
                _parse_operand(operand_element, source_name, context)
                for operand_element in operands_element.findall("Operand")
            ),
            key=lambda operand: operand.order,
        )
    )
    _ensure_unique_names(
        (str(operand.order) for operand in operands),
        f"{context} operand order",
        source_name,
    )
    return AmdgpuIsaInstructionEncoding(
        encoding_name=encoding_name,
        condition_name=condition_name,
        opcode=opcode,
        operands=operands,
    )


def _parse_operand(
    operand_element: ElementTree.Element,
    source_name: str,
    encoding_context: str,
) -> AmdgpuIsaOperand:
    order = _required_integer_attribute(
        operand_element, "Order", source_name, f"{encoding_context}/Operand"
    )
    context = f"{encoding_context}/Operand({order})"
    return AmdgpuIsaOperand(
        order=order,
        field_name=_optional_text(operand_element, "FieldName"),
        data_format_name=_optional_text(operand_element, "DataFormatName"),
        operand_type=_required_text(operand_element, "OperandType", source_name),
        size_bits=_required_integer(operand_element, "OperandSize", source_name),
        is_input=_required_boolean_attribute(
            operand_element, "Input", source_name, context
        ),
        is_output=_required_boolean_attribute(
            operand_element, "Output", source_name, context
        ),
        is_implicit=_required_boolean_attribute(
            operand_element, "IsImplicit", source_name, context
        ),
        is_binary_microcode_required=_required_boolean_attribute(
            operand_element,
            "IsBinaryMicrocodeRequired",
            source_name,
            context,
        ),
    )


def _parse_functional_group(
    functional_group_element: ElementTree.Element,
    source_name: str,
    instruction_context: str,
) -> AmdgpuIsaFunctionalGroup:
    name = _required_text(functional_group_element, "Name", source_name)
    subgroups_element = _required_child(
        functional_group_element, "FunctionalSubgroups", source_name
    )
    subgroups = tuple(
        _required_element_text(
            subgroup_element,
            source_name,
            f"{instruction_context}/FunctionalGroup({name})/Subgroup",
        )
        for subgroup_element in subgroups_element.findall("Subgroup")
    )
    if not subgroups:
        raise AmdgpuIsaXmlError(
            f"{source_name}: {instruction_context}/FunctionalGroup({name}) "
            "has no subgroups"
        )
    return AmdgpuIsaFunctionalGroup(name=name, subgroups=subgroups)


def _summarize_instruction_encoding(
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> AmdgpuIsaInstructionEncodingSummary:
    explicit_operand_count = sum(
        1 for operand in encoding.operands if not operand.is_implicit
    )
    implicit_operand_count = sum(
        1 for operand in encoding.operands if operand.is_implicit
    )
    functional_group_names = tuple(
        sorted({group.name for group in instruction.functional_groups})
    )
    functional_subgroup_names = tuple(
        sorted(
            {
                subgroup
                for group in instruction.functional_groups
                for subgroup in group.subgroups
            }
        )
    )
    return AmdgpuIsaInstructionEncodingSummary(
        instruction_name=instruction.name,
        aliases=instruction.aliases,
        encoding_name=encoding.encoding_name,
        condition_name=encoding.condition_name,
        opcode=encoding.opcode,
        operand_types=tuple(
            sorted({operand.operand_type for operand in encoding.operands})
        ),
        data_format_names=tuple(
            sorted(
                {
                    operand.data_format_name
                    for operand in encoding.operands
                    if operand.data_format_name is not None
                }
            )
        ),
        functional_groups=functional_group_names,
        functional_subgroups=functional_subgroup_names,
        explicit_operand_count=explicit_operand_count,
        implicit_operand_count=implicit_operand_count,
    )


def _insert_unique_instruction_name(
    instructions: dict[str, AmdgpuIsaInstruction],
    name: str,
    instruction: AmdgpuIsaInstruction,
    source_name: str,
) -> None:
    existing_instruction = instructions.get(name)
    if (
        existing_instruction is not None
        and existing_instruction.name != instruction.name
    ):
        raise AmdgpuIsaXmlError(
            f"{source_name}: AMDGPU ISA instruction name or alias '{name}' is "
            f"ambiguous between '{existing_instruction.name}' and "
            f"'{instruction.name}'"
        )
    instructions[name] = instruction


def _ensure_unique_names(names: Iterable[str], noun: str, source_name: str) -> None:
    seen_names: set[str] = set()
    for name in names:
        if name in seen_names:
            raise AmdgpuIsaXmlError(
                f"{source_name}: duplicate AMDGPU ISA {noun} '{name}'"
            )
        seen_names.add(name)


def _required_child(
    element: ElementTree.Element, tag: str, source_name: str
) -> ElementTree.Element:
    child = element.find(tag)
    if child is None:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected <{tag}> under <{element.tag}>"
        )
    return child


def _optional_text(element: ElementTree.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    if not text:
        return None
    return text


def _required_text(element: ElementTree.Element, tag: str, source_name: str) -> str:
    child = _required_child(element, tag, source_name)
    return _required_element_text(child, source_name, f"{element.tag}/{tag}")


def _required_element_text(
    element: ElementTree.Element,
    source_name: str,
    context: str,
) -> str:
    if element.text is None:
        raise AmdgpuIsaXmlError(f"{source_name}: missing text for {context}")
    text = element.text.strip()
    if not text:
        raise AmdgpuIsaXmlError(f"{source_name}: empty text for {context}")
    return text


def _required_integer(element: ElementTree.Element, tag: str, source_name: str) -> int:
    child = _required_child(element, tag, source_name)
    return _parse_integer_element(child, source_name, f"{element.tag}/{tag}")


def _required_integer_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> int:
    text = _required_attribute(element, attribute_name, source_name, context)
    return _parse_integer_text(text, 10, source_name, f"{context}@{attribute_name}")


def _parse_integer_element(
    element: ElementTree.Element,
    source_name: str,
    context: str,
) -> int:
    text = _required_element_text(element, source_name, context)
    radix_text = element.get("Radix")
    radix = (
        10
        if radix_text is None
        else _parse_integer_text(radix_text, 10, source_name, f"{context}@Radix")
    )
    return _parse_integer_text(text, radix, source_name, context)


def _parse_integer_text(
    text: str,
    radix: int,
    source_name: str,
    context: str,
) -> int:
    try:
        return int(text, radix)
    except ValueError as exc:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected base-{radix} integer for {context}, "
            f"found {text!r}"
        ) from exc


def _required_boolean(
    element: ElementTree.Element,
    tag: str,
    source_name: str,
    context: str,
) -> bool:
    text = _required_text(element, tag, source_name)
    return _parse_boolean_text(text, source_name, f"{context}/{tag}")


def _required_boolean_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> bool:
    text = _required_attribute(element, attribute_name, source_name, context)
    return _parse_boolean_text(text, source_name, f"{context}@{attribute_name}")


def _parse_boolean_text(text: str, source_name: str, context: str) -> bool:
    normalized_text = text.lower()
    if normalized_text == "true":
        return True
    if normalized_text == "false":
        return False
    raise AmdgpuIsaXmlError(
        f"{source_name}: expected boolean for {context}, found {text!r}"
    )


def _required_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> str:
    value = element.get(attribute_name)
    if value is None:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected attribute {attribute_name!r} on {context}"
        )
    return value


def _context(noun: str, name: str | None) -> str:
    if name is None:
        return noun
    return f"{noun}({name})"
