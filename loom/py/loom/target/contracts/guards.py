# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selection guard schema for target contract rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import (
    ATTR_TYPE_ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    EnumCase,
    Op,
)
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import (
    _require_attr,
    _require_operand,
    _require_value,
)
from loom.target.low_descriptors import Descriptor


@unique
class GuardKind(Enum):
    """Selection guard kind for descriptor-rule contracts."""

    VALUE_TYPE = "value_type"
    ATTR_KIND = "attr_kind"
    ENUM_ATTR_EQUALS = "enum_attr_equals"
    I64_RANGE = "i64_range"
    DESCRIPTOR_AVAILABLE = "descriptor_available"
    VALUE_MATERIALIZABLE = "value_materializable"
    LOW_VALUE_REGISTER_CLASS = "low_value_register_class"
    VALUE_STATIC_DIM0_MULTIPLE = "value_static_dim0_multiple"
    LOW_VALUE_REGISTER_UNIT_COUNT_EQ = "low_value_register_unit_count_eq"
    OPERAND_SEGMENT_COUNT = "operand_segment_count"
    I64_ARRAY_COUNT = "i64_array_count"
    I64_ARRAY_ELEMENT_RANGE = "i64_array_element_range"
    I64_ARRAY_ELEMENTS_RANGE = "i64_array_elements_range"
    VALUE_SIGNED_BIT_COUNT = "value_signed_bit_count"
    VALUE_UNSIGNED_BIT_COUNT = "value_unsigned_bit_count"
    VALUE_EXACT_I64 = "value_exact_i64"
    VALUE_I64_RANGE = "value_i64_range"


@dataclass(frozen=True, slots=True)
class GuardDiagnostic:
    """Authored diagnostic text for a guard failure."""

    subject_kind: str
    subject_name: str
    reason: str

    def __post_init__(self) -> None:
        if not self.subject_kind:
            raise ValueError("guard diagnostic subject kind must be non-empty")
        if not self.subject_name:
            raise ValueError("guard diagnostic subject name must be non-empty")
        if not self.reason:
            raise ValueError("guard diagnostic reason must be non-empty")


@dataclass(frozen=True, slots=True)
class Guard:
    """Selection predicate evaluated before a contract row can match."""

    kind: GuardKind
    field: str
    other_field: str | None = None
    type_pattern: TypePattern | None = None
    attr_type: str | None = None
    enum_keyword: str | None = None
    count: int | None = None
    element: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    descriptor: Descriptor | None = None
    register_class: str | None = None
    materializer: str | None = None
    diagnostic: GuardDiagnostic | None = None

    @classmethod
    def value_type(
        cls,
        field: str,
        type_pattern: TypePattern,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_TYPE,
            field=field,
            type_pattern=type_pattern,
            diagnostic=diagnostic,
        )

    @classmethod
    def enum_attr_equals(
        cls,
        field: str,
        enum_case: str | EnumCase,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        keyword = enum_case.keyword if isinstance(enum_case, EnumCase) else enum_case
        return cls(
            kind=GuardKind.ENUM_ATTR_EQUALS,
            field=field,
            enum_keyword=keyword,
            diagnostic=diagnostic,
        )

    @classmethod
    def attr_kind(
        cls,
        field: str,
        attr_type: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.ATTR_KIND,
            field=field,
            attr_type=attr_type,
            diagnostic=diagnostic,
        )

    @classmethod
    def i64_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
            diagnostic=diagnostic,
        )

    @classmethod
    def descriptor_available(
        cls,
        descriptor: Descriptor,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.DESCRIPTOR_AVAILABLE,
            field=descriptor.key,
            descriptor=descriptor,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_materializable(
        cls,
        field: str,
        materializer: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_MATERIALIZABLE,
            field=field,
            materializer=materializer,
            diagnostic=diagnostic,
        )

    @classmethod
    def low_value_register_class(
        cls,
        field: str,
        register_class: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.LOW_VALUE_REGISTER_CLASS,
            field=field,
            register_class=register_class,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_static_dim0_multiple(
        cls,
        field: str,
        multiple: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
            field=field,
            count=multiple,
            diagnostic=diagnostic,
        )

    @classmethod
    def low_value_register_unit_count_eq(
        cls,
        field: str,
        other_field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
            field=field,
            other_field=other_field,
            diagnostic=diagnostic,
        )

    @classmethod
    def operand_segment_count(cls, field: str, count: int) -> Self:
        return cls(kind=GuardKind.OPERAND_SEGMENT_COUNT, field=field, count=count)

    @classmethod
    def i64_array_count(cls, field: str, count: int) -> Self:
        return cls(kind=GuardKind.I64_ARRAY_COUNT, field=field, count=count)

    @classmethod
    def i64_array_element_range(
        cls,
        field: str,
        element: int,
        minimum: int,
        maximum: int,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_ARRAY_ELEMENT_RANGE,
            field=field,
            element=element,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def i64_array_elements_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
    ) -> Self:
        return cls(
            kind=GuardKind.I64_ARRAY_ELEMENTS_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def value_signed_bit_count(
        cls,
        field: str,
        bit_count: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_SIGNED_BIT_COUNT,
            field=field,
            count=bit_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_unsigned_bit_count(
        cls,
        field: str,
        bit_count: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_UNSIGNED_BIT_COUNT,
            field=field,
            count=bit_count,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_exact_i64(
        cls,
        field: str,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_EXACT_I64,
            field=field,
            diagnostic=diagnostic,
        )

    @classmethod
    def value_i64_range(
        cls,
        field: str,
        minimum: int,
        maximum: int,
        *,
        diagnostic: GuardDiagnostic | None = None,
    ) -> Self:
        return cls(
            kind=GuardKind.VALUE_I64_RANGE,
            field=field,
            minimum=minimum,
            maximum=maximum,
            diagnostic=diagnostic,
        )

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(f"{self.kind.value} guard requires a field")
        if self.other_field is not None and not self.other_field:
            raise ValueError(f"{self.kind.value} other field must be non-empty")
        if self.attr_type is not None and not self.attr_type:
            raise ValueError(f"{self.kind.value} attr type must be non-empty")
        if self.enum_keyword is not None and not self.enum_keyword:
            raise ValueError(f"{self.kind.value} enum keyword must be non-empty")
        if self.count is not None and self.count < 0:
            raise ValueError(f"{self.kind.value} count must be non-negative")
        if self.element is not None and self.element < 0:
            raise ValueError(f"{self.kind.value} element must be non-negative")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.kind.value} range minimum exceeds maximum")
        if self.register_class is not None and not self.register_class:
            raise ValueError(f"{self.kind.value} register class must be non-empty")
        if self.materializer is not None and not self.materializer:
            raise ValueError(f"{self.kind.value} materializer must be non-empty")

    def validate(self, source_op: Op) -> None:
        subject = f"guard {self.kind.value}"
        if self.kind == GuardKind.VALUE_TYPE:
            _require_value(source_op, self.field, subject)
            if self.type_pattern is None:
                raise ValueError(f"{source_op.name}: {subject} needs a type pattern")
            return
        if self.kind == GuardKind.ATTR_KIND:
            _require_attr(source_op, self.field, subject)
            if self.attr_type is None:
                raise ValueError(f"{source_op.name}: {subject} needs an attr type")
            return
        if self.kind == GuardKind.ENUM_ATTR_EQUALS:
            attr = _require_attr(source_op, self.field, subject)
            if attr.attr_type != ATTR_TYPE_ENUM:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be an enum attr"
                )
            if self.enum_keyword is None:
                raise ValueError(f"{source_op.name}: {subject} needs an enum keyword")
            enum_def = attr.enum_def
            if enum_def is None:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "has no enum definition"
                )
            if self.enum_keyword not in enum_def.keywords:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    f"has no enum case '{self.enum_keyword}'"
                )
            return
        if self.kind == GuardKind.I64_RANGE:
            attr = _require_attr(source_op, self.field, subject)
            if attr.attr_type not in (ATTR_TYPE_I64, ATTR_TYPE_ANY):
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be an i64 or any attr"
                )
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
            return
        if self.kind == GuardKind.DESCRIPTOR_AVAILABLE:
            if self.descriptor is None:
                raise ValueError(f"{source_op.name}: {subject} needs a descriptor")
            return
        if self.kind == GuardKind.VALUE_MATERIALIZABLE:
            _require_operand(source_op, self.field, subject)
            if self.materializer is None:
                raise ValueError(f"{source_op.name}: {subject} needs a materializer")
            return
        if self.kind == GuardKind.LOW_VALUE_REGISTER_CLASS:
            _require_value(source_op, self.field, subject)
            if self.register_class is None:
                raise ValueError(f"{source_op.name}: {subject} needs a register class")
            return
        if self.kind == GuardKind.VALUE_STATIC_DIM0_MULTIPLE:
            _require_value(source_op, self.field, subject)
            if self.count is None or self.count <= 0:
                raise ValueError(
                    f"{source_op.name}: {subject} needs a positive multiple"
                )
            return
        if self.kind == GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ:
            _require_value(source_op, self.field, subject)
            if self.other_field is None:
                raise ValueError(f"{source_op.name}: {subject} needs another value")
            _require_value(source_op, self.other_field, subject)
            return
        if self.kind == GuardKind.OPERAND_SEGMENT_COUNT:
            operand = _require_operand(source_op, self.field, subject)
            if not operand.variadic:
                raise ValueError(
                    f"{source_op.name}: {subject} field '{self.field}' "
                    "must be a variadic operand"
                )
            if self.count is None:
                raise ValueError(f"{source_op.name}: {subject} needs a count")
            return
        if self.kind in (
            GuardKind.VALUE_SIGNED_BIT_COUNT,
            GuardKind.VALUE_UNSIGNED_BIT_COUNT,
        ):
            _require_value(source_op, self.field, subject)
            if self.count is None or self.count <= 0:
                raise ValueError(
                    f"{source_op.name}: {subject} needs a positive bit count"
                )
            return
        if self.kind in (GuardKind.VALUE_EXACT_I64, GuardKind.VALUE_I64_RANGE):
            _validate_value_i64_fact_guard(self, source_op, subject)
            return
        _validate_i64_array_guard(self, source_op, subject)


def _validate_value_i64_fact_guard(
    guard: Guard,
    source_op: Op,
    subject: str,
) -> None:
    _require_value(source_op, guard.field, subject)
    if guard.kind == GuardKind.VALUE_I64_RANGE and (
        guard.minimum is None or guard.maximum is None
    ):
        raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")


def _validate_i64_array_guard(
    guard: Guard,
    source_op: Op,
    subject: str,
) -> None:
    attr = _require_attr(source_op, guard.field, subject)
    if attr.attr_type != ATTR_TYPE_I64_ARRAY:
        raise ValueError(
            f"{source_op.name}: {subject} field '{guard.field}' "
            "must be an i64_array attr"
        )
    if guard.kind == GuardKind.I64_ARRAY_COUNT:
        if guard.count is None:
            raise ValueError(f"{source_op.name}: {subject} needs a count")
        return
    if guard.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE:
        if guard.element is None or guard.minimum is None or guard.maximum is None:
            raise ValueError(
                f"{source_op.name}: {subject} needs element/minimum/maximum"
            )
        return
    if guard.minimum is None or guard.maximum is None:
        raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
