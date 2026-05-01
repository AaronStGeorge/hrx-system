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

from loom.dsl import ATTR_TYPE_ENUM, ATTR_TYPE_I64_ARRAY, EnumCase, Op
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import (
    _require_attr,
    _require_operand,
    _require_value,
)


@unique
class GuardKind(Enum):
    """Selection guard kind for descriptor-rule contracts."""

    VALUE_TYPE = "value_type"
    ENUM_ATTR_EQUALS = "enum_attr_equals"
    OPERAND_SEGMENT_COUNT = "operand_segment_count"
    I64_ARRAY_COUNT = "i64_array_count"
    I64_ARRAY_ELEMENT_RANGE = "i64_array_element_range"
    I64_ARRAY_ELEMENTS_RANGE = "i64_array_elements_range"


@dataclass(frozen=True, slots=True)
class Guard:
    """Selection predicate evaluated before a contract row can match."""

    kind: GuardKind
    field: str
    type_pattern: TypePattern | None = None
    enum_keyword: str | None = None
    count: int | None = None
    element: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    @classmethod
    def value_type(cls, field: str, type_pattern: TypePattern) -> Self:
        return cls(
            kind=GuardKind.VALUE_TYPE,
            field=field,
            type_pattern=type_pattern,
        )

    @classmethod
    def enum_attr_equals(cls, field: str, enum_case: str | EnumCase) -> Self:
        keyword = enum_case.keyword if isinstance(enum_case, EnumCase) else enum_case
        return cls(
            kind=GuardKind.ENUM_ATTR_EQUALS,
            field=field,
            enum_keyword=keyword,
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

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(f"{self.kind.value} guard requires a field")
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

    def validate(self, source_op: Op) -> None:
        subject = f"guard {self.kind.value}"
        if self.kind == GuardKind.VALUE_TYPE:
            _require_value(source_op, self.field, subject)
            if self.type_pattern is None:
                raise ValueError(f"{source_op.name}: {subject} needs a type pattern")
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
        attr = _require_attr(source_op, self.field, subject)
        if attr.attr_type != ATTR_TYPE_I64_ARRAY:
            raise ValueError(
                f"{source_op.name}: {subject} field '{self.field}' "
                "must be an i64_array attr"
            )
        if self.kind == GuardKind.I64_ARRAY_COUNT:
            if self.count is None:
                raise ValueError(f"{source_op.name}: {subject} needs a count")
            return
        if self.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE:
            if self.element is None or self.minimum is None or self.maximum is None:
                raise ValueError(
                    f"{source_op.name}: {subject} needs element/minimum/maximum"
                )
            return
        if self.minimum is None or self.maximum is None:
            raise ValueError(f"{source_op.name}: {subject} needs minimum/maximum")
