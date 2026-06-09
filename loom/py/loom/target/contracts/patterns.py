# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type patterns used by target contract guards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

type CTypeExpression = int | str
type ScalarElementPattern = str | Sequence[str]


@dataclass(frozen=True, slots=True)
class TypePattern:
    """Type predicate requested by a target contract guard."""

    kind: str
    element: str | None = None
    elements: tuple[str, ...] = ()
    lanes: int | None = None
    minimum_lanes: CTypeExpression | None = None
    maximum_lanes: CTypeExpression | None = None

    @classmethod
    def scalar(cls, element: ScalarElementPattern) -> Self:
        return cls(kind="scalar", elements=_normalize_elements(element))

    @classmethod
    def vector(
        cls,
        element: ScalarElementPattern,
        *,
        lanes: int | None = None,
        minimum_lanes: CTypeExpression | None = None,
        maximum_lanes: CTypeExpression | None = None,
    ) -> Self:
        return cls(
            kind="vector",
            elements=_normalize_elements(element),
            lanes=lanes,
            minimum_lanes=minimum_lanes,
            maximum_lanes=maximum_lanes,
        )

    @classmethod
    def view(cls, element: ScalarElementPattern) -> Self:
        return cls(kind="view", elements=_normalize_elements(element))

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        if self.element is not None:
            if elements and elements != (self.element,):
                raise ValueError("type pattern element and elements disagree")
            elements = (self.element,)
        object.__setattr__(self, "elements", elements)
        if len(elements) == 1:
            object.__setattr__(self, "element", elements[0])
        if self.kind not in {"scalar", "vector", "view"}:
            raise ValueError(f"unknown type pattern kind '{self.kind}'")
        if not elements:
            raise ValueError(f"{self.kind} type pattern requires an element")
        if len(set(elements)) != len(elements):
            raise ValueError(f"{self.kind} type pattern elements must be unique")
        for element in elements:
            if not isinstance(element, str):
                raise ValueError(f"{self.kind} type pattern element must be a string")
            if not element:
                raise ValueError(f"{self.kind} type pattern element must be non-empty")
        if self.kind == "scalar":
            if (
                self.lanes is not None
                or self.minimum_lanes is not None
                or self.maximum_lanes is not None
            ):
                raise ValueError("scalar type patterns cannot constrain lanes")
            return
        if self.kind == "view":
            if (
                self.lanes is not None
                or self.minimum_lanes is not None
                or self.maximum_lanes is not None
            ):
                raise ValueError("view type patterns cannot constrain lanes")
            return
        if self.lanes is not None:
            if self.minimum_lanes is not None or self.maximum_lanes is not None:
                raise ValueError(
                    "vector type pattern cannot mix exact and ranged lanes"
                )
            if self.lanes < 0:
                raise ValueError("vector lane count must be non-negative")
        elif self.minimum_lanes is not None or self.maximum_lanes is not None:
            if self.minimum_lanes is None or self.maximum_lanes is None:
                raise ValueError("vector lane range requires minimum and maximum")
            _validate_lane_bound(self.minimum_lanes, "minimum vector lane count")
            _validate_lane_bound(self.maximum_lanes, "maximum vector lane count")


def Scalar(element: ScalarElementPattern) -> TypePattern:
    """Returns a scalar type pattern."""

    return TypePattern.scalar(element)


def Vector(
    element: ScalarElementPattern,
    *,
    lanes: int | None = None,
    minimum_lanes: CTypeExpression | None = None,
    maximum_lanes: CTypeExpression | None = None,
) -> TypePattern:
    """Returns a vector type pattern."""

    return TypePattern.vector(
        element,
        lanes=lanes,
        minimum_lanes=minimum_lanes,
        maximum_lanes=maximum_lanes,
    )


def View(element: ScalarElementPattern) -> TypePattern:
    """Returns a view type pattern."""

    return TypePattern.view(element)


def _normalize_elements(element: ScalarElementPattern) -> tuple[str, ...]:
    if isinstance(element, str):
        return (element,)
    return tuple(element)


def _validate_lane_bound(value: CTypeExpression, subject: str) -> None:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{subject} must be non-negative")
        return
    if not value:
        raise ValueError(f"{subject} C expression must be non-empty")
