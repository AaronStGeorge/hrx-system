# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type patterns used by target contract guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

type CTypeExpression = int | str


@dataclass(frozen=True, slots=True)
class TypePattern:
    """Type predicate requested by a target contract guard."""

    kind: str
    element: str | None = None
    lanes: int | None = None
    minimum_lanes: CTypeExpression | None = None
    maximum_lanes: CTypeExpression | None = None

    @classmethod
    def scalar(cls, element: str) -> Self:
        return cls(kind="scalar", element=element)

    @classmethod
    def vector(
        cls,
        element: str,
        *,
        lanes: int | None = None,
        minimum_lanes: CTypeExpression | None = None,
        maximum_lanes: CTypeExpression | None = None,
    ) -> Self:
        return cls(
            kind="vector",
            element=element,
            lanes=lanes,
            minimum_lanes=minimum_lanes,
            maximum_lanes=maximum_lanes,
        )

    def __post_init__(self) -> None:
        if self.kind not in {"scalar", "vector"}:
            raise ValueError(f"unknown type pattern kind '{self.kind}'")
        if not self.element:
            raise ValueError(f"{self.kind} type pattern requires an element")
        if self.kind == "scalar":
            if (
                self.lanes is not None
                or self.minimum_lanes is not None
                or self.maximum_lanes is not None
            ):
                raise ValueError("scalar type patterns cannot constrain lanes")
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


def Scalar(element: str) -> TypePattern:
    """Returns a scalar type pattern."""

    return TypePattern.scalar(element)


def Vector(
    element: str,
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


def _validate_lane_bound(value: CTypeExpression, subject: str) -> None:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{subject} must be non-negative")
        return
    if not value:
        raise ValueError(f"{subject} C expression must be non-empty")
