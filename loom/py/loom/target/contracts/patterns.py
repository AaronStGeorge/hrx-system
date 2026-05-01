# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type patterns used by target contract guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class TypePattern:
    """Type predicate requested by a target contract guard."""

    kind: str
    element: str | None = None
    lanes: int | None = None

    @classmethod
    def scalar(cls, element: str) -> Self:
        return cls(kind="scalar", element=element)

    @classmethod
    def vector(cls, element: str, *, lanes: int | None = None) -> Self:
        return cls(kind="vector", element=element, lanes=lanes)

    def __post_init__(self) -> None:
        if self.kind not in {"scalar", "vector"}:
            raise ValueError(f"unknown type pattern kind '{self.kind}'")
        if not self.element:
            raise ValueError(f"{self.kind} type pattern requires an element")
        if self.lanes is not None and self.lanes < 0:
            raise ValueError("vector lane count must be non-negative")


def Scalar(element: str) -> TypePattern:
    """Returns a scalar type pattern."""

    return TypePattern.scalar(element)


def Vector(element: str, *, lanes: int | None = None) -> TypePattern:
    """Returns a vector type pattern."""

    return TypePattern.vector(element, lanes=lanes)
