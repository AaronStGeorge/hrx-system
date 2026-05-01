# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor immediate projections for target contract rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from loom.dsl import ATTR_TYPE_I64_ARRAY, Op
from loom.target.contracts.descriptors import _require_immediate
from loom.target.contracts.source import _require_attr
from loom.target.low_descriptors import Descriptor


@unique
class AttrProjectKind(Enum):
    """Projection from source op attributes to descriptor immediates."""

    I64_ARRAY_ELEMENT = "i64_array_element"
    EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES = "expand_lane_i64_array_to_byte_lanes"


@dataclass(frozen=True, slots=True)
class AttrProject:
    """Descriptor immediate projection from a source attribute."""

    kind: AttrProjectKind
    source_attr: str
    element: int | None = None
    source_lane_count: int | None = None
    bytes_per_lane: int | None = None
    target_names: tuple[str, ...] = ()

    @classmethod
    def i64_array_element(cls, source_attr: str, *, element: int) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ARRAY_ELEMENT,
            source_attr=source_attr,
            element=element,
        )

    @classmethod
    def expand_lane_i64_array_to_byte_lanes(
        cls,
        *,
        source_attr: str,
        source_lane_count: int,
        bytes_per_lane: int,
        target_names: Sequence[str],
    ) -> Self:
        return cls(
            kind=AttrProjectKind.EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES,
            source_attr=source_attr,
            source_lane_count=source_lane_count,
            bytes_per_lane=bytes_per_lane,
            target_names=tuple(target_names),
        )

    def __post_init__(self) -> None:
        if not self.source_attr:
            raise ValueError(f"{self.kind.value} projection requires a source attr")
        if self.element is not None and self.element < 0:
            raise ValueError(f"{self.kind.value} element must be non-negative")
        if self.source_lane_count is not None and self.source_lane_count < 0:
            raise ValueError(
                f"{self.kind.value} source lane count must be non-negative"
            )
        if self.bytes_per_lane is not None and self.bytes_per_lane <= 0:
            raise ValueError(f"{self.kind.value} bytes per lane must be positive")

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        attr = _require_attr(source_op, self.source_attr, subject)
        if attr.attr_type != ATTR_TYPE_I64_ARRAY:
            raise ValueError(
                f"{source_op.name}: {subject} source attr '{self.source_attr}' "
                "must be an i64_array attr"
            )
        if self.kind == AttrProjectKind.I64_ARRAY_ELEMENT:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            if self.element is None:
                raise ValueError(f"{source_op.name}: {subject} needs an element")
            return
        if bound_immediate_name is not None:
            raise ValueError(
                f"{source_op.name}: {subject} expands multiple immediates and "
                "must not be bound through a single immediate name"
            )
        if self.source_lane_count is None or self.bytes_per_lane is None:
            raise ValueError(
                f"{source_op.name}: {subject} needs source_lane_count and "
                "bytes_per_lane"
            )
        expected_count = self.source_lane_count * self.bytes_per_lane
        if len(self.target_names) != expected_count:
            raise ValueError(
                f"{source_op.name}: {subject} produces {expected_count} "
                f"immediates, got {len(self.target_names)}"
            )
        for name in self.target_names:
            _require_immediate(descriptor, name, subject)
