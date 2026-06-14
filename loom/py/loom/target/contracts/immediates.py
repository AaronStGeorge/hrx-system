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
from loom.target.contracts.source import _require_attr, _require_value
from loom.target.low_descriptors import Descriptor


@unique
class AttrProjectKind(Enum):
    """Projection from source op attributes to descriptor immediates."""

    DIRECT = "direct"
    I64_ARRAY_ELEMENT = "i64_array_element"
    I64_ARRAY_PACK_ELEMENTS = "i64_array_pack_elements"
    EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES = "expand_lane_i64_array_to_byte_lanes"


@unique
class ValueProjectKind(Enum):
    """Projection from source value facts to descriptor immediates."""

    EXACT_I64 = "exact_i64"
    EXACT_I64_NEGATE = "exact_i64_negate"
    EXACT_I64_LOG2 = "exact_i64_log2"
    EXACT_I64_MINUS_ONE = "exact_i64_minus_one"
    U32_DIVISOR_MAGIC_MULTIPLIER = "u32_divisor_magic_multiplier"
    U32_DIVISOR_MAGIC_SHIFT = "u32_divisor_magic_shift"
    I32_AS_U32_BITS = "i32_as_u32_bits"
    F64_AS_F16_BITS = "f64_as_f16_bits"
    F64_AS_BF16_BITS = "f64_as_bf16_bits"
    F64_AS_F32_BITS = "f64_as_f32_bits"
    F64_AS_F64_BITS = "f64_as_f64_bits"


@unique
class SourceMemoryProjectKind(Enum):
    """Projection from a selected source-memory access to descriptor immediates."""

    STATIC_BYTE_OFFSET = "static_byte_offset"
    DYNAMIC_BYTE_STRIDE = "dynamic_byte_stride"


@dataclass(frozen=True, slots=True)
class AttrProject:
    """Descriptor immediate projection from a source attribute."""

    kind: AttrProjectKind
    source_attr: str
    element: int | None = None
    count: int | None = None
    bit_width: int | None = None
    target_bit_offset: int = 0
    source_lane_count: int | None = None
    bytes_per_lane: int | None = None
    target_names: tuple[str, ...] = ()

    @classmethod
    def direct(cls, source_attr: str) -> Self:
        return cls(kind=AttrProjectKind.DIRECT, source_attr=source_attr)

    @classmethod
    def i64_array_element(
        cls,
        source_attr: str,
        *,
        element: int,
        target_bit_offset: int = 0,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ARRAY_ELEMENT,
            source_attr=source_attr,
            element=element,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i64_array_pack_elements(
        cls,
        source_attr: str,
        *,
        element: int,
        count: int,
        bit_width: int,
        target_bit_offset: int = 0,
    ) -> Self:
        return cls(
            kind=AttrProjectKind.I64_ARRAY_PACK_ELEMENTS,
            source_attr=source_attr,
            element=element,
            count=count,
            bit_width=bit_width,
            target_bit_offset=target_bit_offset,
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
        if self.count is not None and self.count <= 0:
            raise ValueError(f"{self.kind.value} count must be positive")
        if self.bit_width is not None and self.bit_width <= 0:
            raise ValueError(f"{self.kind.value} bit width must be positive")
        if self.target_bit_offset < 0:
            raise ValueError(
                f"{self.kind.value} target bit offset must be non-negative"
            )
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
        if self.kind == AttrProjectKind.DIRECT:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            return
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
        if self.kind == AttrProjectKind.I64_ARRAY_PACK_ELEMENTS:
            if bound_immediate_name is None:
                raise ValueError(
                    f"{source_op.name}: {subject} must bind one descriptor immediate"
                )
            _require_immediate(descriptor, bound_immediate_name, subject)
            if self.element is None or self.count is None or self.bit_width is None:
                raise ValueError(
                    f"{source_op.name}: {subject} needs element/count/bit_width"
                )
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


@dataclass(frozen=True, slots=True)
class ValueProject:
    """Descriptor immediate projection from source value facts."""

    kind: ValueProjectKind
    source_value: str
    target_bit_offset: int = 0

    @classmethod
    def exact_i64(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_negate(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_NEGATE,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_log2(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_LOG2,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def exact_i64_minus_one(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.EXACT_I64_MINUS_ONE,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def u32_divisor_magic_multiplier(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.U32_DIVISOR_MAGIC_MULTIPLIER,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def u32_divisor_magic_shift(
        cls, source_value: str, *, target_bit_offset: int = 0
    ) -> Self:
        return cls(
            kind=ValueProjectKind.U32_DIVISOR_MAGIC_SHIFT,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def i32_as_u32_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.I32_AS_U32_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f16_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F16_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_bf16_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_BF16_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f32_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F32_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    @classmethod
    def f64_as_f64_bits(cls, source_value: str, *, target_bit_offset: int = 0) -> Self:
        return cls(
            kind=ValueProjectKind.F64_AS_F64_BITS,
            source_value=source_value,
            target_bit_offset=target_bit_offset,
        )

    def __post_init__(self) -> None:
        if not self.source_value:
            raise ValueError(f"{self.kind.value} projection requires a source value")
        if self.target_bit_offset < 0:
            raise ValueError(
                f"{self.kind.value} target bit offset must be non-negative"
            )

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        _require_value(source_op, self.source_value, subject)
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        _require_immediate(descriptor, bound_immediate_name, subject)


@dataclass(frozen=True, slots=True)
class SourceMemoryProject:
    """Descriptor immediate projection from the selected source-memory access."""

    kind: SourceMemoryProjectKind
    dynamic_term_index: int = 0

    @classmethod
    def static_byte_offset(cls) -> Self:
        return cls(kind=SourceMemoryProjectKind.STATIC_BYTE_OFFSET)

    @classmethod
    def dynamic_byte_stride(cls, *, term: int = 0) -> Self:
        return cls(
            kind=SourceMemoryProjectKind.DYNAMIC_BYTE_STRIDE,
            dynamic_term_index=term,
        )

    def __post_init__(self) -> None:
        if self.dynamic_term_index < 0:
            raise ValueError(
                f"{self.kind.value} dynamic term index must be non-negative"
            )
        if (
            self.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET
            and self.dynamic_term_index != 0
        ):
            raise ValueError("static byte offset projection must not name a term")

    def validate(
        self,
        source_op: Op,
        descriptor: Descriptor,
        bound_immediate_name: str | None,
    ) -> None:
        subject = f"immediate projection {self.kind.value}"
        if bound_immediate_name is None:
            raise ValueError(
                f"{source_op.name}: {subject} must bind one descriptor immediate"
            )
        _require_immediate(descriptor, bound_immediate_name, subject)
