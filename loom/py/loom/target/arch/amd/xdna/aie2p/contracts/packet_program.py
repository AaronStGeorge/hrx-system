# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P lane-wise descriptor programs over one vector packet."""

from __future__ import annotations

from typing import Literal

from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

IntegerSignedness = Literal["signed", "unsigned"]


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


class PacketProgram:
    """Builds lane-wise descriptor programs over one X carrier."""

    def __init__(self, element_bits: int, temporary_prefix: str = "") -> None:
        if element_bits not in (8, 16, 32):
            raise ValueError("packet program element width must be 8, 16, or 32")
        self.emits: list[ContractEmit] = []
        self.element_bits = element_bits
        self.temporary_prefix = temporary_prefix
        # Constants already materialized in the shift-register class.
        self.shift_values: dict[int, ValueRef] = {}
        # Last emitted value of each persistent instruction-state field.
        self.state_values: dict[str, int] = {}

    def temporary(self, name: str) -> ValueRef:
        return ValueRef.temporary(f"{self.temporary_prefix}{name}")

    def constant(
        self,
        name: str,
        value: int | ValueProject,
        *,
        descriptor_key: str | None = None,
    ) -> ValueRef:
        if descriptor_key is None:
            descriptor_key = (
                "amd.xdna.aie2p.constant.i32.short"
                if isinstance(value, int) and -1024 <= value <= 1023
                else "amd.xdna.aie2p.constant.i32"
            )
        result = self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(descriptor_key),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def operation(
        self,
        name: str | None,
        descriptor_key: str,
        result_field: str,
        *,
        immediates: dict[str, int] | None = None,
        copy_operands: tuple[str, ...] = (),
        **operands: ValueRef,
    ) -> ValueRef:
        result = ValueRef.result("result") if name is None else self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amd.xdna.aie2p.{descriptor_key}"),
                operands=operands,
                results={result_field: result},
                result_types=(
                    None if name is None else {result_field: DescriptorResultType()}
                ),
                immediates={} if immediates is None else immediates,
                form=DescriptorEmitForm.OP,
                copy_operands=copy_operands,
            )
        )
        return result

    def state(self, descriptor_key: str, value: int) -> None:
        """Sets one instruction-state field consumed by later descriptors."""

        if self.state_values.get(descriptor_key) == value:
            return
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(
                    f"amd.xdna.aie2p.state.{descriptor_key}.immediate"
                ),
                immediates={"i": value},
                form=DescriptorEmitForm.OP,
            )
        )
        self.state_values[descriptor_key] = value

    def splat(
        self,
        name: str,
        value: int | ValueProject,
        *,
        element_bits: int | None = None,
        descriptor_key: str | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        scalar = self.constant(f"{name}_scalar", value, descriptor_key=descriptor_key)
        return self.operation(
            name,
            f"splat.i{element_bits}x{512 // element_bits}",
            "dst",
            src=scalar,
        )

    def shift(self, value: int) -> ValueRef:
        """Returns one shared shift-register constant for |value|."""

        shift = self.shift_values.get(value)
        if shift is None:
            shift = self.constant(
                f"shift_{value}",
                value,
                descriptor_key="amd.xdna.aie2p.constant.i32.shift",
            )
            self.shift_values[value] = shift
        return shift

    def binary(
        self,
        name: str,
        descriptor_key: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(name, descriptor_key, "d", s1=lhs, s2=rhs)

    def _complete_comparison(
        self,
        name: str,
        descriptor_key: str,
        element_bits: int,
        **operands: ValueRef,
    ) -> ValueRef:
        """Completes one comparison into an X-sized predicate."""

        comparison = self.operation(
            name if element_bits == 8 else f"{name}_low",
            descriptor_key,
            "cmp",
            **operands,
        )
        if element_bits == 8:
            return comparison
        result = self.temporary(name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.predicate.complete.zero.high32"),
                operands={"storage": comparison},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 0},
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    def compare_zero(
        self,
        name: str,
        value: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        suffix = "" if element_bits == 8 else ".el.low32"
        return self._complete_comparison(
            name,
            f"cmp.eqz.i{element_bits}x{512 // element_bits}{suffix}",
            element_bits,
            s2=value,
        )

    def compare_unsigned_less_than(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        suffix = "" if element_bits == 8 else ".el.low32"
        return self._complete_comparison(
            name,
            f"cmp.lt.unsigned.i{element_bits}x{512 // element_bits}{suffix}",
            element_bits,
            s1=lhs,
            s2=rhs,
        )

    def compare_unsigned_greater_equal(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        suffix = "" if element_bits == 8 else ".el.low32"
        return self._complete_comparison(
            name,
            f"cmp.ge.unsigned.i{element_bits}x{512 // element_bits}{suffix}",
            element_bits,
            s1=lhs,
            s2=rhs,
        )

    def compare_signed_less_than(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        suffix = "" if element_bits == 8 else ".el.low32"
        return self._complete_comparison(
            name,
            f"cmp.lt.signed.i{element_bits}x{512 // element_bits}{suffix}",
            element_bits,
            s1=lhs,
            s2=rhs,
        )

    def select(
        self,
        name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
        *,
        element_bits: int | None = None,
    ) -> ValueRef:
        if element_bits is None:
            element_bits = self.element_bits
        descriptor_key = (
            "select.i8x64"
            if element_bits == 8
            else f"select.i{element_bits}x{512 // element_bits}.mask64"
        )
        return self.operation(
            name,
            descriptor_key,
            "d",
            s1=false_value,
            s2=true_value,
            sel=condition,
        )


class PairPacketProgram(PacketProgram):
    """Builds i64 lane programs over deinterleaved i32 word packets."""

    def __init__(self, temporary_prefix: str = "") -> None:
        super().__init__(32, temporary_prefix)
        self.i64_word_controls: tuple[ValueRef, ValueRef] | None = None
        self.i64_join_control: ValueRef | None = None

    def _word_controls(self) -> tuple[ValueRef, ValueRef]:
        if self.i64_word_controls is None:
            self.i64_word_controls = tuple(
                self.constant(
                    f"i64_{word}_word_control",
                    control,
                    descriptor_key="amd.xdna.aie2p.constant.i32.mova",
                )
                for word, control in (("low", 4), ("high", 5))
            )
        return self.i64_word_controls

    def select_i64_word(self, source: ValueRef, name: str, word_index: int) -> ValueRef:
        """Selects one packed i32 word from every i64 source lane."""

        if word_index not in (0, 1):
            raise ValueError("i64 packet word index must be zero or one")
        control = self._word_controls()[word_index]
        return self.operation(
            name,
            "shuffle.x.configured",
            "dst",
            s1=source,
            s2=source,
            mod=control,
        )

    def split_i64_words(self, source: ValueRef, name: str) -> tuple[ValueRef, ValueRef]:
        """Deinterleaves packed low and high words from i64 lanes."""

        return (
            self.select_i64_word(source, f"{name}_low_words", 0),
            self.select_i64_word(source, f"{name}_high_words", 1),
        )

    def join_i64_words(
        self,
        low: ValueRef,
        high: ValueRef,
        name: str | None,
    ) -> ValueRef:
        """Interleaves packed i32 word streams into i64 lanes."""

        if self.i64_join_control is None:
            self.i64_join_control = self.constant(
                "i64_join_control",
                16,
                descriptor_key="amd.xdna.aie2p.constant.i32.mova",
            )
        return self.operation(
            name,
            "shuffle.x.configured",
            "dst",
            s1=low,
            s2=high,
            mod=self.i64_join_control,
        )


_INTEGER_SHIFT_SHAPES = {
    8: ("4x.x-to-d", 0),
    16: ("2x.x-to-c", 0),
    32: ("2x.x-to-c", 1),
}


def _configure_integer_shift(program: PacketProgram, *, rounding_mode: int = 0) -> None:
    """Sets the instruction state shared by integer widen/narrow shifts."""

    _, mode = _INTEGER_SHIFT_SHAPES[program.element_bits]
    program.state("saturation", 0)
    program.state("ups-mode", mode)
    program.state("srs-mode", mode)
    program.state("rounding", rounding_mode)


def widen_integer_packet(
    program: PacketProgram,
    name: str,
    source: ValueRef,
    shift: ValueRef,
    *,
    signedness: IntegerSignedness = "unsigned",
) -> ValueRef:
    """Widens and left-shifts one integer packet into accumulators."""

    shape, mode = _INTEGER_SHIFT_SHAPES[program.element_bits]
    program.state("saturation", 0)
    program.state("ups-mode", mode)
    return program.operation(
        name,
        f"widen.{shape}.{signedness}.configured",
        "dst",
        src=source,
        su=shift,
    )


def narrow_integer_packet(
    program: PacketProgram,
    name: str | None,
    source: ValueRef,
    shift: ValueRef,
    *,
    rounding_mode: int = 0,
    signedness: IntegerSignedness = "unsigned",
) -> ValueRef:
    """Right-shifts and narrows one accumulator packet into integer lanes."""

    shape, mode = _INTEGER_SHIFT_SHAPES[program.element_bits]
    factor, widen_carriers = shape.split(".", 1)
    source_carrier, result_carrier = widen_carriers.split("-to-")
    narrow_shape = f"{factor}.{result_carrier}-to-{source_carrier}"
    program.state("saturation", 0)
    program.state("srs-mode", mode)
    program.state("rounding", rounding_mode)
    return program.operation(
        name,
        f"narrow.{narrow_shape}.{signedness}.configured",
        "dst",
        src=source,
        su=shift,
    )


def shift_integer_packet(
    program: PacketProgram,
    name: str | None,
    source: ValueRef,
    upshift: ValueRef,
    downshift: ValueRef,
    *,
    signedness: IntegerSignedness = "unsigned",
) -> ValueRef:
    """Shifts one integer packet through a widened accumulator value."""

    # Configure both halves before emitting either instruction. Besides making
    # the coupled widen/narrow contract explicit, this preserves the canonical
    # state-before-program order used by existing packet recipes.
    _configure_integer_shift(program)
    wide = widen_integer_packet(
        program,
        f"{name}_wide" if name is not None else "result_wide",
        source,
        upshift,
        signedness=signedness,
    )
    return narrow_integer_packet(
        program,
        name,
        wide,
        downshift,
        signedness=signedness,
    )


def shift_integer_packet_fixed(
    program: PacketProgram,
    name: str | None,
    source: ValueRef,
    amount: int,
    *,
    rounding_mode: int = 0,
    signedness: IntegerSignedness = "unsigned",
) -> ValueRef:
    """Shifts one integer packet by a fixed signed amount."""

    maximum_shift = program.element_bits - 1
    if not -maximum_shift <= amount <= maximum_shift:
        raise ValueError(
            f"i{program.element_bits} packet shift must fit the element width"
        )
    # Fixed-shift packet recipes historically configure state before
    # materializing immediates and materialize each immediate immediately
    # before its consumer. Keep that stable as this helper generalizes beyond
    # i32 so unrelated exact lowerings do not churn.
    _configure_integer_shift(program, rounding_mode=rounding_mode)
    wide = widen_integer_packet(
        program,
        f"{name}_wide" if name is not None else "result_wide",
        source,
        program.shift(max(amount, 0)),
        signedness=signedness,
    )
    return narrow_integer_packet(
        program,
        name,
        wide,
        program.shift(max(-amount, 0)),
        rounding_mode=rounding_mode,
        signedness=signedness,
    )
