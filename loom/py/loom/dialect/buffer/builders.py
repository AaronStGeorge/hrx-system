# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Region, Type


class BufferBuilders:
    """Typed builder methods for buffer ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def alloca(self, *, byte_length: ValueRef, base_alignment: int, memory_space: str, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Create a fixed-frame scratch buffer root in workgroup or private memory. Each execution produces a distinct storage identity; identical allocas must not be commoned. The byte length is a physical byte count, and base_alignment is the minimum byte alignment of the root storage base.

        Example::
            %scratch = buffer.alloca %bytes {base_alignment = 64, memory_space = workgroup} : buffer
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["base_alignment"] = base_alignment
        _attributes["memory_space"] = memory_space
        _operands.append(byte_length)
        return cast(ValueRef, self._b.build("buffer.alloca", _operands, results=results, attributes=_attributes, regions=_regions))

    def memory_space(self, *, buffer: ValueRef, memory_space: str, result_types: list[Type]) -> ValueRef:
        """Refine an existing buffer root with a concrete target-independent memory-space fact while preserving the same storage identity, extent, alignment, and nullability facts.

        Example::
            %global = buffer.assume.memory_space %buffer {memory_space = global} : buffer
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["memory_space"] = memory_space
        _operands.append(buffer)
        return cast(ValueRef, self._b.build("buffer.assume.memory_space", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def view(self, *, buffer: ValueRef, byte_offset: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Form a typed non-owning view from an opaque buffer root and base byte offset. The result view type carries the address layout.

        Example::
            %view = buffer.view %buffer[%offset] : buffer -> view<[%M]xf32, %layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(buffer)
        _operands.append(byte_offset)
        return cast(ValueRef, self._b.build("buffer.view", _operands, results=results, attributes=_attributes, regions=_regions))
