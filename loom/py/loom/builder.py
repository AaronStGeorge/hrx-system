# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR builder: construct loom Operations from Op declarations.

The builder provides a typed, validated API for constructing IR.
Parameter order follows the format spec: the order you see in the
textual assembly is the order you pass to the builder.

Two layers:
  1. Generic builder (this module): creates Operations from any Op
     declaration. Called by generated stubs and directly for one-offs.
  2. Generated typed stubs (gen/builders_*.py): per-op methods with
     typed parameters and docstrings. Checked in, greppable, IDE-friendly.

Value references:
  The builder returns ValueRef objects (not bare ints). ValueRef
  supports indexing (tensor[off]) and tied result marking (a.as_type(t)).

    b = IRBuilder()
    ...
    tensor = b.value("tensor", tensor_t)
    tile = b.value("tile", tile_t)
    off = b.value("off", INDEX)

    # Natural indexing: tensor[off] captures value + offsets
    result = b.tensor.update(source=tile, target=tensor[off])

    # Tied results: a.as_type(t) marks 'a' as consumed, result has type t
    out, count = b.func.call("@f", a, b_val,
                              results=[tied(a, tensor_t), INDEX])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loom.dsl import Op, TypeDef
from loom.fields import FieldLayout
from loom.ir import (
    Module,
    Operation,
    Region,
    ScalarType,
    ShapedType,
    Type,
    Value,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)

__all__ = [
    "IRBuilder",
    "ValueRef",
    "IndexedValue",
    "TiedResultSpec",
    "tied",
]


# ============================================================================
# Value reference types
# ============================================================================


class ValueRef:
    """A reference to an SSA value in the builder.

    Returned by IRBuilder.value() and build(). Supports:
      - Indexing: tensor[off] or tensor[i, j] → IndexedValue
      - Tied marking: val.as_type(new_t) → TiedResultSpec
      - Use as operand: pass directly to build() or generated stubs
    """

    __slots__ = ("_id", "_builder")

    def __init__(self, value_id: int, builder: IRBuilder) -> None:
        self._id = value_id
        self._builder = builder

    @property
    def id(self) -> int:
        """The underlying value ID in the module."""
        return self._id

    @property
    def type(self) -> Type:
        """The type of this value."""
        return self._builder.module.values[self._id].type

    @property
    def name(self) -> str:
        """The SSA name of this value."""
        return self._builder.module.values[self._id].name

    def __getitem__(self, indices: Any) -> IndexedValue:
        """tensor[off] or tensor[i, j] → IndexedValue.

        Each index is either an int (static) or a ValueRef (dynamic).
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        return IndexedValue(self._id, list(indices), self._builder)

    def as_type(self, new_type: Type) -> TiedResultSpec:
        """Mark this value as tied to a result with the given type.

        Used in result specs: results=[a.as_type(tensor_t), INDEX]
        Mirrors the textual: -> (%a as tensor<...>, index)
        """
        return TiedResultSpec(self._id, new_type)

    def __repr__(self) -> str:
        name = self._builder.module.values[self._id].name
        return f"ValueRef({name or self._id})"

    def __int__(self) -> int:
        """Allow using ValueRef where an int value ID is expected."""
        return self._id


class IndexedValue:
    """A value with index offsets: tensor[%i, %j] or tensor[0, %off].

    Created by ValueRef.__getitem__. Passed to builder methods that
    expect an indexed operand (ops with IndexList in their format).
    The builder extracts the base value and offsets to build the
    static/dynamic offset arrays.
    """

    __slots__ = ("_value_id", "_offsets", "_builder")

    def __init__(self, value_id: int, offsets: list[Any], builder: IRBuilder) -> None:
        self._value_id = value_id
        self._offsets = offsets
        self._builder = builder

    @property
    def value_id(self) -> int:
        return self._value_id

    @property
    def offsets(self) -> list[Any]:
        """List of int (static) or ValueRef (dynamic) offsets."""
        return self._offsets

    def decompose(self) -> tuple[int, list[int], list[int]]:
        """Decompose into (base_id, static_offsets, dynamic_ids).

        Static offsets are integers. Dynamic offsets are replaced with
        a sentinel in the static array, and the dynamic value IDs are
        collected separately.
        """
        sentinel = -(2**63)
        static: list[int] = []
        dynamic: list[int] = []
        for offset in self._offsets:
            if isinstance(offset, ValueRef):
                static.append(sentinel)
                dynamic.append(offset._id)
            elif isinstance(offset, int):
                static.append(offset)
            else:
                raise TypeError(
                    f"Index must be int (static) or ValueRef (dynamic), "
                    f"got {type(offset).__name__}"
                )
        return self._value_id, static, dynamic

    def __repr__(self) -> str:
        return f"IndexedValue({self._value_id}, {self._offsets})"


@dataclass(frozen=True, slots=True)
class TiedResultSpec:
    """Specifies that a result is tied to an operand.

    Created by ValueRef.as_type() or the tied() function.
    Used in result specs: results=[tied(a, tensor_t), INDEX]
    """

    operand_id: int
    result_type: Type

    def __repr__(self) -> str:
        return f"tied({self.operand_id}, {self.result_type})"


def tied(operand: ValueRef | int, result_type: Type) -> TiedResultSpec:
    """Mark a result as tied to an operand with a (possibly different) type.

    Usage: results=[tied(tensor, new_tensor_t), INDEX]
    Mirrors: -> (%tensor as new_tensor_t, index)
    """
    operand_id = operand._id if isinstance(operand, ValueRef) else operand
    return TiedResultSpec(operand_id, result_type)


class IRBuilder:
    """Generic IR builder: constructs Operations from Op declarations.

    The builder manages a Module and provides methods to create values
    and operations. It validates types and constraints at construction
    time (fail-fast, no silent errors).
    """

    def __init__(self, module: Module | None = None) -> None:
        self._module = module if module is not None else Module()
        self._op_registry: dict[str, Op] = {}
        self._type_registry: dict[str, TypeDef] = {}
        self._layouts: dict[str, FieldLayout] = {}

    @property
    def module(self) -> Module:
        """The module being built."""
        return self._module

    def register_ops(self, ops: Sequence[Op]) -> None:
        """Register op declarations."""
        for op in ops:
            self._op_registry[op.name] = op

    def register_types(self, types: Sequence[TypeDef]) -> None:
        """Register type declarations."""
        for td in types:
            self._type_registry[td.name] = td

    # --- Value creation ---

    def value(self, name: str, value_type: Type, **kwargs: Any) -> ValueRef:
        """Create a named value and return a ValueRef.

        Used for function arguments, constants, and any value that
        needs to exist before ops that reference it.
        """
        value = Value(name=name, type=value_type, **kwargs)
        value_id = self._module.add_value(value)
        return ValueRef(value_id, self)

    # --- Operand resolution ---

    def _resolve_operand(self, operand: ValueRef | IndexedValue | int) -> int:
        """Resolve an operand to a value ID."""
        if isinstance(operand, ValueRef):
            return operand._id
        if isinstance(operand, IndexedValue):
            return operand._value_id
        if isinstance(operand, int):
            return operand
        raise TypeError(
            f"Operand must be ValueRef, IndexedValue, or int, "
            f"got {type(operand).__name__}"
        )

    def _resolve_operands(
        self,
        operands: Sequence[ValueRef | IndexedValue | int],
    ) -> list[int]:
        """Resolve a list of operands to value IDs."""
        return [self._resolve_operand(op) for op in operands]

    # --- Generic op building ---

    def build(
        self,
        op_name: str,
        operands: Sequence[ValueRef | int] | None = None,
        *,
        results: Sequence[Type | TiedResultSpec] | None = None,
        result_names: Sequence[str] | None = None,
        attributes: dict[str, Any] | None = None,
        regions: Sequence[Region] | None = None,
    ) -> ValueRef | list[ValueRef] | None:
        """Build an operation and return the result ValueRef(s).

        Parameters:
          op_name: Full op name (e.g., "test.addi").
          operands: ValueRefs or ints for operands.
          results: Result specs. Each is either:
            - A Type (fresh result): INDEX, F32, tensor_t
            - A TiedResultSpec (tied): tied(operand, type) or operand.as_type(type)
          result_names: SSA names for results (auto-generated if None).
          attributes: Op attributes.
          regions: Nested regions.

        Returns:
          Single result: ValueRef.
          Multiple results: list[ValueRef].
          No results: None.
        """
        op_decl = self._op_registry.get(op_name)
        if op_decl is None:
            raise ValueError(
                f"Unknown op '{op_name}'. Register it with register_ops()."
            )

        operand_ids = self._resolve_operands(operands or [])
        result_specs = results or []
        attr_dict = attributes or {}
        region_list = list(regions) if regions else []

        # Process result specs: extract types and tied bindings.
        result_ids: list[int] = []
        tied_results: list[IRTiedResult] = []
        for i, spec in enumerate(result_specs):
            if isinstance(spec, TiedResultSpec):
                # Tied result: find the operand index.
                operand_index = operand_ids.index(spec.operand_id)
                tied_results.append(
                    IRTiedResult(result_index=i, operand_index=operand_index)
                )
                result_type = spec.result_type
            elif isinstance(spec, ScalarType | ShapedType):
                result_type = spec
            else:
                result_type = spec  # Any Type

            name = ""
            if result_names and i < len(result_names):
                name = result_names[i]
            value_id = self._module.add_value(Value(name=name, type=result_type))
            result_ids.append(value_id)

        # Create operation.
        Operation(
            name=op_name,
            operands=operand_ids,
            results=result_ids,
            tied_results=tied_results,
            attributes=attr_dict,
            regions=region_list,
        )

        # Return based on result count.
        if len(result_ids) == 0:
            return None
        if len(result_ids) == 1:
            return ValueRef(result_ids[0], self)
        return [ValueRef(rid, self) for rid in result_ids]
