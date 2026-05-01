# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small TVM/TIR-shaped objects for importer check corpus fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class Var:
    def __init__(self, name: str, dtype: str = "int32") -> None:
        self.name = name
        self.dtype = dtype

    def __repr__(self) -> str:
        return self.name


class Buffer:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        scope: str = "",
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = Var(f"{name}_data", "handle")
        self._scope = scope

    def scope(self) -> str:
        return self._scope

    def __repr__(self) -> str:
        return self.name


class PrimFunc:
    def __init__(
        self,
        params: Sequence[Var],
        buffer_map: Mapping[Var, Buffer],
        body: object,
        attrs: Mapping[str, object] | None = None,
    ) -> None:
        self.params = params
        self.buffer_map = buffer_map
        self.body = body
        self.attrs = {} if attrs is None else attrs


class SeqStmt:
    def __init__(self, seq: Sequence[object]) -> None:
        self.seq = tuple(seq)


class Evaluate:
    def __init__(self, value: object) -> None:
        self.value = value


class Block:
    def __init__(
        self,
        body: object,
        alloc_buffers: Sequence[object] = (),
        match_buffers: Sequence[object] = (),
        init: object | None = None,
    ) -> None:
        self.body = body
        self.alloc_buffers = tuple(alloc_buffers)
        self.match_buffers = tuple(match_buffers)
        self.init = init


class For:
    def __init__(
        self,
        loop_var: Var,
        minimum: object,
        extent: object,
        body: object,
        *,
        thread_binding: object | None = None,
    ) -> None:
        self.loop_var = loop_var
        self.min = minimum
        self.extent = extent
        self.body = body
        self.thread_binding = thread_binding


class ThreadAxis:
    def __init__(self, name: str, variable: Var) -> None:
        self.thread_tag = name
        self.var = variable

    def __repr__(self) -> str:
        return self.thread_tag


class AttrStmt:
    def __init__(
        self,
        node: object,
        attr_key: str,
        value: object,
        body: object,
    ) -> None:
        self.node = node
        self.attr_key = attr_key
        self.value = value
        self.body = body


class IfThenElse:
    def __init__(self, condition: object, then_case: object, else_case: object) -> None:
        self.condition = condition
        self.then_case = then_case
        self.else_case = else_case


class BufferLoad:
    def __init__(
        self,
        buffer: Buffer,
        indices: Sequence[object],
        dtype: str = "float32",
    ) -> None:
        self.buffer = buffer
        self.indices = tuple(indices)
        self.dtype = dtype


class BufferStore:
    def __init__(
        self,
        buffer: Buffer,
        value: object,
        indices: Sequence[object],
    ) -> None:
        self.buffer = buffer
        self.value = value
        self.indices = tuple(indices)


class IntImm:
    def __init__(self, value: int, dtype: str = "int32") -> None:
        self.value = value
        self.dtype = dtype


class FloatImm:
    def __init__(self, value: float, dtype: str = "float32") -> None:
        self.value = value
        self.dtype = dtype


class Ramp:
    def __init__(
        self,
        base: object,
        stride: object,
        lanes: int,
        dtype: str = "int32",
    ) -> None:
        self.base = base
        self.stride = stride
        self.lanes = lanes
        self.dtype = f"{dtype}x{lanes}"


class Broadcast:
    def __init__(self, value: object, lanes: int, dtype: str = "float32") -> None:
        self.value = value
        self.lanes = lanes
        self.dtype = f"{dtype}x{lanes}"


class Add:
    def __init__(self, lhs: object, rhs: object, dtype: str = "float32") -> None:
        self.a = lhs
        self.b = rhs
        self.dtype = dtype


class LT:
    def __init__(self, lhs: object, rhs: object, dtype: str = "bool") -> None:
        self.a = lhs
        self.b = rhs
        self.dtype = dtype


class Cast:
    def __init__(self, value: object, dtype: str) -> None:
        self.value = value
        self.dtype = dtype


class Select:
    def __init__(
        self,
        condition: object,
        true_value: object,
        false_value: object,
        dtype: str = "float32",
    ) -> None:
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value
        self.dtype = dtype


class Op:
    def __init__(self, name: str) -> None:
        self.name = name


class Call:
    def __init__(
        self,
        op_name: str,
        args: Sequence[object],
        dtype: str = "float32",
        annotations: Mapping[str, object] | None = None,
    ) -> None:
        self.op = Op(op_name)
        self.args = tuple(args)
        self.dtype = dtype
        self.annotations = {} if annotations is None else dict(annotations)

    def __repr__(self) -> str:
        return f"{self.op.name}(...)"
