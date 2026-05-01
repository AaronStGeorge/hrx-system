# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Mapping, Sequence

from loom.importers.check.tilelang import TileLangImportInput
from loom.importers.core import print_loom_module
from loom.importers.tilelang.importer import TileLangImportOptions, import_tilelang


class Var:
    def __init__(self, name: str, dtype: str = "int32") -> None:
        self.name = name
        self.dtype = dtype

    def __repr__(self) -> str:
        return self.name


class Buffer:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype

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
        self.seq = seq


class For:
    def __init__(
        self,
        loop_var: Var,
        minimum: object,
        extent: object,
        body: object,
    ) -> None:
        self.loop_var = loop_var
        self.min = minimum
        self.extent = extent
        self.body = body


class BufferLoad:
    def __init__(
        self,
        buffer: Buffer,
        indices: Sequence[object],
        dtype: str = "float32",
    ) -> None:
        self.buffer = buffer
        self.indices = indices
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
        self.indices = indices


class IntImm:
    def __init__(self, value: int, dtype: str = "int32") -> None:
        self.value = value
        self.dtype = dtype


class FloatImm:
    def __init__(self, value: float, dtype: str = "float32") -> None:
        self.value = value
        self.dtype = dtype


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


class Min:
    def __init__(self, lhs: object, rhs: object, dtype: str = "float32") -> None:
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


class IfThenElse:
    def __init__(self, condition: object, then_case: object, else_case: object) -> None:
        self.condition = condition
        self.then_case = then_case
        self.else_case = else_case


class Block:
    def __init__(
        self,
        body: object,
        init: object | None = None,
        alloc_buffers: Sequence[object] = (),
        match_buffers: Sequence[object] = (),
    ) -> None:
        self.body = body
        self.init = init
        self.alloc_buffers = alloc_buffers
        self.match_buffers = match_buffers


class BlockRealize:
    def __init__(self, block: object, predicate: object | None = None) -> None:
        self.block = block
        self.predicate = IntImm(1, "bool") if predicate is None else predicate


class LetStmt:
    def __init__(self, var: Var, value: object, body: object) -> None:
        self.var = var
        self.value = value
        self.body = body


class AttrStmt:
    def __init__(self, attr_key: str, value: object, body: object) -> None:
        self.attr_key = attr_key
        self.value = value
        self.body = body


def test_import_tilelang_builds_kernel_from_direct_primfunc() -> None:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = BufferStore(
        dst_buffer,
        Add(BufferLoad(src_buffer, [IntImm(0)]), FloatImm(1.0)),
        [IntImm(0)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "add_one"},
    )

    result = import_tilelang(
        prim_func,
        options=TileLangImportOptions(include_report=True),
    )
    text = print_loom_module(result.module)

    assert 'kernel.def target(@tilelang_generic) export("add_one")' in text
    assert "buffer.view" in text
    assert "view.load" in text
    assert "scalar.addf" in text
    assert "view.store" in text
    assert result.report is not None


def test_import_tilelang_builds_scf_for_from_loop() -> None:
    src, dst = Var("src"), Var("dst")
    index = Var("i", "int32")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = For(
        index,
        IntImm(0),
        IntImm(4),
        BufferStore(
            dst_buffer,
            BufferLoad(src_buffer, [index]),
            [index],
        ),
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={
            "global_symbol": "copy",
            "thread_extent": {"threadIdx.x": 64},
        },
    )

    result = import_tilelang(
        TileLangImportInput(source=prim_func, target="hip", name="copy"),
        options=TileLangImportOptions(),
    )
    text = print_loom_module(result.module)

    assert 'kernel.def target(@hip) export("copy") workgroup_size(64, 1, 1)' in text
    assert "scf.for" in text
    assert "view.store" in text


def test_import_tilelang_builds_scalar_control_expressions() -> None:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (4,), "float32")
    select = Select(
        LT(IntImm(0), IntImm(1)),
        Cast(IntImm(4), "float32"),
        Min(FloatImm(2.0), FloatImm(3.0)),
    )
    body = IfThenElse(
        LT(IntImm(0), IntImm(1)),
        BufferStore(dst_buffer, select, [IntImm(0)]),
        BufferStore(dst_buffer, FloatImm(0.0), [IntImm(0)]),
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "scalar_control"},
    )

    result = import_tilelang(prim_func, options=TileLangImportOptions())
    text = print_loom_module(result.module)

    assert "scalar.cmpi slt" in text
    assert "scalar.sitofp" in text
    assert "scalar.minimumf" in text
    assert "scf.select" in text
    assert "scf.if" in text


def test_import_tilelang_normalizes_structural_wrappers() -> None:
    dst = Var("dst")
    tmp = Var("tmp", "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = AttrStmt(
        "thread_extent",
        IntImm(1),
        BlockRealize(
            Block(
                LetStmt(
                    tmp,
                    FloatImm(2.0),
                    BufferStore(dst_buffer, tmp, [IntImm(0)]),
                )
            )
        ),
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "structural"},
    )

    result = import_tilelang(prim_func, options=TileLangImportOptions())
    text = print_loom_module(result.module)

    assert "scalar.constant 2.0" in text
    assert "view.store" in text
