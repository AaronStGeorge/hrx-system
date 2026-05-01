# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from tempfile import TemporaryDirectory

from loom.importers.check.tilelang.runner import (
    TileLangCheckOptions,
    run_tilelang_check,
)


def test_run_tilelang_check_updates_and_verifies_python_fixture() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tilelang_cases.py"
        path.write_text(
            """
from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


class Var:
    def __init__(self, name, dtype="int32"):
        self.name = name
        self.dtype = dtype


class Buffer:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class PrimFunc:
    def __init__(self, params, buffer_map, body, attrs=None):
        self.params = params
        self.buffer_map = buffer_map
        self.body = body
        self.attrs = {} if attrs is None else attrs


class BufferLoad:
    def __init__(self, buffer, indices, dtype="float32"):
        self.buffer = buffer
        self.indices = indices
        self.dtype = dtype


class BufferStore:
    def __init__(self, buffer, value, indices):
        self.buffer = buffer
        self.value = value
        self.indices = indices


class IntImm:
    def __init__(self, value, dtype="int32"):
        self.value = value
        self.dtype = dtype


@tilelang_case(name="copy")
def copy():
    src = Var("src")
    dst = Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = BufferStore(dst_buffer, BufferLoad(src_buffer, [IntImm(0)]), [IntImm(0)])
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "copy"},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="copy")
# ----
# old
"""
        )

        update_results = run_tilelang_check(
            path,
            options=TileLangCheckOptions(update=True),
        )
        updated_source = path.read_text()
        verify_results = run_tilelang_check(
            path,
            options=TileLangCheckOptions(),
        )

    assert [result.status for result in update_results] == ["updated"]
    assert [result.status for result in verify_results] == ["passed"]
    assert '# kernel.def target(@hip) export("copy")' in updated_source
    assert "view.load" in updated_source
    assert "view.store" in updated_source
