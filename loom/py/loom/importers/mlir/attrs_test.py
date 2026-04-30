# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.importers.mlir.attrs import MlirAttributeDecoder


class _TextAttr:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def test_dense_i64_array_decodes_text_binding_gap() -> None:
    decoder = MlirAttributeDecoder()

    assert decoder.dense_i64_array(_TextAttr("array<i64: 0, 4, 7>")) == (0, 4, 7)


def test_dense_i32_array_decodes_text_binding_gap() -> None:
    decoder = MlirAttributeDecoder()

    assert decoder.dense_i32_array(_TextAttr("array<i32: 1, 2, 1, 0>")) == (
        1,
        2,
        1,
        0,
    )


def test_dense_i64_array_rejects_unknown_spelling() -> None:
    decoder = MlirAttributeDecoder()

    with pytest.raises(ValueError, match="DenseIntegerArrayAttr"):
        decoder.dense_i64_array(_TextAttr("[0, 1]"))
