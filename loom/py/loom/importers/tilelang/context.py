# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang-specific import session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loom.builder import ValueRef
from loom.importers.core import SourceImportSession
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import INDEX, OFFSET, Type


@dataclass(slots=True)
class TileLangConversionContext(SourceImportSession):
    """TileLang-specialized import session using Loom dynamic builders."""

    type_converter: TileLangTypeConverter = field(default_factory=TileLangTypeConverter)
    index_values: dict[object, ValueRef] = field(default_factory=dict)

    def type(self, value_type: str) -> Type:
        return self.type_converter.map_dtype(value_type)

    def build_constant(self, value: Any, value_type: str, name: str) -> ValueRef:
        result_type = self.type_converter.map_dtype(
            value_type,
            index_like=value_type == "index",
        )
        if result_type in (INDEX, OFFSET):
            return self.builder.index.constant(
                value=int(value),
                results=[result_type],
                name=name,
            )
        return self.builder.scalar.constant(
            value=value,
            results=[result_type],
            name=name,
        )

    def mapped_index_value(self, source: object) -> ValueRef | None:
        return self.index_values.get(source)

    def map_index_value(self, source: object, ref: ValueRef) -> None:
        self.index_values[source] = ref
        if ref.name:
            self.names.capture(ref.name)

    def fork(self, *, preview_block: object | None = None) -> TileLangConversionContext:
        return TileLangConversionContext(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
            type_converter=self.type_converter,
            index_values=dict(self.index_values),
        )
