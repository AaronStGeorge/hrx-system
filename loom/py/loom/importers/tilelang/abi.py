# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang kernel ABI extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from loom.importers.core import (
    NameAllocator,
    source_key,
)
from loom.importers.tilelang.model import TileLangBinding
from loom.importers.tilelang.nodes import (
    attrs,
    buffer_map,
    dtype,
    lookup_buffer,
    mapping_items,
    node_kind,
    source_name,
)
from loom.importers.tilelang.ops.topology import thread_binding_var
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import BUFFER_TYPE


@dataclass(frozen=True, slots=True)
class DynamicScalarSourceGroup:
    """One semantic scalar symbol plus all source IR objects that spell it."""

    source: object
    aliases: tuple[object, ...]


def extract_bindings(
    prim_func: object,
    type_converter: TileLangTypeConverter,
) -> tuple[TileLangBinding, ...]:
    """Extract deterministic Loom kernel ABI bindings from a TileLang PrimFunc."""

    source_buffer_map = buffer_map(prim_func)
    names = NameAllocator()
    bindings: list[TileLangBinding] = []
    skip_sources: list[object] = []
    for ordinal, param in enumerate(tuple(getattr(prim_func, "params", ()))):
        buffer = lookup_buffer(source_buffer_map, param)
        name = names.reserve_or_fresh(
            source_name(param, fallback=f"arg{ordinal}"),
            fallback=f"arg{ordinal}",
        )
        if buffer is None:
            value_type = type_converter.map_dtype(dtype(param), index_like=False)
        else:
            value_type = BUFFER_TYPE
        bindings.append(
            TileLangBinding(
                ordinal=ordinal,
                name=name,
                source=param,
                type=value_type,
                buffer=buffer,
            )
        )
        skip_sources.append(param)
        if buffer is not None:
            skip_sources.append(buffer)
            data = getattr(buffer, "data", None)
            if data is not None:
                skip_sources.append(data)

    next_ordinal = len(bindings)
    for group in collect_dynamic_scalar_source_groups(prim_func, skip_sources):
        source = group.source
        name = names.reserve_or_fresh(
            source_name(source, fallback=f"arg{next_ordinal}"),
            fallback=f"arg{next_ordinal}",
        )
        bindings.append(
            TileLangBinding(
                ordinal=next_ordinal,
                name=name,
                source=source,
                type=type_converter.map_dtype(dtype(source), index_like=False),
                aliases=group.aliases,
            )
        )
        next_ordinal += 1

    if not bindings:
        raise ValueError("TileLang PrimFunc has no parameters")
    return tuple(bindings)


def collect_dynamic_scalar_sources(
    prim_func: object,
    skip_sources: Iterable[object],
) -> tuple[object, ...]:
    """Collect free scalar TIR variables that must cross the kernel ABI."""

    return tuple(
        group.source
        for group in collect_dynamic_scalar_source_groups(prim_func, skip_sources)
    )


def collect_dynamic_scalar_source_groups(
    prim_func: object,
    skip_sources: Iterable[object],
) -> tuple[DynamicScalarSourceGroup, ...]:
    """Collect free scalar TIR variables grouped by source-level symbol."""

    collector = _DynamicScalarCollector(skip_sources)
    for buffer in buffer_map(prim_func).values():
        collector.visit_buffer_metadata(buffer)
    for value in attrs(prim_func).values():
        collector.visit(value)
    collector.visit(getattr(prim_func, "body", None))
    return collector.groups()


class _DynamicScalarCollector:
    """Walks TIR-like objects and records free scalar variables by source order."""

    def __init__(self, skip_sources: Iterable[object]) -> None:
        self.sources: list[object] = []
        self.aliases: list[list[object]] = []
        self._skip_keys = {
            key for source in skip_sources for key in _source_binding_keys(source)
        }
        self._source_keys: set[object] = set()
        self._semantic_source_indices: dict[tuple[str, str, str], int] = {}
        self._visited_objects: list[object] = []

    def groups(self) -> tuple[DynamicScalarSourceGroup, ...]:
        return tuple(
            DynamicScalarSourceGroup(source=source, aliases=tuple(aliases))
            for source, aliases in zip(self.sources, self.aliases, strict=True)
        )

    def visit(self, value: object, bound_keys: set[object] | None = None) -> None:
        bound = set() if bound_keys is None else bound_keys
        if value is None or _is_leaf(value):
            return
        if _is_scalar_var(value):
            self._record_source(value, bound)
            return
        if any(value is visited for visited in self._visited_objects):
            return
        self._visited_objects.append(value)
        self._visit_structured(value, bound)

    def visit_buffer_metadata(
        self,
        buffer: object,
        bound_keys: set[object] | None = None,
    ) -> None:
        for attr_name in ("shape", "strides", "elem_offset", "offset_factor"):
            self.visit(getattr(buffer, attr_name, None), bound_keys)

    def _record_source(self, value: object, bound_keys: set[object]) -> None:
        key = source_key(value)
        semantic_key = _semantic_var_key(value)
        if (
            key in self._skip_keys
            or semantic_key in self._skip_keys
            or key in bound_keys
            or semantic_key in bound_keys
            or key in self._source_keys
        ):
            return
        self._source_keys.add(key)
        existing_index = self._semantic_source_indices.get(semantic_key)
        if existing_index is not None:
            self.aliases[existing_index].append(value)
            return
        self._semantic_source_indices[semantic_key] = len(self.sources)
        self.sources.append(value)
        self.aliases.append([])

    def _visit_structured(self, value: object, bound_keys: set[object]) -> None:
        kind = node_kind(value)
        if kind == "For":
            self.visit(getattr(value, "min", None), bound_keys)
            self.visit(getattr(value, "extent", None), bound_keys)
            body_bound = self._extend_bound(
                bound_keys,
                getattr(value, "loop_var", None),
                getattr(value, "thread_binding", None),
                thread_binding_var(getattr(value, "thread_binding", None)),
            )
            self.visit(getattr(value, "body", None), body_bound)
            return
        if kind == "LetStmt":
            self.visit(getattr(value, "value", None), bound_keys)
            self.visit(
                getattr(value, "body", None),
                self._extend_bound(bound_keys, getattr(value, "var", None)),
            )
            return
        if kind == "AttrStmt":
            attr_key = str(getattr(value, "attr_key", ""))
            if attr_key == "thread_extent":
                node = getattr(value, "node", None)
                self.visit(getattr(value, "value", None), bound_keys)
                self.visit(
                    getattr(value, "body", None),
                    self._extend_bound(bound_keys, node, thread_binding_var(node)),
                )
                return
            self.visit(getattr(value, "node", None), bound_keys)
            self.visit(getattr(value, "value", None), bound_keys)
            self.visit(getattr(value, "body", None), bound_keys)
            return
        if kind == "Block":
            for buffer in tuple(getattr(value, "alloc_buffers", ()) or ()):
                self.visit_buffer_metadata(buffer, bound_keys)
            block_bound = bound_keys
            for iter_var in tuple(getattr(value, "iter_vars", ()) or ()):
                block_bound = self._extend_bound(
                    block_bound,
                    iter_var,
                    thread_binding_var(iter_var),
                )
            self.visit(getattr(value, "init", None), block_bound)
            self.visit(getattr(value, "body", None), block_bound)
            return
        if kind == "BlockRealize":
            self.visit(getattr(value, "iter_values", ()), bound_keys)
            self.visit(getattr(value, "predicate", None), bound_keys)
            self.visit(getattr(value, "block", None), bound_keys)
            return
        if kind == "Allocate":
            self.visit(getattr(value, "extents", ()), bound_keys)
            self.visit(getattr(value, "condition", None), bound_keys)
            self.visit(
                getattr(value, "body", None),
                self._extend_bound(bound_keys, getattr(value, "buffer_var", None)),
            )
            return
        if kind == "DeclBuffer":
            buffer = getattr(value, "buffer", None)
            self.visit_buffer_metadata(buffer, bound_keys)
            self.visit(getattr(value, "body", None), bound_keys)
            return
        if kind == "BufferRealize":
            self.visit_buffer_metadata(getattr(value, "buffer", None), bound_keys)
            self.visit(getattr(value, "bounds", ()), bound_keys)
            self.visit(getattr(value, "condition", None), bound_keys)
            self.visit(getattr(value, "body", None), bound_keys)
            return
        if kind == "BufferLoad":
            self.visit(getattr(value, "indices", ()), bound_keys)
            return
        if kind == "BufferStore":
            self.visit(getattr(value, "value", None), bound_keys)
            self.visit(getattr(value, "indices", ()), bound_keys)
            return
        self._visit_common_fields(value, bound_keys)

    def _visit_common_fields(self, value: object, bound_keys: set[object]) -> None:
        for attr_name in _COMMON_SOURCE_ATTRS:
            if hasattr(value, attr_name):
                self.visit(getattr(value, attr_name), bound_keys)
        if isinstance(value, Mapping):
            for child in value.values():
                self.visit(child, bound_keys)
            return
        items = mapping_items(value)
        if items:
            for _key, child in items:
                self.visit(child, bound_keys)
            return
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            for child in value:
                self.visit(child, bound_keys)

    def _extend_bound(
        self,
        bound_keys: set[object],
        *sources: object | None,
    ) -> set[object]:
        extended = set(bound_keys)
        for source in sources:
            if source is not None:
                extended.update(_source_binding_keys(source))
        return extended


def _is_scalar_var(value: object) -> bool:
    if node_kind(value) not in ("Var", "SizeVar"):
        return False
    value_dtype = dtype(value)
    return value_dtype != "handle" and not value_dtype.endswith("*")


def _source_binding_keys(value: object) -> tuple[object, ...]:
    keys: list[object] = [source_key(value)]
    if _is_scalar_var(value):
        keys.append(_semantic_var_key(value))
    return tuple(keys)


def _semantic_var_key(value: object) -> tuple[str, str, str]:
    return ("var", source_name(value, fallback=str(value)), dtype(value))


def _is_leaf(value: object) -> bool:
    return isinstance(value, str | bytes | int | float | bool)


_COMMON_SOURCE_ATTRS = (
    "a",
    "b",
    "args",
    "base",
    "block",
    "body",
    "condition",
    "else_case",
    "extent",
    "indices",
    "lanes",
    "message",
    "min",
    "predicate",
    "seq",
    "stride",
    "then_case",
    "true_value",
    "false_value",
    "value",
)
