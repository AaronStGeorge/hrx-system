# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place
# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type
from loom.builders import DialectBuilder

class LowBuilder(DialectBuilder):
    def func_def(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        allocation: str | None = ...,
        schedule: str | None = ...,
        target: str,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        export_symbol: str | None = ...,
        export_attrs: Mapping[str, Any] | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def kernel_def(
        self,
        *,
        allocation: str | None = ...,
        schedule: str | None = ...,
        target: str,
        export_symbol: str | None = ...,
        artifact: str | None = ...,
        export_ordinal: int | None = ...,
        export_linkage: str | None = ...,
        workgroup_size_x: int | None = ...,
        workgroup_size_y: int | None = ...,
        workgroup_size_z: int | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        predicates: list[Predicate] = ...,
        body: Region | None = ...,
    ) -> None: ...
    def decl(
        self,
        *,
        visibility: str | None = ...,
        cc: str | None = ...,
        purity: str | None = ...,
        allocation: str | None = ...,
        schedule: str | None = ...,
        import_kind: str | None = ...,
        code_symbol: str | None = ...,
        target: str,
        abi: str | None = ...,
        abi_attrs: Mapping[str, Any] | None = ...,
        export_symbol: str | None = ...,
        export_attrs: Mapping[str, Any] | None = ...,
        callee: str,
        args: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] = ...,
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def return_(
        self,
        *,
        values: list[ValueRef] = ...,
    ) -> None: ...
    def call(
        self,
        *,
        purity: str | None = ...,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def op(
        self,
        *,
        opcode: str,
        operands: list[ValueRef] = ...,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def const(
        self,
        *,
        opcode: str,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def copy(
        self,
        *,
        source: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def slice(
        self,
        *,
        source: ValueRef,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def concat(
        self,
        *,
        sources: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def invoke(
        self,
        *,
        purity: str | None = ...,
        callee: str,
        operands: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def slot(
        self,
        *,
        symbol: str,
        function: str,
        space: str,
        size: int,
        align: int,
    ) -> None: ...
    def spill(
        self,
        *,
        value: ValueRef,
        slot: str,
        offset: int,
    ) -> None: ...
    def reload(
        self,
        *,
        slot: str,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def frame_index(
        self,
        *,
        slot: str,
        offset: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def br(
        self,
        *,
        dest: Block,
        args: list[ValueRef] = ...,
    ) -> None: ...
    def cond_br(
        self,
        *,
        condition: ValueRef,
        true_dest: Block,
        false_dest: Block,
    ) -> None: ...
    def resource(
        self,
        *,
        import_kind: str,
        index: int,
        semantic_type: Type,
        valid_byte_count: int | None = ...,
        cache_swizzle_stride: int | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def live_in(
        self,
        *,
        source: str,
        attrs: Mapping[str, Any] | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
