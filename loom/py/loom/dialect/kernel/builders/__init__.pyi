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

class KernelBuilder(DialectBuilder):
    def def_(
        self,
        *,
        target: str | None = ...,
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
    def return_(
        self,
    ) -> None: ...
    def barrier(
        self,
        *,
        memory_space: str,
        ordering: str,
        scope: str,
    ) -> None: ...
    def copy(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_copy_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        direction: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_gather(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_gather_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def group(
        self,
        *,
        tokens: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def wait(
        self,
        *,
        group: ValueRef,
        newer_groups: int,
    ) -> None: ...
    def descriptor(
        self,
        *,
        dgroups: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_tensor_load_to_lds(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        descriptor: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_tensor_store_from_lds(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        descriptor: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_cluster_gather(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cluster_mask: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def async_cluster_gather_mask(
        self,
        *,
        source: ValueRef,
        dest: ValueRef,
        cluster_mask: ValueRef,
        predicate: ValueRef,
        cache_scope: str,
        cache_temporal: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def workitem_id(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def workgroup_id(
        self,
        *,
        dimension: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
