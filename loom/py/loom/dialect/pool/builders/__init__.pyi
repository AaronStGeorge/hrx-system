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

class PoolBuilder(DialectBuilder):
    def load(
        self,
        *,
        pool: ValueRef,
        page_id: ValueRef,
        page_bytes: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def store(
        self,
        *,
        pool: ValueRef,
        page_id: ValueRef,
        page_bytes: ValueRef,
        offset_in_page: ValueRef,
        data: ValueRef,
    ) -> None: ...
    def pin(
        self,
        *,
        pool: ValueRef,
        block_id: ValueRef,
    ) -> None: ...
    def unpin(
        self,
        *,
        pool: ValueRef,
        block_id: ValueRef,
    ) -> None: ...
    def buffer(
        self,
        *,
        pool: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
