# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place
# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type

class VectorContractionMixin:
    def dotf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        init: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def dot2f(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        acc: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def dot4i(
        self,
        *,
        kind: str,
        lhs: ValueRef,
        rhs: ValueRef,
        acc: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def dot8i4(
        self,
        *,
        kind: str,
        lhs: ValueRef,
        rhs: ValueRef,
        acc: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def dot4f8(
        self,
        *,
        kind: str,
        lhs: ValueRef,
        rhs: ValueRef,
        acc: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
    def mma(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        init: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
