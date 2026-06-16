# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place
# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type
from loom.builders import DialectBuilder

class SanitizerBuilder(DialectBuilder):
    def access(
        self,
        *,
        kind: str,
        view: ValueRef,
        indices: list[int | ValueRef],
        location_id: int | None = ...,
    ) -> None: ...
    def value(
        self,
        *,
        values: list[ValueRef] = ...,
        predicates: list[Predicate],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> list[ValueRef]: ...
    def op(
        self,
        *,
        values: list[ValueRef] = ...,
        predicates: list[Predicate],
        location_id: int | None = ...,
    ) -> None: ...
    def layout(
        self,
        *,
        view: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
        location_id: int | None = ...,
    ) -> ValueRef: ...
