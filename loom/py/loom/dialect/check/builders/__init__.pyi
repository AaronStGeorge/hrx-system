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

class CheckBuilder(DialectBuilder):
    def case(
        self,
        *,
        visibility: str | None = ...,
        case_symbol: str,
        body: Region | None = ...,
    ) -> None: ...
    def return_(
        self,
    ) -> None: ...
    def requires(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any],
    ) -> None: ...
    def skip_if(
        self,
        *,
        provider: str,
        attrs: Mapping[str, Any],
        reason: str | None = ...,
    ) -> None: ...
    def range(
        self,
        *,
        policy: str,
        lower: Any,
        upper: Any,
        step: Any | None = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def choice(
        self,
        *,
        values: list[int],
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def seed(
        self,
        *,
        base: int,
        count: int,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def literal(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def iota(
        self,
        *,
        offset: Any,
        step: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def fill(
        self,
        *,
        value: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def uniform(
        self,
        *,
        seed: ValueRef,
        lower: Any,
        upper: Any,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def file_read_npy(
        self,
        *,
        path: str,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def file_write_npy(
        self,
        *,
        value: ValueRef,
        path: str,
        mode: str | None = ...,
    ) -> None: ...
    def call(
        self,
        *,
        provider: str,
        callee: str,
        inputs: list[ValueRef] = ...,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> list[ValueRef]: ...
    def equal(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
    ) -> None: ...
    def bitwise(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
    ) -> None: ...
    def close(
        self,
        *,
        actual: ValueRef,
        expected: ValueRef,
        atol: float,
        rtol: float,
        nan: str,
    ) -> None: ...
    def shape(
        self,
        *,
        value: ValueRef,
        dims: list[int | ValueRef],
    ) -> None: ...
    def expect(
        self,
        *,
        provider: str,
        actual: ValueRef,
        expected: ValueRef,
        attrs: Mapping[str, Any] | None = ...,
    ) -> None: ...
    def benchmark(
        self,
        *,
        benchmark: str,
        case_ref: str,
        attrs: Mapping[str, Any],
    ) -> None: ...
