# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place
# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type

class ScalarArithmeticMixin:
    def addi(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def subi(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def muli(
        self,
        *,
        overflow: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def divsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def divui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def remsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def remui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def ceildivsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def ceildivui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def floordivsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def negi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def absi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def minsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def maxsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def minui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def maxui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def fmai(
        self,
        *,
        overflow: str = ...,
        a: ValueRef,
        b: ValueRef,
        c: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def addf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def subf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def mulf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def divf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def remf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def negf(
        self,
        *,
        fastmath: str = ...,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def absf(
        self,
        *,
        fastmath: str = ...,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def minimumf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def maximumf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def minnumf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def maxnumf(
        self,
        *,
        fastmath: str = ...,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def copysignf(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
