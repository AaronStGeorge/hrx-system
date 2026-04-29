# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place
# ruff: noqa

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loom.builder import TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type

class ScalarBitwiseMixin:
    def andi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def ori(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def xori(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def shli(
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
    def shrsi(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def shrui(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def rotli(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def rotri(
        self,
        *,
        lhs: ValueRef,
        rhs: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def ctlzi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def cttzi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
    def ctpopi(
        self,
        *,
        input: ValueRef,
        results: list[Type | TiedResultSpec],
        name: str | None = ...,
        names: Sequence[str] | None = ...,
        result_names: Sequence[str] | None = ...,
    ) -> ValueRef: ...
