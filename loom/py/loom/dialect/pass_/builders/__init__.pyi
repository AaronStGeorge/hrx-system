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

class PassBuilder(DialectBuilder):
    def pipeline(
        self,
        *,
        anchor: str,
        symbol: str,
        body: Region | None = ...,
    ) -> None: ...
    def for_(
        self,
        *,
        anchor: str,
        body: Region | None = ...,
    ) -> None: ...
    def where(
        self,
        *,
        predicate: str,
        attrs: Mapping[str, Any] | None = ...,
        body: Region | None = ...,
    ) -> None: ...
    def repeat(
        self,
        *,
        mode: str,
        count: int | None = ...,
        max_iterations: int | None = ...,
        body: Region | None = ...,
    ) -> None: ...
    def call(
        self,
        *,
        callee: str,
    ) -> None: ...
    def run(
        self,
        *,
        key: str,
        options: Mapping[str, Any] | None = ...,
    ) -> None: ...
    def fail(
        self,
        *,
        message: str,
    ) -> None: ...
    def halt(
        self,
        *,
        message: str,
    ) -> None: ...
    def yield_(
        self,
    ) -> None: ...
