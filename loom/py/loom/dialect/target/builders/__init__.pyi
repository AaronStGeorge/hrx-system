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

class TargetBuilder(DialectBuilder):
    def artifact(
        self,
        *,
        symbol: str,
        target: str,
        artifact_format: str | None = ...,
        abi: str | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def profile(
        self,
        *,
        symbol: str,
        preset: str,
        overrides: Mapping[str, Any] | None = ...,
        location_id: int | None = ...,
    ) -> None: ...
