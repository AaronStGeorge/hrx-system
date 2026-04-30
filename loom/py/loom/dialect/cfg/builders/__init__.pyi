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

class CfgBuilder(DialectBuilder):
    def br(
        self,
        *,
        dest: Block,
        args: list[ValueRef] = ...,
        location_id: int | None = ...,
    ) -> None: ...
    def cond_br(
        self,
        *,
        condition: ValueRef,
        true_dest: Block,
        false_dest: Block,
        location_id: int | None = ...,
    ) -> None: ...
