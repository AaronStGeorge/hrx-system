# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR affine ops."""

from __future__ import annotations

from typing import Any

from loom.builder import ValueRef
from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("affine.apply", convert_affine_apply)


def convert_affine_apply(op: SourceOp, context: MlirConversionContext) -> bool:
    if not context.require_top_level(op, "affine.apply -> index arithmetic"):
        return True
    map_attr = op.attr("map")
    if map_attr is None:
        context.record_blocked(op.text, "affine.apply has no map attribute")
        return True
    affine_map = map_attr.value
    results = tuple(affine_map.results)
    if len(results) != 1 or len(op.results) != 1:
        context.record_blocked(
            op.text, "multi-result affine.apply needs tuple/result-list lowering"
        )
        return True
    lowering = AffineLowering(op, context, affine_map.n_dims)
    lowered = lowering.lower(results[0], root=True)
    if lowered is None:
        context.record_blocked(
            op.text, lowering.failure or "unsupported affine expression"
        )
        return True
    context.map_result(op.result(), lowered, op.result_type())
    context.record_converted(
        op.text,
        tuple(lowering.emitted_targets)
        or f"{context.ssa(lowered)} = affine.apply alias",
    )
    return True


class AffineLowering:
    """Lowers a single affine expression into Loom index arithmetic."""

    def __init__(
        self,
        op: SourceOp,
        context: MlirConversionContext,
        dimension_count: int,
    ) -> None:
        self.op = op
        self.context = context
        self.dimension_count = dimension_count
        self.emitted_targets: list[str] = []
        self.failure: str | None = None

    def lower(self, expr: Any, *, root: bool = False) -> ValueRef | None:
        kind = type(expr).__name__
        if kind == "AffineConstantExpr":
            return self.context.ensure_constant(
                str(expr.value), "index", f"c{expr.value}_index"
            )
        if kind == "AffineDimExpr":
            return self._operand(expr.position)
        if kind == "AffineSymbolExpr":
            return self._operand(self.dimension_count + expr.position)
        if kind == "AffineAddExpr":
            return self._lower_binary(
                "index.add", expr.lhs, expr.rhs, self._root_name(root, "sum")
            )
        if kind == "AffineMulExpr":
            return self._lower_binary(
                "index.mul", expr.lhs, expr.rhs, self._mul_name(expr)
            )
        if kind == "AffineFloorDivExpr":
            return self._lower_binary(
                "index.div", expr.lhs, expr.rhs, self._div_name(expr)
            )
        self.failure = f"unsupported affine expression node `{kind}`"
        return None

    def _operand(self, index: int) -> ValueRef | None:
        if index < 0 or index >= len(self.op.operands):
            self.failure = f"affine operand index {index} is out of range"
            return None
        operand = self.op.operand(index)
        mapped = self.context.mapped(operand)
        if mapped is None:
            self.failure = f"missing converted affine operand: {operand.get_name()}"
            return None
        return mapped

    def _lower_binary(
        self,
        target_op: str,
        lhs_expr: Any,
        rhs_expr: Any,
        name: str,
    ) -> ValueRef | None:
        lhs = self.lower(lhs_expr)
        rhs = self.lower(rhs_expr)
        if lhs is None or rhs is None:
            return None
        result = self.context.build_binary(
            target_op,
            lhs,
            rhs,
            "index",
            self.context.result_name(self.op.result(), name)
            if name == "row"
            else self.context.fresh_name(name),
        )
        self.emitted_targets.append(f"{self.context.ssa(result)} = {target_op} ...")
        return result

    def _root_name(self, root: bool, fallback: str) -> str:
        return "row" if root else fallback

    def _mul_name(self, expr: Any) -> str:
        lhs = self._name_fragment(expr.lhs)
        rhs = self._name_fragment(expr.rhs)
        return f"{lhs}_x{rhs}"

    def _div_name(self, expr: Any) -> str:
        lhs = self._name_fragment(expr.lhs)
        rhs = self._name_fragment(expr.rhs)
        return f"{lhs}_div{rhs}"

    def _name_fragment(self, expr: Any) -> str:
        kind = type(expr).__name__
        if kind == "AffineConstantExpr":
            return str(expr.value)
        if kind == "AffineDimExpr":
            value = self.op.operand(expr.position)
            mapped = self.context.mapped(value)
            return (
                self.context.ssa(mapped) if mapped else value.get_name()
            ).removeprefix("%")
        if kind == "AffineSymbolExpr":
            value = self.op.operand(self.dimension_count + expr.position)
            mapped = self.context.mapped(value)
            return (
                self.context.ssa(mapped) if mapped else value.get_name()
            ).removeprefix("%")
        return "expr"
