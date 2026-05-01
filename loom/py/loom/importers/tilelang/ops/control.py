# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured control-flow TileLang/TIR node converters."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import node_text, source_name
from loom.ir import INDEX


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("SeqStmt", convert_seq_stmt)
    registry.register_statement("For", convert_for)
    registry.register_statement("IfThenElse", convert_if_then_else)
    registry.register_statement("Evaluate", convert_evaluate)


def convert_seq_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR statement sequence by preserving source order."""

    for child in tuple(getattr(stmt, "seq", ())):
        converter.convert_stmt(child, context)


def convert_for(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR for-loop as scf.for."""

    loop_var = getattr(stmt, "loop_var", None)
    loop_name = sanitize_identifier(source_name(loop_var, fallback="iv"))
    lower = converter.convert_expr(
        getattr(stmt, "min", None),
        context,
        index_like=True,
    )
    extent = converter.convert_expr(
        getattr(stmt, "extent", None),
        context,
        index_like=True,
    )
    if lower is None or extent is None:
        context.record_blocked(node_text(stmt), "loop bounds are not mapped")
        return
    upper = context.builder.index.add(
        lhs=lower,
        rhs=extent,
        results=[INDEX],
        name=context.fresh_name(f"{loop_name}_ub"),
    )
    step = context.ensure_constant("1", "index", "c1")
    body = context.builder.region(args=[(loop_name, INDEX)])
    loop_var_ref = ValueRef(body.blocks[0].arg_ids[0], context.builder.ir)
    child = context.fork(preview_block=body.blocks[0])
    child.map_value(loop_var, loop_var_ref, "index")
    with context.builder.insertion_block(body.blocks[0]):
        converter.convert_stmt(getattr(stmt, "body", None), child)
        context.builder.scf.yield_()
    context.merge_child_records(child)
    context.builder.scf.for_(
        lower_bound=lower,
        upper_bound=upper,
        step=step,
        iter_args=[],
        results=[],
        body=body,
    )
    context.record_converted(node_text(stmt), "scf.for")


def convert_if_then_else(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR if-then-else as scf.if."""

    condition = converter.convert_expr(getattr(stmt, "condition", None), context)
    if condition is None:
        context.record_blocked(node_text(stmt), "if condition is not mapped")
        return
    then_region = context.builder.region()
    else_region = context.builder.region()
    then_child = context.fork(preview_block=then_region.blocks[0])
    else_child = context.fork(preview_block=else_region.blocks[0])
    with context.builder.insertion_block(then_region.blocks[0]):
        converter.convert_stmt(getattr(stmt, "then_case", None), then_child)
        context.builder.scf.yield_()
    with context.builder.insertion_block(else_region.blocks[0]):
        converter.convert_stmt(getattr(stmt, "else_case", None), else_child)
        context.builder.scf.yield_()
    context.merge_child_records(then_child)
    context.merge_child_records(else_child)
    context.builder.scf.if_(
        condition=condition,
        results=[],
        then_region=then_region,
        else_region=else_region,
    )
    context.record_converted(node_text(stmt), "scf.if")


def convert_evaluate(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import an effect expression wrapper."""

    value = getattr(stmt, "value", None)
    if value is not None:
        converter.convert_expr(value, context)
