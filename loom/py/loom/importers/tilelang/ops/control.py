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
from loom.importers.tilelang.nodes import node_kind, node_text, source_name
from loom.importers.tilelang.ops.calls import is_thread_return_call
from loom.importers.tilelang.ops.topology import (
    map_thread_axis,
    mapped_thread_axis_sources,
    thread_axis_from_binding,
    thread_axis_name,
)
from loom.ir import INDEX


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("SeqStmt", convert_seq_stmt)
    registry.register_statement("For", convert_for)
    registry.register_statement("While", convert_while)
    registry.register_statement("IfThenElse", convert_if_then_else)
    registry.register_statement("Evaluate", convert_evaluate)


def convert_seq_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR statement sequence by preserving source order."""

    for child in tuple(getattr(stmt, "seq", ())):
        if convert_thread_return_prefix_guard(child, context, converter):
            continue
        converter.convert_stmt(child, context)


def convert_for(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR for-loop as scf.for."""

    loop_var = getattr(stmt, "loop_var", None)
    thread_binding = getattr(stmt, "thread_binding", None)
    thread_axis = thread_axis_from_binding(thread_binding)
    if thread_binding is not None and thread_axis is None:
        context.record_blocked(node_text(stmt), "thread binding axis is not mapped")
        return
    if thread_axis is not None:
        thread_value = map_thread_axis(
            axis=thread_axis,
            sources=mapped_thread_axis_sources(
                loop_var=loop_var,
                binding=thread_binding,
            ),
            context=context,
            converter=converter,
            lower=getattr(stmt, "min", None),
            name=thread_axis_name(loop_var, thread_axis),
        )
        if thread_value is None:
            context.record_blocked(
                node_text(stmt),
                "thread binding lower bound is not mapped",
            )
            return
        converter.convert_stmt(getattr(stmt, "body", None), context)
        operation_name = f"kernel.{thread_axis.loom_scope}.id<{thread_axis.dimension}>"
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(thread_value)} = {operation_name}",
        )
        return

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


def convert_while(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR while-loop as resultless scf.while."""

    before_region = context.builder.region()
    before_child = context.fork(preview_block=before_region.blocks[0])
    with context.builder.insertion_block(before_region.blocks[0]):
        condition = converter.convert_expr(
            getattr(stmt, "condition", None),
            before_child,
        )
        if condition is None:
            before_child.record_blocked(
                node_text(stmt), "while condition is not mapped"
            )
            context.merge_child_records(before_child)
            return
        if str(condition.type) != "i1":
            before_child.record_blocked(
                node_text(stmt),
                f"while condition must be i1, got {condition.type}",
            )
            context.merge_child_records(before_child)
            return
        context.builder.scf.condition(condition=condition, forwarded=[])
    context.merge_child_records(before_child)

    after_region = context.builder.region()
    after_child = context.fork(preview_block=after_region.blocks[0])
    with context.builder.insertion_block(after_region.blocks[0]):
        converter.convert_stmt(getattr(stmt, "body", None), after_child)
        context.builder.scf.yield_()
    context.merge_child_records(after_child)
    context.builder.scf.while_(
        iter_args=[],
        results=[],
        before=before_region,
        after=after_region,
    )
    context.record_converted(node_text(stmt), "scf.while")


def convert_if_then_else(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR if-then-else as scf.if."""

    if convert_thread_return_prefix_guard(stmt, context, converter):
        return

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

    if convert_thread_return_prefix_guard(stmt, context, converter):
        return

    value = getattr(stmt, "value", None)
    if value is not None:
        converter.convert_expr(value, context, effect=True)


def convert_thread_return_prefix_guard(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    """Import a top-level `if cond: tir.thread_return()` as kernel.exit."""

    if not context.can_emit_kernel_exit():
        return False

    if node_kind(stmt) == "Evaluate" and _is_thread_return_statement(stmt):
        exit_condition = context.ensure_constant("1", "bool", "thread_return")
        context.builder.kernel.exit(condition=exit_condition)
        context.record_converted(node_text(stmt), "kernel.exit")
        return True

    if node_kind(stmt) != "IfThenElse":
        return False
    if getattr(stmt, "else_case", None) is not None:
        return False

    exit_path = _thread_return_exit_path(getattr(stmt, "then_case", None))
    if exit_path is None:
        return False

    condition = converter.convert_expr(getattr(stmt, "condition", None), context)
    if condition is None:
        context.record_blocked(
            node_text(stmt),
            "thread_return guard condition is not mapped",
        )
        return True
    if str(condition.type) != "i1":
        context.record_blocked(
            node_text(stmt),
            f"thread_return guard condition must be i1, got {condition.type}",
        )
        return True

    body = None
    if exit_path:
        body = context.builder.region()
        child = context.fork(preview_block=body.blocks[0])
        with context.builder.insertion_block(body.blocks[0]):
            for body_stmt in exit_path:
                converter.convert_stmt(body_stmt, child)
            context.builder.kernel.return_()
        context.merge_child_records(child)

    context.builder.kernel.exit(condition=condition, body=body)
    context.record_converted(node_text(stmt), "kernel.exit")
    return True


def _thread_return_exit_path(stmt: object | None) -> tuple[object, ...] | None:
    if stmt is None:
        return None
    if _is_thread_return_statement(stmt):
        return ()
    if node_kind(stmt) != "SeqStmt":
        return None
    statements = tuple(getattr(stmt, "seq", ()))
    if not statements or not _is_thread_return_statement(statements[-1]):
        return None
    return statements[:-1]


def _is_thread_return_statement(stmt: object) -> bool:
    if node_kind(stmt) != "Evaluate":
        return False
    return is_thread_return_call(getattr(stmt, "value", None))
