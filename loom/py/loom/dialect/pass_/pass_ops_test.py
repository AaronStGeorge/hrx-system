# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the pass pipeline control dialect."""

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.pass_ import (
    ALL_PASS_OPS,
    PassAnchor,
    PassRepeatMode,
    pass_call,
    pass_fail,
    pass_for,
    pass_halt,
    pass_pipeline,
    pass_repeat,
    pass_run,
    pass_where,
    pass_yield,
)
from loom.dsl import (
    ISOLATED_FROM_ABOVE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    ImplicitTerminator,
    Op,
)
from loom.format.bytecode.reader import BytecodeReader
from loom.format.bytecode.writer import BytecodeWriter
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer


def _op_by_name() -> dict[str, Op]:
    return {op.name: op for op in ALL_PASS_OPS}


def _roundtrip_text(source: str) -> str:
    parser = Parser()
    parser.register_ops(ALL_PASS_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(source)
    printer = Printer()
    printer.register_ops(ALL_PASS_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


class TestAllPassOps:
    def test_registered_ops(self) -> None:
        assert [op.name for op in ALL_PASS_OPS] == [
            "pass.pipeline",
            "pass.for",
            "pass.where",
            "pass.repeat",
            "pass.call",
            "pass.run",
            "pass.fail",
            "pass.halt",
            "pass.yield",
        ]

    def test_specific_exports(self) -> None:
        assert pass_pipeline in ALL_PASS_OPS
        assert pass_for in ALL_PASS_OPS
        assert pass_where in ALL_PASS_OPS
        assert pass_repeat in ALL_PASS_OPS
        assert pass_call in ALL_PASS_OPS
        assert pass_run in ALL_PASS_OPS
        assert pass_fail in ALL_PASS_OPS
        assert pass_halt in ALL_PASS_OPS
        assert pass_yield in ALL_PASS_OPS

    def test_pipeline_shape(self) -> None:
        ops = _op_by_name()
        pipeline = ops["pass.pipeline"]
        assert pipeline.group is not None
        assert pipeline.group.name == "pass"
        assert pipeline.group.dialect_id == 0x15
        assert any(trait == SYMBOL_DEFINE for trait in pipeline.traits)
        assert any(trait == ISOLATED_FROM_ABOVE for trait in pipeline.traits)
        assert any(trait == ImplicitTerminator("pass.yield") for trait in pipeline.traits)
        assert pipeline.symbol_def is not None
        assert pipeline.symbol_def.interfaces == ("record",)
        assert pipeline.symbol_def.bytecode_kind == "LOOM_SYMBOL_RECORD"
        assert pipeline.regions[0].terminator == "pass.yield"

    def test_control_ops_have_pipeline_regions(self) -> None:
        ops = _op_by_name()
        for name in ("pass.pipeline", "pass.for", "pass.where", "pass.repeat"):
            op = ops[name]
            assert len(op.regions) == 1
            assert op.regions[0].terminator == "pass.yield"
            assert any(trait == ImplicitTerminator("pass.yield") for trait in op.traits)

    def test_leaf_shapes(self) -> None:
        ops = _op_by_name()
        run = ops["pass.run"]
        assert not run.operands
        assert not run.results
        assert not run.regions
        assert any(trait == UNKNOWN_EFFECTS for trait in run.traits)
        assert ops["pass.yield"].traits == (TERMINATOR,)

    def test_enums(self) -> None:
        assert [case.keyword for case in PassAnchor.cases] == ["module", "func"]
        assert [case.keyword for case in PassRepeatMode.cases] == [
            "fixed",
            "until_converged",
        ]


class TestPassText:
    def test_pipeline_roundtrip(self) -> None:
        source = """pass.pipeline<module> @cleanup pipeline {
  canonicalize(max_iterations = 10)
  repeat until_converged(max_iterations = 8) {
    cse
    dce
  }
}
"""
        assert _roundtrip_text(source) == source

    def test_func_pipeline_roundtrip(self) -> None:
        source = """pass.pipeline<func> @function_cleanup pipeline {
  for func {
    where name(value = "matmul") {
      vector-memory-footprint(budget_bytes = 4096)
    }
  }
}
"""
        assert _roundtrip_text(source) == source

    def test_debug_ops_roundtrip(self) -> None:
        source = """pass.pipeline<module> @debug pipeline {
  call @cleanup
  fail "expected cleanup to run"
  halt "inspect IR"
}

pass.pipeline<module> @cleanup pipeline {
}
"""
        assert _roundtrip_text(source) == source

    def test_canonical_pipeline_region_still_parses(self) -> None:
        source = """pass.pipeline<module> @cleanup {
  pass.run<canonicalize> {max_iterations = 10}
  pass.yield
}
"""
        expected = """pass.pipeline<module> @cleanup pipeline {
  canonicalize(max_iterations = 10)
}
"""
        assert _roundtrip_text(source) == expected


class TestPassBytecode:
    def test_pipeline_bytecode_roundtrip(self) -> None:
        source = """pass.pipeline<module> @cleanup pipeline {
  unknown-pass(message = "preserve structurally")
}
"""
        parser = Parser()
        parser.register_ops(ALL_PASS_OPS)
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(source)

        data = BytecodeWriter(module, op_decls=ALL_PASS_OPS).write()
        loaded = BytecodeReader(data, op_decls=ALL_PASS_OPS).read()

        printer = Printer()
        printer.register_ops(ALL_PASS_OPS)
        printer.register_types(ALL_BUILTIN_TYPES)
        assert printer.print_module(loaded) == source
