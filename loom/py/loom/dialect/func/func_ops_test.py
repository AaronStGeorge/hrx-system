# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the func dialect ops.

Validates op declarations (structure, format specs, examples) and
round-trip behavior (parse -> print -> identical output) for all
seven func dialect ops.
"""

import pytest

from loom.assembly import (
    OpRef,
    SymbolRef,
)
from loom.dialect.func import (
    ALL_FUNC_OPS,
    func_call,
    func_decl,
    func_def,
    func_ops,
    func_return,
    func_template,
    func_ukernel,
)

# ============================================================================
# Structural tests
# ============================================================================


class TestAllOpsRegistered:
    def test_count(self) -> None:
        assert len(ALL_FUNC_OPS) == 7

    def test_unique_names(self) -> None:
        names = [op.name for op in ALL_FUNC_OPS]
        assert len(set(names)) == len(names)

    def test_all_in_func_namespace(self) -> None:
        for op in ALL_FUNC_OPS:
            assert op.namespace == "func", f"{op.name} not in func namespace"

    def test_all_have_group(self) -> None:
        for op in ALL_FUNC_OPS:
            assert op.group is func_ops

    def test_all_have_docs(self) -> None:
        for op in ALL_FUNC_OPS:
            assert op.doc, f"{op.name} missing doc"

    def test_all_have_format(self) -> None:
        for op in ALL_FUNC_OPS:
            assert op.format, f"{op.name} missing format spec"

    def test_all_have_examples(self) -> None:
        for op in ALL_FUNC_OPS:
            assert op.examples, f"{op.name} missing examples"

    def test_dialect_id(self) -> None:
        assert func_ops.dialect_id == 0x06


class TestOpStructure:
    def test_func_def_has_region(self) -> None:
        assert len(func_def.regions) == 1
        assert func_def.regions[0].name == "body"

    def test_func_decl_no_region(self) -> None:
        assert len(func_decl.regions) == 0

    def test_func_template_has_region(self) -> None:
        assert len(func_template.regions) == 1

    def test_func_ukernel_no_region(self) -> None:
        assert len(func_ukernel.regions) == 0

    def test_func_template_has_opref(self) -> None:
        has_opref = any(isinstance(e, OpRef) for e in func_template.format)
        assert has_opref

    def test_func_ukernel_has_opref(self) -> None:
        has_opref = any(isinstance(e, OpRef) for e in func_ukernel.format)
        assert has_opref

    def test_func_call_has_symbol_ref(self) -> None:
        has_sym = any(isinstance(e, SymbolRef) for e in func_call.format)
        assert has_sym

    def test_func_return_is_terminator(self) -> None:
        assert any(t.name == "Terminator" for t in func_return.traits)

    def test_func_call_variadic_operands(self) -> None:
        assert any(o.variadic for o in func_call.operands)

    def test_func_call_variadic_results(self) -> None:
        assert any(r.variadic for r in func_call.results)


# ============================================================================
# Round-trip tests
# ============================================================================


def _roundtrip(text: str) -> None:
    """Parse text, print, assert identical."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.dialect.test import ALL_TEST_OPS
    from loom.format.text.parser import Parser
    from loom.format.text.printer import Printer

    all_ops = list(ALL_TEST_OPS) + list(ALL_FUNC_OPS)

    parser = Parser()
    parser.register_ops(all_ops)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(text)

    printer = Printer()
    printer.register_ops(all_ops)
    printer.register_types(ALL_BUILTIN_TYPES)
    printed = printer.print_module(module)

    assert printed == text, f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"


class TestFuncDefRoundTrip:
    def test_simple(self) -> None:
        _roundtrip("func.def @add(%a: f32, %b: f32) -> (f32) {\n  %r = test.addi %a, %b : f32\n  func.return %r : f32\n}\n")

    def test_no_results(self) -> None:
        _roundtrip("func.def @noop(%a: f32) {\n  func.return %a : f32\n}\n")

    def test_public(self) -> None:
        _roundtrip("func.def public @entry(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_public_device(self) -> None:
        _roundtrip("func.def public device @kernel(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_device_only(self) -> None:
        _roundtrip("func.def device @helper(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_with_predicates(self) -> None:
        _roundtrip("func.def @constrained(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)] {\n  func.return %a : tensor<[%M]xf32>\n}\n")


class TestFuncDeclRoundTrip:
    def test_simple(self) -> None:
        _roundtrip("func.decl @extern_add(%a: f32, %b: f32) -> (f32)\n")

    def test_public(self) -> None:
        _roundtrip("func.decl public @exported(%a: f32) -> (f32)\n")

    def test_with_predicates(self) -> None:
        _roundtrip("func.decl @constrained(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)]\n")

    def test_no_results(self) -> None:
        _roundtrip("func.decl @sink(%a: f32)\n")

    def test_def_without_body_is_error(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.text.parser import Parser
        from loom.format.text.tokenizer import ParseError

        parser = Parser()
        parser.register_ops(list(ALL_TEST_OPS) + list(ALL_FUNC_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        with pytest.raises(ParseError, match="LBRACE"):
            parser.parse("func.def @bad(%a: f32) -> (f32)\n")

    def test_decl_with_body_is_error(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.text.parser import Parser
        from loom.format.text.tokenizer import ParseError

        parser = Parser()
        parser.register_ops(list(ALL_TEST_OPS) + list(ALL_FUNC_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        with pytest.raises(ParseError, match="LBRACE|top-level op"):
            parser.parse("func.decl @bad(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")


class TestFuncCallApplyRoundTrip:
    """func.call and func.apply are body ops — they appear inside functions."""

    def test_call_in_function(self) -> None:
        _roundtrip("func.def @caller(%a: f32) -> (f32) {\n  %r = func.call @callee(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_apply_in_function(self) -> None:
        _roundtrip("func.def @test_template(%a: f32) -> (f32) {\n  %r = func.apply @my_template(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_call_multiple_args_and_results(self) -> None:
        _roundtrip("func.def @multi(%a: f32, %b: i32) -> (f32, i32) {\n  %r0, %r1 = func.call @process(%a, %b) : (f32, i32) -> (f32, i32)\n  func.return %r0, %r1 : f32, i32\n}\n")


class TestFuncReturnRoundTrip:
    def test_no_operands(self) -> None:
        _roundtrip("func.def @empty() {\n  func.return\n}\n")

    def test_no_operands_initializer(self) -> None:
        _roundtrip("func.def initializer @setup() {\n  func.return\n}\n")

    def test_single_value(self) -> None:
        _roundtrip("func.def @ret(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n")

    def test_multiple_values(self) -> None:
        _roundtrip("func.def @multi_ret(%a: f32, %b: i32) -> (f32, i32) {\n  func.return %a, %b : f32, i32\n}\n")


class TestMultipleFunctions:
    def test_decl_then_def(self) -> None:
        _roundtrip("func.decl @extern(%a: f32) -> (f32)\n\nfunc.def @main(%a: f32) -> (f32) {\n  %r = func.call @extern(%a) : (f32) -> (f32)\n  func.return %r : f32\n}\n")

    def test_multiple_defs(self) -> None:
        _roundtrip("func.def @f1(%a: f32) -> (f32) {\n  func.return %a : f32\n}\n\nfunc.def @f2(%a: i32) -> (i32) {\n  func.return %a : i32\n}\n")
