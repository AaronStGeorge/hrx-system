# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the test dialect ops.

Validates that every test op is well-formed: correct field counts,
format specs reference valid fields, constraints reference valid
fields, traits are sensible, and examples exist.
"""

from loom.assembly import (
    Attr,
    AttrDict,
    BindingList,
    FormatElement,
    FuncArgs,
    IndexList,
    Keyword,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypeOf,
    TypesOf,
)
from loom.dialect.test import (
    ALL_TEST_OPS,
    cmp_predicates,
    test_addi,
    test_attrs,
    test_branch,
    test_cast,
    test_cmp,
    test_constant,
    test_convert,
    test_func,
    test_implicit_yield,
    test_invoke,
    test_loop,
    test_map,
    test_neg,
    test_ops,
    test_slice,
    test_update,
    test_yield,
)
from loom.dsl import (
    Op,
    TypeConstraint,
)


class TestAllOpsRegistered:
    def test_count(self) -> None:
        assert len(ALL_TEST_OPS) > 0

    def test_unique_names(self) -> None:
        names = [op.name for op in ALL_TEST_OPS]
        assert len(set(names)) == len(names), f"Duplicate op names: {names}"

    def test_all_in_test_namespace(self) -> None:
        for op in ALL_TEST_OPS:
            assert op.namespace == "test", f"{op.name} not in test namespace"

    def test_all_have_group(self) -> None:
        for op in ALL_TEST_OPS:
            assert op.group is test_ops, f"{op.name} missing group"

    def test_all_have_docs(self) -> None:
        for op in ALL_TEST_OPS:
            assert op.doc, f"{op.name} missing doc"

    def test_all_have_format(self) -> None:
        for op in ALL_TEST_OPS:
            if op.format:
                continue
            assert not op.operands
            assert not op.results
            assert not op.attrs
            assert not op.regions

    def test_all_have_examples(self) -> None:
        for op in ALL_TEST_OPS:
            assert op.examples, f"{op.name} missing examples"


class TestFormatFieldsMatchDeclarations:
    """Every field referenced in a format spec must exist on the op."""

    def _collect_fields(self, elements: tuple[FormatElement, ...]) -> set[str]:
        """Recursively collect all field names from format elements."""
        fields: set[str] = set()
        for elem in elements:
            match elem:
                case Ref(field=f) | Refs(field=f):
                    fields.add(f)
                case Attr(field=f):
                    fields.add(f)
                case SymbolRef(field=f):
                    fields.add(f)
                case TypeOf(field=f) | TypesOf(field=f):
                    fields.add(f)
                case ResultType(field=f) | ResultTypeList(field=f):
                    fields.add(f)
                case Region(field=f):
                    fields.add(f)
                case BindingList(field=f) | FuncArgs(field=f):
                    fields.add(f)
                case PredicateList(field=f):
                    fields.add(f)
                case IndexList(dynamic=d, static=s):
                    fields.add(d)
                    fields.add(s)
                case OptionalGroup(elements=inner):
                    fields |= self._collect_fields(inner)
                case Scope(elements=inner):
                    fields |= self._collect_fields(inner)
                case Keyword() | AttrDict():
                    pass
        return fields

    def _declared_fields(self, op: Op) -> set[str]:
        """All field names declared on an op."""
        fields: set[str] = set()
        for o in op.operands:
            fields.add(o.name)
        for result in op.results:
            fields.add(result.name)
        for attr in op.attrs:
            fields.add(attr.name)
        for region in op.regions:
            fields.add(region.name)
        return fields

    def test_all_ops_format_fields_valid(self) -> None:
        """Every field in every format spec must be a declared field,
        or a well-known implicit field (iv, args)."""
        # Fields that are implicit (not in operands/results/attrs/regions
        # but valid in format specs).
        implicit_fields = {"iv", "args"}

        for op in ALL_TEST_OPS:
            format_fields = self._collect_fields(op.format)
            declared = self._declared_fields(op) | implicit_fields
            unknown = format_fields - declared
            assert not unknown, f"{op.name}: format references undeclared fields: {unknown}. Declared: {declared}"


class TestConstraintFieldsMatchDeclarations:
    """Every field referenced in constraints must exist on the op."""

    def test_all_ops_constraint_fields_valid(self) -> None:
        implicit_fields = {"iv", "args"}
        for op in ALL_TEST_OPS:
            declared: set[str] = set()
            for operand in op.operands:
                declared.add(operand.name)
            for result in op.results:
                declared.add(result.name)
            for attr in op.attrs:
                declared.add(attr.name)
            for region in op.regions:
                declared.add(region.name)
            declared |= implicit_fields

            for constraint in op.constraints:
                for arg in constraint.args:
                    assert arg in declared, f"{op.name}: constraint {constraint} references undeclared field '{arg}'. Declared: {declared}"


class TestFormatElementCoverage:
    """Every format element type must appear in at least one test op."""

    def _all_element_types(self) -> set[type]:
        """Recursively collect all element types used across all ops."""
        types: set[type] = set()
        for op in ALL_TEST_OPS:
            self._collect_types(op.format, types)
        return types

    def _collect_types(
        self,
        elements: tuple[FormatElement, ...],
        types: set[type],
    ) -> None:
        for elem in elements:
            types.add(type(elem))
            if isinstance(elem, OptionalGroup | Scope):
                self._collect_types(elem.elements, types)

    def test_all_element_types_covered(self) -> None:
        used = self._all_element_types()
        required = {
            Ref,
            Refs,
            Attr,
            SymbolRef,
            TypeOf,
            TypesOf,
            ResultType,
            ResultTypeList,
            Keyword,
            AttrDict,
            Region,
            IndexList,
            BindingList,
            FuncArgs,
            PredicateList,
            OptionalGroup,
        }
        missing = required - used
        assert not missing, f"Format element types not covered: {missing}"


class TestSpecificOps:
    """Spot-check specific ops for expected structure."""

    def test_addi(self) -> None:
        assert test_addi.is_pure
        assert test_addi.is_commutative
        assert len(test_addi.operands) == 2
        assert len(test_addi.results) == 1
        assert len(test_addi.constraints) == 1

    def test_neg(self) -> None:
        assert test_neg.is_pure
        assert test_neg.has_trait("Involution")
        assert len(test_neg.operands) == 1

    def test_cast(self) -> None:
        assert test_cast.operands[0].type_constraint == TypeConstraint.INTEGER
        assert test_cast.results[0].type_constraint == TypeConstraint.FLOAT

    def test_constant(self) -> None:
        assert test_constant.has_trait("ConstantLike")
        assert len(test_constant.attrs) == 1
        assert test_constant.attrs[0].name == "value"

    def test_cmp(self) -> None:
        assert test_cmp.attrs[0].enum_def is cmp_predicates
        assert len(cmp_predicates.cases) == 6
        assert cmp_predicates.keywords[2] == "lt"

    def test_map(self) -> None:
        assert test_map.has_trait("Elementwise")
        assert test_map.operands[0].variadic
        assert len(test_map.regions) == 1
        assert test_map.regions[0].single_block
        assert len(test_map.constraints) == 5

    def test_update(self) -> None:
        assert test_update.operand("source") is not None
        assert test_update.operand("target") is not None
        offsets_operand = test_update.operand("offsets")
        assert offsets_operand is not None
        assert offsets_operand.variadic
        assert test_update.attr("static_offsets") is not None

    def test_invoke(self) -> None:
        callee_attr = test_invoke.attr("callee")
        assert callee_attr is not None
        assert callee_attr.attr_type == "symbol"
        operands_operand = test_invoke.operand("operands")
        assert operands_operand is not None
        assert operands_operand.variadic
        results_result = test_invoke.result("results")
        assert results_result is not None
        assert results_result.variadic

    def test_slice(self) -> None:
        offsets_operand = test_slice.operand("offsets")
        assert offsets_operand is not None
        assert offsets_operand.variadic
        static_offsets_attr = test_slice.attr("static_offsets")
        assert static_offsets_attr is not None
        assert static_offsets_attr.attr_type == "i64_array"

    def test_loop(self) -> None:
        assert test_loop.operand("lower_bound") is not None
        assert test_loop.operand("upper_bound") is not None
        assert test_loop.operand("step") is not None
        iter_args_operand = test_loop.operand("iter_args")
        assert iter_args_operand is not None
        assert iter_args_operand.variadic
        results_result = test_loop.result("results")
        assert results_result is not None
        assert results_result.variadic
        assert len(test_loop.regions) == 1
        # Format has OptionalGroups for iter_args and results.
        optional_elements = [e for e in test_loop.format if isinstance(e, OptionalGroup)]
        assert len(optional_elements) == 2

    def test_branch(self) -> None:
        assert test_branch.operand("condition") is not None
        assert len(test_branch.regions) == 2
        # Only results is optional — the else region is mandatory.
        optional_elements = [e for e in test_branch.format if isinstance(e, OptionalGroup)]
        assert len(optional_elements) == 1

    def test_yield(self) -> None:
        assert test_yield.is_terminator
        values_operand = test_yield.operand("values")
        assert values_operand is not None
        assert values_operand.variadic
        assert len(test_yield.results) == 0

    def test_implicit_yield(self) -> None:
        assert test_implicit_yield.is_terminator
        assert len(test_implicit_yield.operands) == 0
        assert len(test_implicit_yield.results) == 0
        assert len(test_implicit_yield.attrs) == 0
        assert len(test_implicit_yield.regions) == 0

    def test_func(self) -> None:
        # Top-level optional groups: visibility, cc.
        optional_elements = [e for e in test_func.format if isinstance(e, OptionalGroup)]
        assert len(optional_elements) == 2
        signature_scope = [e for e in test_func.format if isinstance(e, Scope)]
        assert len(signature_scope) == 1
        # Signature-scope optional groups: results, predicates.
        signature_optional_elements = [e for e in signature_scope[0].elements if isinstance(e, OptionalGroup)]
        assert len(signature_optional_elements) == 2
        # Body is a Region, not an OptionalGroup (always present for test.func).
        assert test_func.region("body") is not None

    def test_attrs(self) -> None:
        # Format includes AttrDict.
        has_attr_dict = any(isinstance(e, AttrDict) for e in test_attrs.format)
        assert has_attr_dict

    def test_convert(self) -> None:
        assert test_convert.is_pure
        assert len(test_convert.operands) == 1
        assert len(test_convert.results) == 1
        # Format uses ResultType (bare, no parens), not ResultTypeList.
        has_result_type = any(isinstance(e, ResultType) for e in test_convert.format)
        assert has_result_type
        has_result_type_list = any(isinstance(e, ResultTypeList) for e in test_convert.format)
        assert not has_result_type_list


class TestCanonicalExamples:
    """Verify examples follow canonical form (one op per line, regions indented)."""

    def test_no_leading_whitespace(self) -> None:
        """Canonical examples have no leading indentation."""
        for op in ALL_TEST_OPS:
            for example in op.examples:
                first_line = example.split("\n")[0]
                assert not first_line.startswith("  "), f"{op.name}: example has leading indent: {first_line!r}"

    def test_region_ops_indented(self) -> None:
        """Ops inside regions are indented by 2 spaces."""
        # Lines that are structural (closing brace, else separator) are
        # at the outer indentation level, not indented.
        structural_prefixes = ("}", "} else {")
        for op in ALL_TEST_OPS:
            for example in op.examples:
                lines = example.split("\n")
                if len(lines) > 1:
                    for line in lines[1:-1]:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if any(stripped == p for p in structural_prefixes):
                            continue
                        assert line.startswith("  "), f"{op.name}: region body line not indented: {line!r}"
