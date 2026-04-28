# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from collections.abc import Mapping
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Block, Predicate, Region, Type


class TestBuilders:
    """Typed builder methods for test ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def addi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Test binary integer op.

        Example::
            %result = test.addi %lhs, %rhs : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("test.addi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def neg(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Test unary float op.

        Example::
            %result = test.neg %input : f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("test.neg", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def cast(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Test cast op.

        Example::
            %result = test.cast %input : i32 to f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("test.cast", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def constant(self, *, value: Any, result_types: list[Type]) -> ValueRef:
        """Test constant materialization.

        Example::
            %c42 = test.constant 42 : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["value"] = value
        return cast(ValueRef, self._b.build("test.constant", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def use(self, *, values: list[ValueRef]) -> None:
        """Side-effecting sink that observes values without producing results. Not DCE-able. Use in tests to keep values alive for inspection.

        Example::
            test.use %a : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(values)
        self._b.build("test.use", _operands, attributes=_attributes, regions=_regions)

    def cmp(self, *, predicate: str, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Test comparison op.

        Example::
            %result = test.cmp lt, %lhs, %rhs : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["predicate"] = predicate
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("test.cmp", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def map(self, *, inputs: list[ValueRef], body: Region | None = None, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test region-capture elementwise op.

        Example::
            %result = test.map(%element = %input : tile<4xf32>) {
              %negated = test.neg %element : f32
              test.yield %negated : f32
            } -> (tile<4xf32>)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        _operands.extend(inputs)
        return cast(ValueRef, self._b.build("test.map", _operands, results=results, attributes=_attributes, regions=_regions))

    def update(self, *, source: ValueRef, target: ValueRef, offsets: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test tied result with index list.

        Example::
            %result = test.update %tile, %tensor[%offset] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        _operands.append(target)
        _sentinel = -(2**63)
        _static = []
        for _idx in offsets:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_offsets"] = _static
        return cast(ValueRef, self._b.build("test.update", _operands, results=results, attributes=_attributes, regions=_regions))

    def invoke(self, *, callee: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test variadic call-like op with tied results. The verifier checks that the invoke signature matches the referenced function declaration or definition.

        Example::
            %output, %count = test.invoke @callee(%weights, %input) : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["callee"] = callee
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("test.invoke", _operands, results=results, attributes=_attributes, regions=_regions))

    def low_call(self, *, callee: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test call-like op classified like a target-low internal call.

        Example::
            test.low_call @callee() : ()
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["callee"] = callee
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("test.low_call", _operands, results=results, attributes=_attributes, regions=_regions))

    def low_invoke(self, *, callee: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test call-like op classified like an explicit target-low invocation.

        Example::
            test.low_invoke @callee() : ()
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["callee"] = callee
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("test.low_invoke", _operands, results=results, attributes=_attributes, regions=_regions))

    def slice(self, *, source: ValueRef, offsets: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test index list with mixed static/dynamic offsets.

        Example::
            %subtile = test.slice %source[0, %offset] : tile<64x64xf16> -> (tile<16x16xf16>)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        _sentinel = -(2**63)
        _static = []
        for _idx in offsets:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_offsets"] = _static
        return cast(ValueRef, self._b.build("test.slice", _operands, results=results, attributes=_attributes, regions=_regions))

    def loop(self, *, lower_bound: ValueRef, upper_bound: ValueRef, step: ValueRef, iter_args: list[ValueRef], results: list[Type | TiedResultSpec], body: Region | None = None) -> list[ValueRef]:
        """Test for-loop with iter_args and tied results.

        Example::
            %result = test.loop %i = %c0 to %count step %c1 iter_args(%accumulator = %init : f32) -> (%init as f32) {
              %next = test.neg %accumulator : f32
              test.yield %next : f32
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        _operands.append(lower_bound)
        _operands.append(upper_bound)
        _operands.append(step)
        _operands.extend(iter_args)
        return cast(list[ValueRef], self._b.build("test.loop", _operands, results=results, attributes=_attributes, regions=_regions))

    def block_args(self, *, inputs: list[ValueRef], body: Region | None = None) -> None:
        """Test op with explicit BlockArgs syntax for a region entry block.

        Example::
            test.block_args %value : i32 do(%arg: i32) {
              test.yield
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        _operands.extend(inputs)
        self._b.build("test.block_args", _operands, attributes=_attributes, regions=_regions)

    def branch(self, *, condition: ValueRef, results: list[Type | TiedResultSpec], then_region: Region | None = None, else_region: Region | None = None) -> list[ValueRef]:
        """Test if/else with both regions always present.

        Example::
            %result = test.branch %condition -> (f32) {
              test.yield %true_value : f32
            } else {
              test.yield %false_value : f32
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if then_region is not None:
            _regions.append(then_region)
        if else_region is not None:
            _regions.append(else_region)
        _operands.append(condition)
        return cast(list[ValueRef], self._b.build("test.branch", _operands, results=results, attributes=_attributes, regions=_regions))

    def implicit_yield(self) -> None:
        """Dedicated zero-field implicit terminator synthesized for elidable test regions.

        Example::
            test.implicit_yield
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        self._b.build("test.implicit_yield", _operands, attributes=_attributes, regions=_regions)

    def yield_(self, *, values: list[ValueRef]) -> None:
        """Test yield terminator.

        Example::
            test.yield
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(values)
        self._b.build("test.yield", _operands, attributes=_attributes, regions=_regions)

    def br(self, *, dest: Block) -> None:
        """Test CFG branch terminator with a semantic successor edge.

        Example::
            test.br ^ dest
        """
        _operands: list[ValueRef | int] = []
        _successors: list[Block] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _successors.append(dest)
        self._b.build("test.br", _operands, successors=_successors, attributes=_attributes, regions=_regions)

    def func(
        self,
        *,
        visibility: str | None = None,
        cc: str | None = None,
        callee: str,
        args: list[ValueRef] | None = None,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] | None = None,
        body: Region | None = None,
    ) -> list[ValueRef]:
        """Test function definition with body always present.

        Example::
            test.func @identity(%input: f32) -> (f32) {
              test.yield %input : f32
            }
        """
        _operands: list[ValueRef | int] = []
        _func_args: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if visibility is not None:
            _attributes["visibility"] = visibility
        if cc is not None:
            _attributes["cc"] = cc
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        if predicates:
            _attributes["predicates"] = predicates
        if body is not None:
            _regions.append(body)
        return cast(list[ValueRef], self._b.build("test.func", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def decl(self, *, visibility: str | None = None, cc: str | None = None, callee: str, args: list[ValueRef] | None = None, results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test function declaration with no body and signature arguments stored as op operands.

        Example::
            test.decl @identity(%input: f32) -> (%input as f32)
        """
        _operands: list[ValueRef | int] = []
        _func_args: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if visibility is not None:
            _attributes["visibility"] = visibility
        if cc is not None:
            _attributes["cc"] = cc
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        return cast(list[ValueRef], self._b.build("test.decl", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def record(self, *, kind: str | None = None, symbol: str, dict: Mapping[str, Any] | None = None) -> None:
        """Test named module record with generic symbol payload metadata.

        Example::
            test.record target @target {arch = "gfx1100", lanes = 64}
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if kind is not None:
            _attributes["kind"] = kind
        _attributes["symbol"] = symbol
        if dict is not None:
            _attributes["dict"] = dict
        self._b.build("test.record", _operands, attributes=_attributes, regions=_regions)

    def attrs(self, *, input: ValueRef, dict: Mapping[str, Any] | None = None, result_types: list[Type]) -> ValueRef:
        """Test op with attribute dictionary.

        Example::
            %result = test.attrs %input {axis = 0, label = "foo"} : f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if dict is not None:
            _attributes["dict"] = dict
        _operands.append(input)
        return cast(ValueRef, self._b.build("test.attrs", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def operand_dict(self, *, input: ValueRef, params: dict[str, ValueRef], result_types: list[Type]) -> ValueRef:
        """Test op with a keyed SSA operand dictionary.

        Example::
            %result = test.operand_dict %input {alpha = %a : i32, beta = %b : f32} : f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        if params:
            _operand_dict_names: builtins.dict[str, int] = {}
            for _name in sorted(params):
                _operand_dict_names[_name] = len(_operand_dict_names)
                _operands.append(params[_name])
            _attributes["param_names"] = _operand_dict_names
        return cast(ValueRef, self._b.build("test.operand_dict", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def attr_table(self, *, selector: ValueRef, case_keys: list[int], values: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test op with a static-attribute-keyed SSA value table.

        Example::
            %a, %b = test.attr_table %selector {0 = (%a0, %b0), 1 = (%a1, %b1)} default(%ad, %bd) : i32, f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["case_keys"] = case_keys
        _operands.append(selector)
        _operands.extend(values)
        return cast(list[ValueRef], self._b.build("test.attr_table", _operands, results=results, attributes=_attributes, regions=_regions))

    def region_table(self, *, selector: ValueRef, case_keys: list[int], default_region: Region, case_regions: list[Region]) -> None:
        """Test op with a static-attribute-keyed region table.

        Example::
            test.region_table %selector {
              case 0 {
                test.yield
              }
              default {
                test.yield
              }
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["case_keys"] = case_keys
        _regions.append(default_region)
        _regions.extend(case_regions)
        _operands.append(selector)
        self._b.build("test.region_table", _operands, attributes=_attributes, regions=_regions)

    def deflate(self, *, input: ValueRef, results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Test op with result type referencing a co-result dim.

        Example::
            %output, %length = test.deflate %input : tensor<[%M]xf32> -> (tensor<[%length]xf32>, index)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(list[ValueRef], self._b.build("test.deflate", _operands, results=results, attributes=_attributes, regions=_regions))

    def assume(self, *, values: list[ValueRef], predicates: list[Predicate], result_types: list[Type]) -> list[ValueRef]:
        """Test predicate-constrained identity (SSA assume).

        Example::
            %M2 = test.assume %M [mul(%M, 16)] : index
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["predicates"] = predicates
        _operands.extend(values)
        return cast(list[ValueRef], self._b.build("test.assume", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def convert(self, *, input: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test op with bare result type (no parentheses).

        Example::
            %result = test.convert %input : i32 -> f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("test.convert", _operands, results=results, attributes=_attributes, regions=_regions))

    def reduce(self, *, inputs: list[ValueRef], result_types: list[Type]) -> ValueRef:
        """Test variadic operands with SameType constraint across variadic and result.

        Example::
            %sum = test.reduce %a, %b, %c : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(inputs)
        return cast(ValueRef, self._b.build("test.reduce", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def read_resource(self, *, source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test op that reads from a resource operand.

        Example::
            %tile = test.read_resource %pool : pool<[%BS]> -> tile<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        return cast(ValueRef, self._b.build("test.read_resource", _operands, results=results, attributes=_attributes, regions=_regions))

    def write_resource(self, *, target: ValueRef, data: ValueRef) -> None:
        """Test op that writes to a resource operand.

        Example::
            test.write_resource %pool, %tile : pool<[%BS]>, tile<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(target)
        _operands.append(data)
        self._b.build("test.write_resource", _operands, attributes=_attributes, regions=_regions)

    def mutate_resource(self, *, target: ValueRef, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test op that atomically reads and writes a resource operand.

        Example::
            %old = test.mutate_resource %pool, %delta : pool<[%BS]>, i32 -> i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(target)
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.mutate_resource", _operands, results=results, attributes=_attributes, regions=_regions))

    def alloc(self, *, size: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test allocation op. Each execution produces a distinct identity even with identical operands. Prevents CSE but allows DCE when unused.

        Example::
            %pool = test.alloc %sz : index -> pool<[%BS]>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(size)
        return cast(ValueRef, self._b.build("test.alloc", _operands, results=results, attributes=_attributes, regions=_regions))

    def isolated_region(self, *, results: list[Type | TiedResultSpec], body: Region | None = None) -> list[ValueRef]:
        """Test op with an isolated single-block region. Values from the enclosing scope are not visible inside the body.

        Example::
            %r = test.isolated_region -> (i32) {
              %c = test.constant 42 : i32
              test.yield %c : i32
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        return cast(list[ValueRef], self._b.build("test.isolated_region", _operands, results=results, attributes=_attributes, regions=_regions))

    def counter(self, *, value: int, result_types: list[Type]) -> ValueRef:
        """Test op for canonicalize multi-step and error path testing. Canonicalize: value < 0 returns error, value > 0 decrements, value == 0 is fixed point.

        Example::
            %c = test.counter 3 : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["value"] = value
        return cast(ValueRef, self._b.build("test.counter", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def dim(self, *, source: ValueRef, dim_index: int, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Test dimension query to exercise ATTR_IN_RANGE_RANK constraint.

        Example::
            %d = test.dim %t[0] : tile<4xf32> -> index
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["dim_index"] = dim_index
        _operands.append(source)
        return cast(ValueRef, self._b.build("test.dim", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_range_lo(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the analysis range lower bound as an i64 constant.

        Example::
            %lo = test.fact_range_lo %x : index -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_range_lo", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_range_hi(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the analysis range upper bound as an i64 constant.

        Example::
            %hi = test.fact_range_hi %x : index -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_range_hi", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_divisor(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the analysis known divisor as an i64 constant.

        Example::
            %div = test.fact_divisor %x : index -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_divisor", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_non_negative(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input is provably non-negative, 0 otherwise.

        Example::
            %nn = test.fact_non_negative %x : index -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_non_negative", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_non_zero(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input is provably non-zero, 0 otherwise.

        Example::
            %nz = test.fact_non_zero %x : index -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_non_zero", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_positive(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input is provably positive (> 0), 0 otherwise.

        Example::
            %pos = test.fact_positive %x : index -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_positive", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_power_of_two(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input is provably a power of two, 0 otherwise.

        Example::
            %p2 = test.fact_power_of_two %x : index -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_power_of_two", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_is_vector_iota(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input has a vector.iota analysis summary, 0 otherwise.

        Example::
            %is = test.fact_is_vector_iota %x : vector<[%n]xindex> -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_is_vector_iota", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_is_vector_prefix_mask(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input has a vector.mask.range analysis summary, 0 otherwise.

        Example::
            %is = test.fact_is_vector_prefix_mask %x : vector<[%n]xi1> -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_is_vector_prefix_mask", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_encoding_layout_kind(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes an encoding-summary address-layout kind as an i64 constant.

        Example::
            %kind = test.fact_encoding_layout_kind %layout : encoding<layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_encoding_layout_kind", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_encoding_layout_stride_hi(self, *, value: ValueRef, axis: int, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes an encoding-summary strided-layout stride upper bound as an i64 constant.

        Example::
            %hi = test.fact_encoding_layout_stride_hi %layout[0] : encoding<layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["axis"] = axis
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_encoding_layout_stride_hi", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_encoding_matrix_field(self, *, value: ValueRef, field: str, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a packed matrix storage-schema summary field as an i64 constant. Supported fields are format, scale_kind, scale_format, scale_placement, scale_conversion, packed_registers, packed_elements, zero_scale_fallback, and static_spec.

        Example::
            %format = test.fact_encoding_matrix_field %schema["format"] : encoding<schema> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["field"] = field
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_encoding_matrix_field", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_is_buffer_reference(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input has a buffer-reference analysis summary, 0 otherwise.

        Example::
            %is = test.fact_is_buffer_reference %buffer : buffer -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_is_buffer_reference", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_is_view_reference(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input has a view-reference analysis summary, 0 otherwise.

        Example::
            %is = test.fact_is_view_reference %view : view<4xf32, %layout> -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_is_view_reference", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_buffer_memory_space(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a buffer-reference memory-space enum value as an i64 constant, or -1 when unknown.

        Example::
            %space = test.fact_buffer_memory_space %buffer : buffer -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_buffer_memory_space", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_memory_space(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a view-reference memory-space enum value as an i64 constant, or -1 when unknown.

        Example::
            %space = test.fact_view_memory_space %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_memory_space", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_root_matches(self, *, view: ValueRef, root: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if a view reference and another reference share the same root identity.

        Example::
            %same = test.fact_view_root_matches %view, %buffer : view<4xf32, %layout>, buffer -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(view)
        _operands.append(root)
        return cast(ValueRef, self._b.build("test.fact_view_root_matches", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_alias_scope_known(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if the input has a comparable storage alias scope, 0 otherwise.

        Example::
            %known = test.fact_alias_scope_known %buffer : buffer -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_alias_scope_known", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_alias_scope_matches(self, *, lhs: ValueRef, rhs: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Returns 1 if both inputs have the same comparable storage alias scope, 0 otherwise.

        Example::
            %same = test.fact_alias_scope_matches %lhs, %rhs : buffer, buffer -> i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("test.fact_alias_scope_matches", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_byte_offset_lo(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a view-reference byte-offset lower bound as an i64 constant.

        Example::
            %lo = test.fact_view_byte_offset_lo %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_byte_offset_lo", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_byte_offset_hi(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a view-reference byte-offset upper bound as an i64 constant.

        Example::
            %hi = test.fact_view_byte_offset_hi %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_byte_offset_hi", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_byte_length_lo(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a view-reference footprint byte-length lower bound as an i64 constant.

        Example::
            %lo = test.fact_view_byte_length_lo %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_byte_length_lo", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_byte_length_hi(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes a view-reference footprint byte-length upper bound as an i64 constant.

        Example::
            %hi = test.fact_view_byte_length_hi %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_byte_length_hi", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_min_alignment(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the minimum provable view byte-offset alignment as an i64 constant.

        Example::
            %align = test.fact_view_min_alignment %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_min_alignment", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_buffer_min_alignment(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the minimum provable buffer root byte alignment as an i64 constant.

        Example::
            %align = test.fact_buffer_min_alignment %buffer : buffer -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_buffer_min_alignment", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_root_min_alignment(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the minimum provable view root byte alignment as an i64 constant.

        Example::
            %align = test.fact_view_root_min_alignment %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_root_min_alignment", _operands, results=results, attributes=_attributes, regions=_regions))

    def fact_view_element_bytes(self, *, value: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Exposes the static addressed element byte count, or -1 when unknown.

        Example::
            %bytes = test.fact_view_element_bytes %view : view<4xf32, %layout> -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        return cast(ValueRef, self._b.build("test.fact_view_element_bytes", _operands, results=results, attributes=_attributes, regions=_regions))

    def region_syntax(self, *, body: Region | None = None) -> None:
        """Test op whose body uses an alternate declarative region syntax while preserving ordinary region storage.

        Example::
            test.region_syntax do {
              test.yield
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        self._b.build("test.region_syntax", _operands, attributes=_attributes, regions=_regions)

    def low_asm_region(self, *, body: Region | None = None) -> None:
        """Test op whose body uses descriptor-backed target-low assembly syntax while preserving ordinary region storage.

        Example::
            test.low_asm_region asm<test.low.core> {
              return
            }
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if body is not None:
            _regions.append(body)
        self._b.build("test.low_asm_region", _operands, attributes=_attributes, regions=_regions)

    def clause_constant(self, *, value: Any, result_types: list[Type]) -> ValueRef:
        """Test constant materialization using a named value clause.

        Example::
            %c42 = test.clause_constant value(42) : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["value"] = value
        return cast(ValueRef, self._b.build("test.clause_constant", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def clause_copy(self, *, source: ValueRef, target: ValueRef) -> None:
        """Test dynamic operand clauses that model source/target-style syntax.

        Example::
            test.clause_copy source(%src) target(%dst) : i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        _operands.append(target)
        self._b.build("test.clause_copy", _operands, attributes=_attributes, regions=_regions)

    def typed_use(self, *, values: list[ValueRef]) -> None:
        """Side-effecting sink with adjacent SSA type annotations in its format.

        Example::
            test.typed_use %a: i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(values)
        self._b.build("test.typed_use", _operands, attributes=_attributes, regions=_regions)
