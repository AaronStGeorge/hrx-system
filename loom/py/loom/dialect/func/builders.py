# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from collections.abc import Mapping
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Predicate, Region, Type


class FuncBuilders:
    """Typed builder methods for func ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def def_(
        self,
        *,
        visibility: str | None = None,
        cc: str | None = None,
        purity: str | None = None,
        target: str | None = None,
        abi: str | None = None,
        abi_attrs: Mapping[str, Any] | None = None,
        export_symbol: str | None = None,
        export_attrs: Mapping[str, Any] | None = None,
        callee: str,
        args: list[ValueRef] | None = None,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] | None = None,
        body: Region | None = None,
    ) -> list[ValueRef]:
        """Function definition. Callable by name via func.call.

        Example::
            func.def @negate(%input: f32) -> (f32) {
              func.return %input : f32
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
        if purity is not None:
            _attributes["purity"] = purity
        if target is not None:
            _attributes["target"] = target
        if abi is not None:
            _attributes["abi"] = abi
        if abi_attrs is not None:
            _attributes["abi_attrs"] = abi_attrs
        if export_symbol is not None:
            _attributes["export_symbol"] = export_symbol
        if export_attrs is not None:
            _attributes["export_attrs"] = export_attrs
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        if predicates:
            _attributes["predicates"] = predicates
        if body is not None:
            _regions.append(body)
        return cast(list[ValueRef], self._b.build("func.def", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def decl(
        self,
        *,
        visibility: str | None = None,
        import_module: str | None = None,
        import_symbol: str | None = None,
        cc: str | None = None,
        purity: str | None = None,
        target: str | None = None,
        abi: str | None = None,
        abi_attrs: Mapping[str, Any] | None = None,
        export_symbol: str | None = None,
        export_attrs: Mapping[str, Any] | None = None,
        callee: str,
        args: list[ValueRef] | None = None,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] | None = None,
    ) -> list[ValueRef]:
        """External function declaration. Callable by name via func.call.

        Example::
            func.decl @extern_matmul(%a: tensor<[%M]xf32>, %b: tensor<[%K]xf32>) -> (tensor<[%M]xf32>)
        """
        _operands: list[ValueRef | int] = []
        _func_args: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if visibility is not None:
            _attributes["visibility"] = visibility
        if import_module is not None:
            _attributes["import_module"] = import_module
        if import_symbol is not None:
            _attributes["import_symbol"] = import_symbol
        if cc is not None:
            _attributes["cc"] = cc
        if purity is not None:
            _attributes["purity"] = purity
        if target is not None:
            _attributes["target"] = target
        if abi is not None:
            _attributes["abi"] = abi
        if abi_attrs is not None:
            _attributes["abi_attrs"] = abi_attrs
        if export_symbol is not None:
            _attributes["export_symbol"] = export_symbol
        if export_attrs is not None:
            _attributes["export_attrs"] = export_attrs
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        if predicates:
            _attributes["predicates"] = predicates
        return cast(list[ValueRef], self._b.build("func.decl", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def template(
        self,
        *,
        implements: str,
        visibility: str | None = None,
        cc: str | None = None,
        purity: str | None = None,
        priority: int | None = None,
        callee: str,
        args: list[ValueRef] | None = None,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] | None = None,
        body: Region | None = None,
    ) -> list[ValueRef]:
        """Constraint-matched visible implementation of an abstract op.

        Example::
            func.template<tile.contract> device @vnni_q8(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)] {
              func.return %x : tensor<[%K]xf32>
            }
        """
        _operands: list[ValueRef | int] = []
        _func_args: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["implements"] = implements
        if visibility is not None:
            _attributes["visibility"] = visibility
        if cc is not None:
            _attributes["cc"] = cc
        if purity is not None:
            _attributes["purity"] = purity
        if priority is not None:
            _attributes["priority"] = priority
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        if predicates:
            _attributes["predicates"] = predicates
        if body is not None:
            _regions.append(body)
        return cast(list[ValueRef], self._b.build("func.template", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def ukernel(
        self,
        *,
        implements: str,
        visibility: str | None = None,
        cc: str | None = None,
        purity: str | None = None,
        priority: int | None = None,
        callee: str,
        args: list[ValueRef] | None = None,
        results: list[Type | TiedResultSpec],
        predicates: list[Predicate] | None = None,
    ) -> list[ValueRef]:
        """Constraint-matched opaque implementation of an abstract op.

        Example::
            func.ukernel<tile.contract> device @vnni_q8_asm(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)]
        """
        _operands: list[ValueRef | int] = []
        _func_args: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["implements"] = implements
        if visibility is not None:
            _attributes["visibility"] = visibility
        if cc is not None:
            _attributes["cc"] = cc
        if purity is not None:
            _attributes["purity"] = purity
        if priority is not None:
            _attributes["priority"] = priority
        _attributes["callee"] = callee
        if args is not None:
            _func_args.extend(args)
        if predicates:
            _attributes["predicates"] = predicates
        return cast(list[ValueRef], self._b.build("func.ukernel", _operands, func_args=_func_args, results=results, attributes=_attributes, regions=_regions))

    def call(self, *, purity: str | None = None, callee: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Runtime function call. Target must be func.def or func.decl.

        Example::
            %r = func.call @add(%a, %b) : (f32, f32) -> (f32)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if purity is not None:
            _attributes["purity"] = purity
        _attributes["callee"] = callee
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("func.call", _operands, results=results, attributes=_attributes, regions=_regions))

    def apply(self, *, purity: str | None = None, callee: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Compile-time template expansion. Target must be func.template.

        Example::
            %r = func.apply @vnni_q8_matvec(%w, %x) : (tensor<16x32xi8>, tensor<32xf32>) -> (tensor<16xf32>)
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if purity is not None:
            _attributes["purity"] = purity
        _attributes["callee"] = callee
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("func.apply", _operands, results=results, attributes=_attributes, regions=_regions))

    def return_(self, *, operands: list[ValueRef]) -> None:
        """Return values from function body. Types must match enclosing function's result types.

        Example::
            func.return
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(operands)
        self._b.build("func.return", _operands, attributes=_attributes, regions=_regions)
