# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Region, Type


class LlvmIrBuilders:
    """Typed builder methods for llvmir ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def inline_asm(self, *, flags: str | None = None, asm_template: str, constraints: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Structured LLVM inline assembly call. The asm template and constraint strings use LLVM inline asm syntax; operands/results remain ordinary typed Loom SSA values.

        Example::
            %sum = llvmir.inline_asm<sideeffect> "addl $2, $0", "=r,r,r"(%lhs, %rhs) : (i32, i32) -> i32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if flags is not None:
            _attributes["flags"] = flags
        _attributes["asm_template"] = asm_template
        _attributes["constraints"] = constraints
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("llvmir.inline_asm", _operands, results=results, attributes=_attributes, regions=_regions))

    def intrinsic(self, *, kind: str, operands: list[ValueRef], results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Structured call to a supported LLVM intrinsic. The intrinsic spelling is a string so target-family providers can recognize their own intrinsics without extending a central enum.

        Example::
            %ticks = llvmir.intrinsic<llvm.x86.rdtsc> () : () -> i64
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _operands.extend(operands)
        return cast(list[ValueRef], self._b.build("llvmir.intrinsic", _operands, results=results, attributes=_attributes, regions=_regions))
