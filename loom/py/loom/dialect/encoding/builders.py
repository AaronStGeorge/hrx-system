# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Region, Type


class EncodingBuilders:
    """Typed builder methods for encoding ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def layout_dense(self, *, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Construct a dense row-major address layout. The consuming view type provides the rank and logical extents.

        Example::
            %layout = encoding.layout.dense : encoding<layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        return cast(ValueRef, self._b.build("encoding.layout.dense", _operands, results=results, attributes=_attributes, regions=_regions))

    def layout_strided(self, *, strides: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Construct an address layout from per-dimension element strides. Static and dynamic stride values are interleaved in one bracket list.

        Example::
            %layout = encoding.layout.strided [%row_stride, 1] : encoding<layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _sentinel = -(2**63)
        _static = []
        for _idx in strides:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_strides"] = _static
        return cast(ValueRef, self._b.build("encoding.layout.strided", _operands, results=results, attributes=_attributes, regions=_regions))

    def define(self, *, spec: Any, params: dict[str, ValueRef], result_types: list[Type]) -> ValueRef:
        """Create an encoding value from a static encoding specification.

        Example::
            %enc = encoding.define #q8_0<block=32> : encoding<schema>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["spec"] = spec
        if params:
            _operand_dict_names: builtins.dict[str, int] = {}
            for _name in sorted(params):
                _operand_dict_names[_name] = len(_operand_dict_names)
                _operands.append(params[_name])
            _attributes["param_names"] = _operand_dict_names
        return cast(ValueRef, self._b.build("encoding.define", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def isa(self, *, enc: ValueRef, category: str, result_types: list[Type]) -> ValueRef:
        """Test if an encoding belongs to a category.

        Example::
            %is_quantized = encoding.isa %enc, "quantized" : i1
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["category"] = category
        _operands.append(enc)
        return cast(ValueRef, self._b.build("encoding.isa", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def layout_assume_dense(self, *, layout: ValueRef, result_types: list[Type]) -> ValueRef:
        """Refine an existing address-layout encoding value with the fact that it is dense row-major. The result is the same encoding value in SSA form with stronger local facts.

        Example::
            %dense = encoding.layout.assume.dense %layout : encoding<layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(layout)
        return cast(ValueRef, self._b.build("encoding.layout.assume.dense", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def layout_assume_strided(self, *, layout: ValueRef, rank: int, result_types: list[Type]) -> ValueRef:
        """Refine an existing address-layout encoding value with the fact that it is strided and has the given rank. Per-axis stride values remain unknown unless a concrete encoding.layout.strided value is available.

        Example::
            %strided = encoding.layout.assume.strided %layout {rank = 2} : encoding<layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["rank"] = rank
        _operands.append(layout)
        return cast(ValueRef, self._b.build("encoding.layout.assume.strided", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def spec(self, *, enc: ValueRef, spec: Any, result_types: list[Type]) -> ValueRef:
        """Refine an existing encoding value with an exact static encoding specification. Dynamic values remain ordinary SSA operands elsewhere; this op only states the selected static family and static parameters.

        Example::
            %schema2 = encoding.assume.spec %schema, #ggml_q4_0<block_elems=32, storage_bytes=18> : encoding<schema>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["spec"] = spec
        _operands.append(enc)
        return cast(ValueRef, self._b.build("encoding.assume.spec", _operands, results=result_types, attributes=_attributes, regions=_regions))
