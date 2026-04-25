# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.builders.
# Regenerate: python3 loom/py/loom/gen/run.py builders

from __future__ import annotations

import builtins
from typing import Any, cast

from loom.builder import IRBuilder, TiedResultSpec, ValueRef
from loom.ir import Region, Type


class VectorBuilders:
    """Typed builder methods for vector ops."""

    __test__ = False

    def __init__(self, builder: IRBuilder) -> None:
        self._b = builder

    def constant(self, *, value: Any, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Materialize a compile-time vector value whose every lane has the same scalar attribute payload. The result type supplies both the vector shape and the element type used to interpret the payload.

        Example::
            %v = vector.constant 0.0 : vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["value"] = value
        return cast(ValueRef, self._b.build("vector.constant", _operands, results=results, attributes=_attributes, regions=_regions))

    def poison(self, *, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Materialize a typed Loom poison vector. Poison represents an invalid vector value and propagates through pure vector ops until dead-code elimination removes it or a boundary diagnoses it. A zero-lane vector such as vector<0xf32> is not poison: it is an empty aggregate whose pure lane-wise computation and zero-lane memory effects should canonicalize away. Poison is introduced when IR observes something that cannot exist, such as a lane extracted from a vector proven to have zero lanes.

        Example::
            %p = vector.poison : vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        return cast(ValueRef, self._b.build("vector.poison", _operands, results=results, attributes=_attributes, regions=_regions))

    def empty(self, *, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Materialize the unique empty aggregate value for a static zero-lane vector type. Empty vectors are ordinary values, not poison, and pure zero-lane computation canonicalizes to this op.

        Example::
            %v = vector.empty : vector<0xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        return cast(ValueRef, self._b.build("vector.empty", _operands, results=results, attributes=_attributes, regions=_regions))

    def splat(self, *, scalar: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Replicate one scalar value to every lane of a vector result. The annotation after ':' is the result vector type; the scalar operand must already have the same element type, so conversions must be spelled with scalar/vector cast ops before or after the splat.

        Example::
            %vec = vector.splat %scalar : vector<16xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(scalar)
        return cast(ValueRef, self._b.build("vector.splat", _operands, results=results, attributes=_attributes, regions=_regions))

    def iota(self, *, base: ValueRef, step: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Construct a vector of lane-coordinate values. Lane order is the logical row-major order of the result shape; result lane ordinal i contains base + i * step. The result element type must be index or a non-i1 integer payload, and base/step must be scalar values with the same element type. Dynamic result extents are allowed: the result type supplies the lane count symbolically and later specialization fixes the concrete number of produced coordinates.

        Example::
            %lanes = vector.iota %c0 step %c1 : vector<16xindex>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(base)
        _operands.append(step)
        return cast(ValueRef, self._b.build("vector.iota", _operands, results=results, attributes=_attributes, regions=_regions))

    def range(self, *, lower_bound: ValueRef, upper_bound: ValueRef, step: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Construct an i1 tail mask from an explicit scalar coordinate range. For logical lane ordinal i, the lane is true when lower_bound + i * step is strictly less than upper_bound using the coordinate domain's signed ordering. The bracketed syntax mirrors scf.for ranges because the same inclusive-lower, exclusive-upper semantics are being tested; the result vector type supplies the number and shape of lanes to test.

        Example::
            %mask = vector.mask.range [%iv to %n step %c1] : index -> vector<16xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lower_bound)
        _operands.append(upper_bound)
        _operands.append(step)
        return cast(ValueRef, self._b.build("vector.mask.range", _operands, results=results, attributes=_attributes, regions=_regions))

    def broadcast(self, *, source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Broadcast a vector value to a larger-rank or same-rank vector result. Source axes align with the trailing result axes, and each static source extent must either be 1 or match the corresponding result extent.

        Example::
            %wide = vector.broadcast %v : vector<4xf32> -> vector<16x4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.broadcast", _operands, results=results, attributes=_attributes, regions=_regions))

    def from_elements(self, *, elements: list[ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Build an all-static vector from scalar element operands in logical lane order. The result vector type defines both the lane count and element type: the number of operands must equal the static element count, and every operand must have the vector element type.

        Example::
            %v = vector.from_elements %a, %b, %c, %d : vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.extend(elements)
        return cast(ValueRef, self._b.build("vector.from_elements", _operands, results=results, attributes=_attributes, regions=_regions))

    def extract(self, *, source: ValueRef, indices: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Extract a scalar or tail subvector from a vector at explicit leading indices. Supplying one index consumes the first source axis, two indices consume the first two axes, and consuming all axes produces a scalar element.

        Example::
            %x = vector.extract %v[%i] : vector<[%n]xf32> -> f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.extract", _operands, results=results, attributes=_attributes, regions=_regions))

    def insert(self, *, value: ValueRef, dest: ValueRef, indices: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Insert a scalar or tail subvector into a vector at explicit leading indices. The inserted value must match the destination tail shape remaining after the supplied indices, and the result type is the same as the destination type.

        Example::
            %r = vector.insert %x into %v[%i] : f32, vector<[%n]xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(value)
        _operands.append(dest)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.insert", _operands, results=results, attributes=_attributes, regions=_regions))

    def slice(self, *, source: ValueRef, offsets: list[int | ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Extract a rank-preserving contiguous register subvector at explicit offsets. The offset list has one entry per source axis; each result axis extent describes how many lanes are kept from that source axis.

        Example::
            %tail = vector.slice %v[%i] : vector<[%n]xf32> -> vector<4xf32>
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
        return cast(ValueRef, self._b.build("vector.slice", _operands, results=results, attributes=_attributes, regions=_regions))

    def concat(self, *, axis: int, inputs: list[ValueRef], results: list[Type | TiedResultSpec]) -> ValueRef:
        """Concatenate one or more same-rank vectors along the template axis. All non-concatenated axes must match the result shape, and when static the result axis extent must equal the sum of input extents.

        Example::
            %wide = vector.concat<0> %a, %b : vector<4xf32>, vector<4xf32> -> vector<8xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["axis"] = axis
        _operands.extend(inputs)
        return cast(ValueRef, self._b.build("vector.concat", _operands, results=results, attributes=_attributes, regions=_regions))

    def transpose(self, *, permutation: list[int], source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Permute vector register axes. The template list maps each result axis to a source axis: permutation[i] is the source axis used for result axis i, so <[1, 0]> maps vector<MxN> to vector<NxM>. This does not touch memory layout; it only reorders lanes in the register value.

        Example::
            %t = vector.transpose<[1, 0]> %v : vector<4x8xf32> -> vector<8x4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["permutation"] = permutation
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.transpose", _operands, results=results, attributes=_attributes, regions=_regions))

    def shuffle(self, *, source_lanes: list[int], source: ValueRef, result_types: list[Type]) -> ValueRef:
        """Reorder a static rank-1 vector with a static lane map. Entry i of source_lanes selects the source lane for result lane i; duplicate source lanes are allowed, but the result type is the same as the source type.

        Example::
            %rev = vector.shuffle<[3, 2, 1, 0]> %v : vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["source_lanes"] = source_lanes
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.shuffle", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def interleave(self, *, axis: int, even: ValueRef, odd: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Interleave two same-typed vectors along the template axis. Result positions with even coordinates along that axis come from the first operand, odd coordinates come from the second operand, and the result extent on that axis is doubled.

        Example::
            %r = vector.interleave<0> %lo, %hi : vector<16xi8>, vector<16xi8> -> vector<32xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["axis"] = axis
        _operands.append(even)
        _operands.append(odd)
        return cast(ValueRef, self._b.build("vector.interleave", _operands, results=results, attributes=_attributes, regions=_regions))

    def deinterleave(self, *, axis: int, source: ValueRef, results: list[Type | TiedResultSpec]) -> list[ValueRef]:
        """Split one vector along the template axis into two same-typed results. The first result receives even coordinates along that axis, the second receives odd coordinates, and each result extent on that axis is half of the source extent.

        Example::
            %lo, %hi = vector.deinterleave<0> %r : vector<32xi8> -> vector<16xi8>, vector<16xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["axis"] = axis
        _operands.append(source)
        return cast(list[ValueRef], self._b.build("vector.deinterleave", _operands, results=results, attributes=_attributes, regions=_regions))

    def lookup(self, *, table: ValueRef, indices: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Select values from a rank-1 register table using integer index lanes. Each result lane reads table[indices lane]; the result shape matches the index vector shape and the result element type matches the table element type.

        Example::
            %values = vector.table.lookup %grid[%codes] : vector<16xf16>, vector<32xi8> -> vector<32xf16>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(table)
        _operands.append(indices)
        return cast(ValueRef, self._b.build("vector.table.lookup", _operands, results=results, attributes=_attributes, regions=_regions))

    def quantize(self, *, input: ValueRef, thresholds: ValueRef, nan: str, tie: str, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Map floating-point lanes to integer ordinal code lanes using an ordered rank-1 threshold table. For each input lane, the result code is the selected quantization bin; nan and tie attributes make NaN and threshold equality behavior explicit.

        Example::
            %codes = vector.table.quantize %values, %thresholds {nan = zero, tie = lower} : vector<32xf32>, vector<15xf32> -> vector<32xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["nan"] = nan
        _attributes["tie"] = tie
        _operands.append(input)
        _operands.append(thresholds)
        return cast(ValueRef, self._b.build("vector.table.quantize", _operands, results=results, attributes=_attributes, regions=_regions))

    def transform(self, *, source: ValueRef, transform: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Apply an explicit numeric transform descriptor to vector register lanes. The transform operand is an encoding<transform> value that names the numeric mapping, such as scale/zero-point decode, whitening, or projection; verifier rules keep supported transform families and shape-changing parameters explicit. Hadamard-like families act along the last axis. `hadamard_sign` applies either an explicit per-lane sign table or deterministic seed-derived signs from the low bit of SplitMix64(seed + input lane) before the Hadamard. `sign_permute_hadamard` applies explicit signs to source lanes, gathers lanes through the explicit permutation vector, then applies the Hadamard.

        Example::
            %r = vector.transform %v, %xf : vector<128xf32>, encoding<transform> -> vector<128xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(source)
        _operands.append(transform)
        return cast(ValueRef, self._b.build("vector.transform", _operands, results=results, attributes=_attributes, regions=_regions))

    def load(self, *, view: ValueRef, indices: list[int | ValueRef], cache_scope: str | None = None, cache_temporal: str | None = None, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Load a vector footprint from a typed view at a full-rank logical origin. The index list addresses the origin in view coordinates; vector axes map onto the trailing view axes, so leading view axes select a slice and trailing axes describe the loaded footprint.

        Example::
            %v = vector.load %view[%row, %col] : view<[%m]x[%n]xf32, %layout> -> vector<4x8xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(view)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.load", _operands, results=results, attributes=_attributes, regions=_regions))

    def store(self, *, value: ValueRef, view: ValueRef, indices: list[int | ValueRef], cache_scope: str | None = None, cache_temporal: str | None = None) -> None:
        """Store a vector footprint into a typed view at a full-rank logical origin. The index list addresses the origin in view coordinates; vector axes map onto the trailing view axes, matching vector.load.

        Example::
            vector.store %v, %view[%row, %col] : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.store", _operands, attributes=_attributes, regions=_regions)

    def load_mask(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Masked vector load from a typed view. Mask lanes with true values perform the same access as vector.load, while false lanes do not access memory and instead take the corresponding passthrough lane.

        Example::
            %v = vector.load.mask %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>, vector<4x8xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(view)
        _operands.append(mask)
        _operands.append(passthrough)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.load.mask", _operands, results=results, attributes=_attributes, regions=_regions))

    def store_mask(self, *, value: ValueRef, view: ValueRef, indices: list[int | ValueRef], mask: ValueRef, cache_scope: str | None = None, cache_temporal: str | None = None) -> None:
        """Masked vector store into a typed view. True mask lanes store the corresponding value lane, and false mask lanes do not access memory and leave the destination unchanged.

        Example::
            vector.store.mask %v, %view[%row, %col], %mask : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(mask)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.store.mask", _operands, attributes=_attributes, regions=_regions)

    def expand(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Rank-1 masked expand load from consecutive view elements. Active lanes consume memory densely in increasing lane order; inactive lanes do not consume memory and take the corresponding passthrough lane.

        Example::
            %v = vector.load.expand %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xi1>, vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(view)
        _operands.append(mask)
        _operands.append(passthrough)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.load.expand", _operands, results=results, attributes=_attributes, regions=_regions))

    def compress(self, *, value: ValueRef, view: ValueRef, indices: list[int | ValueRef], mask: ValueRef, cache_scope: str | None = None, cache_temporal: str | None = None) -> None:
        """Rank-1 masked compress store to consecutive view elements. Active lanes write densely in increasing lane order; inactive lanes do not produce memory elements.

        Example::
            vector.store.compress %v, %view[%row, %col], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(mask)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.store.compress", _operands, attributes=_attributes, regions=_regions)

    def gather(
        self, *, view: ValueRef, indices: list[int | ValueRef], offsets: ValueRef, cache_scope: str | None = None, cache_temporal: str | None = None, results: list[Type | TiedResultSpec]
    ) -> ValueRef:
        """Gather a vector from per-lane signed logical offsets added to the last view axis of a full-rank view origin. Each result lane reads origin with the final coordinate adjusted by offsets[lane]; the offset vector shape matches the result shape.

        Example::
            %v = vector.gather %view[%row, %col][%offsets] : view<[%m]x[%n]xf32, %layout>, vector<4xindex> -> vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(view)
        _operands.append(offsets)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.gather", _operands, results=results, attributes=_attributes, regions=_regions))

    def scatter(self, *, value: ValueRef, view: ValueRef, indices: list[int | ValueRef], offsets: ValueRef, cache_scope: str | None = None, cache_temporal: str | None = None) -> None:
        """Non-atomic scatter of a vector to per-lane signed logical offsets added to the last view axis of a full-rank view origin. Each lane writes origin with the final coordinate adjusted by offsets[lane], and active lane addresses must be distinct because no atomic conflict resolution is implied.

        Example::
            vector.scatter %v, %view[%row, %col][%offsets] : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.scatter", _operands, attributes=_attributes, regions=_regions)

    def gather_mask(
        self,
        *,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        passthrough: ValueRef,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Masked vector gather from per-lane signed logical offsets added to the last view axis. True mask lanes read the adjusted coordinate, while false mask lanes do not access memory and take the corresponding passthrough lane.

        Example::
            %v = vector.gather.mask %view[%row, %col][%offsets], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(view)
        _operands.append(offsets)
        _operands.append(mask)
        _operands.append(passthrough)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.gather.mask", _operands, results=results, attributes=_attributes, regions=_regions))

    def scatter_mask(
        self, *, value: ValueRef, view: ValueRef, indices: list[int | ValueRef], offsets: ValueRef, mask: ValueRef, cache_scope: str | None = None, cache_temporal: str | None = None
    ) -> None:
        """Masked non-atomic scatter. True mask lanes write the full-rank origin with the last coordinate adjusted by offsets[lane], false mask lanes do not access memory, and active lane addresses must be distinct.

        Example::
            vector.scatter.mask %v, %view[%row, %col][%offsets], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _operands.append(mask)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.scatter.mask", _operands, attributes=_attributes, regions=_regions)

    def atomic_reduce(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
    ) -> None:
        """Atomic no-result scatter reduction/update into per-lane signed element offsets. Each lane atomically combines its value into origin + offsets[lane]; duplicate active addresses are valid and are serialized by the required ordering and scope attributes.

        Example::
            vector.atomic.reduce<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _attributes["ordering"] = ordering
        _attributes["scope"] = scope
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.atomic.reduce", _operands, attributes=_attributes, regions=_regions)

    def atomic_reduce_mask(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
    ) -> None:
        """Masked atomic no-result scatter reduction/update. True mask lanes perform vector.atomic.reduce, while false mask lanes do not access memory.

        Example::
            vector.atomic.reduce.mask<addf> %v, %view[%row, %col][%offsets], %mask {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _attributes["ordering"] = ordering
        _attributes["scope"] = scope
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _operands.append(mask)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        self._b.build("vector.atomic.reduce.mask", _operands, attributes=_attributes, regions=_regions)

    def rmw(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Atomic read-modify-write at per-lane signed element offsets. Each lane atomically combines its value with origin + offsets[lane] and the result lane is the old memory value observed by that atomic operation.

        Example::
            %old = vector.atomic.rmw<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _attributes["ordering"] = ordering
        _attributes["scope"] = scope
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.atomic.rmw", _operands, results=results, attributes=_attributes, regions=_regions))

    def atomic_rmw_mask(
        self,
        *,
        kind: str,
        value: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        mask: ValueRef,
        passthrough: ValueRef,
        ordering: str,
        scope: str,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Masked atomic read-modify-write. True mask lanes perform vector.atomic.rmw, while false mask lanes do not access memory and take the corresponding passthrough lane in the result.

        Example::
            %old = vector.atomic.rmw.mask<addf> %v, %view[%row, %col][%offsets], %mask, %passthrough {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _attributes["ordering"] = ordering
        _attributes["scope"] = scope
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(value)
        _operands.append(view)
        _operands.append(offsets)
        _operands.append(mask)
        _operands.append(passthrough)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.atomic.rmw.mask", _operands, results=results, attributes=_attributes, regions=_regions))

    def cmpxchg(
        self,
        *,
        expected: ValueRef,
        replacement: ValueRef,
        view: ValueRef,
        indices: list[int | ValueRef],
        offsets: ValueRef,
        success_ordering: str,
        failure_ordering: str,
        scope: str,
        cache_scope: str | None = None,
        cache_temporal: str | None = None,
        results: list[Type | TiedResultSpec],
    ) -> ValueRef:
        """Atomic compare-exchange at per-lane signed element offsets. Each lane compares origin + offsets[lane] with expected[lane], writes replacement[lane] on success, and returns the old memory value. Success lanes are derived by comparing old == expected.

        Example::
            %old = vector.atomic.cmpxchg %expected, %replacement, %view[%row, %col][%offsets] {success_ordering = acq_rel, failure_ordering = acquire, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex> -> vector<4xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["success_ordering"] = success_ordering
        _attributes["failure_ordering"] = failure_ordering
        _attributes["scope"] = scope
        if cache_scope is not None:
            _attributes["cache_scope"] = cache_scope
        if cache_temporal is not None:
            _attributes["cache_temporal"] = cache_temporal
        _operands.append(expected)
        _operands.append(replacement)
        _operands.append(view)
        _operands.append(offsets)
        _sentinel = -(2**63)
        _static = []
        for _idx in indices:
            if isinstance(_idx, ValueRef):
                _static.append(_sentinel)
                _operands.append(_idx)
            else:
                _static.append(_idx)
        _attributes["static_indices"] = _static
        return cast(ValueRef, self._b.build("vector.atomic.cmpxchg", _operands, results=results, attributes=_attributes, regions=_regions))

    def select(self, *, condition: ValueRef, true_value: ValueRef, false_value: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise select from two same-typed vector values using an i1 mask vector. True condition lanes choose true_value; false lanes choose false_value.

        Example::
            %r = vector.select %mask, %a, %b : vector<16xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(condition)
        _operands.append(true_value)
        _operands.append(false_value)
        return cast(ValueRef, self._b.build("vector.select", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def cmpi(self, *, predicate: str, lhs: ValueRef, rhs: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Lanewise integer comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpi predicate names and applies independently to each lane.

        Example::
            %m = vector.cmpi slt, %lhs, %rhs : vector<16xi32> -> vector<16xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["predicate"] = predicate
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.cmpi", _operands, results=results, attributes=_attributes, regions=_regions))

    def cmpf(self, *, predicate: str, lhs: ValueRef, rhs: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Lanewise floating-point comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpf ordered/unordered predicate names and applies independently to each lane.

        Example::
            %m = vector.cmpf olt, %lhs, %rhs : vector<16xf32> -> vector<16xi1>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["predicate"] = predicate
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.cmpf", _operands, results=results, attributes=_attributes, regions=_regions))

    def addf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point addition of same-typed vector operands. Optional assumptions flags constrain lane value domains; they do not change the required element type or shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.addf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def subf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point subtraction of same-typed vector operands. Optional assumptions flags constrain lane value domains; they do not change the required element type or shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.subf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def mulf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point multiplication of same-typed vector operands. Optional assumptions flags constrain lane value domains; they do not imply fusion with neighboring operations."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.mulf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def divf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point division of same-typed vector operands. Optional assumptions flags constrain lane value domains; they do not change division-by-zero or NaN semantics."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.divf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def remf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point remainder with C fmod semantics over same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.remf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def negf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point negation of a same-typed vector operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.negf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def absf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point absolute value of a same-typed vector operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.absf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def minimumf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise IEEE 754 floating-point minimum of same-typed vector operands; NaN lanes propagate."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.minimumf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def maximumf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise IEEE 754 floating-point maximum of same-typed vector operands; NaN lanes propagate."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.maximumf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def minnumf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise C99 fmin-style floating-point minimum of same-typed vector operands; NaN lanes select the non-NaN operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.minnumf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def maxnumf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise C99 fmax-style floating-point maximum of same-typed vector operands; NaN lanes select the non-NaN operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.maxnumf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def copysignf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise copy sign of rhs lanes onto lhs lane magnitudes."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.copysignf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def fmaf(self, *, assumptions: str | None = None, a: ValueRef, b: ValueRef, c: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise fused multiply-add of same-typed floating-point vectors. Each result lane computes a*b + c with one final rounding; use separate vector.mulf/vector.addf when unfused rounding is required.

        Example::
            %r = vector.fmaf %a, %b, %c : vector<16xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(a)
        _operands.append(b)
        _operands.append(c)
        return cast(ValueRef, self._b.build("vector.fmaf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def addi(self, *, overflow: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer addition of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if overflow is not None:
            _attributes["overflow"] = overflow
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.addi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def subi(self, *, overflow: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer subtraction of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if overflow is not None:
            _attributes["overflow"] = overflow
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.subi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def muli(self, *, overflow: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer multiplication of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if overflow is not None:
            _attributes["overflow"] = overflow
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.muli", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def divsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer division of same-typed vector operands; each lane rounds toward zero."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.divsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def divui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer division of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.divui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def remsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer remainder of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.remsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def remui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer remainder of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.remui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ceildivsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer division rounding toward positive infinity."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.ceildivsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ceildivui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer division rounding toward positive infinity."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.ceildivui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def floordivsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer division rounding toward negative infinity."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.floordivsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def negi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer negation of a same-typed vector operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.negi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def absi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer absolute value of a same-typed vector operand."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.absi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def minsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer minimum of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.minsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def maxsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer maximum of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.maxsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def minui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer minimum of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.minui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def maxui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer maximum of same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.maxui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def fmai(self, *, overflow: str | None = None, a: ValueRef, b: ValueRef, c: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise fused integer multiply-add a*b + c over same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane.

        Example::
            %r = vector.fmai %a, %b, %c : vector<16xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if overflow is not None:
            _attributes["overflow"] = overflow
        _operands.append(a)
        _operands.append(b)
        _operands.append(c)
        return cast(ValueRef, self._b.build("vector.fmai", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def andi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise bitwise AND of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.andi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ori(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise bitwise OR of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.ori", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def xori(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise bitwise XOR of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.xori", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def shli(self, *, overflow: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise left shift of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if overflow is not None:
            _attributes["overflow"] = overflow
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.shli", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def shrsi(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise arithmetic right shift of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.shrsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def shrui(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise logical right shift of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.shrui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def rotli(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise left rotate of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.rotli", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def rotri(self, *, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise right rotate of same-typed integer vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.rotri", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ctlzi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise count leading zeros over integer lanes."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.ctlzi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def cttzi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise count trailing zeros over integer lanes."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.cttzi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ctpopi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise population count over integer lanes. Each result lane is the number of set bits in the corresponding input lane and has the same integer element type as the input."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.ctpopi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def expf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise natural exponential e^x."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.expf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def exp2f(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise base-2 exponential 2^x."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.exp2f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def expm1f(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise exp(x)-1, preserving the scalar operation's near-zero numerical semantics."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.expm1f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def logf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise natural logarithm ln(x)."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.logf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def log2f(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise base-2 logarithm."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.log2f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def log10f(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise base-10 logarithm."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.log10f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def log1pf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise log(1+x), preserving the scalar operation's near-zero numerical semantics."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.log1pf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def powf(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point power lhs^rhs over same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.powf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def sqrtf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point square root. Optional assumptions flags constrain lane value domains for optimization and lowering."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.sqrtf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def rsqrtf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise reciprocal square root 1/sqrt(x)."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.rsqrtf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def cbrtf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise cube root."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.cbrtf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def sinf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise sine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.sinf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def cosf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise cosine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.cosf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def tanf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise tangent."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.tanf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def asinf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise arcsine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.asinf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def acosf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise arccosine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.acosf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def atanf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise arctangent."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.atanf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def atan2f(self, *, assumptions: str | None = None, lhs: ValueRef, rhs: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise two-argument arctangent atan2(lhs, rhs) over same-typed vector operands."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(lhs)
        _operands.append(rhs)
        return cast(ValueRef, self._b.build("vector.atan2f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def sinhf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise hyperbolic sine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.sinhf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def coshf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise hyperbolic cosine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.coshf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def tanhf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise hyperbolic tangent."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.tanhf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def asinhf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise inverse hyperbolic sine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.asinhf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def acoshf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise inverse hyperbolic cosine."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.acoshf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def atanhf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise inverse hyperbolic tangent."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.atanhf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def erff(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise error function, used by GeLU-style activations."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.erff", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def erfcf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise complementary error function 1-erf(x)."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.erfcf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def ceilf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise round toward positive infinity."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.ceilf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def floorf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise round toward negative infinity."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.floorf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def roundf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise round to nearest, ties away from zero."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.roundf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def roundevenf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise round to nearest, ties to even."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.roundevenf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def truncf(self, *, assumptions: str | None = None, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise round toward zero."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        if assumptions is not None:
            _attributes["assumptions"] = assumptions
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.truncf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def isnanf(self, *, input: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Lanewise floating-point NaN test producing an i1 mask vector."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.isnanf", _operands, results=results, attributes=_attributes, regions=_regions))

    def isinff(self, *, input: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Lanewise floating-point infinity test producing an i1 mask vector."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.isinff", _operands, results=results, attributes=_attributes, regions=_regions))

    def isfinitef(self, *, input: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Lanewise floating-point finite test producing an i1 mask vector."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.isfinitef", _operands, results=results, attributes=_attributes, regions=_regions))

    def signf(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point sign, returning -1.0, 0.0, or 1.0 per lane."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.signf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def signi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer sign, returning -1, 0, or 1 per lane."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.signi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def extf(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point precision extension. Source and result shapes match exactly; only the floating-point element type widens."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.extf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def fptrunc(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point precision truncation. Source and result shapes match exactly; only the floating-point element type narrows."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.fptrunc", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def extsi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer extension. Source and result shapes match exactly, and each source lane is sign-extended to the result element width."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.extsi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def extui(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer extension. Source and result shapes match exactly, and each source lane is zero-extended to the result element width."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.extui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def trunci(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise integer truncation. Source and result shapes match exactly, and each lane keeps the low bits required by the result element width."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.trunci", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def sitofp(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise signed integer to floating-point conversion with unchanged shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.sitofp", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def uitofp(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise unsigned integer to floating-point conversion with unchanged shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.uitofp", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def fptosi(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point to signed integer conversion with unchanged shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.fptosi", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def fptoui(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Lanewise floating-point to unsigned integer conversion with unchanged shape."""
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.fptoui", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def bitcast(self, *, input: ValueRef, result_types: list[Type]) -> ValueRef:
        """Bitwise reinterpretation between vector register types with the same total bit count. No numeric conversion is performed; only the lane shape and element interpretation change.

        Example::
            %r = vector.bitcast %input : vector<16xf32> to vector<16xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(input)
        return cast(ValueRef, self._b.build("vector.bitcast", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def extractu(self, *, source: ValueRef, offset: int, width: int, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Extract one fixed bitfield from each integer source lane and zero-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width.

        Example::
            %lo = vector.bitfield.extractu %bytes {offset = 0, width = 4} : vector<16xi8> -> vector<16xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["offset"] = offset
        _attributes["width"] = width
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.bitfield.extractu", _operands, results=results, attributes=_attributes, regions=_regions))

    def extracts(self, *, source: ValueRef, offset: int, width: int, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Extract one fixed bitfield from each integer source lane and sign-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width.

        Example::
            %signed = vector.bitfield.extracts %bytes {offset = 4, width = 4} : vector<16xi8> -> vector<16xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["offset"] = offset
        _attributes["width"] = width
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.bitfield.extracts", _operands, results=results, attributes=_attributes, regions=_regions))

    def bitfield_insert(self, *, field: ValueRef, base: ValueRef, offset: int, width: int, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Insert the low bits of each integer field lane into a fixed bitfield of the corresponding integer base lane. Bits outside the target field are preserved from the base lane.

        Example::
            %packed = vector.bitfield.insert %lo into %zero {offset = 0, width = 4} : vector<16xi32>, vector<16xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["offset"] = offset
        _attributes["width"] = width
        _operands.append(field)
        _operands.append(base)
        return cast(ValueRef, self._b.build("vector.bitfield.insert", _operands, results=results, attributes=_attributes, regions=_regions))

    def bitpack(self, *, width: int, source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Pack the low bits of each integer source lane into a contiguous little-endian bitstream stored in integer result lanes. Source lanes are consumed in logical lane order and width gives the number of bits taken from each source lane.

        Example::
            %packed = vector.bitpack<4> %codes : vector<32xi8> -> vector<16xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["width"] = width
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.bitpack", _operands, results=results, attributes=_attributes, regions=_regions))

    def bitunpacku(self, *, width: int, source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Unpack unsigned fixed-width fields from a contiguous little-endian integer bitstream into zero-extended integer result lanes. Result lanes are produced in logical lane order.

        Example::
            %codes = vector.bitunpacku<4> %packed : vector<16xi8> -> vector<32xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["width"] = width
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.bitunpacku", _operands, results=results, attributes=_attributes, regions=_regions))

    def bitunpacks(self, *, width: int, source: ValueRef, results: list[Type | TiedResultSpec]) -> ValueRef:
        """Unpack signed fixed-width fields from a contiguous little-endian integer bitstream into sign-extended integer result lanes. Result lanes are produced in logical lane order.

        Example::
            %deltas = vector.bitunpacks<3> %packed : vector<12xi8> -> vector<32xi8>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["width"] = width
        _operands.append(source)
        return cast(ValueRef, self._b.build("vector.bitunpacks", _operands, results=results, attributes=_attributes, regions=_regions))

    def dotf(self, *, lhs: ValueRef, rhs: ValueRef, init: ValueRef, result_types: list[Type]) -> ValueRef:
        """Compute a same-element floating-point dot product with an explicit scalar accumulator. Semantics are equivalent to accumulating scalar.fmaf(lhs_lane, rhs_lane, acc) over lanes in logical lane order; use vector.mulf followed by vector.reduce<addf> when separately rounded products and additions are required. The source vectors must have the same shape and element type, and the init/result scalar type matches that element type. Zero-lane inputs return init.

        Example::
            %r = vector.dotf %lhs, %rhs, %acc : vector<16xf32>, vector<16xf32>, f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        _operands.append(init)
        return cast(ValueRef, self._b.build("vector.dotf", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def dot2f(self, *, lhs: ValueRef, rhs: ValueRef, acc: ValueRef, result_types: list[Type]) -> ValueRef:
        """Group adjacent two-lane f16 or bf16 products along the last axis and add each two-product fused sum into an f32 accumulator lane. Semantics are equivalent to extending each source lane to f32, then accumulating scalar.fmaf(lhs0_f32, rhs0_f32, acc) followed by scalar.fmaf(lhs1_f32, rhs1_f32, partial) for each result lane. This models AMDGPU fdot2-style widened register dots without making f16 dot accumulation implicit in vector.dotf.

        Example::
            %r = vector.dot2f %lhs, %rhs, %acc : vector<16xf16>, vector<16xf16>, vector<8xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _operands.append(lhs)
        _operands.append(rhs)
        _operands.append(acc)
        return cast(ValueRef, self._b.build("vector.dot2f", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def dot4i(self, *, kind: str, lhs: ValueRef, rhs: ValueRef, acc: ValueRef, result_types: list[Type]) -> ValueRef:
        """Group adjacent four-lane i8 products along the last axis and add each four-product sum into an i32 accumulator lane. The signedness template chooses how lhs and rhs i8 lanes are interpreted, matching dp4a/VNNI-style hardware operations.

        Example::
            %r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<16xi8>, vector<16xi8>, vector<4xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _operands.append(lhs)
        _operands.append(rhs)
        _operands.append(acc)
        return cast(ValueRef, self._b.build("vector.dot4i", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def dot8i4(self, *, kind: str, lhs: ValueRef, rhs: ValueRef, acc: ValueRef, result_types: list[Type]) -> ValueRef:
        """Treat each i32 source lane as a little-endian pack of eight 4-bit integer fields, multiply corresponding packed fields using the signedness template, and add the eight-product sum into the matching i32 accumulator lane. This is a packed-storage register dot: use vector.bitpack<4> when starting from unpacked byte lanes. The semantics match AMDGPU sdot8/udot8/sudot8 with clamp disabled.

        Example::
            %r = vector.dot8i4<s4s4> %lhs, %rhs, %acc : vector<4xi32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _operands.append(lhs)
        _operands.append(rhs)
        _operands.append(acc)
        return cast(ValueRef, self._b.build("vector.dot8i4", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def dot4f8(self, *, kind: str, lhs: ValueRef, rhs: ValueRef, acc: ValueRef, result_types: list[Type]) -> ValueRef:
        """Treat each i32 source lane as a little-endian pack of four 8-bit floating-point fields, decode fields according to the fp8/bf8 template, and add the four-product fused sum into the matching f32 accumulator lane. The fp8 spelling names the E4M3 primitive float format and bf8 names the E5M2 primitive float format. This is a packed-storage register dot matching AMDGPU dot4.f32.fp8/bf8 families without requiring unpacked f8 vector source lanes.

        Example::
            %r = vector.dot4f8<fp8bf8> %lhs, %rhs, %acc : vector<4xi32>, vector<4xf32>
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _operands.append(lhs)
        _operands.append(rhs)
        _operands.append(acc)
        return cast(ValueRef, self._b.build("vector.dot4f8", _operands, results=result_types, attributes=_attributes, regions=_regions))

    def reduce(self, *, kind: str, input: ValueRef, init: ValueRef, result_types: list[Type]) -> ValueRef:
        """Reduce all lanes of a vector into a scalar accumulator/result using the template combining kind. The init operand and result have the same scalar type, and the combining kind must be valid for the input element type.

        Example::
            %sum = vector.reduce<addf> %v, %zero : vector<16xf32>, f32
        """
        _operands: list[ValueRef | int] = []
        _attributes: builtins.dict[str, Any] = {}
        _regions: list[Region] = []
        _attributes["kind"] = kind
        _operands.append(input)
        _operands.append(init)
        return cast(ValueRef, self._b.build("vector.reduce", _operands, results=result_types, attributes=_attributes, regions=_regions))
