# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom in-memory IR representation.

Structurally compatible with the C IR (loom/src/loom/ir/). Python
uses dataclasses and lists instead of arenas and packed structs, but
the concepts, fields, and ID-based references are identical.

Hierarchy: Context -> Module -> Symbol -> Function -> Block -> Operation -> Value

All cross-references use integer IDs (indices into module tables),
not object references. This ensures compatibility with serialization
and the C implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import Any

__all__ = [
    # Scalar types.
    "ScalarTypeKind",
    "ScalarType",
    "scalar_type_name",
    "parse_scalar_type_kind",
    "INDEX",
    "OFFSET",
    "I1",
    "I8",
    "I16",
    "I32",
    "I64",
    "F16",
    "BF16",
    "F32",
    "F64",
    # Dimensions.
    "StaticDim",
    "DynamicDim",
    "Dim",
    # Type kinds and types.
    "TypeKind",
    "ShapedType",
    "GroupScope",
    "GROUP_SCOPE_BY_NAME",
    "GroupType",
    "PoolType",
    "FunctionType",
    "NoneType",
    "DialectType",
    "EncodingType",
    "ENCODING_TYPE",
    "NONE_TYPE",
    "Type",
    "binding_element_type",
    # Locations.
    "LocationKind",
    "FileLocation",
    "FusedLocation",
    "OpaqueLocation",
    "LocationData",
    "LOCATION_UNKNOWN",
    "LOCATION_FLAG_SYNTHETIC",
    # Encoding instances.
    "DynamicEncoding",
    "EncodingInstance",
    # Predicates.
    "PredicateArg",
    "Predicate",
    "PREDICATE_KINDS",
    "evaluate_predicate",
    "evaluate_predicates",
    # Tied results.
    "TiedResult",
    # Use-def.
    "Use",
    "VALUE_FLAG_BLOCK_ARG",
    "VALUE_FLAG_CONSUMED",
    # IR graph.
    "Value",
    "Operation",
    "Block",
    "Region",
    # Symbols.
    "SymbolKind",
    "Symbol",
    "SymbolRef",
    "SYMBOL_FLAG_IMPORT",
    "SYMBOL_FLAG_PUBLIC",
    # Tables.
    "StringTable",
    "TypeTable",
    "LocationTable",
    # Module and context.
    "Module",
    "Context",
]


# ============================================================================
# Scalar types
# ============================================================================


@unique
class ScalarTypeKind(IntEnum):
    """Scalar element type kind.

    Values match loom_scalar_type_e in ir/types.h. These are internal
    compiler values, NOT bytecode-stable.

    Ordered: address types, integers by width, floats by width.
    """

    INDEX = 0
    OFFSET = 1
    I1 = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    F8E4M3 = 7
    F8E5M2 = 8
    F16 = 9
    BF16 = 10
    F32 = 11
    F64 = 12


# Scalar type name -> enum mapping.
_SCALAR_NAMES: dict[str, ScalarTypeKind] = {
    "index": ScalarTypeKind.INDEX,
    "offset": ScalarTypeKind.OFFSET,
    "i1": ScalarTypeKind.I1,
    "i8": ScalarTypeKind.I8,
    "i16": ScalarTypeKind.I16,
    "i32": ScalarTypeKind.I32,
    "i64": ScalarTypeKind.I64,
    "f8E4M3": ScalarTypeKind.F8E4M3,
    "f8E5M2": ScalarTypeKind.F8E5M2,
    "f16": ScalarTypeKind.F16,
    "bf16": ScalarTypeKind.BF16,
    "f32": ScalarTypeKind.F32,
    "f64": ScalarTypeKind.F64,
}

# Reverse mapping.
_SCALAR_KIND_NAMES: dict[ScalarTypeKind, str] = {v: k for k, v in _SCALAR_NAMES.items()}


def scalar_type_name(kind: ScalarTypeKind) -> str:
    """Return the textual name for a scalar type kind."""
    return _SCALAR_KIND_NAMES[kind]


def parse_scalar_type_kind(name: str) -> ScalarTypeKind | None:
    """Parse a scalar type name, returning None if not recognized."""
    return _SCALAR_NAMES.get(name)


# ============================================================================
# Dimensions
# ============================================================================


@dataclass(frozen=True, slots=True)
class StaticDim:
    """A static (compile-time known) dimension size."""

    size: int

    def __repr__(self) -> str:
        return str(self.size)


@dataclass(frozen=True, slots=True)
class DynamicDim:
    """A dynamic dimension. The actual size is bound at the use site,
    not in the type. Types are structural — DynamicDim carries no
    value reference."""

    def __repr__(self) -> str:
        return "?"


# A dimension is either static or dynamic.
type Dim = StaticDim | DynamicDim


# ============================================================================
# Type kinds
# ============================================================================


@unique
class TypeKind(IntEnum):
    """Type kind tag. Matches loom_type_kind_e in ir/types.h."""

    NONE = 0
    SCALAR = 1
    TILE = 2
    TENSOR = 3
    GROUP = 4
    FUNCTION = 5
    DIALECT = 6
    ENCODING = 7
    POOL = 8
    PLACEHOLDER = 9


# ============================================================================
# Types — frozen, internable, structurally compared
# ============================================================================


@dataclass(frozen=True, slots=True)
class NoneType:
    """Absence of a type (for ops with no results)."""

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.NONE

    def __repr__(self) -> str:
        return "none"


# Singleton for no-type.
NONE_TYPE = NoneType()


@dataclass(frozen=True, slots=True)
class ScalarType:
    """A scalar type: f16, f32, i8, index, etc."""

    kind: ScalarTypeKind

    def __repr__(self) -> str:
        return scalar_type_name(self.kind)

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.SCALAR

    @property
    def bitwidth(self) -> int:
        widths = {
            ScalarTypeKind.INDEX: 64,
            ScalarTypeKind.OFFSET: 64,
            ScalarTypeKind.I1: 1,
            ScalarTypeKind.I8: 8,
            ScalarTypeKind.I16: 16,
            ScalarTypeKind.I32: 32,
            ScalarTypeKind.I64: 64,
            ScalarTypeKind.F8E4M3: 8,
            ScalarTypeKind.F8E5M2: 8,
            ScalarTypeKind.F16: 16,
            ScalarTypeKind.BF16: 16,
            ScalarTypeKind.F32: 32,
            ScalarTypeKind.F64: 64,
        }
        return widths[self.kind]


# Common scalar type singletons.
INDEX = ScalarType(ScalarTypeKind.INDEX)
OFFSET = ScalarType(ScalarTypeKind.OFFSET)
I1 = ScalarType(ScalarTypeKind.I1)
I8 = ScalarType(ScalarTypeKind.I8)
I16 = ScalarType(ScalarTypeKind.I16)
I32 = ScalarType(ScalarTypeKind.I32)
I64 = ScalarType(ScalarTypeKind.I64)
F16 = ScalarType(ScalarTypeKind.F16)
BF16 = ScalarType(ScalarTypeKind.BF16)
F32 = ScalarType(ScalarTypeKind.F32)
F64 = ScalarType(ScalarTypeKind.F64)


@dataclass(frozen=True, slots=True)
class ShapedType:
    """A tile or tensor type with shape, element type, and optional encoding.

    Types are structural: dynamic dims are DynamicDim() flags, not
    references to specific SSA values. The binding of dynamic dims to
    values happens on the Value that carries this type.

    encoding is None for dense (no encoding), an EncodingInstance for
    a statically-known encoding, or DynamicEncoding for an encoding
    bound via SSA value. Since all variants are frozen and hashable,
    ShapedType remains internable.
    """

    type_kind: TypeKind  # TILE or TENSOR
    element_type: ScalarType
    dims: tuple[Dim, ...]
    encoding: EncodingInstance | DynamicEncoding | None = None

    def __post_init__(self) -> None:
        if self.type_kind not in (TypeKind.TILE, TypeKind.TENSOR):
            raise ValueError(
                f"ShapedType type_kind must be TILE or TENSOR, got {self.type_kind}"
            )

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def has_encoding(self) -> bool:
        return self.encoding is not None

    @property
    def has_dynamic_encoding(self) -> bool:
        return isinstance(self.encoding, DynamicEncoding)

    @property
    def has_static_encoding(self) -> bool:
        return isinstance(self.encoding, EncodingInstance)

    @property
    def is_all_static(self) -> bool:
        return all(isinstance(d, StaticDim) for d in self.dims)

    def __repr__(self) -> str:
        kind_name = "tile" if self.type_kind == TypeKind.TILE else "tensor"
        if self.dims:
            dim_strs = []
            for d in self.dims:
                match d:
                    case StaticDim(size=s):
                        dim_strs.append(str(s))
                    case DynamicDim():
                        dim_strs.append("?")
            shape = "x".join(dim_strs)
            return f"{kind_name}<{shape}x{self.element_type}>"
        return f"{kind_name}<{self.element_type}>"


class GroupScope(IntEnum):
    """Group scope kind. Must match loom_group_scope_t in types.h."""

    WORKGROUP = 0
    SUBGROUP = 1

    @property
    def text(self) -> str:
        """The canonical text representation for printing."""
        return _GROUP_SCOPE_NAMES[self]


_GROUP_SCOPE_NAMES: dict[GroupScope, str] = {
    GroupScope.WORKGROUP: "workgroup",
    GroupScope.SUBGROUP: "subgroup",
}

# Reverse mapping for parsing: "workgroup" -> GroupScope.WORKGROUP.
GROUP_SCOPE_BY_NAME: dict[str, GroupScope] = {
    name: scope for scope, name in _GROUP_SCOPE_NAMES.items()
}


@dataclass(frozen=True, slots=True)
class GroupType:
    """A group type: group<scope>."""

    scope: GroupScope

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.GROUP

    def __repr__(self) -> str:
        return f"group<{self.scope.text}>"


@dataclass(frozen=True, slots=True)
class FunctionType:
    """A function type: (arg_types) -> (result_types)."""

    arg_types: tuple[Type, ...]
    result_types: tuple[Type, ...]

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.FUNCTION

    def __repr__(self) -> str:
        args = ", ".join(repr(t) for t in self.arg_types)
        results = ", ".join(repr(t) for t in self.result_types)
        return f"({args}) -> ({results})"


@dataclass(frozen=True, slots=True)
class DialectType:
    """A dialect-defined type (hal.buffer, vm.ref<T>, etc.).

    Catch-all for types defined by dialects outside the built-in set.
    Built-in types (ScalarType, ShapedType, GroupType, FunctionType)
    keep their specific dataclasses for performance and ergonomics.
    Dialect types use DialectType with the TypeDef driving print/parse.

    name: Full type name as it appears in text ("hal.buffer", "vm.ref").
    params: Type parameters (e.g., the hal.buffer in vm.ref<hal.buffer>).
    """

    name: str
    params: tuple[Type, ...] = ()

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.DIALECT

    def __repr__(self) -> str:
        if self.params:
            param_strs = ", ".join(repr(p) for p in self.params)
            return f"{self.name}<{param_strs}>"
        return self.name


@dataclass(frozen=True, slots=True)
class EncodingType:
    """The type of an encoding SSA value.

    encoding.define produces a value of this type. Values of
    EncodingType are referenced in the encoding position of
    tile/tensor types (tile<4xf32, %enc>).
    """

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.ENCODING

    def __repr__(self) -> str:
        return "encoding"


# Singleton for encoding type.
ENCODING_TYPE = EncodingType()


@dataclass(frozen=True, slots=True)
class PoolType:
    """A block-managed device memory pool: pool<[%block_size]>.

    One parameter: the block size in bytes, which may be static or
    dynamic. The pool carries no capacity, no element type, no
    encoding — it's untyped bytes. Element type and encoding are
    imposed by pool ops at access time.

    Dynamic block_size uses the same DynamicDim/dim_bindings mechanism
    as shaped types: the Value carrying this type binds the dim at
    position 0 to an index-typed SSA value.
    """

    block_size: Dim

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.POOL

    @property
    def has_dynamic_block_size(self) -> bool:
        return isinstance(self.block_size, DynamicDim)

    def __repr__(self) -> str:
        match self.block_size:
            case StaticDim(size=size):
                return f"pool<{size}>"
            case DynamicDim():
                return "pool<?>"


@dataclass(frozen=True, slots=True)
class PlaceholderType:
    """A placeholder type for forward references in signatures.

    Used only during assembly parsing when a name is used in a type dim
    before it has been defined in the argument or result list. All
    placeholders must be resolved by the end of the signature.
    """

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.PLACEHOLDER

    def __repr__(self) -> str:
        return "<<placeholder>>"


# Union of all type kinds.
type Type = (
    ScalarType
    | ShapedType
    | GroupType
    | FunctionType
    | DialectType
    | EncodingType
    | PoolType
    | PlaceholderType
    | NoneType
)


def binding_element_type(operand_type: Type) -> Type:
    """Extract the element type for an 'element' binding.

    For shaped types (tile, tensor): returns the scalar element type.
    For scalar types: returns the type itself (identity).
    For dialect types: returns the first type parameter if available,
      otherwise the type itself. Custom types can override this by
      implementing element extraction on their TypeDef.

    Used by the parser when BindingList kind='element' to determine
    the block arg type from the operand type.
    """
    match operand_type:
        case ShapedType(element_type=elem):
            return elem
        case ScalarType():
            return operand_type
        case DialectType(params=params) if params:
            return params[0]
        case _:
            return operand_type


# ============================================================================
# Source locations
# ============================================================================


@unique
class LocationKind(IntEnum):
    """Location kind tag. Matches loom_location_kind_e."""

    NONE = 0
    FILE = 1
    FUSED = 2
    OPAQUE = 3


# Location flag bits (matches loom_location_flag_bits_e in ir.h).
LOCATION_FLAG_SYNTHETIC = 1 << 0


@dataclass(frozen=True, slots=True)
class FileLocation:
    """Source range location: source:start_line:start_col to end_line:end_col."""

    source_id: int
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    flags: int = 0


@dataclass(frozen=True, slots=True)
class FusedLocation:
    """Derived from multiple source locations."""

    children: tuple[int, ...]
    flags: int = 0


@dataclass(frozen=True, slots=True)
class OpaqueLocation:
    """External system identifier (torch node, JAX trace)."""

    source_id: int
    data: bytes
    flags: int = 0


# Union of location data.
type LocationData = FileLocation | FusedLocation | OpaqueLocation | None

LOCATION_UNKNOWN = 0  # Location ID 0 is always unknown.


# ============================================================================
# Encoding instances
# ============================================================================


@dataclass(frozen=True, slots=True)
class DynamicEncoding:
    """A dynamic encoding bound at the use site via an SSA value of
    type EncodingType. The Value carrying this type holds the binding
    in its encoding_binding field.

    Analogous to DynamicDim for shape dimensions: the type is
    structural and carries no value reference. The binding lives on
    the Value, not the Type.
    """


@dataclass(frozen=True, slots=True)
class EncodingInstance:
    """A parameterized encoding instance in the module's encoding table.

    Each unique encoding usage in a program gets one entry. The type's
    encoding_instance field indexes into this table (1-based, 0 = none).

    name: Encoding name ("q8_0", "q6_k", "dense"). Always present.
        This is the name used in textual format: #q8_0, #q6_k.
    encoding_kind: Index into the context's encoding vtable array.
        Used for runtime operations (storage_size, encode, decode).
    alias: Attribute alias for pretty-printing ("#enc"), or empty.
        When set, the printer uses the alias instead of the full
        #name<params> form. Defined at file level: #enc = #q8_0<block=32>
    params: Structured parameters as key-value pairs.
        Each entry is (name, value): [("block", "32"), ("group_size", "128")].
    """

    name: str
    encoding_kind: int = 0
    alias: str = ""
    params: tuple[tuple[str, str], ...] = ()


# ============================================================================
# Predicates
# ============================================================================

# Valid predicate kinds. Maps kind name to expected argument count
# (None means variable, e.g., range takes 3).
PREDICATE_KINDS: dict[str, int | None] = {
    "eq": 2,
    "lt": 2,
    "le": 2,
    "gt": 2,
    "ge": 2,
    "mul": 2,  # mul(a, n) — a is a multiple of n.
    "min": 2,  # min(a, n) — a >= n.
    "max": 2,  # max(a, n) — a <= n.
    "pow2": 1,  # pow2(a) — a is a power of 2.
    "range": 3,  # range(a, lo, hi) — lo <= a <= hi.
}


@dataclass(frozen=True, slots=True)
class PredicateArg:
    """A single argument to a predicate.

    tag: "value" for SSA value references, "ordinal" for result
         ordinals, "const" for integer constants.
    value: For "value" tag, the bare SSA name as a string (no % prefix).
           For "ordinal" tag, the ordinal index as an int.
           For "const" tag, the integer constant value.
    """

    tag: str  # "value", "ordinal", "const"
    value: int | str


@dataclass(frozen=True, slots=True)
class Predicate:
    """A predicate constraint on dynamic dimension values.

    Used in function where clauses and scalar.assume ops to express
    constraints like "M is a multiple of 16" or "K is between 32 and 512".

    kind: Predicate kind name ("eq", "lt", "mul", "pow2", "range", etc.).
    args: Tuple of predicate arguments.
    """

    kind: str
    args: tuple[PredicateArg, ...]


def _resolve_predicate_arg(arg: PredicateArg, values: dict[str, int]) -> int | None:
    """Resolve a predicate argument to a concrete integer.

    Returns None if the argument cannot be resolved (ordinal with no
    call-site context, or value name not in the values dict).
    """
    match arg.tag:
        case "const":
            assert isinstance(arg.value, int)
            return arg.value
        case "value":
            return values.get(str(arg.value))
        case "ordinal":
            return None  # Ordinals need call-site context.
        case _:
            raise ValueError(f"unknown predicate arg tag: {arg.tag!r}")


def evaluate_predicate(predicate: Predicate, values: dict[str, int]) -> bool:
    """Evaluate a predicate against concrete dimension values.

    values maps bare SSA names ("M", "K") to their integer values.
    Returns True if the predicate is satisfied or if any argument
    cannot be resolved (ordinal args defer to the call site).
    """
    resolved = [_resolve_predicate_arg(a, values) for a in predicate.args]
    if any(v is None for v in resolved):
        return True  # Unresolvable args — defer judgment.
    args = [v for v in resolved if v is not None]

    match predicate.kind:
        case "eq":
            return args[0] == args[1]
        case "lt":
            return args[0] < args[1]
        case "le":
            return args[0] <= args[1]
        case "gt":
            return args[0] > args[1]
        case "ge":
            return args[0] >= args[1]
        case "mul":
            return args[1] != 0 and args[0] % args[1] == 0
        case "min":
            return args[0] >= args[1]
        case "max":
            return args[0] <= args[1]
        case "pow2":
            return args[0] > 0 and (args[0] & (args[0] - 1)) == 0
        case "range":
            return args[1] <= args[0] <= args[2]
        case _:
            raise ValueError(f"unknown predicate kind: {predicate.kind!r}")


def evaluate_predicates(predicates: list[Predicate], values: dict[str, int]) -> bool:
    """Evaluate all predicates. Returns True iff all are satisfied."""
    return all(evaluate_predicate(p, values) for p in predicates)


# ============================================================================
# Tied results
# ============================================================================


@dataclass(frozen=True, slots=True)
class TiedResult:
    """A result that reuses an operand's storage.

    result_index: which result is tied.
    operand_index: which operand is consumed.
    has_type_change: True if the result type differs from the operand type.
    """

    result_index: int
    operand_index: int
    has_type_change: bool = False


# ============================================================================
# Use-def tracking
# ============================================================================


@dataclass(frozen=True, slots=True)
class Use:
    """Records a use of a value by an operation."""

    user_op_index: int
    operand_index: int
    block_index: int


# ============================================================================
# Values
# ============================================================================

# Value flags.
VALUE_FLAG_BLOCK_ARG = 1 << 0
VALUE_FLAG_CONSUMED = 1 << 1


@dataclass(slots=True)
class Value:
    """An SSA value (operation result or block argument).

    Values live in the module's value table, accessed by integer ID.
    The type is structural (no SSA value references in dims).
    dim_bindings maps dynamic dim positions to the value IDs that
    provide the runtime sizes.

    name: Bare name without the '%' sigil. A value named "x" prints
    as %x; a value named "" is unnamed and gets an auto-name from
    its value ID (%0, %1, ...).

    encoding_binding: when the value's type has DynamicEncoding, this
    holds the value_id of the encoding-typed SSA value that provides
    the encoding at runtime. -1 means no binding.
    """

    name: str
    type: Type
    flags: int = 0
    def_op_index: int = -1
    def_block_index: int = 0
    def_result_index: int = 0
    dim_bindings: dict[int, int] = field(default_factory=dict)
    encoding_binding: int = -1
    location_id: int = LOCATION_UNKNOWN
    uses: list[Use] = field(default_factory=list)

    @property
    def is_block_arg(self) -> bool:
        return (self.flags & VALUE_FLAG_BLOCK_ARG) != 0

    @property
    def is_consumed(self) -> bool:
        return (self.flags & VALUE_FLAG_CONSUMED) != 0


# ============================================================================
# Operations
# ============================================================================


@dataclass(slots=True)
class Operation:
    """A single IR operation.

    Operands and results are value IDs (indices into module's value table).
    The op kind is an integer ID indexing into the context's op vtable.
    """

    kind: int = 0
    operands: list[int] = field(default_factory=list)
    results: list[int] = field(default_factory=list)
    tied_results: list[TiedResult] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    location_id: int = LOCATION_UNKNOWN
    name: str = ""
    is_dead: bool = False


# ============================================================================
# Blocks and regions
# ============================================================================


@dataclass(slots=True)
class Block:
    """A basic block: a sequence of operations with optional arguments."""

    label: str = ""
    arg_ids: list[int] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)


@dataclass(slots=True)
class Region:
    """An ordered list of blocks."""

    blocks: list[Block] = field(default_factory=list)


# ============================================================================
# Symbols
# ============================================================================


@unique
class SymbolKind(IntEnum):
    """Symbol kind. Matches loom_symbol_kind_e."""

    FUNC_DEF = 0
    FUNC_DECL = 1
    FUNC_TEMPLATE = 2
    FUNC_UKERNEL = 3
    GLOBAL = 4
    EXECUTABLE = 5


# Symbol flags.
SYMBOL_FLAG_PUBLIC = 1 << 0
SYMBOL_FLAG_IMPORT = 1 << 1


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """Cross-module symbol reference."""

    module_id: int
    symbol_id: int


@dataclass(slots=True)
class Symbol:
    """A module-level named entity.

    Import symbols (SYMBOL_FLAG_IMPORT set) are forward declarations
    of symbols defined in another module. source_module names the
    module; source_symbol names the symbol in that module (defaults
    to name if empty, supporting import aliasing).

    op is the defining Operation (func.def, func.decl, etc.). All
    function properties (args, results, body, attributes) are read
    directly from op rather than from any intermediate struct.
    """

    name: str = ""
    kind: SymbolKind = SymbolKind.FUNC_DEF
    flags: int = 0
    op: Operation | None = None
    source_module: str = ""
    source_symbol: str = ""
    uses: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def is_import(self) -> bool:
        return (self.flags & SYMBOL_FLAG_IMPORT) != 0

    @property
    def is_public(self) -> bool:
        return (self.flags & SYMBOL_FLAG_PUBLIC) != 0


# ============================================================================
# Tables
# ============================================================================


class StringTable:
    """Interned string table. Deduplicates strings by content."""

    __slots__ = ("_strings", "_index")

    def __init__(self) -> None:
        self._strings: list[str] = []
        self._index: dict[str, int] = {}

    def intern(self, s: str) -> int:
        """Intern a string, returning its ID."""
        if s in self._index:
            return self._index[s]
        idx = len(self._strings)
        self._strings.append(s)
        self._index[s] = idx
        return idx

    def get(self, string_id: int) -> str:
        """Look up a string by ID."""
        return self._strings[string_id]

    def __len__(self) -> int:
        return len(self._strings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._strings)


class TypeTable:
    """Interned type table. Deduplicates types by structural equality."""

    __slots__ = ("_types", "_index")

    def __init__(self) -> None:
        self._types: list[Type] = []
        self._index: dict[Type, int] = {}

    def intern(self, t: Type) -> int:
        """Intern a type, returning its ID."""
        if t in self._index:
            return self._index[t]
        idx = len(self._types)
        self._types.append(t)
        self._index[t] = idx
        return idx

    def get(self, type_id: int) -> Type:
        """Look up a type by ID."""
        return self._types[type_id]

    def __len__(self) -> int:
        return len(self._types)

    def __iter__(self) -> Iterator[Type]:
        return iter(self._types)


class LocationTable:
    """Interned location table. Entry 0 is always None (unknown).
    O(1) dedup via hash map on frozen LocationData."""

    __slots__ = ("_locations", "_index")

    def __init__(self) -> None:
        self._locations: list[LocationData] = [None]  # Index 0 = unknown.
        self._index: dict[LocationData, int] = {None: 0}

    def add(self, loc: LocationData) -> int:
        """Add a location, returning its ID. Deduplicates."""
        if loc in self._index:
            return self._index[loc]
        loc_id = len(self._locations)
        self._locations.append(loc)
        self._index[loc] = loc_id
        return loc_id

    def get(self, location_id: int) -> LocationData:
        """Look up a location by ID."""
        return self._locations[location_id]

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self) -> Iterator[LocationData]:
        return iter(self._locations)


# ============================================================================
# Module
# ============================================================================


@dataclass
class Module:
    """Top-level IR container.

    Owns all IR through its tables. The module is the unit of
    serialization, linking, and compilation.
    """

    name: str = ""
    flags: int = 0

    # Tables.
    strings: StringTable = field(default_factory=StringTable)
    types: TypeTable = field(default_factory=TypeTable)
    locations: LocationTable = field(default_factory=LocationTable)
    values: list[Value] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    encodings: list[EncodingInstance] = field(default_factory=list)

    # Source table is on the context, but for standalone modules
    # we keep a local source list.
    sources: list[str] = field(default_factory=list)

    def add_value(self, value: Value) -> int:
        """Add a value to the module, returning its ID."""
        value_id = len(self.values)
        self.values.append(value)
        return value_id

    def add_location(self, loc: LocationData) -> int:
        """Add a location, returning its ID. O(1) dedup."""
        return self.locations.add(loc)

    def add_symbol(self, symbol: Symbol) -> int:
        """Add a symbol, returning its ID."""
        sym_id = len(self.symbols)
        self.symbols.append(symbol)
        return sym_id

    def add_encoding(self, instance: EncodingInstance) -> int:
        """Add an encoding instance, returning its index. Deduplicates."""
        for i, existing in enumerate(self.encodings):
            if existing.name == instance.name and existing.params == instance.params:
                return i
        idx = len(self.encodings)
        self.encodings.append(instance)
        return idx


# ============================================================================
# Context
# ============================================================================


@dataclass
class Context:
    """Global state: vtables, source table, module registry.

    Immutable during compilation. Created once at startup.
    """

    sources: list[str] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)

    def add_source(self, name: str) -> int:
        """Register a source identifier, returning its ID."""
        for i, existing in enumerate(self.sources):
            if existing == name:
                return i
        source_id = len(self.sources)
        self.sources.append(name)
        return source_id
