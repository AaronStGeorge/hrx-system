# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom op declaration DSL.

This module defines the types used to declare loom operations in Python.
These declarations are the single source of truth: generators consume
them to produce C headers, typed builder APIs, JSON databases, and
documentation.

Op declarations are dataclass instances (not class hierarchies). Each
declaration is pure data describing an op's interface: what it takes,
what it produces, how it looks in textual assembly, what constraints
it enforces.

Dialect files import the specific types they need from this module and
from loom.assembly. ruff manages import lists automatically.

Quick reference — declaring an op:

    scalar_addi = Op(
        name="scalar.addi",
        group=scalar_ops,
        doc="Integer addition.",
        operands=[
            Operand("lhs", INTEGER),
            Operand("rhs", INTEGER),
        ],
        results=[Result("result", INTEGER)],
        constraints=[SameType("lhs", "rhs", "result")],
        traits=[PURE, COMMUTATIVE],
        format=[Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")],
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, NamedTuple

from loom.assembly import FormatElement
from loom.errors import ErrorDef

__all__ = [
    # Type constraints.
    "TypeConstraint",
    "TILE",
    "TENSOR",
    "VECTOR",
    "VIEW",
    "BUFFER",
    "INTEGER",
    "INTEGER_ELEMENT",
    "FLOAT",
    "FLOAT_ELEMENT",
    "I1_ELEMENT",
    "SCALAR",
    "INDEX",
    "OFFSET",
    "ANY",
    "GROUP",
    "ANY_ENCODING",
    "ENCODING_LAYOUT",
    "ENCODING_SCHEMA",
    "ENCODING_STORAGE",
    "ENCODING_TRANSFORM",
    "POOL",
    "I1",
    # Type constraint helpers.
    "type_constraint_name",
    # Field descriptors.
    "Operand",
    "Result",
    "TiedResult",
    "AttrDef",
    "AttrType",
    "ATTR_TYPE_I64",
    "ATTR_TYPE_F64",
    "ATTR_TYPE_STRING",
    "ATTR_TYPE_BOOL",
    "ATTR_TYPE_ENUM",
    "ATTR_TYPE_TYPE",
    "ATTR_TYPE_I64_ARRAY",
    "ATTR_TYPE_ENCODING",
    "ATTR_TYPE_ANY",
    "ATTR_TYPE_SYMBOL",
    "ATTR_TYPE_FLAGS",
    "ATTR_TYPE_PREDICATE_LIST",
    "ATTR_TYPE_DICT",
    "RegionDef",
    # Enum support.
    "EnumCase",
    "EnumDef",
    # Traits.
    "Trait",
    "PURE",
    "COMMUTATIVE",
    "IDEMPOTENT",
    "INVOLUTION",
    "TERMINATOR",
    "CONSTANT_LIKE",
    "ELEMENTWISE",
    "DECOMPOSABLE",
    "SYMBOL_DEFINE",
    "ISOLATED_FROM_ABOVE",
    "NON_DETERMINISTIC",
    "UNKNOWN_EFFECTS",
    "HINT",
    # Trait constructors.
    "AllTypesMatch",
    "HasParent",
    "ImplicitTerminator",
    # Memory effects.
    "EffectKind",
    "Effect",
    "Reads",
    "Writes",
    "ReadWrites",
    # Constraints.
    "Constraint",
    "SameType",
    "SameKind",
    "SameElementType",
    "SameEncoding",
    "SameShape",
    "RanksMatch",
    "HasIntegerElement",
    "HasFloatElement",
    "HasI1Element",
    "OffsetCountMatchesRank",
    "DimIndexInBounds",
    "AllShapesMatch",
    "BlockArgCount",
    "BlockArgsMatchElementTypes",
    "YieldCountMatchesResults",
    "YieldTypesMatchResults",
    "YieldElementTypesMatchResults",
    "IterArgsMatchResults",
    # Op group.
    "Dialect",
    # Interfaces.
    "FuncLikeInterface",
    "LoopLikeInterface",
    "RegionBranchInterface",
    # Op declaration.
    "Op",
    # Type declaration.
    "TypeDef",
    "TypeParam",
    "ShapeParam",
    "ScalarParam",
    "EncodingParam",
    # Helpers.
    "binary_op",
    "unary_op",
    "cast_op",
    "comparison_op",
]


# ============================================================================
# Type constraints
# ============================================================================


@unique
class TypeConstraint(Enum):
    """Constraint on the type of an operand or result.

    These are abstract categories, not concrete types. The verifier
    checks that the actual type of a value satisfies the constraint.
    The builder generator maps them to Python type hints.

    Constraint → ir.py types that satisfy it:
      TILE     → ShapedType with type_kind=TILE
      TENSOR   → ShapedType with type_kind=TENSOR
      VECTOR   → ShapedType with type_kind=VECTOR
      VIEW     → ShapedType with type_kind=VIEW
      BUFFER   → BufferType
      INTEGER  → ScalarType with kind in {I1, I8, I16, I32, I64}
      FLOAT    → ScalarType with kind in {F8*, F16, BF16, F32, F64}
      INTEGER_ELEMENT → ShapedType with integer element type
      FLOAT_ELEMENT   → ShapedType with float element type
      I1_ELEMENT      → ShapedType with element type i1
      SCALAR   → any ScalarType (INTEGER | FLOAT | INDEX | OFFSET)
      INDEX    → ScalarType with kind=INDEX
      OFFSET   → ScalarType with kind=OFFSET
      ANY      → any type
      GROUP    → GroupType
      ANY_ENCODING → any EncodingType
      ENCODING_LAYOUT → EncodingType with role=layout
      ENCODING_SCHEMA → EncodingType with role=schema
      ENCODING_STORAGE → EncodingType with role=storage
      ENCODING_TRANSFORM → EncodingType with role=transform
      POOL     → PoolType

    Element-qualified constraints are shaped-only: tile, tensor, vector,
    and view types can satisfy them, while scalar values continue to use
    INTEGER/FLOAT/I1. Combine a shaped kind constraint with SameElementType
    when an op needs both a specific shaped kind and a shared element family.
    """

    TILE = "tile"
    TENSOR = "tensor"
    VECTOR = "vector"
    VIEW = "view"
    BUFFER = "buffer"
    INTEGER = "integer"
    FLOAT = "float"
    INTEGER_ELEMENT = "integer_element"
    FLOAT_ELEMENT = "float_element"
    I1_ELEMENT = "i1_element"
    SCALAR = "scalar"
    INDEX = "index"
    OFFSET = "offset"
    ANY = "any"
    GROUP = "group"
    ANY_ENCODING = "encoding"
    ENCODING_LAYOUT = "encoding<layout>"
    ENCODING_SCHEMA = "encoding<schema>"
    ENCODING_STORAGE = "encoding<storage>"
    ENCODING_TRANSFORM = "encoding<transform>"
    POOL = "pool"
    I1 = "i1"


# Singletons for use in op declarations.
TILE = TypeConstraint.TILE
TENSOR = TypeConstraint.TENSOR
VECTOR = TypeConstraint.VECTOR
VIEW = TypeConstraint.VIEW
BUFFER = TypeConstraint.BUFFER
INTEGER = TypeConstraint.INTEGER
FLOAT = TypeConstraint.FLOAT
INTEGER_ELEMENT = TypeConstraint.INTEGER_ELEMENT
FLOAT_ELEMENT = TypeConstraint.FLOAT_ELEMENT
I1_ELEMENT = TypeConstraint.I1_ELEMENT
SCALAR = TypeConstraint.SCALAR
INDEX = TypeConstraint.INDEX
OFFSET = TypeConstraint.OFFSET
ANY = TypeConstraint.ANY
GROUP = TypeConstraint.GROUP
ANY_ENCODING = TypeConstraint.ANY_ENCODING
ENCODING_LAYOUT = TypeConstraint.ENCODING_LAYOUT
ENCODING_SCHEMA = TypeConstraint.ENCODING_SCHEMA
ENCODING_STORAGE = TypeConstraint.ENCODING_STORAGE
ENCODING_TRANSFORM = TypeConstraint.ENCODING_TRANSFORM
POOL = TypeConstraint.POOL
I1 = TypeConstraint.I1


def type_constraint_name(constraint: TypeConstraint) -> str:
    """Return the display name for a type constraint."""
    return constraint.value


# ============================================================================
# Field descriptors
# ============================================================================


@dataclass(frozen=True, slots=True)
class Operand:
    """An SSA value operand (input to the op at runtime).

    name: Field name, used in format specs and builders.
    type_constraint: What types this operand accepts.
    doc: Human-readable description.
    variadic: If True, this is zero-or-more values (list[Value]).
    optional: If True, this operand may be absent.
    """

    name: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    optional: bool = False


@dataclass(frozen=True, slots=True)
class Result:
    """An SSA value result (output of the op).

    name: Field name, used in format specs and builders.
    type_constraint: What types this result produces.
    doc: Human-readable description.
    variadic: If True, this is zero-or-more result values.
    allocates: If True, this result is a freshly allocated resource
        that cannot alias any pre-existing resource.
    """

    name: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    allocates: bool = False


@dataclass(frozen=True, slots=True)
class TiedResult:
    """A result that is always tied to a specific operand.

    The result reuses the operand's storage. The result type may differ
    from the operand type (e.g., reshape or encoding change over the
    same storage).

    name: Field name for the result.
    tied_to: Name of the operand this result is tied to.
    type_constraint: What types this result produces.
    doc: Human-readable description.
    variadic: For protocol compatibility with Result. Tied results
        cannot be variadic.
    allocates: Always False. Tied results reuse existing storage.
    """

    name: str
    tied_to: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    allocates: bool = False


# Allowed attribute type strings. Using Literal for static checking.
type AttrType = str  # One of the ATTR_TYPE_* constants below.

ATTR_TYPE_I64 = "i64"
ATTR_TYPE_F64 = "f64"
ATTR_TYPE_STRING = "string"
ATTR_TYPE_BOOL = "bool"
ATTR_TYPE_ENUM = "enum"
ATTR_TYPE_TYPE = "type"
ATTR_TYPE_I64_ARRAY = "i64_array"
ATTR_TYPE_ENCODING = "encoding"
ATTR_TYPE_ANY = "any"
ATTR_TYPE_SYMBOL = "symbol"
ATTR_TYPE_FLAGS = "flags"
ATTR_TYPE_PREDICATE_LIST = "predicate_list"
ATTR_TYPE_DICT = "dict"  # Named attribute dictionary.

_VALID_ATTR_TYPES = frozenset(
    {
        ATTR_TYPE_I64,
        ATTR_TYPE_F64,
        ATTR_TYPE_STRING,
        ATTR_TYPE_BOOL,
        ATTR_TYPE_ENUM,
        ATTR_TYPE_TYPE,
        ATTR_TYPE_I64_ARRAY,
        ATTR_TYPE_ENCODING,
        ATTR_TYPE_ANY,
        ATTR_TYPE_SYMBOL,
        ATTR_TYPE_FLAGS,
        ATTR_TYPE_PREDICATE_LIST,
        ATTR_TYPE_DICT,
    }
)


@dataclass(frozen=True, slots=True)
class AttrDef:
    """A compile-time constant attribute on an op.

    name: Attribute name (key in the attribute dictionary or
          positional in the format spec).
    attr_type: The kind of attribute value. Must be one of the
        ATTR_TYPE_* constants: "i64", "f64", "string", "bool",
        "enum", "type", "i64_array", "encoding", "any".
    doc: Human-readable description.
    default: Default value (None = required, not optional).
    enum_def: For enum attrs, the EnumDef describing valid values.
        Required when attr_type is "enum".
    optional: If True, this attribute may be absent.
    """

    name: str
    attr_type: AttrType
    doc: str = ""
    default: Any = None
    enum_def: EnumDef | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.attr_type not in _VALID_ATTR_TYPES:
            raise ValueError(
                f"AttrDef '{self.name}': invalid attr_type "
                f"'{self.attr_type}', must be one of {sorted(_VALID_ATTR_TYPES)}"
            )
        if self.attr_type == ATTR_TYPE_ENUM and self.enum_def is None:
            raise ValueError(
                f"AttrDef '{self.name}': attr_type='enum' requires enum_def"
            )
        if self.attr_type == ATTR_TYPE_FLAGS and self.enum_def is None:
            raise ValueError(
                f"AttrDef '{self.name}': attr_type='flags' requires enum_def"
            )


@dataclass(frozen=True, slots=True)
class RegionDef:
    """A nested region on an op.

    name: Field name, used in format specs.
    doc: Human-readable description.
    single_block: If True, the region must have exactly one block.
    implicit_args: Implicit block arguments created by the builder
        but not derived from operands. Each entry is (name, type_keyword)
        where type_keyword is a scalar type name (e.g., "index").
        These are prepended before any BindingList-derived args.
        Example: loop IV is ("iv", "index").
    """

    name: str
    doc: str = ""
    single_block: bool = False
    implicit_args: tuple[tuple[str, str], ...] = ()


# ============================================================================
# Enum support
# ============================================================================


@dataclass(frozen=True, slots=True)
class EnumCase:
    """A single case in an enum attribute.

    keyword: The textual keyword in assembly (e.g., "slt", "left").
    value: Integer value for binary encoding.
    doc: Human-readable description.
    """

    keyword: str
    value: int
    doc: str = ""


@dataclass(frozen=True, slots=True)
class EnumDef:
    """An enum attribute type with named cases.

    name: Enum type name (e.g., "CmpIPredicate", "BroadcastDir").
    cases: The valid cases.
    doc: Human-readable description.
    """

    name: str
    cases: tuple[EnumCase, ...]
    doc: str = ""

    def __init__(
        self,
        name: str,
        cases: list[EnumCase] | tuple[EnumCase, ...],
        doc: str = "",
    ) -> None:
        object.__setattr__(self, "name", name)
        frozen_cases = tuple(cases)
        object.__setattr__(self, "cases", frozen_cases)
        object.__setattr__(self, "doc", doc)
        # Validate no duplicate keywords or values.
        seen_keywords: set[str] = set()
        seen_values: set[int] = set()
        for case in frozen_cases:
            if case.keyword in seen_keywords:
                raise ValueError(
                    f"EnumDef '{name}': duplicate keyword '{case.keyword}'"
                )
            if case.value in seen_values:
                raise ValueError(f"EnumDef '{name}': duplicate value {case.value}")
            seen_keywords.add(case.keyword)
            seen_values.add(case.value)

    @property
    def keywords(self) -> tuple[str, ...]:
        """All valid keyword strings."""
        return tuple(c.keyword for c in self.cases)


# ============================================================================
# Traits
# ============================================================================


@dataclass(frozen=True, slots=True)
class Trait:
    """A structural property of an operation.

    Traits describe invariants that the verifier checks and that
    optimizations may exploit. Simple traits are singletons (PURE,
    COMMUTATIVE). Parameterized traits are constructed with helper
    functions (AllTypesMatch, HasParent).

    name: Trait identifier.
    args: Optional arguments (field names, op names, etc.).
    """

    name: str
    args: tuple[str, ...] = ()

    def __init__(self, name: str, *args: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", args)

    def __repr__(self) -> str:
        if self.args:
            return f"{self.name}({', '.join(self.args)})"
        return self.name


# Standard singleton traits.
PURE = Trait("Pure")
COMMUTATIVE = Trait("Commutative")
IDEMPOTENT = Trait("Idempotent")
INVOLUTION = Trait("Involution")
TERMINATOR = Trait("Terminator")
CONSTANT_LIKE = Trait("ConstantLike")
ELEMENTWISE = Trait("Elementwise")
DECOMPOSABLE = Trait("Decomposable")
SYMBOL_DEFINE = Trait("SymbolDefine")
# Op's regions cannot reference values from the enclosing scope.
# Values enter the region only through block arguments. Passes must
# not substitute inner values with outer definitions.
ISOLATED_FROM_ABOVE = Trait("IsolatedFromAbove")
# Same inputs may produce different outputs (RNG, timestamps). Prevents
# CSE and LICM but not DCE — unused non-deterministic results are dead
# as long as the op has no write effects.
NON_DETERMINISTIC = Trait("NonDeterministic")
# Effects depend on runtime state (e.g., func.call depends on the callee).
# Passes treat this conservatively as both READS_MEMORY and WRITES_MEMORY.
UNKNOWN_EFFECTS = Trait("UnknownEffects")
# Each execution produces a result with a distinct identity, even when
# operands and attributes are identical. Prevents CSE but allows DCE
# (unused identity with no write effects is dead) and LICM. Derived
# automatically when any result has allocates=True, but can also be
# declared explicitly.
UNIQUE_IDENTITY = Trait("UniqueIdentity")
# Compiler hint with no semantic memory effects. Hint ops are preserved by
# canonicalization/DCE and removed only by an explicit hint-stripping pass.
HINT = Trait("Hint")


# ============================================================================
# Memory effects
# ============================================================================


@unique
class EffectKind(Enum):
    """Kind of memory effect on a resource-typed operand."""

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


@dataclass(frozen=True, slots=True)
class Effect:
    """Declares a memory effect on a specific operand.

    operand: Name of the resource-typed operand being accessed.
    kind: What the op does to the resource (read, write, or both).
    """

    operand: str
    kind: EffectKind


def Reads(operand: str) -> Effect:
    """Op reads from the named resource operand."""
    return Effect(operand, EffectKind.READ)


def Writes(operand: str) -> Effect:
    """Op writes to the named resource operand."""
    return Effect(operand, EffectKind.WRITE)


def ReadWrites(operand: str) -> Effect:
    """Op performs an atomic read-modify-write on the named resource operand."""
    return Effect(operand, EffectKind.READWRITE)


# Type constraints that represent mutable resources (as opposed to
# pure SSA values). Effects may only reference operands with one of
# these constraints.
_RESOURCE_TYPE_CONSTRAINTS = frozenset(
    {
        TypeConstraint.POOL,
        TypeConstraint.BUFFER,
        TypeConstraint.VIEW,
        TypeConstraint.TENSOR,
        TypeConstraint.GROUP,
        TypeConstraint.ANY,
    }
)


# Parameterized trait constructors.
def AllTypesMatch(*fields: str) -> Trait:
    """All named fields must have identical types."""
    return Trait("AllTypesMatch", *fields)


def HasParent(op_name: str) -> Trait:
    """This op must be directly nested inside the named op."""
    return Trait("HasParent", op_name)


def ImplicitTerminator(op_name: str) -> Trait:
    """Regions of this op have an implicit terminator of the given type."""
    return Trait("ImplicitTerminator", op_name)


# ============================================================================
# Constraints
# ============================================================================


# Validate signature: receives a dict mapping field names to values,
# returns (ok, message). Used by the Python validator/oracle.
type ValidateFn = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class Constraint:
    """A verification constraint on an op's fields.

    Constraints express relationships between operands, results, and
    attributes that the verifier checks. Each constraint has:

      name: Identifier for code generation and diagnostics.
      args: Field names this constraint references.
      error: Structured error definition emitted on failure.
      validate: Optional Python predicate for the oracle/validator.

    Constraints are defined as module-level constructor functions
    (SameType, RanksMatch, etc.) that return Constraint instances
    with appropriate validation closures.
    """

    name: str
    args: tuple[str, ...]
    error: ErrorDef | None = None
    validate: ValidateFn | None = None

    def __init__(
        self,
        name: str,
        args: tuple[str, ...],
        error: ErrorDef | None = None,
        validate: ValidateFn | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "validate", validate)

    def check(self, fields: dict[str, Any]) -> tuple[bool, str]:
        """Run the validation predicate. Returns (ok, message)."""
        if self.validate is None:
            return (True, "")
        return self.validate(fields)

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


# --- Type and shape constraints ---


def _flatten_field(name: str, value: Any) -> list[tuple[str, Any]]:
    """Expand a field value to (display_name, item) pairs.

    For scalar fields (a single Value), returns [(name, value)].
    For variadic fields (a list of Values), returns
    [(name[0], items[0]), (name[1], items[1]), ...].

    None values and empty lists produce an empty result.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [(f"{name}[{i}]", item) for i, item in enumerate(value)]
    return [(name, value)]


def _field_value_type(item: Any) -> Any:
    """Returns the type carried by a field item, or the item if it is a type."""
    return item.type if hasattr(item, "type") else item


def _field_value_kind(item: Any) -> Any:
    """Returns a type-kind-like discriminator for a field item."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "type_kind"):
        return value_type.type_kind
    if hasattr(value_type, "kind"):
        return value_type.kind
    return None


def _field_element_type(item: Any) -> Any:
    """Returns the scalar element type carried by a scalar or shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "element_type"):
        return value_type.element_type
    if hasattr(value_type, "dtype"):
        return value_type.dtype
    if hasattr(value_type, "type_kind") and hasattr(value_type, "kind"):
        return value_type
    return None


def _field_shape(item: Any) -> Any:
    """Returns the shape carried by a shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "dims"):
        return value_type.dims
    if hasattr(value_type, "shape"):
        return value_type.shape
    return None


def _field_rank(item: Any) -> Any:
    """Returns the rank carried by a shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "rank"):
        return value_type.rank
    if hasattr(value_type, "ndim"):
        return value_type.ndim
    return None


def SameType(*fields: str) -> Constraint:
    """All named fields must have identical types.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        types = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                if hasattr(item, "type"):
                    types.append((display_name, item.type))
        if len(types) < 2:
            return (True, "")
        first_name, first_type = types[0]
        for entry_name, value_type in types[1:]:
            if value_type != first_type:
                return (
                    False,
                    f"'{entry_name}' type {value_type}"
                    f" != '{first_name}' type {first_type}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_001

    return Constraint(
        "SameType",
        fields,
        error=ERR_TYPE_001,
        validate=_validate,
    )


def SameKind(*fields: str) -> Constraint:
    """All named fields must have the same type kind."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        kinds = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                kind = _field_value_kind(item)
                if kind is not None:
                    kinds.append((display_name, kind))
        if len(kinds) < 2:
            return (True, "")
        first_name, first_kind = kinds[0]
        for entry_name, kind in kinds[1:]:
            if kind != first_kind:
                return (
                    False,
                    f"'{entry_name}' kind {kind} != '{first_name}' kind {first_kind}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_001

    return Constraint(
        "SameKind",
        fields,
        error=ERR_TYPE_001,
        validate=_validate,
    )


def SameElementType(*fields: str) -> Constraint:
    """All named shaped fields must have the same element type.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        element_types: list[tuple[str, Any]] = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                element_type = _field_element_type(item)
                if element_type is not None:
                    element_types.append((display_name, element_type))
        if len(element_types) < 2:
            return (True, "")
        first_name, first_dtype = element_types[0]
        for entry_name, dtype in element_types[1:]:
            if dtype != first_dtype:
                return (
                    False,
                    f"'{entry_name}' element type {dtype} != "
                    f"'{first_name}' element type {first_dtype}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_002

    return Constraint(
        "SameElementType",
        fields,
        error=ERR_TYPE_002,
        validate=_validate,
    )


def SameEncoding(*fields: str) -> Constraint:
    """All named fields must have the same encoding.

    Encodings are a type-system concept not present in numpy arrays.
    The Python validator is a no-op; the C verifier checks encoding
    attributes on the actual types.
    """
    from loom.error.encoding import ERR_ENCODING_001

    return Constraint(
        "SameEncoding",
        fields,
        error=ERR_ENCODING_001,
    )


def SameShape(*fields: str) -> Constraint:
    """All named shaped fields must have identical shapes.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        shapes: list[tuple[str, Any]] = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                shape = _field_shape(item)
                if shape is not None:
                    shapes.append((display_name, shape))
        if len(shapes) < 2:
            return (True, "")
        first_name, first_shape = shapes[0]
        for entry_name, shape in shapes[1:]:
            if shape != first_shape:
                return (
                    False,
                    f"'{entry_name}' shape {shape}"
                    f" != '{first_name}' shape {first_shape}",
                )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_002

    return Constraint(
        "SameShape",
        fields,
        error=ERR_SHAPE_002,
        validate=_validate,
    )


def RanksMatch(a: str, b: str) -> Constraint:
    """Two shaped fields must have the same rank.

    Handles variadic fields: if either field's value is a list, each
    element is checked pairwise against all elements of the other field.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        items_a = [
            (display_name, rank)
            for display_name, item in _flatten_field(a, values.get(a))
            if (rank := _field_rank(item)) is not None
        ]
        items_b = [
            (display_name, rank)
            for display_name, item in _flatten_field(b, values.get(b))
            if (rank := _field_rank(item)) is not None
        ]
        if not items_a or not items_b:
            return (True, "")
        # Check all pairs: every item in a against every item in b.
        for name_a, rank_a in items_a:
            for name_b, rank_b in items_b:
                if rank_a != rank_b:
                    return (
                        False,
                        f"'{name_a}' rank {rank_a} != '{name_b}' rank {rank_b}",
                    )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_001

    return Constraint(
        "RanksMatch",
        (a, b),
        error=ERR_SHAPE_001,
        validate=_validate,
    )


def _type_satisfies_element_constraint(
    value_type: Any, constraint: TypeConstraint
) -> bool:
    from loom.ir import ScalarTypeKind, ShapedType

    if not isinstance(value_type, ShapedType):
        return False
    element_kind = value_type.element_type.kind
    if constraint == INTEGER_ELEMENT:
        return element_kind in {
            ScalarTypeKind.I1,
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
        }
    if constraint == FLOAT_ELEMENT:
        return element_kind in {
            ScalarTypeKind.F8E4M3,
            ScalarTypeKind.F8E5M2,
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
            ScalarTypeKind.F32,
            ScalarTypeKind.F64,
        }
    if constraint == I1_ELEMENT:
        return element_kind == ScalarTypeKind.I1
    return False


def _has_element_constraint(name: str, constraint: TypeConstraint) -> Constraint:
    """A shaped field's element type must satisfy a family constraint."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        for display_name, item in _flatten_field(name, values.get(name)):
            value_type = _field_value_type(item)
            if _type_satisfies_element_constraint(value_type, constraint):
                continue
            return (
                False,
                f"'{display_name}' type {value_type} does not satisfy "
                f"{constraint.value}",
            )
        return (True, "")

    from loom.error.type import ERR_TYPE_003

    return Constraint(
        f"Has{constraint.name.title().replace('_', '')}",
        (name,),
        error=ERR_TYPE_003,
        validate=_validate,
    )


def HasIntegerElement(field: str) -> Constraint:
    """A shaped field must have an integer element type."""

    return _has_element_constraint(field, INTEGER_ELEMENT)


def HasFloatElement(field: str) -> Constraint:
    """A shaped field must have a floating-point element type."""

    return _has_element_constraint(field, FLOAT_ELEMENT)


def HasI1Element(field: str) -> Constraint:
    """A shaped field must have an i1 element type."""

    return _has_element_constraint(field, I1_ELEMENT)


def OffsetCountMatchesRank(shaped: str, offsets: str) -> Constraint:
    """Offset count must equal the shaped field's rank."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        shaped_val = values.get(shaped)
        offsets_val = values.get(offsets, [])
        if shaped_val is None or not hasattr(shaped_val, "ndim"):
            return (True, "")
        if len(offsets_val) != shaped_val.ndim:
            return (
                False,
                f"'{shaped}' rank {shaped_val.ndim} != offset count {len(offsets_val)}",
            )
        return (True, "")

    from loom.error.subrange import ERR_SUBRANGE_001

    return Constraint(
        "OffsetCountMatchesRank",
        (shaped, offsets),
        error=ERR_SUBRANGE_001,
        validate=_validate,
    )


def DimIndexInBounds(source: str, index: str) -> Constraint:
    """Dimension index must be in [0, rank) when statically known."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        source_val = values.get(source)
        index_val = values.get(index)
        if source_val is None or index_val is None:
            return (True, "")
        if not isinstance(index_val, int):
            return (True, "")
        if not hasattr(source_val, "ndim"):
            return (True, "")
        if index_val < 0 or index_val >= source_val.ndim:
            return (
                False,
                f"dimension index {index_val} out of bounds for rank {source_val.ndim}",
            )
        return (True, "")

    from loom.error.subrange import ERR_SUBRANGE_002

    return Constraint(
        "DimIndexInBounds",
        (source, index),
        error=ERR_SUBRANGE_002,
        validate=_validate,
    )


def AllShapesMatch(field: str) -> Constraint:
    """All values in a variadic field must have identical shapes."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        items = values.get(field, [])
        if len(items) < 2:
            return (True, "")
        if not hasattr(items[0], "shape"):
            return (True, "")
        first_shape = items[0].shape
        for i, item in enumerate(items[1:], 1):
            if item.shape != first_shape:
                return (
                    False,
                    f"input {i} shape {item.shape} != input 0 shape {first_shape}",
                )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_003

    return Constraint(
        "AllShapesMatch",
        (field,),
        error=ERR_SHAPE_003,
        validate=_validate,
    )


# --- Region constraints (structural, no-op in Python validator) ---


def BlockArgCount(region: str, inputs: str) -> Constraint:
    """Region block must have one argument per input."""
    from loom.error.structure import ERR_STRUCTURE_007

    return Constraint(
        "BlockArgCount",
        (region, inputs),
        error=ERR_STRUCTURE_007,
    )


def BlockArgsMatchElementTypes(region: str, inputs: str) -> Constraint:
    """Each block argument type must match its input's element type."""
    from loom.error.type import ERR_TYPE_008

    return Constraint(
        "BlockArgsMatchElementTypes",
        (region, inputs),
        error=ERR_TYPE_008,
    )


def YieldCountMatchesResults(region: str, results: str) -> Constraint:
    """Region terminator must yield the same number of values as results."""
    from loom.error.structure import ERR_STRUCTURE_008

    return Constraint(
        "YieldCountMatchesResults",
        (region, results),
        error=ERR_STRUCTURE_008,
    )


def YieldTypesMatchResults(region: str, results: str) -> Constraint:
    """Yielded value types must match result types."""
    from loom.error.type import ERR_TYPE_009

    return Constraint(
        "YieldTypesMatchResults",
        (region, results),
        error=ERR_TYPE_009,
    )


def YieldElementTypesMatchResults(region: str, results: str) -> Constraint:
    """Yielded value element types must match result element types.

    For elementwise ops where the body operates at scalar granularity:
    the yield produces scalar values and the result is a shaped type.
    This constraint checks that the element types match, not the full
    types.
    """
    from loom.error.type import ERR_TYPE_009

    return Constraint(
        "YieldElementTypesMatchResults",
        (region, results),
        error=ERR_TYPE_009,
    )


def IterArgsMatchResults(iter_args: str, results: str) -> Constraint:
    """Two variadic value fields must agree on count and per-position type.

    Used by ops that thread loop-carried state through a region (e.g.,
    scf.for): the iter_args operands provide initial values, the body
    yields the next iteration's values, and the results expose the
    final values. The yield-to-results match is enforced by
    YieldTypesMatchResults; this constraint enforces that iter_args
    and results agree directly so a count or type mismatch is reported
    on the loop op itself, not just the terminator.

    Both fields must be variadic value fields (operands or results).
    A count mismatch produces a single ERR_STRUCTURE_013; per-position
    type mismatches each produce an ERR_TYPE_001.
    """
    from loom.error.structure import ERR_STRUCTURE_013

    return Constraint(
        "IterArgsMatchResults",
        (iter_args, results),
        error=ERR_STRUCTURE_013,
    )


# ============================================================================
# Op group
# ============================================================================


@dataclass(frozen=True, slots=True)
class Dialect:
    """A logical grouping of related operations.

    Corresponds to a namespace prefix in op names (scalar, tile, scf,
    func, etc.) and a subdirectory in the dialect file layout. Groups
    carry documentation, an optional dialect ID for bytecode encoding,
    and optionally shared enum definitions.

    Op names within a group share the group prefix: if the group is
    "scalar" and the op is "addi", the full op name is "scalar.addi".

    The dialect_id is a stable integer used in bytecode format to
    identify which dialect an op belongs to. Once assigned, it must
    never change. Unassigned groups use dialect_id=0.
    """

    name: str
    dialect_id: int
    doc: str = ""
    enums: tuple[EnumDef, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        dialect_id: int = 0,
        doc: str = "",
        enums: list[EnumDef] | tuple[EnumDef, ...] = (),
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dialect_id", dialect_id)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "enums", tuple(enums))


# ============================================================================
# Type declarations
# ============================================================================


@dataclass(frozen=True, slots=True)
class TypeParam:
    """A type parameter — another type nested inside this type.

    Used for parameterized types like vm.ref<hal.buffer> where the
    inner type (hal.buffer) is a type parameter.
    """

    name: str
    constraint: TypeConstraint = TypeConstraint.ANY
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ShapeParam:
    """Shape dimensions parameter for shaped types.

    Represents the dimension list in tile<4x4xf32>. Each dim is
    either a static integer or a dynamic dim reference [%name].
    """

    name: str
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ScalarParam:
    """Scalar element type parameter for shaped types.

    Represents the element type in tile<4x4xf32> — the f32 part.
    """

    name: str
    doc: str = ""


@dataclass(frozen=True, slots=True)
class EncodingParam:
    """Encoding parameter for shaped types.

    Represents the optional encoding suffix in tile<256xi8, #q8_0<block=32>>.
    """

    name: str
    optional: bool = True
    doc: str = ""


# Union of type parameter kinds.
type TypeParamDef = TypeParam | ShapeParam | ScalarParam | EncodingParam


@dataclass(frozen=True, slots=True)
class TypeDef:
    """Declarative type definition.

    Types are declared with the same pattern as ops: a name, optional
    parameters, and a format spec describing the textual syntax.

    The format describes the interior of name<...>:
      - Empty format = opaque type (no angle brackets): hal.buffer
      - Non-empty format = parameterized: vm.ref<hal.buffer>, tile<4x4xf32>

    Scalar types (f32, i32, index) are NOT TypeDefs — they are
    keywords handled by a fixed name table. TypeDefs are for
    structured types with angle-bracket syntax.

    Examples:
        # Opaque type:
        TypeDef(name="hal.buffer")
        # Prints: hal.buffer

        # Single type parameter:
        TypeDef(name="vm.ref",
                params=[TypeParam("object", ANY)],
                format=[TypeOf("object")])
        # Prints: vm.ref<hal.buffer>

        # Shaped type with dims, element, encoding:
        TypeDef(name="tile",
                params=[ShapeParam("dims"), ScalarParam("element_type"),
                        EncodingParam("encoding")],
                format=[ShapeOf("dims"), kw("x"), ScalarOf("element_type"),
                        OptionalGroup([COMMA, EncodingOf("encoding")],
                                      anchor="encoding")])
        # Prints: tile<4x4xf32, #q8_0>
    """

    name: str
    doc: str = ""
    params: tuple[TypeParamDef, ...] = ()
    format: tuple[FormatElement, ...] = ()
    ir_kind: str = "dialect"  # "tile", "tensor", "vector", "view", etc.

    def __init__(
        self,
        name: str,
        *,
        doc: str = "",
        params: list[TypeParamDef] | tuple[TypeParamDef, ...] = (),
        format: list[FormatElement] | tuple[FormatElement, ...] = (),
        ir_kind: str = "dialect",
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "params", tuple(params))
        object.__setattr__(self, "format", tuple(format))
        object.__setattr__(self, "ir_kind", ir_kind)

    def __repr__(self) -> str:
        return f"TypeDef({self.name!r})"

    @property
    def is_opaque(self) -> bool:
        """True if this type has no parameters (no angle brackets)."""
        return len(self.format) == 0

    @property
    def is_parameterized(self) -> bool:
        """True if this type has angle-bracket syntax."""
        return len(self.format) > 0

    def param(self, name: str) -> TypeParamDef | None:
        """Find a parameter by name."""
        for p in self.params:
            if p.name == name:
                return p
        return None


# ============================================================================
# Format field validation
# ============================================================================

# Fields that are implicit (not declared as operands/results/attrs/regions
# but valid in format specs). These are created by the printer/parser from
# context: "iv" is the induction variable for loops, "args" are function
# parameters, "predicates" are where-clause items.
_IMPLICIT_FORMAT_FIELDS = frozenset({"iv", "args", "predicates"})


def _collect_format_fields(elements: tuple[FormatElement, ...]) -> set[str]:
    """Recursively collect all field names referenced by format elements."""
    from loom.assembly import (
        Attr,
        AttrDict,
        BindingList,
        EncodingOf,
        Flags,
        FuncArgs,
        Glue,
        IndexList,
        Keyword,
        OperandDict,
        OpRef,
        OptionalGroup,
        PredicateList,
        Ref,
        Refs,
        Region,
        ResultType,
        ResultTypeList,
        ScalarOf,
        Scope,
        ShapeOf,
        SymbolRef,
        TemplateParam,
        TypeOf,
        TypesOf,
    )

    fields: set[str] = set()
    for elem in elements:
        match elem:
            case Ref(field=f) | Refs(field=f):
                fields.add(f)
            case (
                Attr(field=f)
                | SymbolRef(field=f)
                | Flags(field=f)
                | OpRef(field=f)
                | TemplateParam(field=f)
            ):
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
            case OperandDict(operands=operands, names=names):
                fields.add(operands)
                fields.add(names)
            case OptionalGroup(elements=inner):
                fields |= _collect_format_fields(inner)
            case Scope(elements=inner):
                fields |= _collect_format_fields(inner)
            case ShapeOf(field=f) | ScalarOf(field=f) | EncodingOf(field=f):
                fields.add(f)
            case Keyword() | AttrDict() | Glue():
                pass
    return fields


def _validate_no_nested_scope(
    op_name: str,
    elements: tuple[FormatElement, ...],
    depth: int = 0,
) -> None:
    """Validate that Scope elements are not nested."""
    from loom.assembly import OptionalGroup, Scope

    for elem in elements:
        if isinstance(elem, Scope):
            if depth > 0:
                raise ValueError(
                    f"Op '{op_name}': nested Scope is not supported. "
                    f"Each op may have at most one Scope level."
                )
            _validate_no_nested_scope(op_name, elem.elements, depth + 1)
        elif isinstance(elem, OptionalGroup):
            _validate_no_nested_scope(op_name, elem.elements, depth)


def _validate_format_fields(
    op_name: str,
    format_elements: tuple[FormatElement, ...],
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    attrs: tuple[AttrDef, ...],
    regions: tuple[RegionDef, ...],
) -> None:
    """Validate that all fields in format elements are declared on the op."""
    _validate_no_nested_scope(op_name, format_elements)
    declared = (
        {o.name for o in operands}
        | {r.name for r in results}
        | {a.name for a in attrs}
        | {r.name for r in regions}
        | _IMPLICIT_FORMAT_FIELDS
    )
    referenced = _collect_format_fields(format_elements)
    unknown = referenced - declared
    if unknown:
        raise ValueError(
            f"Op '{op_name}': format references undeclared fields: "
            f"{sorted(unknown)}. Declared: {sorted(declared)}"
        )


# ============================================================================
# Effect validation
# ============================================================================


def _validate_effects(
    op_name: str,
    effects: tuple[Effect, ...],
    operands: tuple[Operand, ...],
    traits: tuple[Trait, ...],
) -> None:
    """Validate memory effect declarations against operands and traits."""
    trait_names = {t.name for t in traits}
    operand_map = {o.name: o for o in operands}

    # PURE and effects are mutually exclusive.
    if "Pure" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both traits=[PURE] and effects=["
            f"{', '.join(repr(e) for e in effects)}]. "
            f"An op with memory effects cannot be pure."
        )

    # UNKNOWN_EFFECTS and explicit effects are mutually exclusive.
    if "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both UNKNOWN_EFFECTS and explicit "
            f"effects. If the effects are known, declare them; if unknown, "
            f"use UNKNOWN_EFFECTS alone."
        )
    if "Hint" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and explicit effects. "
            f"A hint is not a semantic memory effect; attach policy to the "
            f"real memory op instead."
        )

    for effect in effects:
        # Effect must reference an existing operand.
        if effect.operand not in operand_map:
            raise ValueError(
                f"Op '{op_name}': effect references operand "
                f"'{effect.operand}' which is not declared. "
                f"Declared operands: {sorted(operand_map.keys())}"
            )

        # Referenced operand must have a resource-like type constraint.
        operand = operand_map[effect.operand]
        if operand.type_constraint not in _RESOURCE_TYPE_CONSTRAINTS:
            allowed = sorted(c.value for c in _RESOURCE_TYPE_CONSTRAINTS)
            raise ValueError(
                f"Op '{op_name}': effect on operand "
                f"'{effect.operand}' with type constraint "
                f"'{operand.type_constraint.value}' is not "
                f"allowed. Effects may only reference "
                f"resource-typed operands "
                f"({', '.join(allowed)})."
            )


def _validate_no_effect_conflicts(
    op_name: str,
    traits: tuple[Trait, ...],
) -> None:
    """Validate trait-only ops for effect-related conflicts."""
    trait_names = {t.name for t in traits}
    if "Pure" in trait_names and "NonDeterministic" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and NON_DETERMINISTIC. "
            f"A pure op is deterministic by definition."
        )
    if "Pure" in trait_names and "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and UNKNOWN_EFFECTS. "
            f"A pure op has no effects."
        )
    if "Pure" in trait_names and "UniqueIdentity" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and UNIQUE_IDENTITY. "
            f"A pure op can be CSE'd; a unique identity op cannot."
        )
    if "Hint" in trait_names and "Pure" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and PURE. "
            f"Hints are semantically discardable but intentionally preserved "
            f"until an explicit strip pass."
        )
    if "Hint" in trait_names and "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and UNKNOWN_EFFECTS. "
            f"Hints are not semantic effects."
        )
    if "Hint" in trait_names and "NonDeterministic" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and NON_DETERMINISTIC. "
            f"Hints do not produce observable values."
        )


# ============================================================================
# Interfaces
# ============================================================================


class FuncLikeInterface(NamedTuple):
    """Interface for function-like ops (def, decl, template, ukernel).

    Each field is the name of an attr or region on the op, or None if
    the op doesn't have that field. The generator resolves names to
    attr/region indices and emits a loom_func_like_vtable_t in .rodata.
    """

    # Symbol ref attr that names this function (required).
    callee: str
    # Visibility enum attr (e.g., public). None if not applicable.
    visibility: str | None = None
    # Calling convention enum attr. None if not applicable.
    cc: str | None = None
    # Purity enum attr. None if not applicable.
    purity: str | None = None
    # Predicate list attr for where-clause constraints. None if absent.
    predicates: str | None = None
    # Region name for the function body. None for bodyless ops that
    # only declare a signature without providing an implementation.
    body: str | None = None
    # String attr naming the abstract op this function implements
    # (for template/ukernel dispatch). None for def/decl.
    implements: str | None = None
    # I64 attr for dispatch priority among competing implementations.
    # None for def/decl.
    priority: str | None = None
    # When True, function arguments are stored as the op's operands
    # rather than as block arguments of the body region. Used for
    # ops that have a signature but no body (the parser stores parsed
    # FUNC_ARGS as operands when no REGION follows).
    args_as_operands: bool = False


class LoopLikeInterface(NamedTuple):
    """Interface for loop-like ops that iterate a body region.

    Each field is the name of a region, an implicit block argument,
    or an operand on the op (or None where applicable). The generator
    resolves names to indices and emits a loom_loop_like_vtable_t in
    .rodata. Used by LICM, loop-invariant sinking, trip count
    analysis, and loop transformation passes.
    """

    # Region name for the primary loop body. For scf.for this is the
    # one body region; for scf.while this is the "after" region that
    # contains the iteration body.
    body: str
    # Operand name where loop-carried state begins. For scf.for this
    # is the iter_args variadic operand following lower_bound,
    # upper_bound, step; for scf.while iter_args are the only
    # operands and this names that variadic.
    iter_args: str
    # Implicit block argument name for the induction variable on the
    # body region's entry block. None for loops without an IV (while
    # loops). For scf.for this is "iv" — the implicit_args entry on
    # the body region.
    iv: str | None = None
    # Region name for the condition region, if the loop has one
    # separate from the body. For scf.for this is None; for scf.while
    # this is the "before" region that computes the loop condition.
    condition_region: str | None = None


class RegionBranchInterface(NamedTuple):
    """Interface for ops whose regions are mutually-exclusive
    conditional branches of a single decision.

    Implementing this interface declares that every region of the op
    is an alternative of the same decision (then/else for scf.if,
    case/default for scf.switch). Ops with iterating body regions do
    NOT implement this interface — they implement LoopLikeInterface
    instead.

    The generator resolves the selector operand name to its index
    and emits a loom_region_branch_vtable_t in .rodata.
    """

    # Operand name that drives the branch decision. For scf.if this
    # is the i1 condition; for scf.switch this is the index selector.
    selector: str


# ============================================================================
# Op declaration
# ============================================================================


@dataclass(frozen=True, slots=True)
class Op:
    """A complete operation declaration.

    This is the core type of the DSL. Each Op instance fully describes
    an operation's interface: what it takes, what it produces, how it
    looks in textual assembly, what constraints it enforces, and its
    structural properties.

    name: Full dotted name as it appears in text format.
          Examples: "scalar.addi", "tile.contract", "func.def"
    group: Dialect this op belongs to (for organization/docs).
    doc: Human-readable documentation string.
    operands: List of Operand descriptors.
    results: List of Result descriptors.
    attrs: List of AttrDef descriptors.
    regions: List of RegionDef descriptors.
    constraints: List of Constraint instances.
    traits: List of Trait instances.
    format: Format element list describing textual assembly.
    examples: List of example IR strings for documentation.

    The format field is the critical connection point. It drives:
      - The text printer (walks elements, emits tokens).
      - The text parser (walks elements, consumes tokens).
      - Builder parameter ordering (format order = parameter order).
      - C code generation (same elements drive C printer/parser).
    """

    name: str
    group: Dialect | None = None
    doc: str = ""
    operands: tuple[Operand, ...] = ()
    results: tuple[Result | TiedResult, ...] = ()
    attrs: tuple[AttrDef, ...] = ()
    regions: tuple[RegionDef, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    traits: tuple[Trait, ...] = ()
    effects: tuple[Effect, ...] = ()
    canonicalize: str = ""  # C function name for canonicalization, or "".
    effective_traits: str = (
        ""  # C function name for per-instance trait computation, or "".
    )
    fold: str = ""  # C function name for fold/transfer, or "".
    verify: str = ""  # C function name for op-specific verification, or "".
    interfaces: tuple[
        Any, ...
    ] = ()  # Interface implementations (FuncLikeInterface, etc.).
    format: tuple[FormatElement, ...] = ()
    examples: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        group: Dialect | None = None,
        doc: str = "",
        operands: list[Operand] | tuple[Operand, ...] = (),
        results: list[Result | TiedResult] | tuple[Result | TiedResult, ...] = (),
        attrs: list[AttrDef] | tuple[AttrDef, ...] = (),
        regions: list[RegionDef] | tuple[RegionDef, ...] = (),
        constraints: list[Constraint] | tuple[Constraint, ...] = (),
        traits: list[Trait] | tuple[Trait, ...] = (),
        effects: list[Effect] | tuple[Effect, ...] = (),
        canonicalize: str = "",
        effective_traits: str = "",
        fold: str = "",
        verify: str = "",
        interfaces: list[Any] | tuple[Any, ...] = (),
        format: list[FormatElement] | tuple[FormatElement, ...] = (),
        examples: list[str] | tuple[str, ...] = (),
    ) -> None:
        # Accept lists for ergonomics, store as tuples for immutability.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "doc", doc)
        frozen_operands = tuple(operands)
        frozen_results = tuple(results)
        frozen_attrs = tuple(attrs)
        frozen_regions = tuple(regions)
        frozen_effects = tuple(effects)
        frozen_format = tuple(format)
        object.__setattr__(self, "operands", frozen_operands)
        object.__setattr__(self, "results", frozen_results)
        object.__setattr__(self, "attrs", frozen_attrs)
        object.__setattr__(self, "regions", frozen_regions)
        object.__setattr__(self, "constraints", tuple(constraints))
        object.__setattr__(self, "traits", tuple(traits))
        object.__setattr__(self, "effects", frozen_effects)
        object.__setattr__(self, "canonicalize", canonicalize)
        object.__setattr__(self, "effective_traits", effective_traits)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "verify", verify)
        object.__setattr__(self, "interfaces", tuple(interfaces))
        object.__setattr__(self, "format", frozen_format)
        object.__setattr__(self, "examples", tuple(examples))
        # Validate memory effect declarations.
        if frozen_effects:
            _validate_effects(name, frozen_effects, frozen_operands, tuple(traits))
        else:
            _validate_no_effect_conflicts(name, tuple(traits))
        # Validate that format elements reference declared fields.
        if frozen_format:
            _validate_format_fields(
                name,
                frozen_format,
                frozen_operands,
                frozen_results,
                frozen_attrs,
                frozen_regions,
            )

    def __repr__(self) -> str:
        return f"Op({self.name!r})"

    # --- Lookup helpers ---

    def operand(self, name: str) -> Operand | None:
        """Find an operand by name."""
        for op in self.operands:
            if op.name == name:
                return op
        return None

    def result(self, name: str) -> Result | TiedResult | None:
        """Find a result by name."""
        for r in self.results:
            if r.name == name:
                return r
        return None

    def attr(self, name: str) -> AttrDef | None:
        """Find an attribute by name."""
        for a in self.attrs:
            if a.name == name:
                return a
        return None

    def region(self, name: str) -> RegionDef | None:
        """Find a region by name."""
        for r in self.regions:
            if r.name == name:
                return r
        return None

    def has_trait(self, trait_name: str) -> bool:
        """Check if this op has a trait by name."""
        return any(t.name == trait_name for t in self.traits)

    @property
    def is_pure(self) -> bool:
        """True if the op has no memory effects and is deterministic.

        An op is pure if it explicitly declares traits=[PURE], or if it
        has no effects, no ALLOCATES results, and no HINT,
        NON_DETERMINISTIC, UNKNOWN_EFFECTS, or UNIQUE_IDENTITY traits.
        """
        if self.has_trait("Pure"):
            return True
        if self.effects:
            return False
        if any(r.allocates for r in self.results):
            return False
        if self.has_trait("UniqueIdentity"):
            return False
        if self.has_trait("Hint"):
            return False
        if self.has_trait("NonDeterministic") or self.has_trait("UnknownEffects"):
            return False
        return True

    @property
    def is_terminator(self) -> bool:
        return self.has_trait("Terminator")

    @property
    def is_commutative(self) -> bool:
        return self.has_trait("Commutative")

    @property
    def namespace(self) -> str:
        """The namespace prefix (everything before the last dot)."""
        dot = self.name.rfind(".")
        return self.name[:dot] if dot >= 0 else ""

    @property
    def short_name(self) -> str:
        """The short name (everything after the last dot)."""
        dot = self.name.rfind(".")
        return self.name[dot + 1 :] if dot >= 0 else self.name


# ============================================================================
# Op declaration helpers
# ============================================================================
# These reduce boilerplate for common op patterns. They produce Op
# instances — no class hierarchies, no metaclasses.


def binary_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    doc: str,
    commutative: bool = False,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a binary op: (lhs, rhs) -> result, all same type.

    Format: %r = name %a, %b : type
    Or with flags: %r = name<flag1|flag2> %a, %b : type
    """
    from loom.assembly import COLON, COMMA, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_traits = [PURE]
    if commutative:
        op_traits.append(COMMUTATIVE)
    if traits:
        op_traits.extend(traits)

    op_attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend([Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")])

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[
            Operand("lhs", type_constraint),
            Operand("rhs", type_constraint),
        ],
        results=[Result("result", type_constraint)],
        attrs=op_attrs,
        constraints=[SameType("lhs", "rhs", "result")],
        traits=op_traits,
        format=fmt,
        **kwargs,
    )


def unary_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    doc: str,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a unary op: (input) -> result, same type.

    Format: %r = name %x : type
    Or with flags: %r = name<flag1|flag2> %x : type
    """
    from loom.assembly import COLON, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_traits = [PURE]
    if traits:
        op_traits.extend(traits)

    op_attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend([Ref("input"), COLON, TypeOf("result")])

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[Operand("input", type_constraint)],
        results=[Result("result", type_constraint)],
        attrs=op_attrs,
        constraints=[SameType("input", "result")],
        traits=op_traits,
        format=fmt,
        **kwargs,
    )


def cast_op(
    name: str,
    *,
    group: Dialect,
    from_constraint: TypeConstraint,
    to_constraint: TypeConstraint,
    doc: str,
    **kwargs: Any,
) -> Op:
    """Declare a type-casting op: (input) -> result, different types.

    Format: %r = name %x : input_type to result_type
    """
    from loom.assembly import COLON, Ref, TypeOf, kw

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[Operand("input", from_constraint)],
        results=[Result("result", to_constraint)],
        traits=[PURE],
        format=[
            Ref("input"),
            COLON,
            TypeOf("input"),
            kw("to"),
            TypeOf("result"),
        ],
        **kwargs,
    )


def comparison_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    predicates: EnumDef,
    doc: str,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a comparison op: (predicate, lhs, rhs) -> i1.

    Format: %r = name pred, %a, %b : operand_type
    Or with flags: %r = name<flag1|flag2> pred, %a, %b : operand_type

    Operands must have matching types (both satisfy type_constraint).
    The result is always i1 (integer predicate). The format prints the
    operand type after ':', and the result type (i1) is inferred.
    The builder takes both operand_type and result_type so the caller
    can construct the i1 result explicitly.
    """
    from loom.assembly import COLON, COMMA, Attr, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_attrs: list[AttrDef] = [AttrDef("predicate", "enum", enum_def=predicates)]
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend(
        [
            Attr("predicate"),
            COMMA,
            Ref("lhs"),
            COMMA,
            Ref("rhs"),
            COLON,
            TypeOf("lhs"),
        ]
    )

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[
            Operand("lhs", type_constraint),
            Operand("rhs", type_constraint),
        ],
        results=[Result("result", I1)],
        attrs=op_attrs,
        constraints=[SameType("lhs", "rhs")],
        traits=[PURE],
        format=fmt,
        **kwargs,
    )
