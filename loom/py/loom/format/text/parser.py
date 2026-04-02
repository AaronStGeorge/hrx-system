# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Format-element-driven parser for loom IR text format.

Mirrors the printer: walks an op's format element list consuming
tokens instead of emitting them. Types are parsed through a type
registry (TypeDef declarations), not hardcoded tables.

Architecture (bottom to top):
  1. NameScope — SSA name → value ID mapping with parent chain
  2. Type parser — registry-driven, walks TypeDef format specs
  3. Format walk — mirrors printer's _walk_format, consumes tokens
  4. Structure parser — module/function/block/op orchestration
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, unique
from typing import Any

from loom.assembly import (
    Attr,
    AttrDict,
    BindingList,
    Flags,
    FormatElement,
    FuncArgs,
    Glue,
    IndexList,
    Keyword,
    OpRef,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypeOf,
    TypesOf,
)
from loom.assembly import (
    Region as RegionFmt,
)
from loom.dsl import (
    AttrDef,
    Op,
    ShapeParam,
    TypeDef,
)
from loom.fields import FieldKind, FieldLayout, compute_layout
from loom.format.text.tokenizer import (
    ParseError,
    SourceLocation,
    Token,
    Tokenizer,
    TokenKind,
)
from loom.ir import (
    ENCODING_TYPE,
    GROUP_SCOPE_BY_NAME,
    NONE_TYPE,
    PREDICATE_KINDS,
    SYMBOL_FLAG_IMPORT,
    SYMBOL_FLAG_PUBLIC,
    Block,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    FileLocation,
    FunctionType,
    FusedLocation,
    GroupType,
    Module,
    OpaqueLocation,
    Operation,
    PlaceholderType,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    Symbol,
    SymbolKind,
    Type,
    TypeKind,
    Value,
    binding_element_type,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)

__all__ = [
    "ParseError",
    "Parser",
    "NameScope",
    "parse_type_string",
]


# ============================================================================
# Parser-level state (set by Parser during parse(), read by module-level fns)
# ============================================================================

# These are set by Parser.parse() and read by _parse_encoding() during type
# parsing. This avoids threading alias/encoding state through every function
# in the call chain. Reset to None/empty between parses.
_CURRENT_ALIASES: dict[str, EncodingInstance] | None = None
_CURRENT_KNOWN_ENCODINGS: set[str] | None = None


# ============================================================================
# Scalar type name table (fixed, finite set — not pluggable)
# ============================================================================

_SCALAR_NAMES: dict[str, ScalarTypeKind] = {
    "index": ScalarTypeKind.INDEX,
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


# Maps symbol-defining op names to their SymbolKind. Used when registering
# symbols after parsing top-level ops. Ops not in this map (e.g., test.func)
# default to FUNC_DEF.
_OP_NAME_TO_SYMBOL_KIND: dict[str, SymbolKind] = {
    "func.def": SymbolKind.FUNC_DEF,
    "func.decl": SymbolKind.FUNC_DECL,
    "func.template": SymbolKind.FUNC_TEMPLATE,
    "func.ukernel": SymbolKind.FUNC_UKERNEL,
}


# ============================================================================
# Name scope
# ============================================================================


class NameScope:
    """Maps SSA names to value IDs during parsing.

    Parent-chain for nested regions: inner scopes can see outer
    names, outer scopes cannot see inner names after the region ends.
    """

    __slots__ = ("_names", "_parent")

    def __init__(self, parent: NameScope | None = None) -> None:
        self._names: dict[str, int] = {}
        self._parent = parent

    def define(self, name: str, value_id: int) -> None:
        """Define a new SSA name in this scope."""
        if name in self._names:
            raise ValueError(
                f"SSA name '%{name}' already defined in this scope "
                f"(existing value ID {self._names[name]}, "
                f"new value ID {value_id})"
            )
        self._names[name] = value_id

    def lookup(self, name: str) -> int:
        """Look up an SSA name, searching parent scopes."""
        if name in self._names:
            return self._names[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise KeyError(f"undefined SSA value '%{name}'")

    def push(self) -> NameScope:
        """Create a child scope for entering a region."""
        return NameScope(parent=self)


# ============================================================================
# Type parse mode
# ============================================================================


@unique
class TypeParseMode(Enum):
    """Context for how dynamic dim references are resolved."""

    BODY = "body"  # Op body: [%M] looks up existing values.
    SIGNATURE = "signature"  # Function signature: [%M] creates placeholders.


# ============================================================================
# Type reference resolution
# ============================================================================


def _resolve_type_value(
    name: str,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    token: Token,
    filename: str,
) -> int:
    """Resolve an SSA value reference in a type context.

    |name| is the bare name (without '%' sigil). In SIGNATURE mode,
    creates a PlaceholderType value if not found. In BODY mode, raises
    ParseError for undefined names.
    """
    try:
        return scope.lookup(name)
    except KeyError:
        pass

    if mode == TypeParseMode.SIGNATURE:
        value_id = module.add_value(Value(name=name, type=PlaceholderType()))
        scope.define(name, value_id)
        return value_id

    raise ParseError(
        f"undefined SSA value '%{name}'",
        token.location,
        filename,
    )


def _parse_dim_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    filename: str,
) -> tuple[StaticDim | DynamicDim, int | None]:
    """Parse a single dimension: INTEGER (static) or [SSA_VALUE] (dynamic).

    Returns (dim, binding_value_id). binding_value_id is None for static dims.
    """
    token = tokenizer.peek()
    if token.kind == TokenKind.INTEGER:
        tokenizer.next()
        return StaticDim(int(token.text)), None
    if token.kind == TokenKind.LBRACKET:
        tokenizer.next()  # consume [
        name_token = tokenizer.expect(TokenKind.SSA_VALUE)
        value_id = _resolve_type_value(
            name_token.text, scope, module, mode, name_token, filename
        )
        tokenizer.expect(TokenKind.RBRACKET)
        return DynamicDim(), value_id
    raise ParseError(
        f"expected integer or '[' for dimension, got {token.kind.name} {token.text!r}",
        token.location,
        filename,
    )


def _parse_type_encoding_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    filename: str,
) -> tuple[EncodingInstance | DynamicEncoding | None, int]:
    """Parse a type encoding after the comma in a shaped type.

    Returns (encoding, encoding_value_id). encoding_value_id is -1
    for static encodings, or the SSA value ID for dynamic (%enc).
    """
    token = tokenizer.peek()
    if token.kind == TokenKind.SSA_VALUE:
        tokenizer.next()
        value_id = _resolve_type_value(token.text, scope, module, mode, token, filename)
        # Verify the referenced value has EncodingType.
        if value_id < len(module.values):
            value = module.values[value_id]
            if not isinstance(value.type, EncodingType | PlaceholderType):
                raise ParseError(
                    f"encoding reference '%{token.text}' has type "
                    f"'{value.type}', expected encoding",
                    token.location,
                    filename,
                )
        return DynamicEncoding(), value_id

    if token.kind == TokenKind.HASH_ATTR:
        tokenizer.next()
        # Reconstruct full text with '#' for alias lookup.
        full_text = f"#{token.text}"

        # Check alias table.
        if _CURRENT_ALIASES and full_text in _CURRENT_ALIASES:
            aliased = _CURRENT_ALIASES[full_text]
            instance = EncodingInstance(
                name=aliased.name, alias=full_text, params=aliased.params
            )
            module.add_encoding(instance)
            return instance, -1

        # Parse inline encoding: #name or #name<params>.
        name = token.text
        params: tuple[tuple[str, str], ...] = ()
        if tokenizer.try_consume(TokenKind.LANGLE):
            param_list: list[tuple[str, str]] = []
            while not tokenizer.at(TokenKind.RANGLE) and not tokenizer.at(
                TokenKind.EOF
            ):
                key_token = tokenizer.expect(TokenKind.BARE_IDENT)
                tokenizer.expect(TokenKind.EQUALS)
                value_token = tokenizer.next()
                param_list.append((key_token.text, value_token.text))
                tokenizer.try_consume(TokenKind.COMMA)
            tokenizer.expect(TokenKind.RANGLE)
            params = tuple(param_list)

        # Validate encoding name if registry is available.
        if (
            _CURRENT_KNOWN_ENCODINGS is not None
            and name not in _CURRENT_KNOWN_ENCODINGS
        ):
            raise ParseError(
                f"unknown encoding '{name}'. "
                f"Known encodings: {sorted(_CURRENT_KNOWN_ENCODINGS)}",
                token.location,
                filename,
            )

        instance = EncodingInstance(name=name, params=params)
        module.add_encoding(instance)
        return instance, -1

    if token.kind == TokenKind.RESULT_ORDINAL:
        tokenizer.next()
        raise ParseError(
            "encoding ordinals are no longer supported; use %name",
            token.location,
            filename,
        )

    raise ParseError(
        f"expected encoding (%name or #name), got {token.kind.name} {token.text!r}",
        token.location,
        filename,
    )


# ============================================================================
# Type parser — registry-driven
# ============================================================================


def _is_type_start(token: Token, type_registry: dict[str, TypeDef]) -> bool:
    """Check if a token could start a type expression."""
    if token.kind == TokenKind.BARE_IDENT:
        if token.text in _SCALAR_NAMES:
            return True
        if token.text == "encoding":
            return True
        if token.text in type_registry:
            return True
        return False
    if token.kind == TokenKind.OP_NAME:
        # Dotted type names like hal.buffer, vm.ref.
        return token.text in type_registry
    if token.kind == TokenKind.LPAREN:
        return True  # Function type.
    return False


def parse_type_from_tokens(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode = TypeParseMode.BODY,
) -> tuple[Type, dict[int, int]]:
    """Parse a type from the token stream.

    Dispatches through the type registry for structured types.
    Returns (type, dim_bindings).
    """
    token = tokenizer.peek()

    # Scalar type keyword?
    if token.kind == TokenKind.BARE_IDENT:
        scalar_kind = _SCALAR_NAMES.get(token.text)
        if scalar_kind is not None:
            tokenizer.next()
            return ScalarType(scalar_kind), {}

    # Encoding type keyword?
    if token.kind == TokenKind.BARE_IDENT and token.text == "encoding":
        tokenizer.next()
        return ENCODING_TYPE, {}

    # Registered type (BARE_IDENT like "tile" or OP_NAME like "hal.buffer")?
    if token.kind in (TokenKind.BARE_IDENT, TokenKind.OP_NAME):
        type_def = type_registry.get(token.text)
        if type_def is not None:
            tokenizer.next()
            if type_def.is_opaque:
                return DialectType(type_def.name), {}
            tokenizer.expect(TokenKind.LANGLE)
            # Shaped types (tile, tensor, pool) parse from the token
            # stream using in_dim_list for 'x' separators. Other types
            # use the interior tokenizer approach.
            has_shape = any(isinstance(p, ShapeParam) for p in type_def.params)
            if has_shape:
                result = _parse_shaped_type_from_tokens(
                    type_def,
                    tokenizer,
                    scope,
                    module,
                    mode,
                )
                tokenizer.in_dim_list = False  # cleanup on error paths
                return result
            interior = tokenizer.scan_to_matching_angle_bracket()
            return _parse_type_interior(
                type_def,
                interior,
                scope,
                module,
                type_registry,
                mode,
                token.location,
                tokenizer._filename,
            )

    # Function type: (types) -> (types)
    if token.kind == TokenKind.LPAREN:
        return _parse_function_type(tokenizer, scope, module, type_registry, mode)

    raise ParseError(
        f"expected type, got {token.kind.name} {token.text!r}",
        token.location,
        tokenizer._filename,
    )


def parse_type_string(
    text: str,
    type_registry: dict[str, TypeDef] | None = None,
    scope: NameScope | None = None,
    module: Module | None = None,
    mode: TypeParseMode | None = None,
) -> tuple[Type, dict[int, int]]:
    """Parse a type from a string. Convenience for testing."""
    tokenizer = Tokenizer(text)
    if scope is None:
        scope = NameScope()
    if module is None:
        module = Module()
    if type_registry is None:
        from loom.builtin_types import ALL_BUILTIN_TYPES

        type_registry = {td.name: td for td in ALL_BUILTIN_TYPES}
    if mode is None:
        mode = TypeParseMode.BODY
    result = parse_type_from_tokens(tokenizer, scope, module, type_registry, mode)
    tokenizer.expect(TokenKind.EOF)
    return result


def _parse_function_type(
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
) -> tuple[FunctionType, dict[int, int]]:
    """Parse (arg_types) -> (result_types)."""
    tokenizer.expect(TokenKind.LPAREN)
    arg_types: list[Type] = []
    if not tokenizer.at(TokenKind.RPAREN):
        arg_type, _ = parse_type_from_tokens(
            tokenizer, scope, module, type_registry, mode
        )
        arg_types.append(arg_type)
        while tokenizer.try_consume(TokenKind.COMMA):
            arg_type, _ = parse_type_from_tokens(
                tokenizer, scope, module, type_registry, mode
            )
            arg_types.append(arg_type)
    tokenizer.expect(TokenKind.RPAREN)
    tokenizer.expect(TokenKind.ARROW)
    tokenizer.expect(TokenKind.LPAREN)
    result_types: list[Type] = []
    if not tokenizer.at(TokenKind.RPAREN):
        result_type, _ = parse_type_from_tokens(
            tokenizer, scope, module, type_registry, mode
        )
        result_types.append(result_type)
        while tokenizer.try_consume(TokenKind.COMMA):
            result_type, _ = parse_type_from_tokens(
                tokenizer, scope, module, type_registry, mode
            )
            result_types.append(result_type)
    tokenizer.expect(TokenKind.RPAREN)
    return FunctionType(tuple(arg_types), tuple(result_types)), {}


# ============================================================================
# Type interior parser — format-driven
# ============================================================================


def _expect_keyword(tokenizer: Tokenizer, text: str) -> None:
    """Consume a keyword token, handling both punctuation and words."""
    keyword_map = {
        ",": TokenKind.COMMA,
        ":": TokenKind.COLON,
        "->": TokenKind.ARROW,
        "=": TokenKind.EQUALS,
        "(": TokenKind.LPAREN,
        ")": TokenKind.RPAREN,
        "[": TokenKind.LBRACKET,
        "]": TokenKind.RBRACKET,
        "{": TokenKind.LBRACE,
        "}": TokenKind.RBRACE,
        "<": TokenKind.LANGLE,
        ">": TokenKind.RANGLE,
    }
    kind = keyword_map.get(text)
    if kind is not None:
        tokenizer.expect(kind)
    else:
        tokenizer.expect(TokenKind.BARE_IDENT, text)


def _type_optional_present(
    tokenizer: Tokenizer,
    inner_elements: tuple[FormatElement, ...],
    type_registry: dict[str, TypeDef],
) -> bool:
    """Peek to decide if an OptionalGroup is present in a type interior."""
    if not inner_elements:
        return False
    first = inner_elements[0]
    match first:
        case Keyword(text=","):
            result: bool = tokenizer.at(TokenKind.COMMA)
            return result
        case Keyword(text="->"):
            result = tokenizer.at(TokenKind.ARROW)
            return result
        case Keyword(text=text):
            result = tokenizer.at(TokenKind.BARE_IDENT, text)
            return result
        case TypeOf() | TypesOf():
            return _is_type_start(tokenizer.peek(), type_registry)
        case _:
            return not tokenizer.at(TokenKind.EOF)


def _parse_type_interior(
    type_def: TypeDef,
    interior: str,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
    location: SourceLocation,
    filename: str,
) -> tuple[Type, dict[int, int]]:
    """Parse the interior of a parameterized type.

    The type_def's format spec drives the parse. For built-in shaped
    types (tile, tensor), the format contains ShapeOf, ScalarOf,
    EncodingOf elements. For dialect types (vm.ref), it contains
    TypeOf elements.
    """
    # For shaped types, we use the character-level sub-parser since
    # the interior (4x4xf32, [%M]x4xf32, etc.) doesn't tokenize
    # cleanly with the main tokenizer.
    has_shape = any(isinstance(p, ShapeParam) for p in type_def.params)
    if has_shape:
        return _parse_shaped_interior(
            type_def, interior, scope, module, type_registry, mode, location, filename
        )

    # For non-shaped types, tokenize the interior and walk the format.
    interior_tokenizer = Tokenizer(interior, filename)
    parsed_params: list[Type] = []
    parsed_attrs: dict[str, Any] = {}

    def walk_type_format(elements: tuple[FormatElement, ...]) -> None:
        """Walk type format elements, consuming tokens."""
        for element in elements:
            match element:
                case TypeOf():
                    param_type, _ = parse_type_from_tokens(
                        interior_tokenizer, scope, module, type_registry, mode
                    )
                    parsed_params.append(param_type)
                case TypesOf():
                    # Comma-separated types.
                    if _is_type_start(interior_tokenizer.peek(), type_registry):
                        t, _ = parse_type_from_tokens(
                            interior_tokenizer, scope, module, type_registry, mode
                        )
                        parsed_params.append(t)
                        while interior_tokenizer.try_consume(TokenKind.COMMA):
                            t, _ = parse_type_from_tokens(
                                interior_tokenizer, scope, module, type_registry, mode
                            )
                            parsed_params.append(t)
                case Attr(field=name):
                    tok = interior_tokenizer.expect(TokenKind.BARE_IDENT)
                    parsed_attrs[name] = tok.text
                case SymbolRef(field=name):
                    tok = interior_tokenizer.expect(TokenKind.SYMBOL)
                    parsed_attrs[name] = tok.text
                case Keyword(text=text):
                    _expect_keyword(interior_tokenizer, text)
                case OptionalGroup(elements=inner, anchor=_anchor):
                    if _type_optional_present(interior_tokenizer, inner, type_registry):
                        walk_type_format(inner)
                case Glue():
                    pass
                case Ref(field=name):
                    tok = interior_tokenizer.expect(TokenKind.SSA_VALUE)
                    parsed_attrs[name] = tok.text
                case Refs(field=name):
                    if interior_tokenizer.at(TokenKind.SSA_VALUE):
                        refs = [interior_tokenizer.next().text]
                        while interior_tokenizer.try_consume(TokenKind.COMMA):
                            refs.append(
                                interior_tokenizer.expect(TokenKind.SSA_VALUE).text
                            )
                        parsed_attrs[name] = refs

    walk_type_format(type_def.format)

    # Dispatch based on ir_kind.
    if type_def.ir_kind == "group" and "scope" in parsed_attrs:
        scope_name = parsed_attrs["scope"]
        group_scope = GROUP_SCOPE_BY_NAME.get(scope_name)
        if group_scope is None:
            raise ValueError(f"unknown group scope '{scope_name}'")
        return GroupType(group_scope), {}

    return DialectType(type_def.name, tuple(parsed_params)), {}


def _parse_shaped_type_from_tokens(
    type_def: TypeDef,
    tokenizer: Tokenizer,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
) -> tuple[ShapedType | PoolType, dict[int, int]]:
    """Parse a shaped type (tile, tensor, pool) from the token stream.

    Called after LANGLE has been consumed. Consumes tokens through
    RANGLE. Uses in_dim_list on the tokenizer to handle 'x' as a
    dimension separator.
    """
    filename = tokenizer._filename

    # Pool: single dim, no element type, no encoding.
    if type_def.ir_kind == "pool":
        token = tokenizer.peek()
        if token.kind not in (TokenKind.INTEGER, TokenKind.LBRACKET):
            raise ParseError(
                f"expected integer or '[' for pool dim, "
                f"got {token.kind.name} {token.text!r}",
                token.location,
                filename,
            )
        dim, binding_id = _parse_dim_from_tokens(
            tokenizer, scope, module, mode, filename
        )
        tokenizer.expect(TokenKind.RANGLE)
        dim_bindings: dict[int, int] = {}
        if binding_id is not None:
            dim_bindings[0] = binding_id
        return PoolType(block_size=dim), dim_bindings

    # Shaped type (tile, tensor).
    _IR_KIND_TO_TYPE_KIND = {"tile": TypeKind.TILE, "tensor": TypeKind.TENSOR}
    type_kind = _IR_KIND_TO_TYPE_KIND.get(type_def.ir_kind)
    if type_kind is None:
        token = tokenizer.peek()
        raise ParseError(
            f"TypeDef '{type_def.name}' has ShapeParam but ir_kind "
            f"'{type_def.ir_kind}' is not a recognized shaped type kind.",
            token.location,
            filename,
        )

    # Parse dimensions. in_dim_list must be true from the start so
    # that '0x' in '0xf32' is scanned as INTEGER(0) + DIM_X(x) +
    # BARE_IDENT(f32), not as hex INTEGER(0xf32).
    dims: list[StaticDim | DynamicDim] = []
    dim_bindings = {}
    tokenizer.in_dim_list = True
    token = tokenizer.peek()
    if token.kind in (TokenKind.INTEGER, TokenKind.LBRACKET):
        dim, binding_id = _parse_dim_from_tokens(
            tokenizer, scope, module, mode, filename
        )
        dims.append(dim)
        if binding_id is not None:
            dim_bindings[len(dims) - 1] = binding_id

        while tokenizer.at(TokenKind.DIM_X):
            tokenizer.next()  # consume 'x'
            tokenizer.in_dim_list = False
            token = tokenizer.peek()
            if token.kind not in (TokenKind.INTEGER, TokenKind.LBRACKET):
                break  # element type follows
            tokenizer.in_dim_list = True
            dim, binding_id = _parse_dim_from_tokens(
                tokenizer, scope, module, mode, filename
            )
            dims.append(dim)
            if binding_id is not None:
                dim_bindings[len(dims) - 1] = binding_id
    else:
        # Rank 0 — no dims. Clear in_dim_list before element type.
        tokenizer.in_dim_list = False

    # Parse element type (in_dim_list is false here).
    element_token = tokenizer.expect(TokenKind.BARE_IDENT)
    scalar_kind = _SCALAR_NAMES.get(element_token.text)
    if scalar_kind is None:
        raise ParseError(
            f"unknown element type '{element_token.text}' in shaped type",
            element_token.location,
            filename,
        )
    element_type = ScalarType(scalar_kind)

    # Parse optional encoding.
    encoding: EncodingInstance | DynamicEncoding | None = None
    encoding_binding = -1
    if tokenizer.try_consume(TokenKind.COMMA):
        encoding, encoding_binding = _parse_type_encoding_from_tokens(
            tokenizer, scope, module, mode, filename
        )
    if encoding_binding >= 0:
        dim_bindings[-1] = encoding_binding

    tokenizer.expect(TokenKind.RANGLE)

    shaped = ShapedType(
        type_kind=type_kind,
        element_type=element_type,
        dims=tuple(dims),
        encoding=encoding,
    )
    return shaped, dim_bindings


# The old character-level sub-parsers below are no longer called from
# the main type parsing path. They are retained temporarily for
# reference until the migration is verified complete.


def _parse_shaped_interior(
    type_def: TypeDef,
    interior: str,
    scope: NameScope,
    module: Module,
    type_registry: dict[str, TypeDef],
    mode: TypeParseMode,
    location: SourceLocation,
    filename: str,
) -> tuple[ShapedType | PoolType, dict[int, int]]:
    """Parse the interior of tile<...> or tensor<...>.

    Character-level sub-parser for: 4x4xf32, [%M]x4xf32,
    256xi8, #q8_0<block=32>, f32 (0-d), etc.

    DEPRECATED: Use _parse_shaped_type_from_tokens instead.
    """
    # Pool type: single dim, no element type, no encoding.
    if type_def.ir_kind == "pool":
        return _parse_pool_interior(interior, scope, module, mode, location, filename)

    _IR_KIND_TO_TYPE_KIND = {
        "tile": TypeKind.TILE,
        "tensor": TypeKind.TENSOR,
    }
    type_kind = _IR_KIND_TO_TYPE_KIND.get(type_def.ir_kind)
    if type_kind is None:
        raise ParseError(
            f"TypeDef '{type_def.name}' has ShapeParam but ir_kind "
            f"'{type_def.ir_kind}' is not a recognized shaped type kind. "
            f"Use ir_kind='tile' or ir_kind='tensor', or implement "
            f"DialectType-based shaped type support.",
            location,
            filename,
        )

    # Split shape+element from encoding on first ',' outside nested <>.
    shape_part, encoding_part = _split_on_comma(interior)

    # Parse shape+element.
    dims, element_type, dim_bindings = _parse_dims_and_element(
        shape_part.strip(), scope, module, mode, location, filename
    )

    # Parse encoding if present.
    encoding: EncodingInstance | DynamicEncoding | None = None
    if encoding_part is not None:
        encoding_text = encoding_part.strip()
        if encoding_text.startswith("%"):
            # SSA value reference — dynamic encoding bound at use site.
            enc_bare_name = encoding_text[1:]  # Strip % sigil.
            try:
                enc_value_id = scope.lookup(enc_bare_name)
            except KeyError:
                raise ParseError(
                    f"undefined encoding value '{encoding_text}' in type",
                    location,
                    filename,
                ) from None
            # Verify the referenced value has EncodingType.
            if enc_value_id < len(module.values):
                enc_value = module.values[enc_value_id]
                if not isinstance(enc_value.type, EncodingType):
                    raise ParseError(
                        f"encoding reference '{encoding_text}' has type "
                        f"'{enc_value.type}', expected encoding",
                        location,
                        filename,
                    )
            encoding = DynamicEncoding()
            # Use sentinel key -1 in dim_bindings for encoding binding.
            dim_bindings[-1] = enc_value_id
        else:
            encoding = _parse_encoding_instance(
                encoding_text,
                module,
                location,
                filename,
                aliases=_CURRENT_ALIASES,
                known_encodings=_CURRENT_KNOWN_ENCODINGS,
            )

    shaped = ShapedType(
        type_kind=type_kind,
        element_type=element_type,
        dims=tuple(dims),
        encoding=encoding,
    )
    return shaped, dim_bindings


def _split_on_comma(text: str) -> tuple[str, str | None]:
    """Split on first ',' not inside nested <>."""
    depth = 0
    for i, character in enumerate(text):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            return text[:i], text[i + 1 :]
    return text, None


def _parse_dims_and_element(
    text: str,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    location: SourceLocation,
    filename: str,
) -> tuple[list[StaticDim | DynamicDim], ScalarType, dict[int, int]]:
    """Parse '4x4xf32' or '[%M]x4xf32' or 'f32' (0-d)."""
    dim_bindings: dict[int, int] = {}
    segments = _split_on_x(text)

    if not segments:
        raise ParseError(f"empty shaped type interior: '{text}'", location, filename)

    # Last segment is the element type.
    element_text = segments[-1].strip()
    scalar_kind = _SCALAR_NAMES.get(element_text)
    if scalar_kind is None:
        raise ParseError(
            f"unknown element type '{element_text}' in shaped type", location, filename
        )
    element_type = ScalarType(scalar_kind)

    # Preceding segments are dims.
    dims: list[StaticDim | DynamicDim] = []
    for i, segment in enumerate(segments[:-1]):
        segment = segment.strip()
        if segment.startswith("[") and segment.endswith("]"):
            inner = segment[1:-1].strip()
            if inner.startswith("%"):
                bare_name = inner[1:]  # Strip % sigil for scope lookup.
                if mode == TypeParseMode.SIGNATURE:
                    # In signature mode, define a placeholder if not found.
                    try:
                        value_id = scope.lookup(bare_name)
                    except KeyError:
                        value_id = module.add_value(
                            Value(name=bare_name, type=PlaceholderType())
                        )
                        scope.define(bare_name, value_id)
                    dim_bindings[i] = value_id
                else:
                    # BODY mode: look up existing value.
                    try:
                        value_id = scope.lookup(bare_name)
                        dim_bindings[i] = value_id
                    except KeyError:
                        raise ParseError(
                            f"undefined dim name '{inner}' in type", location, filename
                        ) from None
            else:
                raise ParseError(f"invalid dynamic dim '{segment}'", location, filename)
            dims.append(DynamicDim())
        else:
            try:
                size = int(segment)
            except ValueError:
                raise ParseError(
                    f"invalid dimension '{segment}' in shaped type", location, filename
                ) from None
            dims.append(StaticDim(size))

    return dims, element_type, dim_bindings


def _split_on_x(text: str) -> list[str]:
    """Split shaped type interior on 'x' boundaries, respecting [...]."""
    segments: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for character in text:
        if character == "[":
            bracket_depth += 1
            current.append(character)
        elif character == "]":
            bracket_depth -= 1
            current.append(character)
        elif character == "x" and bracket_depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(character)
    if current:
        segments.append("".join(current))
    return segments


def _parse_encoding_instance(
    text: str,
    module: Module,
    location: SourceLocation,
    filename: str,
    aliases: dict[str, EncodingInstance] | None = None,
    known_encodings: set[str] | None = None,
) -> EncodingInstance:
    """Parse '#name' or '#name<key=val, ...>' encoding reference.

    Returns an EncodingInstance (self-describing, no indices needed).
    Also adds the instance to the module's encoding table for bytecode
    serialization.
    """
    text = text.strip()
    if not text.startswith("#"):
        raise ParseError(
            f"expected encoding starting with '#', got '{text}'", location, filename
        )

    # Check alias table first.
    if aliases and text in aliases:
        aliased = aliases[text]
        instance = EncodingInstance(
            name=aliased.name, alias=text, params=aliased.params
        )
        module.add_encoding(instance)
        return instance

    # Not an alias — parse directly.
    bare = text[1:]  # strip #
    name = bare
    params: tuple[tuple[str, str], ...] = ()
    angle_pos = bare.find("<")
    if angle_pos >= 0:
        name = bare[:angle_pos]
        params_text = bare[angle_pos + 1 :]
        if params_text.endswith(">"):
            params_text = params_text[:-1]
        params = _parse_encoding_params(params_text, location, filename)

    # Validate encoding name if registry is available.
    if known_encodings and name not in known_encodings:
        raise ParseError(
            f"unknown encoding '{name}'. Known encodings: {sorted(known_encodings)}",
            location,
            filename,
        )

    instance = EncodingInstance(name=name, params=params)
    module.add_encoding(instance)
    return instance


def _parse_encoding_params(
    text: str,
    location: SourceLocation,
    filename: str,
) -> tuple[tuple[str, str], ...]:
    """Parse 'key=val, key=val' parameter list."""
    if not text.strip():
        return ()
    params: list[tuple[str, str]] = []
    for entry in text.split(","):
        entry = entry.strip()
        if "=" not in entry:
            raise ParseError(
                f"invalid encoding parameter '{entry}' — expected 'key=value' format",
                location,
                filename,
            )
        key, value = entry.split("=", 1)
        params.append((key.strip(), value.strip()))
    return tuple(params)


def _parse_pool_interior(
    interior: str,
    scope: NameScope,
    module: Module,
    mode: TypeParseMode,
    location: SourceLocation,
    filename: str,
) -> tuple[PoolType, dict[int, int]]:
    """Parse the interior of pool<...>.

    The interior is a single dimension: a static integer or [%name].
    No element type, no encoding, no 'x' separator.
    """
    dim_bindings: dict[int, int] = {}
    text = interior.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner.startswith("%"):
            bare_name = inner[1:]  # Strip % sigil for scope lookup.
            if mode == TypeParseMode.SIGNATURE:
                # In signature mode, define a placeholder if not found.
                try:
                    value_id = scope.lookup(bare_name)
                except KeyError:
                    value_id = module.add_value(
                        Value(name=bare_name, type=PlaceholderType())
                    )
                    scope.define(bare_name, value_id)
                dim_bindings[0] = value_id
            else:
                # BODY mode: look up existing value.
                try:
                    value_id = scope.lookup(bare_name)
                    dim_bindings[0] = value_id
                except KeyError:
                    raise ParseError(
                        f"undefined dim name '{inner}' in pool type",
                        location,
                        filename,
                    ) from None
            return PoolType(block_size=DynamicDim()), dim_bindings
        raise ParseError(
            f"invalid pool dim '{text}' — expected [%name]",
            location,
            filename,
        )
    try:
        size = int(text)
    except ValueError:
        raise ParseError(
            f"invalid pool block size '{text}' — expected integer or [%name]",
            location,
            filename,
        ) from None
    return PoolType(block_size=StaticDim(size)), dim_bindings


# ============================================================================
# Parser (placeholder — format walk and structure parsing go here)
# ============================================================================


class ParsedFields:
    """Mutable accumulator filled during the format walk.

    Mirror of ResolvedFields: where ResolvedFields reads from an
    existing Operation, ParsedFields builds one.

    func_arg_ids holds value IDs created by FuncArgs parsing. After
    the format walk, they are either consumed as region entry-block
    pre-args (body ops) or moved to operand_ids (declaration ops).
    """

    __slots__ = (
        "operand_ids",
        "result_ids",
        "result_types",
        "result_bindings",
        "attributes",
        "regions",
        "tied_results",
        "implicit_values",
        "func_arg_ids",
    )

    def __init__(self) -> None:
        self.operand_ids: list[int] = []
        self.result_ids: list[int | None] = []
        self.result_types: list[Type] = []
        self.result_bindings: list[dict[int, int]] = []
        self.attributes: dict[str, Any] = {}
        self.regions: list[Region] = []
        self.tied_results: list[IRTiedResult] = []
        self.implicit_values: dict[str, int] = {}
        self.func_arg_ids: list[int] = []


class Parser:
    """Format-element-driven parser for loom IR text format.

    Usage:
        parser = Parser()
        parser.register_ops(ALL_TEST_OPS)
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(source_text)
    """

    def __init__(self) -> None:
        self._op_registry: dict[str, Op] = {}
        self._type_registry: dict[str, TypeDef] = {}
        self._layouts: dict[str, FieldLayout] = {}
        self._scope: NameScope = NameScope()
        self._module: Module = Module()
        self._tokenizer: Tokenizer = Tokenizer("")
        self._encoding_aliases: dict[str, EncodingInstance] = {}
        self._known_encodings: set[str] = set()
        self._reserved_result_names: list[str] = []
        self._reserved_result_ids: list[int] = []
        self._definition_scope_depth: int = 0

    def register_ops(self, ops: Sequence[Op]) -> None:
        """Register op declarations."""
        for op in ops:
            self._op_registry[op.name] = op

    def register_types(self, types: Sequence[TypeDef]) -> None:
        """Register type declarations."""
        for td in types:
            self._type_registry[td.name] = td

    def register_encodings(self, names: Sequence[str]) -> None:
        """Register known encoding names for validation."""
        self._known_encodings.update(names)

    # --- Top-level parsing ---

    def parse(self, source: str, filename: str = "<input>") -> Module:
        """Parse a complete .loom file into a Module.

        Handles:
          - Attribute aliases: #enc = #q8_0<block=32>
          - Function definitions and declarations
          - Top-level dispatch
        """
        global _CURRENT_ALIASES, _CURRENT_KNOWN_ENCODINGS
        self._tokenizer = Tokenizer(source, filename)
        self._module = Module()
        self._scope = NameScope()
        self._encoding_aliases = {}
        _CURRENT_ALIASES = self._encoding_aliases
        _CURRENT_KNOWN_ENCODINGS = (
            self._known_encodings if self._known_encodings else None
        )
        tok = self._tokenizer

        while not tok.at(TokenKind.EOF):
            # Attribute alias: #name = ...
            if tok.at(TokenKind.HASH_ATTR):
                self._parse_attribute_alias()
                continue

            # Top-level symbol-defining op: func.def, func.decl, test.func, etc.
            if tok.at(TokenKind.OP_NAME):
                op = self._parse_operation()
                self._register_symbol(op)
                continue

            raise ParseError(
                f"expected attribute alias or top-level op, "
                f"got {tok.peek().kind.name} {tok.peek().text!r}",
                tok.peek().location,
                tok._filename,
            )

        _CURRENT_ALIASES = None
        _CURRENT_KNOWN_ENCODINGS = None
        return self._module

    def _parse_attribute_alias(self) -> None:
        """Parse #alias = #encoding<params> at file level."""
        tok = self._tokenizer
        alias_tok = tok.expect(TokenKind.HASH_ATTR)
        alias_name = "#" + alias_tok.text  # re-add # for raw-text lookups
        tok.expect(TokenKind.EQUALS)

        # Parse the encoding value: #name or #name<params>
        enc_tok = tok.expect(TokenKind.HASH_ATTR)
        enc_name = enc_tok.text

        params: tuple[tuple[str, str], ...] = ()
        if tok.at(TokenKind.LANGLE):
            tok.next()
            interior = tok.scan_to_matching_angle_bracket()
            params = _parse_encoding_params(interior, enc_tok.location, tok._filename)

        instance = EncodingInstance(
            name=enc_name,
            encoding_kind=0,  # Resolved at link time.
            alias=alias_name,
            params=params,
        )
        self._encoding_aliases[alias_name] = instance

    def _register_symbol(self, op: Operation) -> None:
        """Register a top-level symbol-defining op in the module's symbol table.

        Called after _parse_operation() for any op appearing at module level.
        Extracts the symbol name, kind, and visibility from the op's attributes
        and adds a Symbol entry pointing to the defining op.
        """
        sym_name = op.attributes.get("callee", "")
        sym_kind = _OP_NAME_TO_SYMBOL_KIND.get(op.name, SymbolKind.FUNC_DEF)

        sym_flags = 0
        if op.attributes.get("visibility") == "public":
            sym_flags |= SYMBOL_FLAG_PUBLIC
        import_module = op.attributes.get("import_module", "")
        if import_module:
            sym_flags |= SYMBOL_FLAG_IMPORT

        symbol = Symbol(
            name=sym_name,
            kind=sym_kind,
            flags=sym_flags,
            op=op,
            source_module=import_module,
            source_symbol=op.attributes.get("import_symbol", ""),
        )
        self._module.add_symbol(symbol)

    def _parse_func_arg(self) -> tuple[str, Type, int]:
        """Parse one function argument: %name: type.

        Returns (name, type, value_id).
        """
        tok = self._tokenizer
        name_tok = tok.expect(TokenKind.SSA_VALUE)
        tok.expect(TokenKind.COLON)
        arg_type, all_bindings = parse_type_from_tokens(
            tok, self._scope, self._module, self._type_registry, TypeParseMode.SIGNATURE
        )
        # Extract bindings: dim_bindings are non-negative keys,
        # encoding_binding uses sentinel key -1.
        dim_bindings = {k: v for k, v in all_bindings.items() if k >= 0}
        encoding_binding = all_bindings.get(-1, -1)

        # If the name was already forward-referenced in another argument's
        # type, update the placeholder value.
        try:
            value_id = self._scope.lookup(name_tok.text)
            value = self._module.values[value_id]
            if not isinstance(value.type, PlaceholderType):
                raise ParseError(
                    f"SSA name '%{name_tok.text}' already defined",
                    name_tok.location,
                    tok._filename,
                )
            value.type = arg_type
            value.dim_bindings = dim_bindings
            value.encoding_binding = encoding_binding
        except KeyError:
            # First occurrence: define the argument value in scope.
            value_id = self._module.add_value(
                Value(
                    name=name_tok.text,
                    type=arg_type,
                    dim_bindings=dim_bindings,
                    encoding_binding=encoding_binding,
                )
            )
            self._scope.define(name_tok.text, value_id)
        return name_tok.text, arg_type, value_id

    def _layout(self, op_decl: Op) -> FieldLayout:
        """Get or compute the field layout for an op kind."""
        layout = self._layouts.get(op_decl.name)
        if layout is None:
            layout = compute_layout(op_decl)
            self._layouts[op_decl.name] = layout
        return layout

    # --- Op parsing ---

    def parse_operation_from_text(
        self,
        text: str,
        module: Module | None = None,
        scope: NameScope | None = None,
    ) -> Operation:
        """Parse a single operation from text. Convenience for testing."""
        self._tokenizer = Tokenizer(text)
        self._module = module if module is not None else Module()
        self._scope = scope if scope is not None else NameScope()
        op = self._parse_operation()
        return op

    def _parse_operation(self) -> Operation:
        """Parse one operation from the token stream.

        Handles the result list, op name lookup, format walk,
        and Operation construction.
        """
        tok = self._tokenizer
        start_loc = tok.current_location()

        # 1. Result list: %r = or %a, %b =
        result_names: list[str] = []
        if tok.at(TokenKind.SSA_VALUE):
            result_names.append(tok.next().text)
            while tok.try_consume(TokenKind.COMMA):
                result_names.append(tok.expect(TokenKind.SSA_VALUE).text)
            tok.expect(TokenKind.EQUALS)

        # 2. Op name.
        op_name_tok = tok.expect(TokenKind.OP_NAME)
        op_name = op_name_tok.text
        op_decl = self._op_registry.get(op_name)
        if op_decl is None:
            raise ParseError(
                f"unknown op '{op_name}'", op_name_tok.location, tok._filename
            )

        # 3. Pre-allocate result values so result type annotations can
        # reference them (e.g., tile<[%m]x[%k]xf32> where %m and %k
        # are co-results). The values start with NoneType and get
        # their real types assigned after the format walk. They are
        # NOT in the main scope — _parse_result_type pushes them
        # into a child scope only during result type parsing, so
        # operand/argument parsing cannot see them.
        parsed = ParsedFields()
        reserved_result_ids: list[int] = []
        if result_names:
            for name in result_names:
                value_id = self._module.add_value(Value(name=name, type=NONE_TYPE))
                reserved_result_ids.append(value_id)
        self._reserved_result_names = result_names
        self._reserved_result_ids = reserved_result_ids

        # 4. Walk format spec.
        self._walk_format(op_decl.format, op_decl, parsed)
        self._reserved_result_names = []
        self._reserved_result_ids = []

        # After format walk: resolve func-arg semantics.
        # For declaration-style ops (no body region), func args become operands.
        # For definition-style ops (has body region), they were already consumed
        # by the RegionFmt case as pre_arg_ids.
        if parsed.func_arg_ids and not parsed.regions:
            parsed.operand_ids.extend(parsed.func_arg_ids)

        # 5. Assign real types to pre-allocated result values and
        # define them in scope.
        if result_names:
            for i, value_id in enumerate(reserved_result_ids):
                value = self._module.values[value_id]
                if i < len(parsed.result_types):
                    bindings = (
                        parsed.result_bindings[i]
                        if i < len(parsed.result_bindings)
                        else {}
                    )
                    dim_bindings = {k: v for k, v in bindings.items() if k >= 0}
                    encoding_binding = bindings.get(-1, -1)
                    value.type = parsed.result_types[i]
                    value.dim_bindings = dim_bindings
                    value.encoding_binding = encoding_binding
                if i < len(parsed.result_ids):
                    parsed.result_ids[i] = value_id
                else:
                    parsed.result_ids.append(value_id)
                self._scope.define(result_names[i], value_id)
        elif parsed.result_types:
            # Symbol-defining op: no LHS names, create anonymous result values
            # for any that weren't already named in the signature.
            for i, result_type in enumerate(parsed.result_types):
                if i < len(parsed.result_ids) and parsed.result_ids[i] is not None:
                    continue

                bindings = (
                    parsed.result_bindings[i] if i < len(parsed.result_bindings) else {}
                )
                dim_bindings = {k: v for k, v in bindings.items() if k >= 0}
                encoding_binding = bindings.get(-1, -1)
                value_id = self._module.add_value(
                    Value(
                        name="",
                        type=result_type,
                        dim_bindings=dim_bindings,
                        encoding_binding=encoding_binding,
                    )
                )
                if i < len(parsed.result_ids):
                    parsed.result_ids[i] = value_id
                else:
                    parsed.result_ids.append(value_id)

        # 5. Build Operation.
        # Default location: implicit source position from tokenizer.
        end_loc = tok.current_location()
        location_id = self._module.add_location(
            FileLocation(
                source_id=0,
                start_line=start_loc.line,
                start_col=start_loc.column,
                end_line=end_loc.line,
                end_col=end_loc.column,
            )
        )

        # Explicit location annotation overrides the implicit one.
        if tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "loc":
            location_id = self._parse_location_annotation()

        # All result slots must be resolved to concrete value IDs by now.
        result_ids: list[int] = []
        for rid in parsed.result_ids:
            if rid is None:
                raise ParseError(
                    "internal error: unresolved result slot",
                    end_loc,
                    tok._filename,
                )
            result_ids.append(rid)

        op = Operation(
            name=op_name,
            operands=parsed.operand_ids,
            results=result_ids,
            tied_results=parsed.tied_results,
            attributes=parsed.attributes,
            regions=parsed.regions,
            location_id=location_id,
        )
        return op

    # --- Location annotation parsing ---

    def _parse_location_annotation(self) -> int:
        """Parse loc(...) annotation and return the location ID.

        Syntax:
          loc("source":start_line:start_col to end_line:end_col)  — FILE
          loc(fused<child, child, ...>)                           — FUSED
          loc(opaque<"tag", "data">)                              — OPAQUE
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "loc")
        tok.expect(TokenKind.LPAREN)

        if tok.at(TokenKind.STRING):
            # FILE location: "source":start_line:start_col to end_line:end_col
            location_id = self._parse_file_location()
        elif tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "fused":
            # FUSED location: fused<child, child, ...>
            location_id = self._parse_fused_location()
        elif tok.at(TokenKind.BARE_IDENT) and tok.peek().text == "opaque":
            # OPAQUE location: opaque<"tag", "data">
            location_id = self._parse_opaque_location()
        else:
            raise ParseError(
                f"expected location kind (string, 'fused', or 'opaque'), "
                f"got {tok.peek().kind.name} {tok.peek().text!r}",
                tok.peek().location,
                tok._filename,
            )

        tok.expect(TokenKind.RPAREN)
        return location_id

    def _find_or_add_source(self, name: str) -> int:
        """Find or add a source name in the module's source table."""
        for i, existing in enumerate(self._module.sources):
            if existing == name:
                return i
        source_id = len(self._module.sources)
        self._module.sources.append(name)
        return source_id

    def _parse_file_location(self) -> int:
        """Parse "source":start_line:start_col to end_line:end_col.

        Called after loc( has been consumed; the closing ) is consumed
        by the caller.
        """
        tok = self._tokenizer
        source_text = tok.expect(TokenKind.STRING).text
        source_name = source_text[1:-1]  # Strip quotes.
        source_id = self._find_or_add_source(source_name)

        tok.expect(TokenKind.COLON)
        start_line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        start_col = int(tok.expect(TokenKind.INTEGER).text)

        _expect_keyword(tok, "to")

        end_line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        end_col = int(tok.expect(TokenKind.INTEGER).text)

        return self._module.add_location(
            FileLocation(
                source_id=source_id,
                start_line=start_line,
                start_col=start_col,
                end_line=end_line,
                end_col=end_col,
            )
        )

    def _parse_fused_location(self) -> int:
        """Parse fused<child, child, ...>.

        Each child is a FILE location: "source":line:col.
        Called after loc( has been consumed.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "fused")
        tok.expect(TokenKind.LANGLE)

        children: list[int] = []
        if not tok.at(TokenKind.RANGLE):
            children.append(self._parse_fused_child())
            while tok.try_consume(TokenKind.COMMA):
                children.append(self._parse_fused_child())

        tok.expect(TokenKind.RANGLE)
        return self._module.add_location(FusedLocation(children=tuple(children)))

    def _parse_fused_child(self) -> int:
        """Parse one fused location child: "source":line:col."""
        tok = self._tokenizer
        source_text = tok.expect(TokenKind.STRING).text
        source_name = source_text[1:-1]
        source_id = self._find_or_add_source(source_name)

        tok.expect(TokenKind.COLON)
        line = int(tok.expect(TokenKind.INTEGER).text)
        tok.expect(TokenKind.COLON)
        col = int(tok.expect(TokenKind.INTEGER).text)

        # Fused children are stored as FILE locations (start = end).
        return self._module.add_location(
            FileLocation(
                source_id=source_id,
                start_line=line,
                start_col=col,
                end_line=line,
                end_col=col,
            )
        )

    def _parse_opaque_location(self) -> int:
        """Parse opaque<"tag", "data">.

        Called after loc( has been consumed.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.BARE_IDENT, "opaque")
        tok.expect(TokenKind.LANGLE)

        tag_text = tok.expect(TokenKind.STRING).text
        tag = tag_text[1:-1]
        source_id = self._find_or_add_source(tag)

        tok.expect(TokenKind.COMMA)

        data_text = tok.expect(TokenKind.STRING).text
        data = data_text[1:-1].encode("utf-8")

        tok.expect(TokenKind.RANGLE)
        return self._module.add_location(OpaqueLocation(source_id=source_id, data=data))

    def _walk_format(
        self,
        elements: tuple[FormatElement, ...],
        op_decl: Op,
        parsed: ParsedFields,
    ) -> None:
        """Walk format elements, consuming tokens into ParsedFields."""
        tok = self._tokenizer
        for element in elements:
            match element:
                case Ref(field=name):
                    if name in ("iv",):
                        # Implicit field: define a new value.
                        ssa_tok = tok.expect(TokenKind.SSA_VALUE)
                        value_id = self._module.add_value(
                            Value(
                                name=ssa_tok.text, type=ScalarType(ScalarTypeKind.INDEX)
                            )
                        )
                        self._scope.define(ssa_tok.text, value_id)
                        parsed.implicit_values[name] = value_id
                    else:
                        ssa_tok = tok.expect(TokenKind.SSA_VALUE)
                        try:
                            value_id = self._scope.lookup(ssa_tok.text)
                        except KeyError:
                            raise ParseError(
                                f"undefined SSA value '%{ssa_tok.text}'",
                                ssa_tok.location,
                                tok._filename,
                            ) from None
                        parsed.operand_ids.append(value_id)

                case Refs(field=name):
                    if tok.at(TokenKind.SSA_VALUE):
                        ssa_tok = tok.next()
                        try:
                            value_id = self._scope.lookup(ssa_tok.text)
                        except KeyError:
                            raise ParseError(
                                f"undefined SSA value '%{ssa_tok.text}'",
                                ssa_tok.location,
                                tok._filename,
                            ) from None
                        parsed.operand_ids.append(value_id)
                        while tok.try_consume(TokenKind.COMMA):
                            if not tok.at(TokenKind.SSA_VALUE):
                                break
                            ssa_tok = tok.next()
                            try:
                                value_id = self._scope.lookup(ssa_tok.text)
                            except KeyError:
                                raise ParseError(
                                    f"undefined SSA value '%{ssa_tok.text}'",
                                    ssa_tok.location,
                                    tok._filename,
                                ) from None
                            parsed.operand_ids.append(value_id)

                case Attr(field=name):
                    attr_def = op_decl.attr(name)
                    value = self._parse_attr_value(attr_def)
                    parsed.attributes[name] = value

                case SymbolRef(field=name):
                    sym_tok = tok.expect(TokenKind.SYMBOL)
                    parsed.attributes[name] = sym_tok.text

                case TypeOf(field=name):
                    parsed_type, bindings = parse_type_from_tokens(
                        tok,
                        self._scope,
                        self._module,
                        self._type_registry,
                        TypeParseMode.BODY,
                    )
                    # Check if this field is a result — store the type.
                    field_desc = self._layout(op_decl).fields.get(name)
                    if field_desc and field_desc.kind == FieldKind.RESULT:
                        parsed.result_types.append(parsed_type)
                        parsed.result_bindings.append(bindings)

                case TypesOf(field=name):
                    field_desc = self._layout(op_decl).fields.get(name)
                    is_result = field_desc and field_desc.kind == FieldKind.RESULT
                    if _is_type_start(tok.peek(), self._type_registry):
                        t, bindings = parse_type_from_tokens(
                            tok,
                            self._scope,
                            self._module,
                            self._type_registry,
                            TypeParseMode.BODY,
                        )
                        if is_result:
                            parsed.result_types.append(t)
                            parsed.result_bindings.append(bindings)
                        while tok.try_consume(TokenKind.COMMA):
                            if not _is_type_start(tok.peek(), self._type_registry):
                                break
                            t, bindings = parse_type_from_tokens(
                                tok,
                                self._scope,
                                self._module,
                                self._type_registry,
                                TypeParseMode.BODY,
                            )
                            if is_result:
                                parsed.result_types.append(t)
                                parsed.result_bindings.append(bindings)

                case ResultType(field=name):
                    self._parse_result_type(parsed)

                case ResultTypeList(field=name):
                    self._parse_result_type_list(parsed, op_decl, name)

                case Keyword(text=text):
                    _expect_keyword(tok, text)

                case AttrDict(field=dict_field):
                    if tok.at(TokenKind.LBRACE):
                        self._parse_attr_dict(parsed, dict_field)

                case RegionFmt(field=name):
                    # Get block arg info from binding list if available.
                    binding_names = parsed.attributes.pop("_binding_arg_names", None)
                    binding_types = parsed.attributes.pop("_binding_arg_types", None)
                    # Func arg IDs (from FuncArgs) become the entry block's args.
                    pre_arg_ids = parsed.func_arg_ids if parsed.func_arg_ids else None
                    region = self._parse_region(
                        block_arg_names=binding_names,
                        block_arg_types=binding_types,
                        pre_arg_ids=pre_arg_ids,
                    )
                    parsed.regions.append(region)

                case IndexList(dynamic=dynamic_field, static=static_field):
                    self._parse_index_list(parsed, dynamic_field, static_field)

                case BindingList(field=name, kind=binding_kind):
                    self._parse_binding_list(parsed, name, binding_kind)

                case FuncArgs(field=name):
                    tok.expect(TokenKind.LPAREN)
                    if not tok.at(TokenKind.RPAREN):
                        _, _, vid = self._parse_func_arg()
                        parsed.func_arg_ids.append(vid)
                        while tok.try_consume(TokenKind.COMMA):
                            _, _, vid = self._parse_func_arg()
                            parsed.func_arg_ids.append(vid)
                    tok.expect(TokenKind.RPAREN)

                case PredicateList(field=name):
                    predicates = self._parse_predicate_list()
                    if predicates:
                        parsed.attributes[name] = predicates

                case OptionalGroup(elements=inner, anchor=_anchor):
                    if self._optional_group_present(inner, op_decl):
                        self._walk_format(inner, op_decl, parsed)

                case Scope(elements=inner):
                    parent_scope = self._scope
                    self._scope = self._scope.push()
                    self._definition_scope_depth += 1
                    try:
                        self._walk_format(inner, op_decl, parsed)
                        # Verify all placeholders defined in this scope are resolved.
                        for name, value_id in self._scope._names.items():
                            value = self._module.values[value_id]
                            if isinstance(value.type, PlaceholderType):
                                raise ParseError(
                                    f"unresolved forward reference to "
                                    f"'%{name}' in signature",
                                    tok.current_location(),
                                    tok._filename,
                                )
                    finally:
                        self._definition_scope_depth -= 1
                        self._scope = parent_scope

                case Flags(field=name):
                    if tok.at(TokenKind.LANGLE):
                        tok.next()  # consume '<'
                        parts: list[str] = []
                        parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        while tok.at(TokenKind.PIPE):
                            tok.next()  # consume '|'
                            parts.append(tok.expect(TokenKind.BARE_IDENT).text)
                        tok.expect(TokenKind.RANGLE)
                        parsed.attributes[name] = "|".join(parts)

                case OpRef(field=name):
                    if tok.at(TokenKind.LANGLE):
                        tok.next()
                        op_name_tok = tok.expect(TokenKind.OP_NAME)
                        tok.expect(TokenKind.RANGLE)
                        parsed.attributes[name] = op_name_tok.text

                case Glue():
                    pass

    def _optional_group_present(
        self,
        inner_elements: tuple[FormatElement, ...],
        op_decl: Op | None = None,
    ) -> bool:
        """Peek to decide if an OptionalGroup is present."""
        if not inner_elements:
            return False
        tok = self._tokenizer
        first = inner_elements[0]
        match first:
            case Keyword(text=","):
                result: bool = tok.at(TokenKind.COMMA)
                return result
            case Keyword(text="->"):
                result = tok.at(TokenKind.ARROW)
                return result
            case Keyword(text="{"):
                result = tok.at(TokenKind.LBRACE)
                return result
            case Keyword(text=text):
                result = tok.at(TokenKind.BARE_IDENT, text)
                return result
            case RegionFmt():
                result = tok.at(TokenKind.LBRACE)
                return result
            case Attr(field=attr_name):
                if not tok.at(TokenKind.BARE_IDENT):
                    return False
                # For enum attrs, only enter the group if the token is a
                # valid keyword for that enum. Adjacent optional enum groups
                # (e.g. visibility then cc) otherwise steal each other's
                # keywords and produce spurious parse errors.
                if op_decl is not None:
                    attr_def = op_decl.attr(attr_name)
                    if (
                        attr_def is not None
                        and attr_def.attr_type == "enum"
                        and attr_def.enum_def is not None
                    ):
                        return tok.peek().text in attr_def.enum_def.keywords
                return True
            case Ref() | Refs():
                return tok.at(TokenKind.SSA_VALUE)
            case _:
                return False

    # --- Attribute parsing ---

    def _parse_attr_value(self, attr_def: AttrDef | None) -> Any:
        """Parse an attribute value based on its AttrDef type."""
        tok = self._tokenizer
        if attr_def is None:
            return self._parse_any_attr_value()

        match attr_def.attr_type:
            case "i64":
                return int(tok.expect(TokenKind.INTEGER).text)
            case "f64":
                if tok.at(TokenKind.FLOAT):
                    return float(tok.next().text)
                return float(tok.expect(TokenKind.INTEGER).text)
            case "string":
                text = tok.expect(TokenKind.STRING).text
                return text[1:-1]  # Strip quotes.
            case "bool":
                ident = tok.expect(TokenKind.BARE_IDENT)
                if ident.text == "true":
                    return True
                if ident.text == "false":
                    return False
                raise ParseError(
                    f"expected 'true' or 'false', got '{ident.text}'",
                    ident.location,
                    tok._filename,
                )
            case "enum":
                ident = tok.expect(TokenKind.BARE_IDENT)
                if attr_def.enum_def and ident.text not in attr_def.enum_def.keywords:
                    raise ParseError(
                        f"invalid enum value '{ident.text}', expected one "
                        f"of {attr_def.enum_def.keywords}",
                        ident.location,
                        tok._filename,
                    )
                return ident.text
            case "symbol":
                sym = tok.expect(TokenKind.SYMBOL)
                return sym.text
            case "i64_array":
                return self._parse_i64_array()
            case "any":
                return self._parse_any_attr_value()
            case _:
                return self._parse_any_attr_value()

    def _parse_any_attr_value(self) -> Any:
        """Parse any attribute value (type-agnostic)."""
        tok = self._tokenizer
        if tok.at(TokenKind.INTEGER):
            return int(tok.next().text)
        if tok.at(TokenKind.FLOAT):
            return float(tok.next().text)
        if tok.at(TokenKind.STRING):
            text = tok.next().text
            return text[1:-1]
        if tok.at(TokenKind.BARE_IDENT):
            text = tok.next().text
            if text == "true":
                return True
            if text == "false":
                return False
            return text
        if tok.at(TokenKind.LBRACE):
            # Nested dict: {key = value, ...}
            tok.next()  # consume '{'
            entries: dict[str, Any] = {}
            while not tok.at(TokenKind.RBRACE):
                key = tok.expect(TokenKind.BARE_IDENT).text
                tok.expect(TokenKind.EQUALS)
                value = self._parse_any_attr_value()
                entries[key] = value
                tok.try_consume(TokenKind.COMMA)
            tok.expect(TokenKind.RBRACE)
            return entries
        raise ParseError(
            f"expected attribute value, got {tok.peek().kind.name}",
            tok.peek().location,
            tok._filename,
        )

    def _parse_i64_array(self) -> list[int]:
        """Parse [int, int, ...] array."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        values: list[int] = []
        if not tok.at(TokenKind.RBRACKET):
            values.append(int(tok.expect(TokenKind.INTEGER).text))
            while tok.try_consume(TokenKind.COMMA):
                values.append(int(tok.expect(TokenKind.INTEGER).text))
        tok.expect(TokenKind.RBRACKET)
        return values

    # --- Result type list ---

    def _parse_result_type(self, parsed: ParsedFields) -> None:
        """Parse a single bare result type (no parentheses).

        Pushes pre-allocated result names into a child scope so the
        type annotation can reference co-results by name (e.g.,
        tile<[%m]x[%k]xf32> where %m and %k are results of this op).
        """
        scope = self._scope
        if self._reserved_result_names:
            scope = scope.push()
            for name, value_id in zip(
                self._reserved_result_names, self._reserved_result_ids, strict=False
            ):
                scope.define(name, value_id)
        result_type, bindings = parse_type_from_tokens(
            self._tokenizer,
            scope,
            self._module,
            self._type_registry,
            TypeParseMode.BODY,
        )
        parsed.result_types.append(result_type)
        parsed.result_bindings.append(bindings)

    def _parse_result_type_list(
        self, parsed: ParsedFields, op_decl: Op, field_name: str
    ) -> None:
        """Parse (type) or (%operand as type, type).

        Pushes pre-allocated result names into a child scope so type
        annotations can reference co-results by name.
        """
        saved_scope = self._scope
        if self._reserved_result_names:
            self._scope = self._scope.push()
            for name, value_id in zip(
                self._reserved_result_names, self._reserved_result_ids, strict=False
            ):
                self._scope.define(name, value_id)
        tok = self._tokenizer
        tok.expect(TokenKind.LPAREN)
        if not tok.at(TokenKind.RPAREN):
            self._parse_one_result_type(parsed)
            while tok.try_consume(TokenKind.COMMA):
                self._parse_one_result_type(parsed)
        tok.expect(TokenKind.RPAREN)
        self._scope = saved_scope

    def _parse_one_result_type(self, parsed: ParsedFields) -> None:
        """Parse one result type entry: type, %name: type, or %operand as type."""
        tok = self._tokenizer
        # Result types use SIGNATURE mode (creating placeholders for
        # unknown dims) only inside a Scope. Outside a Scope, unknown
        # dim names are errors — same as the C parser's
        # definition_scope_depth gating.
        result_mode = (
            TypeParseMode.SIGNATURE
            if self._definition_scope_depth > 0
            else TypeParseMode.BODY
        )
        if tok.at(TokenKind.SSA_VALUE):
            name_tok = tok.next()
            if tok.try_consume(TokenKind.COLON):
                # Named result: %name: type.
                result_type, all_bindings = parse_type_from_tokens(
                    tok,
                    self._scope,
                    self._module,
                    self._type_registry,
                    result_mode,
                )
                dim_bindings = {k: v for k, v in all_bindings.items() if k >= 0}
                encoding_binding = all_bindings.get(-1, -1)
                # Resolve placeholder or define new value.
                try:
                    value_id = self._scope.lookup(name_tok.text)
                    value = self._module.values[value_id]
                    if not isinstance(value.type, PlaceholderType):
                        raise ParseError(
                            f"SSA name '%{name_tok.text}' already defined",
                            name_tok.location,
                            tok._filename,
                        )
                    value.type = result_type
                    value.dim_bindings = dim_bindings
                    value.encoding_binding = encoding_binding
                except KeyError:
                    value_id = self._module.add_value(
                        Value(
                            name=name_tok.text,
                            type=result_type,
                            dim_bindings=dim_bindings,
                            encoding_binding=encoding_binding,
                        )
                    )
                    self._scope.define(name_tok.text, value_id)
                parsed.result_types.append(result_type)
                parsed.result_bindings.append(all_bindings)
                parsed.result_ids.append(value_id)
            elif tok.try_consume(TokenKind.BARE_IDENT, "as"):
                # Tied result: %operand as type.
                operand_name = name_tok.text
                result_type, bindings = parse_type_from_tokens(
                    tok,
                    self._scope,
                    self._module,
                    self._type_registry,
                    TypeParseMode.BODY,
                )
                parsed.result_types.append(result_type)
                parsed.result_bindings.append(bindings)
                parsed.result_ids.append(None)
                # Find the operand index.
                operand_id = self._scope.lookup(operand_name)
                if operand_id in parsed.func_arg_ids:
                    operand_index = parsed.func_arg_ids.index(operand_id)
                elif operand_id in parsed.operand_ids:
                    operand_index = parsed.operand_ids.index(operand_id)
                else:
                    raise ParseError(
                        f"tied result {operand_name!r} not found in args or operands",
                        name_tok.location,
                        tok._filename,
                    )
                result_index = len(parsed.result_types) - 1
                parsed.tied_results.append(
                    IRTiedResult(result_index=result_index, operand_index=operand_index)
                )
            else:
                raise ParseError(
                    f"expected ':' or 'as' after result name {name_tok.text!r}",
                    tok.peek().location,
                    tok._filename,
                )
        else:
            # Fresh result: type.
            result_type, bindings = parse_type_from_tokens(
                tok,
                self._scope,
                self._module,
                self._type_registry,
                result_mode,
            )
            parsed.result_types.append(result_type)
            parsed.result_bindings.append(bindings)
            parsed.result_ids.append(None)

    # --- Index list ---

    def _parse_index_list(
        self, parsed: ParsedFields, dynamic_field: str, static_field: str
    ) -> None:
        """Parse [0, %x, 4] — mixed static/dynamic indices."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        sentinel = -(2**63)
        static_values: list[int] = []
        dynamic_ids: list[int] = []

        if not tok.at(TokenKind.RBRACKET):
            self._parse_one_index_entry(static_values, dynamic_ids, sentinel)
            while tok.try_consume(TokenKind.COMMA):
                self._parse_one_index_entry(static_values, dynamic_ids, sentinel)

        tok.expect(TokenKind.RBRACKET)
        parsed.attributes[static_field] = static_values
        parsed.operand_ids.extend(dynamic_ids)

    def _parse_one_index_entry(
        self,
        static_values: list[int],
        dynamic_ids: list[int],
        sentinel: int,
    ) -> None:
        """Parse one entry in an index list: integer or %value."""
        tok = self._tokenizer
        if tok.at(TokenKind.INTEGER):
            static_values.append(int(tok.next().text))
        elif tok.at(TokenKind.SSA_VALUE):
            ssa_tok = tok.next()
            value_id = self._scope.lookup(ssa_tok.text)
            dynamic_ids.append(value_id)
            static_values.append(sentinel)
        else:
            raise ParseError(
                f"expected integer or SSA value in index list, "
                f"got {tok.peek().kind.name}",
                tok.peek().location,
                tok._filename,
            )

    # --- Binding list ---

    def _parse_binding_list(
        self, parsed: ParsedFields, field_name: str, kind: str = "capture"
    ) -> None:
        """Parse (%block_arg = %operand : type, ...).

        kind determines how block arg types are derived:
          "capture" — block arg has same type as operand.
          "element" — block arg has element type of operand.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LPAREN)
        block_arg_names: list[str] = []
        block_arg_types: list[Type] = []

        if not tok.at(TokenKind.RPAREN):
            name, arg_type = self._parse_one_binding(parsed, kind)
            block_arg_names.append(name)
            block_arg_types.append(arg_type)
            while tok.try_consume(TokenKind.COMMA):
                name, arg_type = self._parse_one_binding(parsed, kind)
                block_arg_names.append(name)
                block_arg_types.append(arg_type)

        tok.expect(TokenKind.RPAREN)
        # Store block arg info for region parsing.
        parsed.attributes["_binding_arg_names"] = block_arg_names
        parsed.attributes["_binding_arg_types"] = block_arg_types

    def _parse_one_binding(
        self,
        parsed: ParsedFields,
        kind: str,
    ) -> tuple[str, Type]:
        """Parse one binding: %block_arg = %operand : type.

        Returns (block_arg_name, block_arg_type) where block_arg_type
        is derived from the operand type according to the binding kind.
        """
        tok = self._tokenizer
        block_arg_name = tok.expect(TokenKind.SSA_VALUE).text
        tok.expect(TokenKind.EQUALS)
        operand_name = tok.expect(TokenKind.SSA_VALUE).text
        tok.expect(TokenKind.COLON)
        operand_type, _ = parse_type_from_tokens(
            tok, self._scope, self._module, self._type_registry, TypeParseMode.BODY
        )
        operand_id = self._scope.lookup(operand_name)
        parsed.operand_ids.append(operand_id)

        # Derive block arg type based on binding kind.
        if kind == "element":
            block_arg_type = binding_element_type(operand_type)
        else:
            block_arg_type = operand_type

        return block_arg_name, block_arg_type

    # --- Predicates ---

    def _parse_predicate_list(self) -> list[Predicate]:
        """Parse [pred(...), pred(...), ...].

        Called for both function where clauses and PredicateList format
        elements. Expects the opening '[' to be the next token.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACKET)
        predicates: list[Predicate] = []
        if not tok.at(TokenKind.RBRACKET):
            predicates.append(self._parse_one_predicate())
            while tok.try_consume(TokenKind.COMMA):
                predicates.append(self._parse_one_predicate())
        tok.expect(TokenKind.RBRACKET)
        return predicates

    def _parse_one_predicate(self) -> Predicate:
        """Parse one predicate: kind(arg, arg, ...).

        Predicate kind is a bare identifier from PREDICATE_KINDS.
        """
        tok = self._tokenizer
        kind_tok = tok.expect(TokenKind.BARE_IDENT)
        kind = kind_tok.text
        if kind not in PREDICATE_KINDS:
            raise ParseError(
                f"unknown predicate kind '{kind}', "
                f"expected one of: {', '.join(sorted(PREDICATE_KINDS))}",
                kind_tok.location,
            )
        tok.expect(TokenKind.LPAREN)
        args: list[PredicateArg] = []
        if not tok.at(TokenKind.RPAREN):
            args.append(self._parse_predicate_arg())
            while tok.try_consume(TokenKind.COMMA):
                args.append(self._parse_predicate_arg())
        tok.expect(TokenKind.RPAREN)
        return Predicate(kind=kind, args=tuple(args))

    def _parse_predicate_arg(self) -> PredicateArg:
        """Parse a single predicate argument: %name or integer."""
        tok = self._tokenizer
        if tok.at(TokenKind.SSA_VALUE):
            name_tok = tok.next()
            return PredicateArg(tag="value", value=name_tok.text)
        if tok.at(TokenKind.RESULT_ORDINAL):
            ordinal_tok = tok.next()
            raise ParseError(
                f"predicate ordinals (#{ordinal_tok.text}) are no longer "
                f"supported; use %name instead",
                ordinal_tok.location,
            )
        if tok.at(TokenKind.INTEGER):
            int_tok = tok.next()
            return PredicateArg(tag="const", value=int(int_tok.text))
        raise ParseError(
            f"expected predicate argument: %name or integer, "
            f"got {tok.peek().kind.name} '{tok.peek().text}'",
            tok.peek().location,
        )

    # --- Region ---

    def _parse_region(
        self,
        block_arg_names: list[str] | None = None,
        block_arg_types: list[Type] | None = None,
        pre_arg_ids: list[int] | None = None,
    ) -> Region:
        """Parse { block+ }.

        block_arg_names/types: pre-defined block args from a BindingList.
          These are NEW values defined in the region's scope.
        pre_arg_ids: already-defined value IDs (from function args).
          These are EXISTING values already in scope — just add to
          the entry block's arg list without re-defining.
        """
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        parent_scope = self._scope
        self._scope = parent_scope.push()

        # For function args: they're already in the parent scope.
        # Copy them into the child scope so the body can see them.
        entry_arg_ids: list[int] = []
        if pre_arg_ids:
            for vid in pre_arg_ids:
                value = self._module.values[vid]
                # Don't re-define — just make visible in child scope.
                self._scope._names[value.name] = vid
                entry_arg_ids.append(vid)

        # For binding list args: define new values in the child scope.
        if block_arg_names and block_arg_types:
            for name, arg_type in zip(block_arg_names, block_arg_types, strict=False):
                value_id = self._module.add_value(Value(name=name, type=arg_type))
                self._scope.define(name, value_id)
                entry_arg_ids.append(value_id)

        blocks: list[Block] = []
        is_first = True
        while not tok.at(TokenKind.RBRACE):
            block = self._parse_block()
            if is_first and entry_arg_ids:
                block.arg_ids = entry_arg_ids + block.arg_ids
                is_first = False
            blocks.append(block)

        tok.expect(TokenKind.RBRACE)
        self._scope = parent_scope
        return Region(blocks=blocks)

    def _parse_block(self) -> Block:
        """Parse a block (optional label, then operations)."""
        tok = self._tokenizer
        label = ""
        arg_ids: list[int] = []

        # Block label: ^name(args):
        if tok.at(TokenKind.BLOCK_LABEL):
            label = tok.next().text
            if tok.at(TokenKind.LPAREN):
                tok.expect(TokenKind.LPAREN)
                while not tok.at(TokenKind.RPAREN):
                    arg_name = tok.expect(TokenKind.SSA_VALUE).text
                    tok.expect(TokenKind.COLON)
                    arg_type, _ = parse_type_from_tokens(
                        tok,
                        self._scope,
                        self._module,
                        self._type_registry,
                        TypeParseMode.BODY,
                    )
                    value_id = self._module.add_value(
                        Value(name=arg_name, type=arg_type)
                    )
                    self._scope.define(arg_name, value_id)
                    arg_ids.append(value_id)
                    tok.try_consume(TokenKind.COMMA)
                tok.expect(TokenKind.RPAREN)
            tok.expect(TokenKind.COLON)

        # Operations.
        ops: list[Operation] = []
        while (
            not tok.at(TokenKind.RBRACE)
            and not tok.at(TokenKind.BLOCK_LABEL)
            and not tok.at(TokenKind.EOF)
            and not tok.at(TokenKind.BARE_IDENT, "else")
        ):
            op = self._parse_operation()
            ops.append(op)

        return Block(label=label, arg_ids=arg_ids, ops=ops)

    # --- Attr dict ---

    def _parse_attr_dict(self, parsed: ParsedFields, field: str) -> None:
        """Parse {key = value, ...} into a named dict attribute."""
        tok = self._tokenizer
        tok.expect(TokenKind.LBRACE)
        entries: dict[str, Any] = {}
        while not tok.at(TokenKind.RBRACE):
            key = tok.expect(TokenKind.BARE_IDENT).text
            tok.expect(TokenKind.EQUALS)
            value = self._parse_any_attr_value()
            entries[key] = value
            tok.try_consume(TokenKind.COMMA)
        tok.expect(TokenKind.RBRACE)
        if field:
            parsed.attributes[field] = entries
        else:
            parsed.attributes.update(entries)
