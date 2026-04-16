# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bytecode reader: deserializes .loombc bytes to ir.py Module.

Mirror of the writer. Reads the file header, section directory,
and each section in order, constructing the in-memory IR.

All data is validated: offsets, lengths, counts, varints. A
malformed .loombc file produces a clear error, never a crash.
"""

from __future__ import annotations

import struct
from typing import Any, ClassVar

from loom.format.bytecode.encoding import decode_signed_varint, decode_varint
from loom.format.bytecode.writer import (
    BYTECODE_IR_KIND_BY_TYPE_KIND,
    FORMAT_VERSION,
    LOCATION_MODE_FULL_LOCATIONS,
    LOCATION_MODE_NO_LOCATIONS,
    LOCATION_MODE_SOURCE_LOCATIONS,
    MAGIC,
    SECTION_ENCODINGS,
    SECTION_IR,
    SECTION_LOCATIONS,
    SECTION_OPS,
    SECTION_SOURCES,
    SECTION_STRINGS,
    SECTION_SYMBOLS,
    SECTION_TYPES,
)
from loom.ir import (
    BUFFER_TYPE,
    ENCODING_TYPE,
    NONE_TYPE,
    Block,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingRole,
    EncodingType,
    FileLocation,
    FunctionType,
    FusedLocation,
    GroupScope,
    GroupType,
    Module,
    OpaqueLocation,
    Operation,
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
    TiedResult,
    Type,
    TypeKind,
    Value,
    rebuild_value_metadata,
)

__all__ = [
    "BytecodeReader",
    "read_module",
]

# Maps the kind byte in the SYMBOLS section to an op name.
# 0=FUNC_DEF, 1=FUNC_DECL, 2=FUNC_TEMPLATE, 3=FUNC_UKERNEL.
_FUNC_KIND_OP_NAMES: list[str] = [
    "func.def",
    "func.decl",
    "func.template",
    "func.ukernel",
]

# Maps the cc byte to a cc attribute string.
# 0=HOST (absent/default), 1=DEVICE, 2=INITIALIZER.
_FUNC_CC_BYTES: list[str | None] = [None, "device", "initializer"]


class BytecodeError(Exception):
    """Error reading bytecode."""


class BytecodeReader:
    """Reads .loombc bytes and constructs an ir.py Module."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0
        self._strings: list[str] = []
        self._sources: list[str] = []
        self._types: list[Type] = []
        self._ops: list[str] = []
        self._encodings: list[EncodingInstance] = []
        self._encoding_families: list[str] = []
        self._location_mode = LOCATION_MODE_SOURCE_LOCATIONS
        self._module_value_count = 0
        self._module_region_count = 0
        self._module_block_count = 0
        self._module_op_count = 0

    def read(self) -> Module:
        """Read and return the module."""
        self._read_file_header()
        module_offset, module_length = self._read_module_directory()
        self._read_file_string_pool()

        # Read module sections.
        self._offset = module_offset
        sections = self._read_section_directory(module_length)

        # Read sections in dependency order.
        if SECTION_STRINGS in sections:
            self._read_strings_section(sections[SECTION_STRINGS])
        if SECTION_SOURCES in sections:
            self._read_sources_section(sections[SECTION_SOURCES])
        if SECTION_ENCODINGS in sections:
            self._read_encodings_section(sections[SECTION_ENCODINGS])
        if SECTION_TYPES in sections:
            self._read_types_section(sections[SECTION_TYPES])
        if SECTION_OPS in sections:
            self._read_ops_section(sections[SECTION_OPS])

        # Build module.
        module = Module()
        module.encodings = list(self._encodings)
        module.sources = list(self._sources)

        if self._location_mode == LOCATION_MODE_NO_LOCATIONS:
            if SECTION_LOCATIONS in sections:
                raise BytecodeError(
                    "NO_LOCATIONS bytecode must not contain a LOCATIONS section"
                )
        elif SECTION_LOCATIONS not in sections:
            raise BytecodeError("bytecode with source locations must have LOCATIONS")

        # Read locations.
        if SECTION_LOCATIONS in sections:
            self._read_locations_section(sections[SECTION_LOCATIONS], module)

        # Read IR and symbols.
        ir_data = sections.get(SECTION_IR, (0, b""))
        symbols_data = sections.get(SECTION_SYMBOLS, (0, b""))
        if isinstance(symbols_data, tuple):
            self._read_symbols_section(symbols_data, ir_data, module)

        allocation_counts = self._module_allocation_counts(module)
        if allocation_counts != (
            self._module_value_count,
            self._module_region_count,
            self._module_block_count,
            self._module_op_count,
        ):
            raise BytecodeError("module allocation summary does not match IR")

        rebuild_value_metadata(module)
        return module

    def _module_allocation_counts(self, module: Module) -> tuple[int, int, int, int]:
        """Return module value, serialized region, block, and op counts."""
        value_count = len(module.values)
        region_count = 0
        block_count = 0
        op_count = 0
        for symbol in module.symbols:
            if symbol.op is None or not symbol.op.regions:
                continue
            counts = self._count_region_tree(symbol.op.regions[0])
            region_count += counts[1]
            block_count += counts[2]
            op_count += counts[3]
        return value_count, region_count, block_count, op_count

    # --- Low-level reads ---

    def _read_u8(self) -> int:
        if self._offset >= len(self._data):
            raise BytecodeError("unexpected end of data")
        value = self._data[self._offset]
        self._offset += 1
        return value

    def _read_u16_le(self) -> int:
        if self._offset + 2 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<H", self._data, self._offset)[0]
        self._offset += 2
        return value

    def _read_u32_le(self) -> int:
        if self._offset + 4 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<I", self._data, self._offset)[0]
        self._offset += 4
        return value

    def _read_u64_le(self) -> int:
        if self._offset + 8 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<Q", self._data, self._offset)[0]
        self._offset += 8
        return value

    def _read_varint(self) -> int:
        value, self._offset = decode_varint(self._data, self._offset)
        return value

    def _read_signed_varint(self) -> int:
        value, self._offset = decode_signed_varint(self._data, self._offset)
        return value

    def _read_bytes(self, length: int) -> bytes:
        if self._offset + length > len(self._data):
            raise BytecodeError("unexpected end of data")
        data = self._data[self._offset : self._offset + length]
        self._offset += length
        return data

    def _read_string(self) -> str:
        length = self._read_varint()
        data = self._read_bytes(length)
        return data.decode("utf-8")

    def _skip_to_alignment(self, alignment: int) -> None:
        remainder = self._offset % alignment
        if remainder != 0:
            self._offset += alignment - remainder

    # --- File structure ---

    def _read_file_header(self) -> None:
        magic = self._read_bytes(4)
        if magic != MAGIC:
            raise BytecodeError(f"invalid magic: expected {MAGIC!r}, got {magic!r}")
        version = self._read_u8()
        if version != FORMAT_VERSION:
            raise BytecodeError(f"unsupported format version: {version}")
        self._location_mode = self._read_u8()
        if self._location_mode not in (
            LOCATION_MODE_SOURCE_LOCATIONS,
            LOCATION_MODE_NO_LOCATIONS,
            LOCATION_MODE_FULL_LOCATIONS,
        ):
            raise BytecodeError(f"unsupported location mode: {self._location_mode}")
        if self._location_mode == LOCATION_MODE_FULL_LOCATIONS:
            raise BytecodeError("FULL_LOCATIONS bytecode is not implemented")
        self._module_count = self._read_u16_le()
        self._string_pool_length = self._read_u32_le()
        self._reserved = self._read_u32_le()
        # Producer string (null-terminated).
        producer_start = self._offset
        while self._offset < len(self._data) and self._data[self._offset] != 0:
            self._offset += 1
        self._producer = self._data[producer_start : self._offset].decode("utf-8")
        self._offset += 1  # skip null
        self._skip_to_alignment(8)

    def _read_module_directory(self) -> tuple[int, int]:
        """Read module directory, return (offset, length) of first module."""
        self._read_u32_le()
        self._read_u16_le()
        self._read_u16_le()
        module_offset = self._read_u64_le()
        module_length = self._read_u64_le()
        return module_offset, module_length

    def _read_file_string_pool(self) -> None:
        """Read the file string pool (module names)."""
        if self._string_pool_length > 0:
            pool_data = self._read_bytes(self._string_pool_length)
            self._module_name = pool_data.decode("utf-8")
        else:
            self._module_name = ""
        self._skip_to_alignment(8)

    def _read_section_directory(
        self, module_length: int
    ) -> dict[int, tuple[int, bytes]]:
        """Read section directory, return {kind: (offset, data)}."""
        sections: dict[int, tuple[int, bytes]] = {}
        module_start = self._offset
        module_end = module_start + module_length
        if module_end > len(self._data):
            raise BytecodeError("module extends past end of file")

        section_count = self._read_varint()
        self._module_value_count = self._read_varint()
        self._module_region_count = self._read_varint()
        self._module_block_count = self._read_varint()
        self._module_op_count = self._read_varint()

        entries: list[tuple[int, int, int]] = []
        seen_kinds: set[int] = set()
        for _ in range(section_count):
            if self._offset + 32 > module_end:
                raise BytecodeError("section directory extends past module")
            kind = struct.unpack_from("<H", self._data, self._offset)[0]
            flags = struct.unpack_from("<H", self._data, self._offset + 2)[0]
            reserved = struct.unpack_from("<I", self._data, self._offset + 4)[0]
            offset = struct.unpack_from("<Q", self._data, self._offset + 8)[0]
            length = struct.unpack_from("<Q", self._data, self._offset + 16)[0]
            uncompressed_length = struct.unpack_from(
                "<Q", self._data, self._offset + 24
            )[0]
            if kind in seen_kinds:
                raise BytecodeError(f"duplicate section kind: {kind}")
            if reserved != 0:
                raise BytecodeError(f"section {kind} reserved field must be zero")
            if flags != 0:
                raise BytecodeError(f"section {kind} has unsupported flags {flags}")
            if uncompressed_length != 0:
                raise BytecodeError(f"section {kind} uncompressed length must be zero")
            seen_kinds.add(kind)
            entries.append((kind, offset, length))
            self._offset += 32

        minimum_section_offset = self._offset - module_start
        previous_end = minimum_section_offset
        for kind, offset, length in entries:
            if offset < previous_end:
                raise BytecodeError("section directory is not sorted by offset")
            section_end = offset + length
            if section_end > module_length:
                raise BytecodeError(f"section {kind} extends past module")
            abs_offset = module_start + offset
            section_data = self._data[abs_offset : abs_offset + length]
            sections[kind] = (abs_offset, section_data)
            previous_end = section_end

        return sections

    # --- Section readers ---

    def _read_strings_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._strings = []
        for _ in range(count):
            length, offset = decode_varint(data, offset)
            text = data[offset : offset + length].decode("utf-8")
            offset += length
            self._strings.append(text)

    def _read_sources_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._sources = []
        for _ in range(count):
            length, offset = decode_varint(data, offset)
            text = data[offset : offset + length].decode("utf-8")
            offset += length
            self._sources.append(text)

    def _read_encodings_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0

        # Encoding family registry.
        family_count, offset = decode_varint(data, offset)
        self._encoding_families = []
        for _ in range(family_count):
            name_id, offset = decode_varint(data, offset)
            self._encoding_families.append(self._strings[name_id])

        # Encoding instances with structured attribute parameters.
        instance_count, offset = decode_varint(data, offset)
        self._encodings = []
        for _ in range(instance_count):
            family_index, offset = decode_varint(data, offset)
            if family_index >= len(self._encoding_families):
                raise BytecodeError(
                    f"encoding family index {family_index} out of range "
                    f"(family table has {len(self._encoding_families)} entries)"
                )
            alias_string_id_plus1, offset = decode_varint(data, offset)
            if alias_string_id_plus1 > len(self._strings):
                alias_string_id = alias_string_id_plus1 - 1
                raise BytecodeError(
                    f"encoding alias string_id {alias_string_id} out of range "
                    f"(string table has {len(self._strings)} entries)"
                )
            param_count, offset = decode_varint(data, offset)

            name = self._encoding_families[family_index]
            alias = (
                self._strings[alias_string_id_plus1 - 1]
                if alias_string_id_plus1 > 0
                else ""
            )
            param_list: list[tuple[str, Any]] = []
            for _ in range(param_count):
                param_name_id, offset = decode_varint(data, offset)
                param_name = self._strings[param_name_id]
                param_value, offset = self._read_attr_value(data, offset)
                param_list.append((param_name, param_value))

            self._encodings.append(
                EncodingInstance(name=name, alias=alias, params=tuple(param_list))
            )

    def _read_types_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._types = []
        for _ in range(count):
            kind = data[offset]
            offset += 1
            try:
                type_kind = BYTECODE_IR_KIND_BY_TYPE_KIND[kind]
            except KeyError as err:
                raise BytecodeError(f"unknown type kind: {kind}") from err
            ir_type: Type
            match type_kind:
                case TypeKind.NONE:
                    ir_type = NONE_TYPE
                case TypeKind.SCALAR:
                    scalar_kind = ScalarTypeKind(data[offset])
                    offset += 1
                    ir_type = ScalarType(scalar_kind)
                case TypeKind.TILE | TypeKind.TENSOR | TypeKind.VECTOR | TypeKind.VIEW:
                    elem_kind = ScalarTypeKind(data[offset])
                    offset += 1
                    rank = data[offset]
                    offset += 1
                    encoding_attachment = data[offset]
                    offset += 1
                    enc_instance, offset = decode_varint(data, offset)

                    dims: list[StaticDim | DynamicDim] = []
                    for _ in range(rank):
                        is_dynamic = data[offset]
                        offset += 1
                        if is_dynamic:
                            dims.append(DynamicDim())
                        else:
                            size, offset = decode_varint(data, offset)
                            dims.append(StaticDim(size))

                    encoding: EncodingInstance | DynamicEncoding | None = None
                    match encoding_attachment:
                        case 0:
                            if enc_instance != 0:
                                raise BytecodeError(
                                    "none encoding attachment must have id 0"
                                )
                        case 1:
                            if enc_instance == 0 or enc_instance > len(self._encodings):
                                raise BytecodeError(
                                    f"static encoding id out of range: {enc_instance}"
                                )
                            encoding = self._encodings[enc_instance - 1]
                        case 2:
                            if enc_instance != 0:
                                raise BytecodeError(
                                    "dynamic encoding attachment must have id 0"
                                )
                            encoding = DynamicEncoding()
                        case _:
                            raise BytecodeError(
                                f"unknown encoding attachment: {encoding_attachment}"
                            )
                    if type_kind == TypeKind.VECTOR and encoding is not None:
                        raise BytecodeError(
                            "vector types must not carry encoding/layout suffixes"
                        )

                    try:
                        ir_type = ShapedType(
                            type_kind=type_kind,
                            element_type=ScalarType(elem_kind),
                            dims=tuple(dims),
                            encoding=encoding,
                        )
                    except ValueError as err:
                        raise BytecodeError(str(err)) from err
                case TypeKind.GROUP:
                    scope_byte = data[offset]
                    offset += 1
                    try:
                        scope = GroupScope(scope_byte)
                    except ValueError as err:
                        raise BytecodeError(
                            f"unknown group scope byte: {scope_byte}"
                        ) from err
                    ir_type = GroupType(scope)
                case TypeKind.FUNCTION:
                    arg_count, offset = decode_varint(data, offset)
                    result_count, offset = decode_varint(data, offset)
                    arg_types = []
                    for _ in range(arg_count):
                        type_idx, offset = decode_varint(data, offset)
                        arg_types.append(self._types[type_idx])
                    result_types = []
                    for _ in range(result_count):
                        type_idx, offset = decode_varint(data, offset)
                        result_types.append(self._types[type_idx])
                    ir_type = FunctionType(tuple(arg_types), tuple(result_types))
                case TypeKind.DIALECT:
                    name_id, offset = decode_varint(data, offset)
                    param_count, offset = decode_varint(data, offset)
                    params = []
                    for _ in range(param_count):
                        type_idx, offset = decode_varint(data, offset)
                        params.append(self._types[type_idx])
                    ir_type = DialectType(self._strings[name_id], tuple(params))
                case TypeKind.ENCODING:
                    role_byte = data[offset]
                    offset += 1
                    try:
                        role = EncodingRole(role_byte)
                    except ValueError as e:
                        raise BytecodeError(
                            f"unsupported encoding role byte: {role_byte}"
                        ) from e
                    ir_type = (
                        ENCODING_TYPE
                        if role == EncodingRole.UNKNOWN
                        else EncodingType(role)
                    )
                case TypeKind.POOL:
                    is_dynamic = data[offset]
                    offset += 1
                    if is_dynamic:
                        ir_type = PoolType(block_size=DynamicDim())
                    else:
                        size, offset = decode_varint(data, offset)
                        ir_type = PoolType(block_size=StaticDim(size))
                case TypeKind.BUFFER:
                    ir_type = BUFFER_TYPE
                case _:
                    raise BytecodeError(f"unsupported type kind: {type_kind.name}")
            self._types.append(ir_type)

    def _read_ops_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._ops = []
        for _ in range(count):
            name_id, offset = decode_varint(data, offset)
            self._ops.append(self._strings[name_id])

    def _read_locations_section(
        self, section: tuple[int, bytes], module: Module
    ) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        for _ in range(count):
            kind = data[offset]
            offset += 1
            flags = data[offset]
            offset += 1
            match kind:
                case 0:  # NONE
                    module.locations.add(None)
                case 1:  # FILE
                    source_id, offset = decode_varint(data, offset)
                    start_line, offset = decode_varint(data, offset)
                    start_col, offset = decode_varint(data, offset)
                    end_line, offset = decode_varint(data, offset)
                    end_col, offset = decode_varint(data, offset)
                    module.locations.add(
                        FileLocation(
                            source_id, start_line, start_col, end_line, end_col, flags
                        )
                    )
                case 2:  # FUSED
                    child_count, offset = decode_varint(data, offset)
                    children = []
                    for _ in range(child_count):
                        child, offset = decode_varint(data, offset)
                        children.append(child)
                    module.locations.add(FusedLocation(tuple(children), flags))
                case 3:  # OPAQUE
                    source_id, offset = decode_varint(data, offset)
                    data_length, offset = decode_varint(data, offset)
                    opaque_data = data[offset : offset + data_length]
                    offset += data_length
                    module.locations.add(OpaqueLocation(source_id, opaque_data, flags))

    def _read_symbols_section(
        self,
        symbols_section: tuple[int, bytes],
        ir_section: tuple[int, bytes],
        module: Module,
    ) -> None:
        _, sym_data = symbols_section
        _, ir_data = ir_section
        offset = 0
        count, offset = decode_varint(sym_data, offset)

        # Import/export offset tables.
        import_count, offset = decode_varint(sym_data, offset)
        export_count, offset = decode_varint(sym_data, offset)
        # Skip the offset tables (uint64 each).
        offset += import_count * 8
        offset += export_count * 8

        for _ in range(count):
            name_id, offset = decode_varint(sym_data, offset)
            kind = sym_data[offset]
            offset += 1
            visibility = sym_data[offset]
            offset += 1
            flags = struct.unpack_from("<H", sym_data, offset)[0]
            offset += 2

            name = self._strings[name_id]

            # Import metadata: source module and symbol for cross-module refs.
            is_import = (flags & 0x0002) != 0
            source_module = ""
            source_symbol = ""
            if is_import:
                source_module_id, offset = decode_varint(sym_data, offset)
                source_symbol_id, offset = decode_varint(sym_data, offset)
                source_module = self._strings[source_module_id]
                source_symbol = self._strings[source_symbol_id]

            if kind <= 3:  # FUNC_DEF, FUNC_DECL, FUNC_TEMPLATE, FUNC_UKERNEL
                cc_byte = sym_data[offset]
                offset += 1
                arg_count, offset = decode_varint(sym_data, offset)
                result_count, offset = decode_varint(sym_data, offset)

                arg_types: list[Type] = []
                for _ in range(arg_count):
                    type_idx, offset = decode_varint(sym_data, offset)
                    arg_types.append(self._types[type_idx])

                result_types: list[Type] = []
                tied_results: list[TiedResult] = []
                for i in range(result_count):
                    is_tied = sym_data[offset]
                    offset += 1
                    type_idx, offset = decode_varint(sym_data, offset)
                    result_types.append(self._types[type_idx])
                    if is_tied:
                        tied_op_idx, offset = decode_varint(sym_data, offset)
                        tied_results.append(
                            TiedResult(result_index=i, operand_index=tied_op_idx)
                        )

                _tied_count, offset = decode_varint(sym_data, offset)
                predicates, offset = self._read_predicate_list(sym_data, offset)

                has_body = sym_data[offset]
                offset += 1
                body = None
                if has_body:
                    ir_offset = struct.unpack_from("<Q", sym_data, offset)[0]
                    offset += 8
                    ir_length = struct.unpack_from("<I", sym_data, offset)[0]
                    offset += 4
                    body = self._read_function_body(
                        ir_data[ir_offset : ir_offset + ir_length], module
                    )

                # Build the attributes dict for this func-like op.
                op_attrs: dict[str, Any] = {"callee": name}
                if visibility:
                    op_attrs["visibility"] = "public"
                cc_str = (
                    _FUNC_CC_BYTES[cc_byte] if cc_byte < len(_FUNC_CC_BYTES) else None
                )
                if cc_str is not None:
                    op_attrs["cc"] = cc_str
                if predicates:
                    op_attrs["predicates"] = predicates
                if source_module:
                    op_attrs["import_module"] = source_module
                    if source_symbol and source_symbol != name:
                        op_attrs["import_symbol"] = source_symbol

                # Create anonymous result value IDs (symbol-level ops have no
                # LHS SSA names; types are carried in the value table).
                result_ids: list[int] = []
                for result_type in result_types:
                    vid = module.add_value(Value(name="", type=result_type))
                    result_ids.append(vid)

                # For declaration-style ops (no body), args are operands.
                # For definition-style ops (with body), args are entry block args.
                operand_ids: list[int] = []
                if body is None:
                    for arg_type in arg_types:
                        vid = module.add_value(Value(name="", type=arg_type))
                        operand_ids.append(vid)

                op = Operation(
                    name=_FUNC_KIND_OP_NAMES[kind],
                    operands=operand_ids,
                    results=result_ids,
                    tied_results=tied_results,
                    attributes=op_attrs,
                    regions=[body] if body is not None else [],
                )
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind(kind),
                    flags=flags,
                    op=op,
                    source_module=source_module,
                    source_symbol=source_symbol,
                )
                module.add_symbol(symbol)

    def _read_function_body(self, data: bytes, module: Module) -> Region:
        """Read a function body region from IR section data.

        The bytecode uses function-local sequential value numbers (0, 1, 2, ...)
        for all SSA references within a function body. The value_map translates
        these numbers to module-level value IDs as block args and op results are
        created during reading.
        """
        offset = 0
        value_map: list[int] = []
        _value_count, offset = decode_varint(data, offset)
        _region_count, offset = decode_varint(data, offset)
        _block_count, offset = decode_varint(data, offset)
        _op_count, offset = decode_varint(data, offset)
        region, offset = self._read_region(data, offset, module, value_map)
        parsed_counts = self._count_region_tree(region)
        if parsed_counts != (_value_count, _region_count, _block_count, _op_count):
            raise BytecodeError("function body allocation summary does not match IR")
        return region

    def _count_region_tree(self, region: Region) -> tuple[int, int, int, int]:
        """Return value, region, block, and op counts for a parsed region tree."""
        value_count = 0
        region_count = 1
        block_count = len(region.blocks)
        op_count = 0
        for block in region.blocks:
            value_count += len(block.arg_ids)
            for op in block.ops:
                value_count += len(op.results)
                op_count += 1
                for nested_region in op.regions:
                    nested_counts = self._count_region_tree(nested_region)
                    value_count += nested_counts[0]
                    region_count += nested_counts[1]
                    block_count += nested_counts[2]
                    op_count += nested_counts[3]
        return value_count, region_count, block_count, op_count

    def _read_region(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
    ) -> tuple[Region, int]:
        block_count, offset = decode_varint(data, offset)
        blocks = []
        for _ in range(block_count):
            block, offset = self._read_block(data, offset, module, value_map)
            blocks.append(block)
        return Region(blocks=blocks), offset

    def _read_block(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
    ) -> tuple[Block, int]:
        has_label = data[offset]
        offset += 1
        label = ""
        if has_label:
            label_id, offset = decode_varint(data, offset)
            label = self._strings[label_id]

        # Block args.
        arg_count, offset = decode_varint(data, offset)
        arg_ids = []
        for _ in range(arg_count):
            name_id, offset = decode_varint(data, offset)
            type_idx, offset = decode_varint(data, offset)
            dim_count, offset = decode_varint(data, offset)
            dim_bindings = {}
            for d in range(dim_count):
                value_ref, offset = decode_signed_varint(data, offset)
                value_ref = (
                    value_map[value_ref] if value_ref < len(value_map) else value_ref
                )
                dim_bindings[d] = value_ref
            # Encoding binding: 0 = none, else 1 + value_number.
            enc_binding_raw, offset = decode_varint(data, offset)
            encoding_binding = (enc_binding_raw - 1) if enc_binding_raw > 0 else -1
            if encoding_binding >= 0 and encoding_binding < len(value_map):
                encoding_binding = value_map[encoding_binding]
            value_name = self._strings[name_id] if name_id < len(self._strings) else ""
            value_type = (
                self._types[type_idx] if type_idx < len(self._types) else NONE_TYPE
            )
            vid = module.add_value(
                Value(
                    name=value_name,
                    type=value_type,
                    dim_bindings=dim_bindings,
                    encoding_binding=encoding_binding,
                )
            )
            arg_ids.append(vid)
            value_map.append(vid)

        # Operations.
        op_count, offset = decode_varint(data, offset)
        ops = []
        for _ in range(op_count):
            op, offset = self._read_operation(data, offset, module, value_map)
            ops.append(op)

        return Block(label=label, arg_ids=arg_ids, ops=ops), offset

    def _read_operation(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
    ) -> tuple[Operation, int]:
        op_table_index_plus1, offset = decode_varint(data, offset)
        data[offset]
        offset += 1
        location_id, offset = decode_varint(data, offset)
        if self._location_mode == LOCATION_MODE_NO_LOCATIONS and location_id != 0:
            raise BytecodeError("NO_LOCATIONS bytecode op location must be 0")

        if op_table_index_plus1 == 0:
            raise BytecodeError("operation op_table_index_plus1 must be nonzero")
        kind_id = op_table_index_plus1 - 1
        op_name = self._ops[kind_id] if kind_id < len(self._ops) else ""

        # Operands: map function-local value numbers to module value IDs.
        operand_count, offset = decode_varint(data, offset)
        operands = []
        for _ in range(operand_count):
            value_ref, offset = decode_varint(data, offset)
            operands.append(
                value_map[value_ref] if value_ref < len(value_map) else value_ref
            )

        # Results.
        result_count, offset = decode_varint(data, offset)
        result_ids = []
        for _ in range(result_count):
            name_id, offset = decode_varint(data, offset)
            type_idx, offset = decode_varint(data, offset)
            dim_count, offset = decode_varint(data, offset)
            dim_bindings = {}
            for d in range(dim_count):
                value_ref, offset = decode_signed_varint(data, offset)
                value_ref = (
                    value_map[value_ref] if value_ref < len(value_map) else value_ref
                )
                dim_bindings[d] = value_ref
            # Encoding binding: 0 = none, else 1 + value_number.
            enc_binding_raw, offset = decode_varint(data, offset)
            encoding_binding = (enc_binding_raw - 1) if enc_binding_raw > 0 else -1
            if encoding_binding >= 0 and encoding_binding < len(value_map):
                encoding_binding = value_map[encoding_binding]
            value_name = self._strings[name_id] if name_id < len(self._strings) else ""
            value_type = (
                self._types[type_idx] if type_idx < len(self._types) else NONE_TYPE
            )
            vid = module.add_value(
                Value(
                    name=value_name,
                    type=value_type,
                    dim_bindings=dim_bindings,
                    encoding_binding=encoding_binding,
                )
            )
            result_ids.append(vid)
            value_map.append(vid)

        # Tied results.
        tied_count, offset = decode_varint(data, offset)
        tied_results = []
        for _ in range(tied_count):
            result_idx, offset = decode_varint(data, offset)
            operand_idx, offset = decode_varint(data, offset)
            tied_results.append(
                TiedResult(result_index=result_idx, operand_index=operand_idx)
            )

        # Attributes.
        attr_count, offset = decode_varint(data, offset)
        attributes, offset = self._read_attr_dict_entries(data, offset, attr_count)

        # Regions.
        region_count, offset = decode_varint(data, offset)
        regions = []
        for _ in range(region_count):
            region, offset = self._read_region(data, offset, module, value_map)
            regions.append(region)

        op = Operation(
            name=op_name,
            operands=operands,
            results=result_ids,
            tied_results=tied_results,
            attributes=attributes,
            regions=regions,
            location_id=location_id,
        )
        return op, offset

    def _read_attr_value(self, data: bytes, offset: int) -> tuple[Any, int]:
        kind = data[offset]
        offset += 1
        match kind:
            case 0:  # I64
                value, offset = decode_signed_varint(data, offset)
                return value, offset
            case 1:  # F64
                value = struct.unpack_from("<d", data, offset)[0]
                return value, offset + 8
            case 2:  # STRING
                string_id, offset = decode_varint(data, offset)
                return self._strings[string_id], offset
            case 3:  # BOOL
                return bool(data[offset]), offset + 1
            case 4:  # ENUM
                string_id, offset = decode_varint(data, offset)
                return self._strings[string_id], offset
            case 5:  # I64_ARRAY
                count, offset = decode_varint(data, offset)
                values = []
                for _ in range(count):
                    v, offset = decode_signed_varint(data, offset)
                    values.append(v)
                return values, offset
            case 6:  # SYMBOL
                string_id, offset = decode_varint(data, offset)
                return self._strings[string_id], offset
            case 7:  # TYPE
                type_idx, offset = decode_varint(data, offset)
                return self._types[type_idx], offset
            case 8:  # PREDICATE_LIST
                predicates, offset = self._read_predicate_list(data, offset)
                return predicates, offset
            case 9:  # DICT
                count, offset = decode_varint(data, offset)
                return self._read_attr_dict_entries(data, offset, count)
            case 10:  # ENCODING
                encoding_id, offset = decode_varint(data, offset)
                if encoding_id == 0 or encoding_id > len(self._encodings):
                    raise BytecodeError(
                        f"encoding attr id {encoding_id} out of range "
                        f"(encoding table has {len(self._encodings)} entries)"
                    )
                return self._encodings[encoding_id - 1], offset
            case _:
                raise BytecodeError(f"unknown attr value kind: {kind}")

    def _read_attr_dict_entries(
        self, data: bytes, offset: int, count: int
    ) -> tuple[CanonicalAttrDict, int]:
        """Read canonical dict attr entries and verify sorted/deduped order."""
        entries: list[tuple[str, Any]] = []
        previous_key: str | None = None
        for _ in range(count):
            key_id, offset = decode_varint(data, offset)
            if key_id >= len(self._strings):
                raise BytecodeError(
                    f"dict attr key string_id {key_id} out of range "
                    f"(string table has {len(self._strings)} entries)"
                )
            key = self._strings[key_id]
            if previous_key is not None and key <= previous_key:
                if key == previous_key:
                    raise BytecodeError(f"duplicate dict attr key: {key!r}")
                raise BytecodeError(
                    "dict attr keys are not in canonical order: "
                    f"{previous_key!r} appears before {key!r}"
                )
            value, offset = self._read_attr_value(data, offset)
            entries.append((key, value))
            previous_key = key
        return CanonicalAttrDict.from_sorted_items(entries), offset

    # Predicate kind byte → name mapping (inverse of writer).
    _PRED_KIND_NAMES: ClassVar[list[str]] = [
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "mul",
        "min",
        "max",
        "pow2",
        "range",
    ]

    def _read_predicate_list(
        self, data: bytes, offset: int
    ) -> tuple[list[Predicate], int]:
        """Read a predicate list: count + per-predicate data."""
        count, offset = decode_varint(data, offset)
        predicates: list[Predicate] = []
        for _ in range(count):
            kind_byte = data[offset]
            offset += 1
            arg_count = data[offset]
            offset += 1
            if kind_byte >= len(self._PRED_KIND_NAMES):
                raise BytecodeError(f"unknown predicate kind byte: {kind_byte}")
            kind = self._PRED_KIND_NAMES[kind_byte]
            args: list[PredicateArg] = []
            for _ in range(arg_count):
                arg, offset = self._read_predicate_arg(data, offset)
                args.append(arg)
            predicates.append(Predicate(kind=kind, args=tuple(args)))
        return predicates, offset

    def _read_predicate_arg(self, data: bytes, offset: int) -> tuple[PredicateArg, int]:
        """Read a single predicate argument: tag + value."""
        tag_byte = data[offset]
        offset += 1
        match tag_byte:
            case 1:  # VALUE
                string_id, offset = decode_varint(data, offset)
                return PredicateArg(tag="value", value=self._strings[string_id]), offset
            case 2:  # CONST
                value, offset = decode_signed_varint(data, offset)
                return PredicateArg(tag="const", value=value), offset
            case _:
                raise BytecodeError(f"unknown predicate arg tag: {tag_byte}")


def read_module(data: bytes) -> Module:
    """Read a module from .loombc bytes."""
    return BytecodeReader(data).read()
