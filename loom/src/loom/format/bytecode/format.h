// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ==========================================================================
// Loom IR Binary Format Specification (.loombc)
// ==========================================================================
//
// This header defines the binary serialization format for loom IR
// modules. It is the authoritative reference for reader and writer
// implementations.
//
// ==========================================================================
// Design goals
// ==========================================================================
//
// - Fast: mmap-friendly, minimal parsing overhead. The symbol table
//   is readable without touching the IR section. Individual function
//   bodies are seekable by offset.
// - Compact: varint encoding, string interning, type interning.
// - Linkable: the symbol table carries full signature metadata for
//   linking without loading IR. One table serves as both the symbol
//   directory and the linker's import/export index.
// - Composable: multiple modules per file (archives). Single-module
//   files are the common case. Modules are independently parseable.
// - 64-bit: offsets are uint64 to support embedded weights (models
//   can be tens of GB).
// - Versionable: format_version in the header. Backward compatible
//   (newer readers read older files). Not forward compatible (older
//   readers reject newer formats with a clear error).
//
// ==========================================================================
// File layout
// ==========================================================================
//
//   +-----------------------+
//   | File header           |  Fixed size. Magic, version, module count.
//   +-----------------------+
//   | Module directory      |  One entry per module. Name + offset + length.
//   +-----------------------+
//   | File string pool      |  Shared strings: module names, source names.
//   +-----------------------+
//   | Module 0              |  Self-contained module data.
//   |   Section directory   |
//   |   Sections...         |
//   +-----------------------+
//   | Module 1 (if archive) |
//   |   ...                 |
//   +-----------------------+
//
// For single-module files (the 95% case), the module directory has
// one entry and the file string pool is minimal.
//
// ==========================================================================
// File header
// ==========================================================================
//
// All multi-byte integers are little-endian.
//
//   offset  size  field
//   0       4     magic: "LOOM" (0x4C 0x4F 0x4F 0x4D)
//   4       1     format_version (currently 1)
//   5       1     flags (see loom_bytecode_file_flags_t)
//   6       2     module_count
//   8       4     file_string_pool_length (bytes)
//   12      4     reserved (must be 0)
//   16      ...   producer (null-terminated UTF-8 string)
//
// The producer string identifies the tool that wrote the file
// (e.g., "loom-compile 1.0", "loom-link 1.0"). For diagnostics only.
// The producer ends at the first null byte. Total header size is
// 16 + strlen(producer) + 1, padded to 8-byte alignment.

#ifndef LOOM_FORMAT_BYTECODE_FORMAT_H_
#define LOOM_FORMAT_BYTECODE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_BYTECODE_MAGIC "LOOM"
#define LOOM_BYTECODE_MAGIC_LENGTH 4
#define LOOM_BYTECODE_FORMAT_VERSION 1

// File-level flags.
enum loom_bytecode_file_flag_bits_e {
  // Source locations have been stripped. No location data in any
  // module. All ops reference LOOM_LOCATION_UNKNOWN.
  LOOM_BYTECODE_FILE_FLAG_STRIPPED_LOCATIONS = 1u << 0,
};
typedef uint8_t loom_bytecode_file_flags_t;

// ==========================================================================
// Module directory
// ==========================================================================
//
// Immediately follows the file header (after alignment padding).
// One entry per module.
//
//   offset  size  field
//   0       4     name_offset (into file string pool)
//   4       2     name_length
//   6       2     flags (see loom_bytecode_module_flags_t)
//   8       8     module_offset (from file start)
//   16      8     module_length (bytes)
//
// Total: 24 bytes per module directory entry.

enum loom_bytecode_module_flag_bits_e {
  // Module contains only declarations (no function bodies).
  // The IR section is empty or absent. Used for fast linking
  // with signature-only information.
  LOOM_BYTECODE_MODULE_FLAG_DECLARATIONS_ONLY = 1u << 0,
};
typedef uint16_t loom_bytecode_module_flags_t;

// ==========================================================================
// Module structure
// ==========================================================================
//
// Each module is a self-contained unit with its own section directory
// and sections. A module can be parsed independently of other modules
// in the same file.
//
//   +-----------------------+
//   | Section directory     |  Section kind + offset + length for each.
//   +-----------------------+
//   | STRINGS section       |  Interned string table.
//   +-----------------------+
//   | SOURCES section       |  Source identifiers (filenames, tags).
//   +-----------------------+
//   | TYPES section         |  Interned type table.
//   +-----------------------+
//   | ENCODINGS section     |  Encoding kind registry + instances.
//   +-----------------------+
//   | OPS section           |  Op kind registry (kind_id -> name).
//   +-----------------------+
//   | LOCATIONS section     |  Location table.
//   +-----------------------+
//   | SYMBOLS section       |  Symbol table with full metadata.
//   +-----------------------+
//   | IR section            |  Function bodies: ops, blocks, regions.
//   +-----------------------+
//   | RESOURCES section     |  Large blobs: weights, executables, etc.
//   |                       |  (last — keeps dense metadata up front,
//   |                       |  potentially multi-GB data at the end)
//   +-----------------------+
//
// Section directory entry:
//
//   offset  size  field
//   0       2     section_kind (see loom_bytecode_section_kind_t)
//   2       2     reserved
//   4       4     reserved
//   8       8     offset (from module start)
//   16      8     length (bytes, compressed if file flag set)
//   24      8     uncompressed_length (0 if not compressed)
//
// Total: 32 bytes per section directory entry.
// Sections can appear in any order. Unknown section kinds are skipped.

typedef enum loom_bytecode_section_kind_e {
  LOOM_BYTECODE_SECTION_STRINGS = 0,    // Interned string table.
  LOOM_BYTECODE_SECTION_SOURCES = 1,    // Source identifiers (filenames, tags).
  LOOM_BYTECODE_SECTION_TYPES = 2,      // Interned type table.
  LOOM_BYTECODE_SECTION_ENCODINGS = 3,  // Encoding kind registry + instances.
  LOOM_BYTECODE_SECTION_OPS = 4,        // Op kind registry (kind_id -> name).
  LOOM_BYTECODE_SECTION_LOCATIONS = 5,  // Location table.
  LOOM_BYTECODE_SECTION_SYMBOLS = 6,    // Symbol table with full metadata.
  LOOM_BYTECODE_SECTION_IR = 7,  // Function bodies: ops, blocks, regions.
  LOOM_BYTECODE_SECTION_RESOURCES =
      8,  // Large binary blobs: compiled executables,
          // embedded weights, constant tensors.
          // Aligned, seekable, mmap-friendly.
          // Referenced by symbols and op attributes.
} loom_bytecode_section_kind_t;

#define LOOM_BYTECODE_SECTION_COUNT 9

// ==========================================================================
// Reader validation requirements
// ==========================================================================
//
// The reader MUST validate all untrusted data before use. A .loombc
// file may be malicious. The following checks are mandatory:
//
// Offset/length validation (use iree checked-math helpers):
//   - offset + length must not overflow uint64.
//   - offset >= end of preceding fixed-size header/directory.
//   - offset + length <= containing region size (file for modules,
//     module for sections).
//   - Sections must not overlap.
//
// Varint validation:
//   - Reject overlong LEB128 encodings (value fits in fewer bytes
//     than used). Prevents bypassing size heuristics.
//   - Bound-check every byte read against the section/file end.
//     A varint that hits EOF without clearing the continuation bit
//     is a parse error, not a segfault.
//   - Count fields (string_count, type_count, etc.) must be checked
//     against the remaining section bytes before allocating.
//     Reject if count * min_entry_size > remaining_bytes.
//
// Count limits (can be increased in future versions):
//   - Max strings per module: 16M (2^24)
//   - Max types per module: 64K (2^16)
//   - Max ops per module: 16M (2^24)
//   - Max symbols per module: 64K (2^16)
//   - Max resources per module: 64K (2^16)
//   - Max string length: 16M bytes (2^24)
//   - Max locations per module: 16M (2^24)
//
// Determinism:
//   - All alignment padding bytes MUST be zero.
//   - Producer string is informational only; identical inputs must
//     produce identical output when --stamp is not set.
//   - File paths in locations/sources should be relative by default
//     (not absolute) for CAS-friendly deterministic artifacts.
//
// Little-endian only. Big-endian hosts are not supported.

// ==========================================================================
// Variable-length integer encoding
// ==========================================================================
//
// Varints are used throughout for compact integer encoding.
// Encoding: LEB128 (same as DWARF, protobuf, MLIR bytecode).
//
//   value 0-127:     1 byte   (high bit clear)
//   value 128-16383: 2 bytes  (high bit set, continue)
//   ... up to 10 bytes for uint64.
//
// Signed varints use zigzag encoding:
//   encoded = (value << 1) ^ (value >> 63)

// ==========================================================================
// STRINGS section
// ==========================================================================
//
// Interned string table. All strings in the module are deduplicated
// here and referenced by varint index elsewhere.
//
//   [string_count: varint]
//   For each string:
//     [length: varint]
//     [utf8_data: length bytes]
//
// Strings are indexed 0..string_count-1. String 0 is typically the
// module name. The order is determined by the writer (typically
// insertion order during IR numbering).

// ==========================================================================
// SOURCES section
// ==========================================================================
//
// Source identifiers for the location table. Same format as STRINGS
// but stored separately because sources are context-level (shared
// across modules during linking) while strings are module-local.
//
//   [source_count: varint]
//   For each source:
//     [length: varint]
//     [utf8_data: length bytes]

// ==========================================================================
// TYPES section
// ==========================================================================
//
// Interned structural type table. Types are serialized independent of
// any runtime representation. The reader constructs runtime types from
// the bytecode data. The writer decomposes runtime types for storage.
//
// Types are structural: a dynamic dim is just "dynamic" (a flag), not
// a reference to a specific SSA value. The binding of dynamic dims to
// SSA values happens in the IR section, on the operations that define
// values with those types.
//
//   [type_count: varint]
//   For each type:
//     [kind: byte]
//       0 = none, 1 = scalar, 2 = tile, 3 = tensor, 4 = group,
//       5 = function, 6 = dialect, 7 = encoding, 8 = pool
//     (SCALAR: [element_type: byte])
//     (TILE/TENSOR:
//       [element_type: byte]
//       [rank: byte]
//       [encoding_kind: byte]     0 = none.
//                                 1 = static (instance index follows).
//                                 2 = SSA dynamic (binding on the Value,
//                                     not the type; instance index is 0).
//       [encoding_instance: varint] (0 = none, else 1-based instance index)
//       For each dim (rank times):
//         [is_dynamic: byte]      (0 = static, 1 = dynamic)
//         (if static: [size: varint]))
//     (GROUP: [scope: byte]       (loom_bytecode_group_scope_t))
//     (FUNCTION:
//       [arg_count: varint]
//       [result_count: varint]
//       For each arg/result: [type_index: varint])
//     (DIALECT:
//       [name_id: varint]         (string table index)
//       [param_count: varint]
//       For each param: [type_index: varint])
//     (ENCODING: no additional data)
//     (POOL:
//       [is_dynamic: byte]          (0 = static, 1 = dynamic)
//       (if static: [size: varint]) (block size in bytes))

// ==========================================================================
// ENCODINGS section
// ==========================================================================
//
// Encoding kind registry and parameterized instances.
//
// Kinds are registered by name. The reader looks up each name in the
// runtime's encoding vtable registry. Unknown kinds are errors.
//
// Instance parameters are opaque byte blobs. The encoding vtable's
// serialize/deserialize functions handle the blob format. This avoids
// encoding the parameter schema in the bytecode — adding a new param
// to an encoding only requires updating the vtable, not the format.
//
//   [encoding_kind_count: varint]
//   For each encoding kind:
//     [name_id: varint]       (string table index: "q6_k", "q8_0", etc.)
//
//   [instance_count: varint]
//   For each instance:
//     [kind_index: varint]    (index into the kind list above)
//     [alias_id: varint]      (string table index for printing, 0 = no alias)
//     [param_count: varint]   (number of named attribute parameters)
//     For each parameter:
//       [name_id: varint]     (string table index: "block", "group_size", etc.)
//       [value_kind: byte]    (same attribute kind bytes as IR section attrs)
//       [value_data: ...]     (kind-specific encoding, same as IR section)

// ==========================================================================
// OPS section
// ==========================================================================
//
// Op kind registry: maps op kind IDs (used in the IR section) to
// op names.
//
//   [op_count: varint]
//   For each op kind:
//     [name_id: varint]       (string table index: "tile.contract", etc.)
//
// Op kind 0 is reserved (LOOM_OP_KIND_UNKNOWN). The reader looks up
// each name in the runtime's op vtable registry. Unknown op names
// are noted but not fatal (the function containing them is skippable).

// ==========================================================================
// LOCATIONS section
// ==========================================================================
//
// Location table. Entry 0 is always kind=NONE (unknown).
// If the file has STRIPPED_LOCATIONS, this section is empty and all
// ops reference location index 0.
//
// Locations are serialized with their own byte layout (not overlaid
// on runtime structs). The reader constructs runtime location entries
// from the bytecode data.
//
//   [location_count: varint]
//   For each location:
//     [kind: byte]   (0=none, 1=file, 2=fused, 3=opaque)
//     [flags: byte]
//     (FILE:
//       [source_index: varint]  (index into SOURCES section)
//       [start_line: varint]
//       [start_col: varint]
//       [end_line: varint]
//       [end_col: varint])
//     (FUSED:
//       [child_count: varint]
//       For each child: [location_index: varint])
//     (OPAQUE:
//       [source_index: varint]  (tag: "torch", "jax", etc.)
//       [data_length: varint]
//       [data: bytes])

// ==========================================================================
// SYMBOLS section
// ==========================================================================
//
// The symbol table is the linker's primary data structure. Every
// module-level named entity (function, global, executable) has an
// entry here with full metadata — enough for linking without
// touching the IR section.
//
// The section header contains import and export offset tables that
// let the linker iterate just the imports or exports without
// scanning all N symbols. Each offset is a byte offset from the
// start of the symbol entries to that symbol's data, enabling
// O(1) access to any import or export.
//
// Section layout:
//
//   [symbol_count: varint]
//   [import_count: varint]
//   [export_count: varint]
//   // Import offset table: byte offsets to import symbol entries.
//   For each import (import_count times):
//     [symbol_entry_offset: uint64]
//   // Export offset table: byte offsets to export symbol entries.
//   For each export (export_count times):
//     [symbol_entry_offset: uint64]
//   // Symbol entries follow.
//
// Imports: A symbol with LOOM_BYTECODE_SYMBOL_FLAG_IMPORT set is a
// forward declaration of a symbol defined in another module.
// The linker resolves (source_module_id, source_symbol_id) against
// exports from other modules. The import carries full signature
// metadata for verification during linking.
//
// Exports: A symbol with PUBLIC visibility and no IMPORT flag.
// The export offset table lets the linker enumerate available
// symbols without scanning private internals.
//
// Symbol entry format:
//
//   [name_id: varint]
//   [kind: byte]            (FUNC_DEF=0, FUNC_DECL=1,
//                            FUNC_TEMPLATE=2, FUNC_UKERNEL=3,
//                            GLOBAL=4, EXECUTABLE=5)
//   [visibility: byte]      (PUBLIC=0, PRIVATE=1)
//   [flags: u16]            (see loom_bytecode_symbol_flag_bits_e)
//
//   // Import metadata. Present when LOOM_BYTECODE_SYMBOL_FLAG_IMPORT is
//   // set in flags. Identifies the source module and symbol name
//   // for cross-module linking.
//   (if LOOM_BYTECODE_SYMBOL_FLAG_IMPORT set:
//     [source_module_id: varint]  (string table index: module name)
//     [source_symbol_id: varint]) (string table index: symbol name
//                                  in the source module — may differ
//                                  from name_id for import aliasing)
//
//   For FUNC_DEF / FUNC_DECL / FUNC_TEMPLATE / FUNC_UKERNEL:
//     [calling_convention: byte]
//     [arg_count: varint]
//     [result_count: varint]
//     For each arg:
//       [type_index: varint]  (into TYPES section)
//     For each result:
//       [is_tied: byte]
//       [type_index: varint]
//       (if tied: [tied_operand_index: varint])
//     [tied_result_count: varint]
//     [predicate_count: varint]
//     For each predicate:
//       [kind: byte]       (0=eq, 1=lt, 2=le, 3=gt, 4=ge,
//                           5=mul, 6=min, 7=max, 8=pow2, 9=range)
//       [arg_count: byte]
//       For each arg:
//         [tag: byte]      (1=value, 2=ordinal, 3=const)
//         (VALUE:   [name_id: varint]  string table index)
//         (ORDINAL: [ordinal: varint])
//         (CONST:   [value: signed_varint])
//
//     // Template/ukernel metadata (FUNC_TEMPLATE or FUNC_UKERNEL):
//     (if template or ukernel:
//       [implements_op_name: varint]  (string index of op name)
//       [priority: varint])
//
//     // IR body reference. A has_body byte precedes the offset/length
//     // pair so that declarations-only modules and stripped modules
//     // can omit IR references even for nominally-bodied kinds.
//     [has_body: byte]
//     (if has_body:
//       [ir_offset: u64]    (from IR section start)
//       [ir_length: u32])
//
//   For GLOBAL symbols:
//     [type_index: varint]
//     [is_mutable: byte]
//     [initializer_offset: u64]  (into RESOURCES section)
//     [initializer_length: u32]
//
//   For EXECUTABLE symbols:
//     // Compiled device code, target metadata, ABI info.
//     // Details TBD as we build the backend.

// ==========================================================================
// IR section
// ==========================================================================
//
// Function bodies: the actual operations, blocks, and regions.
// Each function's IR is a contiguous byte range addressable by
// (ir_offset, ir_length) from its symbol table entry. This enables
// function-level lazy loading.
//
// Within a function's IR:
//
//   [block_count: varint]
//   For each block:
//     [has_label: byte]
//     (if has_label: [label_id: varint])
//     [arg_count: varint]
//     For each block arg (these DEFINE SSA values):
//       [name_id: varint]       (SSA name for round-trip printing)
//       [type_index: varint]    (structural type from TYPES section)
//       [dim_binding_count: varint]  (number of dynamic dims in this type)
//       For each dynamic dim (in shape order, skipping static dims):
//         [value_ref: signed_varint]  Signed zigzag encoding.
//              Non-negative: block-local value number (SSA dim).
//              Negative: result ordinal (-1 = #0, -2 = #1, etc.).
//       [encoding_binding: varint]    0 = no encoding binding.
//              N > 0: the value is (N-1), a block-local value number
//              referencing an encoding-typed SSA value.
//     [op_count: varint]
//     For each op:
//       [kind_id: varint]       (index into OPS section)
//       [flags: byte]           (encoding mask: which fields present)
//       [location_id: varint]
//       [operand_count: varint]
//       For each operand (these REFERENCE existing SSA values):
//         [value_ref: varint]   (just the value ref, no type info)
//       [result_count: varint]
//       For each result (these DEFINE SSA values):
//         [name_id: varint]
//         [type_index: varint]
//         [dim_binding_count: varint]
//         For each dynamic dim:
//           [value_ref: signed_varint]  Same encoding as block args.
//         [encoding_binding: varint]    Same encoding as block args.
//       [tied_result_count: varint]
//       For each tied result:
//         [result_index: varint]
//         [operand_index: varint]
//       [attr_count: varint]
//       For each attribute:
//         [key_id: varint]
//         [value_kind: byte]
//         [value_data: ...]
//       [region_count: varint]
//       For each region:
//         (recursive: block_count, blocks...)

// ==========================================================================
// RESOURCES section
// ==========================================================================
//
// Large binary blobs: compiled executables, embedded weights, constant
// tensors, opaque data. Referenced by index from symbol table entries
// (globals, executables) and op attributes (dense constants).
//
// This section can be very large (GB of model weights). The reader
// mmaps it and accesses by offset rather than loading entirely.
// The section uses a fixed-size directory for O(1) random access
// to any resource by index.
//
// Layout:
//
//   +-----------------------------------------+
//   | loom_bytecode_resource_section_header_t        |  Fixed 8 bytes.
//   +-----------------------------------------+
//   | loom_bytecode_resource_dir_entry_t[entry_count]|  24 bytes each.
//   +-----------------------------------------+
//   | Resource data payloads (aligned)         |
//   +-----------------------------------------+
//
// The header and directory are fixed-size for mmap overlay. The
// reader can cast the section start to the header struct, read
// entry_count, and index directly into the directory array for
// O(1) access to any resource.
//
// Each directory entry stores the byte offset (from section start)
// and length of the resource's data payload. The writer ensures
// each payload starts at the alignment declared in the directory
// entry. The reader uses data_offset to jump directly to the data
// without scanning preceding entries.
//
// All padding bytes between payloads MUST be zero (determinism,
// prevents leaking host memory).

// ==========================================================================
// Linking workflow
// ==========================================================================
//
// Static linking reads only the lightweight sections (STRINGS,
// TYPES, SYMBOLS) from each input module. The IR section is loaded
// only for functions being pulled into the output.
//
//   1. Read SYMBOLS from each input module.
//   2. Resolve: match imports (declarations) to exports (definitions).
//   3. Verify: signatures must be compatible (types, tied, where).
//   4. Compute reachability: from main module's entry points, walk
//      call graph through resolved symbols.
//   5. Load IR only for reachable functions.
//   6. Merge: combine string/type/encoding tables, renumber.
//   7. Internalize: linked symbols become private.
//   8. Write output module.
//
// Recipe pruning: functions with `implements` annotations are live
// as long as the implemented op kind is present in any reachable
// function. After lowering consumes recipes, unmatched ones are DCE'd.
//
// Visibility rules:
//   Archive (merge):  public stays public.
//   Static link:      imported symbols become private.
//   Final link:       only main module's exports stay public.

// ==========================================================================
// Bytecode wire enums
// ==========================================================================
//
// These enums define the byte values used in the bytecode format.
// They are the protocol contract between writers and readers. Internal
// IR enums (loom_type_kind_t, loom_scalar_type_t, etc.) may differ;
// the writer maps internal → wire, the reader maps wire → internal.
//
// Scalar types (loom_scalar_type_t) and these wire values are
// identity-mapped by convention. Both are append-only; do not reorder.

// Type kind byte in the TYPES section. Each type entry begins with
// this byte to identify the type's structure and payload format.
typedef enum loom_bytecode_type_kind_e {
  LOOM_BYTECODE_TYPE_NONE = 0,
  LOOM_BYTECODE_TYPE_SCALAR = 1,
  LOOM_BYTECODE_TYPE_TILE = 2,
  LOOM_BYTECODE_TYPE_TENSOR = 3,
  LOOM_BYTECODE_TYPE_GROUP = 4,
  LOOM_BYTECODE_TYPE_FUNCTION = 5,
  LOOM_BYTECODE_TYPE_DIALECT = 6,
  LOOM_BYTECODE_TYPE_ENCODING = 7,
  LOOM_BYTECODE_TYPE_POOL = 8,
} loom_bytecode_type_kind_t;

// Group scope byte in the TYPES section (GROUP payload).
// Append-only; do not reorder.
typedef enum loom_bytecode_group_scope_e {
  LOOM_BYTECODE_GROUP_SCOPE_WORKGROUP = 0,
  LOOM_BYTECODE_GROUP_SCOPE_SUBGROUP = 1,
} loom_bytecode_group_scope_t;

// Encoding attachment kind byte in the TYPES section (TILE/TENSOR payload).
typedef enum loom_bytecode_encoding_kind_e {
  LOOM_BYTECODE_ENCODING_NONE = 0,
  LOOM_BYTECODE_ENCODING_STATIC = 1,
  LOOM_BYTECODE_ENCODING_SSA = 2,
} loom_bytecode_encoding_kind_t;

// ==========================================================================
// Code definitions
// ==========================================================================
//
// The structs below describe the on-disk byte layout. They are NOT
// the same as runtime IR types (loom_type_t, loom_value_t, etc.).
// The reader constructs runtime types from bytecode data. The writer
// serializes runtime types into bytecode format. The two layouts can
// evolve independently.
//
// Fixed-size structs (file header, section directory) are used for
// mmap overlays. Variable-length data (types, ops, strings) uses
// varint encoding and is parsed sequentially.

// File header. The producer string follows as a flexible array member.
// Total on-disk size: sizeof(header) + strlen(producer) + 1, padded
// to 8-byte alignment.
typedef struct loom_bytecode_file_header_t {
  uint8_t magic[LOOM_BYTECODE_MAGIC_LENGTH];
  uint8_t format_version;
  loom_bytecode_file_flags_t flags;
  uint16_t module_count;
  uint32_t file_string_pool_length;
  uint32_t reserved;
  // Null-terminated UTF-8 string identifying the producing tool
  // (e.g., "loom-compile 1.0"). For diagnostics only.
  char producer[];
} loom_bytecode_file_header_t;

// Module directory entry.
typedef struct loom_bytecode_module_dir_entry_t {
  // Byte offset into the file string pool.
  uint32_t name_offset;
  // Length of the module name in bytes.
  uint16_t name_length;
  loom_bytecode_module_flags_t flags;
  // Absolute byte offset from file start to the module's first byte.
  uint64_t module_offset;
  // Total byte length of the module's data.
  uint64_t module_length;
} loom_bytecode_module_dir_entry_t;

enum loom_bytecode_section_flag_bits_e {
  // Section data is zstd-compressed. The length field holds the
  // compressed size; uncompressed_length holds the original size.
  // RESOURCES sections should NOT be compressed (breaks mmap).
  // Metadata sections (STRINGS, TYPES, OPS, LOCATIONS) may benefit.
  LOOM_BYTECODE_SECTION_FLAG_COMPRESSED = 1u << 0,
};
typedef uint16_t loom_bytecode_section_flags_t;

// Section directory entry within a module.
typedef struct loom_bytecode_section_dir_entry_t {
  // Section kind (loom_bytecode_section_kind_t). Unknown kinds are skipped.
  uint16_t section_kind;
  // Per-section flags (compression, etc.).
  loom_bytecode_section_flags_t flags;
  uint32_t reserved;
  // Byte offset from the module's start.
  uint64_t offset;
  // Byte length. If COMPRESSED flag is set, this is the compressed
  // size; otherwise the raw data size.
  uint64_t length;
  // Original uncompressed size. 0 if section is not compressed.
  uint64_t uncompressed_length;
} loom_bytecode_section_dir_entry_t;

// Symbol flags (in the u16 flags field of each symbol entry).
enum loom_bytecode_symbol_flag_bits_e {
  // Symbol is publicly visible (exported). The linker uses this
  // to determine which symbols are available for cross-module
  // references. Matches SYMBOL_FLAG_PUBLIC in the Python IR.
  LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC = 1u << 0,
  // Symbol is an import: declared here, defined in another module.
  // When set, source_module_id and source_symbol_id varints follow
  // the flags field. The linker resolves the import against exports
  // from the named source module.
  LOOM_BYTECODE_SYMBOL_FLAG_IMPORT = 1u << 1,
};
typedef uint16_t loom_bytecode_symbol_flags_t;

// Resource section header. Fixed-size for mmap overlay.
typedef struct loom_bytecode_resource_section_header_t {
  uint32_t entry_count;
  uint32_t reserved;
} loom_bytecode_resource_section_header_t;

// Resource directory entry. One per resource, stored as a flat array
// immediately after the section header. The reader indexes into this
// array for O(1) access to any resource.
typedef struct loom_bytecode_resource_dir_entry_t {
  // Byte offset from the RESOURCES section start to the resource's
  // data payload. The writer aligns each payload to the declared
  // alignment; the reader jumps directly via this offset.
  uint64_t data_offset;
  // Byte length of the resource data (not including alignment padding).
  uint64_t data_length;
  // Required alignment of the data payload (power of 2, e.g., 64 for
  // cache line, 4096 for page). The writer ensures the data at
  // data_offset satisfies this alignment relative to the section start.
  uint16_t alignment;
  uint16_t reserved_0;
  uint32_t reserved_1;
} loom_bytecode_resource_dir_entry_t;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_FORMAT_H_
