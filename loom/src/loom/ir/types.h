// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom IR type system.
//
// Types describe the shapes and element types of values in loom IR.
// The type system has five categories:
//
//   Scalar types:    f16, f32, f64, bf16, i1, i8, i16, i32, i64, index
//   Tile types:      tile<[%M]x4xf32>     (local compute, register-like)
//   Tensor types:    tensor<[%M]xf32>     (global storage, memory-backed)
//   Group types:     group<scope>         (barrier scoping)
//   Function types:  (f32, i32) -> (f64)  (callable signatures)
//
// Shape dimensions are either static integers, dynamic SSA value
// references, or result ordinal references. Static dims are known
// at compile time. Dynamic dims reference SSA values (index-typed)
// that provide the runtime size. Ordinal dims reference another
// co-result by position, enabling ops that return both data and
// its size:
//
//   tile<4x4xf32>           All static.
//   tile<[%M]x4xf32>        First dim dynamic, named %M.
//   tensor<[%M]x[%K]xf32>   Both dims dynamic.
//   tensor<[#1]xf32>        First dim references result #1.
//
// Every dynamic dim has a name. In function signatures, named dims
// implicitly define index-typed SSA values in the function body.
//
// Types may carry an encoding that describes the physical data layout
// (quantization, packing, etc.):
//
//   tile<256x256xf32, #q6_k>
//   tensor<[%N]x[%K]xi8, #q8_0<block=32>>
//
// The encoding is metadata: it doesn't change the logical shape or
// element type, but determines how bytes map to logical elements.
//
// ==========================================================================
// Memory layout
// ==========================================================================
//
// Types are 24 bytes, fitting 2-3 per cache line. For the common case
// (rank <= 2 tiles/vectors), everything is inline with no pointer
// chases. Types are interned per module, so pointer equality works.
//
//   bytes 0-3:  packed header (kind, element_type, rank, flags)
//   bytes 4-5:  encoding_id
//   bytes 6-7:  encoding_flags
//   bytes 8-23: dims (inline for rank <= 2, overflow pointer otherwise)
//
// Inline dim packing (rank <= 2):
//   Each dim is a uint64_t. Top two bits encode the kind:
//   bits 63:62 = 00: static (bits 0-61 = size).
//   bits 63:62 = 10: dynamic SSA value (bits 0-61 = value_id).
//   bits 63:62 = 11: result ordinal (bits 0-61 = ordinal index).
//
// Overflow (rank > 2):
//   dims[0] = pointer to arena-allocated loom_dim_t array.
//   dims[1] = precomputed hash for fast inequality rejection.
//
// Function types use dims[0] as a pointer to an arena-allocated
// loom_func_type_data_t containing the arg and result type arrays.
// Function types may carry SSA-specific dim bindings within their
// arg/result types, so the embedded types are stored by value (24
// bytes each), not interned.
//
// Type equality: since types are interned, pointer comparison suffices.
// During construction, the type table deduplicates by content, so
// structurally identical types share the same pointer.

#ifndef LOOM_IR_TYPES_H_
#define LOOM_IR_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// IDs
//===----------------------------------------------------------------------===//

// Index into the module's value table (module->values.entries[]).
// Values are 64-byte cache-line-aligned entries. Prefer passing
// value IDs over value pointers for stability across IR mutations.
typedef uint32_t loom_value_id_t;
#define LOOM_VALUE_ID_INVALID ((loom_value_id_t)UINT32_MAX)

// Index into the module's interned string table.
// Strings are deduplicated: identical strings share the same ID.
typedef uint32_t loom_string_id_t;
#define LOOM_STRING_ID_INVALID ((loom_string_id_t)UINT32_MAX)

// Index into the module's symbol table (module->symbols.entries[]).
typedef uint16_t loom_symbol_id_t;

// Index into the module's interned type table (module->types.entries[]).
// Types are deduplicated: structurally identical types share the same ID.
typedef uint32_t loom_type_id_t;
#define LOOM_TYPE_ID_INVALID ((loom_type_id_t)UINT32_MAX)

//===----------------------------------------------------------------------===//
// Scalars, dims, and enums
//===----------------------------------------------------------------------===//

// Scalar element type kind.
//
// These are internal compiler values, NOT bytecode-stable. The bytecode
// format has its own encoding with independent versioning. Mapping
// between internal and bytecode representations lives in bytecode/
// reader.c and writer.c.
//
// Ordered: address types, integers by width, floats by width.
typedef enum loom_scalar_type_e {
  // Signed target-width integer for loop bounds, dimension sizes, and
  // general indexing arithmetic. Arithmetic follows signed semantics
  // (divsi, remsi, signed comparisons).
  LOOM_SCALAR_TYPE_INDEX = 0,
  // Unsigned target-width integer for buffer byte offsets and
  // addressing. Arithmetic follows unsigned semantics (divui, remui,
  // unsigned comparisons). Same bitwidth as index.
  LOOM_SCALAR_TYPE_OFFSET = 1,
  // 1-bit integer. Boolean results (comparisons, predicates).
  LOOM_SCALAR_TYPE_I1 = 2,
  // 8-bit signed integer. Quantized weights, byte-level data.
  LOOM_SCALAR_TYPE_I8 = 3,
  // 16-bit signed integer. Intermediate quantized computations.
  LOOM_SCALAR_TYPE_I16 = 4,
  // 32-bit signed integer. General-purpose integer arithmetic.
  LOOM_SCALAR_TYPE_I32 = 5,
  // 64-bit signed integer. Large counts, hash values.
  LOOM_SCALAR_TYPE_I64 = 6,
  // 8-bit float, E4M3 variant (1 sign, 4 exponent, 3 mantissa).
  // Range [-448, 448], no infinities. FP8 training/inference.
  // IEEE draft: https://arxiv.org/abs/2209.05433
  LOOM_SCALAR_TYPE_F8E4M3 = 7,
  // 8-bit float, E5M2 variant (1 sign, 5 exponent, 2 mantissa).
  // Range [-57344, 57344], has infinities and NaN. FP8 gradients.
  // IEEE draft: https://arxiv.org/abs/2209.05433
  LOOM_SCALAR_TYPE_F8E5M2 = 8,
  // IEEE 754 binary16 half-precision (1 sign, 5 exponent, 10 mantissa).
  // Range [-65504, 65504]. Mobile inference, GPU compute.
  LOOM_SCALAR_TYPE_F16 = 9,
  // bfloat16 (1 sign, 8 exponent, 7 mantissa). Same exponent range
  // as f32, reduced mantissa precision. Training and inference on
  // TPUs, GPUs with bf16 support.
  LOOM_SCALAR_TYPE_BF16 = 10,
  // IEEE 754 binary32 single-precision (1 sign, 8 exponent, 23 mantissa).
  // Standard floating-point type.
  LOOM_SCALAR_TYPE_F32 = 11,
  // IEEE 754 binary64 double-precision (1 sign, 11 exponent, 52 mantissa).
  // High-precision accumulation, loss computation.
  LOOM_SCALAR_TYPE_F64 = 12,
  LOOM_SCALAR_TYPE_COUNT_,
} loom_scalar_type_t;

// Returns the name string for a scalar type (e.g., "f32", "index").
// The returned string is static and does not need to be freed.
const char* loom_scalar_type_name(loom_scalar_type_t type);

// Returns the bitwidth of a scalar type. Index returns 64.
int32_t loom_scalar_type_bitwidth(loom_scalar_type_t type);

// Parses a scalar type name. Returns true on success.
bool loom_scalar_type_parse(iree_string_view_t name,
                            loom_scalar_type_t* out_type);

// Dimension packing uses the top two bits (63:62) to distinguish
// three kinds of dimension:
//
//   bits 63:62 = 00: static dimension (bits 0-61 = size).
//   bits 63:62 = 10: dynamic SSA value (bits 0-61 = value_id).
//   bits 63:62 = 11: result ordinal (bits 0-61 = ordinal index).
//
// Bit pattern 01 is reserved (static sizes never need bit 62 since
// max static size is 2^62 - 1, which is still astronomical).
//
// Result ordinal dims reference another co-result by position:
// tensor<[#1]xf32> means "my first dimension equals whatever result
// #1 is at runtime." This enables operations that return both data
// and its size without requiring separate dispatches or readback.

// Dynamic flag: top bit of a packed dimension value. Set for both
// SSA value dims and result ordinal dims.
#define LOOM_DIM_DYNAMIC_FLAG ((uint64_t)1 << 63)

// Ordinal flag: bits 63:62 both set. Distinguishes result ordinal
// dims from SSA value dims (which have only bit 63 set).
#define LOOM_DIM_ORDINAL_FLAG ((uint64_t)3 << 62)

// Mask covering the payload bits (0-61).
#define LOOM_DIM_PAYLOAD_MASK (((uint64_t)1 << 62) - 1)

// Maximum static dimension size (2^62 - 1).
#define LOOM_DIM_MAX_STATIC_SIZE (((int64_t)1 << 62) - 1)

// Maximum SSA value ID that fits in a packed dim.
#define LOOM_DIM_MAX_VALUE_ID (((uint64_t)1 << 62) - 1)

// Maximum result ordinal that fits in a packed dim.
#define LOOM_DIM_MAX_ORDINAL (((uint32_t)1 << 30) - 1)

// Packs a static dimension into the inline format.
static inline uint64_t loom_dim_pack_static(int64_t size) {
  return (uint64_t)size;  // Top two bits clear = static.
}

// Packs a dynamic dimension (SSA value ID) into the inline format.
static inline uint64_t loom_dim_pack_dynamic(uint64_t value_id) {
  return LOOM_DIM_DYNAMIC_FLAG | value_id;
}

// Packs a result ordinal dimension into the inline format.
static inline uint64_t loom_dim_pack_ordinal(uint32_t ordinal) {
  return LOOM_DIM_ORDINAL_FLAG | (uint64_t)ordinal;
}

// Returns true if a packed dimension is dynamic (either SSA value or
// result ordinal — the size is not statically known in either case).
static inline bool loom_dim_is_dynamic(uint64_t packed) {
  return (packed & LOOM_DIM_DYNAMIC_FLAG) != 0;
}

// Returns true if a packed dimension is a result ordinal reference.
// Ordinal dims have both bits 63 and 62 set.
static inline bool loom_dim_is_ordinal(uint64_t packed) {
  return (packed >> 62) == 3;
}

// Returns the result ordinal index from a packed ordinal dimension.
// Caller must check loom_dim_is_ordinal() first.
static inline uint32_t loom_dim_ordinal(uint64_t packed) {
  return (uint32_t)(packed & LOOM_DIM_PAYLOAD_MASK);
}

// Returns the static size from a packed dimension.
// Caller must check !loom_dim_is_dynamic() first.
static inline int64_t loom_dim_static_size(uint64_t packed) {
  return (int64_t)packed;
}

// Returns the SSA value ID from a packed dynamic dimension.
// Caller must check loom_dim_is_dynamic() && !loom_dim_is_ordinal().
static inline loom_value_id_t loom_dim_value_id(uint64_t packed) {
  uint64_t payload = packed & LOOM_DIM_PAYLOAD_MASK;
  IREE_ASSERT(payload <= UINT32_MAX,
              "dim value ID exceeds loom_value_id_t range");
  return (loom_value_id_t)payload;
}

// Overflow dimension for rank > 2. Same packing as inline dims.
// Arena-allocated array, pointed to by dims[0] in the type struct.
typedef uint64_t loom_overflow_dim_t;

// Type kind tag.
//
// These are internal compiler values. The bytecode format has its own
// wire enum (loom_bytecode_type_kind_t in format.h) with independent
// versioning. Currently identity-mapped; append-only, do not reorder.
typedef enum loom_type_kind_e {
  LOOM_TYPE_NONE = 0,      // Absence of a type (no-result ops).
  LOOM_TYPE_SCALAR = 1,    // f32, i8, index, etc.
  LOOM_TYPE_TILE = 2,      // tile<[%M]x4xf32>
  LOOM_TYPE_TENSOR = 3,    // tensor<[%M]xf32>
  LOOM_TYPE_GROUP = 4,     // group<scope>
  LOOM_TYPE_FUNCTION = 5,  // (types) -> (types)
  LOOM_TYPE_DIALECT = 6,   // hal.buffer, vm.ref<T>, etc.
  LOOM_TYPE_ENCODING = 7,  // encoding (first-class SSA encoding value)
  LOOM_TYPE_POOL = 8,      // pool<[%block_size]> (block-managed memory)
  LOOM_TYPE_COUNT_,
} loom_type_kind_t;

// Group scope kind. Stored in the element_type byte of the type header
// for LOOM_TYPE_GROUP (that byte is unused for non-shaped types).
typedef enum loom_group_scope_e {
  LOOM_GROUP_SCOPE_WORKGROUP = 0,
  LOOM_GROUP_SCOPE_SUBGROUP = 1,
  LOOM_GROUP_SCOPE_COUNT_,
} loom_group_scope_t;

// Returns the name string for a group scope kind, or NULL if invalid.
static inline const char* loom_group_scope_name(loom_group_scope_t scope) {
  static const char* const names[] = {
      [LOOM_GROUP_SCOPE_WORKGROUP] = "workgroup",
      [LOOM_GROUP_SCOPE_SUBGROUP] = "subgroup",
  };
  if (scope < LOOM_GROUP_SCOPE_COUNT_) return names[scope];
  return NULL;
}

// Type flags (packed into header bits 20-23).
enum loom_type_flags_e {
  // Shape dims are stored inline in dims[0..1].
  // Clear = dims[0] is a pointer to overflow array.
  LOOM_TYPE_FLAG_INLINE_DIMS = (1 << 0),
  // Type has all-static dimensions (no dynamic dims).
  // Enables fast-path checks without inspecting individual dims.
  LOOM_TYPE_FLAG_ALL_STATIC = (1 << 1),
};

// Encoding flag bits for the encoding_flags field of loom_type_t.
//
// When LOOM_ENCODING_FLAG_SSA is set, encoding_id holds a value_id
// (uint16_t) referencing an encoding-typed SSA value instead of a
// module encoding table index. This enables dynamic encodings that
// propagate through function signatures as SSA values.
//
// When LOOM_ENCODING_FLAG_ORDINAL is set, encoding_id holds a result
// ordinal index (like dim ordinals [#N]). The referenced co-result
// is encoding-typed. Used in return types: tile<[#1]xf32, #2> means
// result #2 is the encoding value for this tile.
enum loom_encoding_flag_bits_e {
  LOOM_ENCODING_FLAG_SSA = (1u << 0),
  LOOM_ENCODING_FLAG_ORDINAL = (1u << 1),
};
typedef uint16_t loom_encoding_flags_t;

//===----------------------------------------------------------------------===//
// Type struct and accessors
//===----------------------------------------------------------------------===//

// 24-byte interned type. Fits 2-3 per cache line.
// Types are interned per module: pointer equality ≡ structural equality.
typedef struct loom_type_t {
  // Packed header:
  //   [0:7]   loom_type_kind_t
  //   [8:15]  loom_scalar_type_t (shaped types) or loom_group_scope_t (group)
  //   [16:19] rank (0-15 for shaped types, 0 otherwise)
  //   [20:23] loom_type_flags_e (inline_dims, all_static)
  //   [24:31] reserved
  uint32_t header;

  // Index into the module's encoding table (1-based, 0 = no encoding).
  // When encoding_flags has LOOM_ENCODING_FLAG_SSA set, this is instead
  // a value_id referencing an SSA value of type LOOM_TYPE_ENCODING.
  uint16_t encoding_id;

  // Encoding interpretation flags. Zero for most types.
  loom_encoding_flags_t encoding_flags;

  // Inline dim storage for rank <= 2. Each element is a packed dim
  // (see loom_dim_pack_*). For rank > 2, dims[0] is a pointer to an
  // arena-allocated loom_overflow_dim_t array and dims[1] is a
  // precomputed hash for fast inequality rejection. For function
  // types, dims[0] is a pointer to loom_func_type_data_t.
  uint64_t dims[2];
} loom_type_t;

// Static assert: type must be exactly 24 bytes, no padding.
_Static_assert(sizeof(loom_type_t) == 24, "loom_type_t must be 24 bytes");

// Arena-allocated overflow data for function types. Stored via pointer
// in dims[0] of a LOOM_TYPE_FUNCTION loom_type_t. The types array
// holds arg types followed by result types contiguously:
//   types[0 .. arg_count-1]                      = argument types
//   types[arg_count .. arg_count+result_count-1]  = result types
//
// The embedded types are stored by value (not interned) because they
// may carry SSA-specific dim bindings that differ per use site.
typedef struct loom_func_type_data_t {
  uint16_t arg_count;
  uint16_t result_count;
  uint32_t reserved;  // Padding for loom_type_t alignment (8 bytes).
  loom_type_t types[];
} loom_func_type_data_t;

_Static_assert(sizeof(loom_func_type_data_t) == 8,
               "loom_func_type_data_t header must be 8 bytes");

// --- Header accessors ---

static inline loom_type_kind_t loom_type_kind(loom_type_t type) {
  return (loom_type_kind_t)(type.header & 0xFF);
}

static inline loom_scalar_type_t loom_type_element_type(loom_type_t type) {
  return (loom_scalar_type_t)((type.header >> 8) & 0xFF);
}

static inline uint8_t loom_type_rank(loom_type_t type) {
  return (uint8_t)((type.header >> 16) & 0xF);
}

static inline uint8_t loom_type_flags(loom_type_t type) {
  return (uint8_t)((type.header >> 20) & 0xF);
}

static inline bool loom_type_has_inline_dims(loom_type_t type) {
  return (loom_type_flags(type) & LOOM_TYPE_FLAG_INLINE_DIMS) != 0;
}

static inline bool loom_type_is_all_static(loom_type_t type) {
  return (loom_type_flags(type) & LOOM_TYPE_FLAG_ALL_STATIC) != 0;
}

// Returns the group scope for a type. Only valid when kind == LOOM_TYPE_GROUP.
// The scope is stored in the element_type byte of the header.
static inline loom_group_scope_t loom_type_group_scope(loom_type_t type) {
  return (loom_group_scope_t)((type.header >> 8) & 0xFF);
}

static inline uint32_t loom_type_make_header(loom_type_kind_t kind,
                                             loom_scalar_type_t element_type,
                                             uint8_t rank, uint8_t flags) {
  return ((uint32_t)kind & 0xFF) | (((uint32_t)element_type & 0xFF) << 8) |
         (((uint32_t)rank & 0xF) << 16) | (((uint32_t)flags & 0xF) << 20);
}

// --- Dim accessors ---

// Returns the packed dim at the given index.
// For inline types (rank <= 2): reads from dims[index].
// For overflow types: reads from the overflow array.
static inline uint64_t loom_type_dim(loom_type_t type, iree_host_size_t index) {
  if (IREE_LIKELY(loom_type_has_inline_dims(type))) {
    return type.dims[index];
  }
  const loom_overflow_dim_t* overflow =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  return overflow[index];
}

static inline bool loom_type_dim_is_dynamic_at(loom_type_t type,
                                               iree_host_size_t index) {
  return loom_dim_is_dynamic(loom_type_dim(type, index));
}

static inline int64_t loom_type_dim_static_size_at(loom_type_t type,
                                                   iree_host_size_t index) {
  return loom_dim_static_size(loom_type_dim(type, index));
}

static inline loom_value_id_t loom_type_dim_value_id_at(
    loom_type_t type, iree_host_size_t index) {
  return loom_dim_value_id(loom_type_dim(type, index));
}

// --- Function type accessors ---

// Returns the function type data for a LOOM_TYPE_FUNCTION type.
// The pointer is stored in dims[0]. Only valid when kind == LOOM_TYPE_FUNCTION.
static inline const loom_func_type_data_t* loom_type_func_data(
    loom_type_t type) {
  return (const loom_func_type_data_t*)(uintptr_t)type.dims[0];
}

// Returns the number of argument types in a function type.
static inline uint16_t loom_type_func_arg_count(loom_type_t type) {
  return loom_type_func_data(type)->arg_count;
}

// Returns the number of result types in a function type.
static inline uint16_t loom_type_func_result_count(loom_type_t type) {
  return loom_type_func_data(type)->result_count;
}

// Returns a pointer to the argument type array.
static inline const loom_type_t* loom_type_func_arg_types(loom_type_t type) {
  return loom_type_func_data(type)->types;
}

// Returns a pointer to the result type array.
static inline const loom_type_t* loom_type_func_result_types(loom_type_t type) {
  const loom_func_type_data_t* data = loom_type_func_data(type);
  return data->types + data->arg_count;
}

// --- Scalar type classification ---

// Returns true if the scalar type is a pure integer (I1, I8, I16, I32, I64).
// INDEX and OFFSET are address types, not pure integers — they have
// target-dependent width and sign semantics.
static inline bool loom_scalar_type_is_integer(loom_scalar_type_t type) {
  return type >= LOOM_SCALAR_TYPE_I1 && type <= LOOM_SCALAR_TYPE_I64;
}

// Returns true if the scalar type is a floating-point type
// (F8E4M3, F8E5M2, F16, BF16, F32, F64).
static inline bool loom_scalar_type_is_float(loom_scalar_type_t type) {
  return type >= LOOM_SCALAR_TYPE_F8E4M3 && type <= LOOM_SCALAR_TYPE_F64;
}

// --- Kind checks ---

static inline bool loom_type_is_scalar(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_SCALAR;
}

static inline bool loom_type_is_tile(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_TILE;
}

static inline bool loom_type_is_tensor(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_TENSOR;
}

static inline bool loom_type_is_function(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_FUNCTION;
}

static inline bool loom_type_is_encoding(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_ENCODING;
}

static inline bool loom_type_is_dialect(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_DIALECT;
}

static inline bool loom_type_is_pool(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_POOL;
}

// Returns true if the type is shaped (has rank, dims, element type):
// tile or tensor. Scalar types are NOT shaped.
static inline bool loom_type_is_shaped(loom_type_t type) {
  loom_type_kind_t kind = loom_type_kind(type);
  return kind == LOOM_TYPE_TILE || kind == LOOM_TYPE_TENSOR;
}

// --- Type comparison ---

// Returns true if two types have the same element type.
// Meaningful for scalar and shaped types. For non-element-bearing types
// (group, function, encoding, pool), compares the raw header byte.
static inline bool loom_type_element_type_equals(loom_type_t a, loom_type_t b) {
  return loom_type_element_type(a) == loom_type_element_type(b);
}

// Returns true if two types have the same rank.
static inline bool loom_type_rank_equals(loom_type_t a, loom_type_t b) {
  return loom_type_rank(a) == loom_type_rank(b);
}

// Returns true if two types have identical shapes (same rank and all
// dims match — static sizes, dynamic value IDs, and ordinal refs).
static inline bool loom_type_shape_equals(loom_type_t a, loom_type_t b) {
  uint8_t rank_a = loom_type_rank(a);
  if (rank_a != loom_type_rank(b)) return false;
  for (uint8_t i = 0; i < rank_a; ++i) {
    if (loom_type_dim(a, i) != loom_type_dim(b, i)) return false;
  }
  return true;
}

// Returns true if two types are structurally equal: same kind, element
// type, rank, all dims (including overflow dims for rank > 2), and
// encoding. This is the full equality check — use the narrower
// predicates above (element_type_equals, shape_equals, etc.) when
// only a specific aspect needs comparison.
bool loom_type_equal(loom_type_t a, loom_type_t b);

// Returns true if two types have the same encoding (same encoding_id
// and encoding_flags). Both types having no encoding counts as equal.
static inline bool loom_type_encoding_equals(loom_type_t a, loom_type_t b) {
  return a.encoding_id == b.encoding_id && a.encoding_flags == b.encoding_flags;
}

// Returns true if the type carries any encoding (static, SSA, or ordinal).
// Static encodings have encoding_id > 0 with no flags. SSA and ordinal
// encodings may have encoding_id == 0 (valid value_id or ordinal index),
// so we also check encoding_flags.
static inline bool loom_type_has_encoding(loom_type_t type) {
  return type.encoding_id != 0 || type.encoding_flags != 0;
}

// Returns true if the encoding is an SSA value reference rather than
// a static encoding table index.
static inline bool loom_type_has_ssa_encoding(loom_type_t type) {
  return (type.encoding_flags & LOOM_ENCODING_FLAG_SSA) != 0;
}

// Returns true if the encoding is a result ordinal reference (#N in
// encoding position). The encoding_id holds the ordinal index.
static inline bool loom_type_has_ordinal_encoding(loom_type_t type) {
  return (type.encoding_flags & LOOM_ENCODING_FLAG_ORDINAL) != 0;
}

// Returns true if the type has a static (non-SSA, non-ordinal) encoding.
static inline bool loom_type_has_static_encoding(loom_type_t type) {
  return type.encoding_id != 0 && !loom_type_has_ssa_encoding(type) &&
         !loom_type_has_ordinal_encoding(type);
}

// Returns the SSA value_id for a dynamic encoding. Only valid when
// loom_type_has_ssa_encoding() returns true.
static inline uint16_t loom_type_encoding_value_id(loom_type_t type) {
  return type.encoding_id;
}

// Returns the result ordinal index for an ordinal encoding. Only valid
// when loom_type_has_ordinal_encoding() returns true.
static inline uint32_t loom_type_encoding_ordinal(loom_type_t type) {
  return type.encoding_id;
}

//===----------------------------------------------------------------------===//
// Type construction and encodings
//===----------------------------------------------------------------------===//

// Creates an encoding type (the type of encoding SSA values).
// No shape, no element type, no dims — just the kind tag.
static inline loom_type_t loom_type_encoding(void) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(LOOM_TYPE_ENCODING, (loom_scalar_type_t)0,
                                      0, LOOM_TYPE_FLAG_INLINE_DIMS);
  return type;
}

// Creates a pool type with a single block_size dimension.
// Uses rank=1 and stores the packed dim in dims[0], following the same
// packing as shaped types (loom_dim_pack_static / loom_dim_pack_dynamic).
static inline loom_type_t loom_type_pool(uint64_t block_size_dim) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(block_size_dim)) flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header =
      loom_type_make_header(LOOM_TYPE_POOL, (loom_scalar_type_t)0, 1, flags);
  type.dims[0] = block_size_dim;
  return type;
}

// Creates a scalar type (rank 0, no shape, no encoding).
static inline loom_type_t loom_type_scalar(loom_scalar_type_t scalar_type) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(
      LOOM_TYPE_SCALAR, scalar_type, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  return type;
}

// Creates a 0-d (scalar) tile or tensor type.
static inline loom_type_t loom_type_shaped_0d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint16_t encoding_id) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(
      kind, element_type, 0,
      LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
  type.encoding_id = encoding_id;
  return type;
}

// Creates a rank-1 shaped type with one inline dim.
static inline loom_type_t loom_type_shaped_1d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint64_t dim0,
                                              uint16_t encoding_id) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(dim0)) flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header = loom_type_make_header(kind, element_type, 1, flags);
  type.encoding_id = encoding_id;
  type.dims[0] = dim0;
  return type;
}

// Creates a rank-2 shaped type with two inline dims.
static inline loom_type_t loom_type_shaped_2d(loom_type_kind_t kind,
                                              loom_scalar_type_t element_type,
                                              uint64_t dim0, uint64_t dim1,
                                              uint16_t encoding_id) {
  uint8_t flags = LOOM_TYPE_FLAG_INLINE_DIMS;
  if (!loom_dim_is_dynamic(dim0) && !loom_dim_is_dynamic(dim1))
    flags |= LOOM_TYPE_FLAG_ALL_STATIC;
  loom_type_t type = {0};
  type.header = loom_type_make_header(kind, element_type, 2, flags);
  type.encoding_id = encoding_id;
  type.dims[0] = dim0;
  type.dims[1] = dim1;
  return type;
}

// Creates a function type from a pre-allocated loom_func_type_data_t.
// The data must be arena-allocated and outlive the type. The caller
// is responsible for populating the types[] array before use.
static inline loom_type_t loom_type_function(loom_func_type_data_t* func_data) {
  loom_type_t type = {0};
  type.header =
      loom_type_make_header(LOOM_TYPE_FUNCTION, (loom_scalar_type_t)0, 0, 0);
  type.dims[0] = (uint64_t)(uintptr_t)func_data;
  return type;
}

// Allocates and constructs a function type. The arg and result type
// arrays are copied into the allocated data. The allocation is owned
// by the provided allocator (typically an arena).
iree_status_t loom_type_function_build(const loom_type_t* arg_types,
                                       uint16_t arg_count,
                                       const loom_type_t* result_types,
                                       uint16_t result_count,
                                       iree_allocator_t allocator,
                                       loom_type_t* out_type);

//===----------------------------------------------------------------------===//
// Dialect type construction and accessors
//===----------------------------------------------------------------------===//
//
// Dialect types represent types from external dialects (hal.buffer,
// vm.ref<T>, etc.). They reuse the 24-byte loom_type_t layout:
//   encoding_id    → string_id (type name in the module string table)
//   encoding_flags → param_count (number of type parameters)
//   dims[0]        → pointer to loom_type_t* array (params), or 0
//   dims[1]        → unused

// Creates an opaque dialect type (no parameters). For types like
// hal.buffer that have no interior syntax.
static inline loom_type_t loom_type_dialect_opaque(loom_string_id_t name_id) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(LOOM_TYPE_DIALECT, (loom_scalar_type_t)0,
                                      0, LOOM_TYPE_FLAG_INLINE_DIMS);
  type.encoding_id = (uint16_t)name_id;
  type.encoding_flags = 0;
  return type;
}

// Creates a parameterized dialect type. The params array must be
// arena-allocated and outlive the type.
static inline loom_type_t loom_type_dialect(loom_string_id_t name_id,
                                            uint16_t param_count,
                                            const loom_type_t* params) {
  loom_type_t type = {0};
  type.header = loom_type_make_header(LOOM_TYPE_DIALECT, (loom_scalar_type_t)0,
                                      0, LOOM_TYPE_FLAG_INLINE_DIMS);
  type.encoding_id = (uint16_t)name_id;
  type.encoding_flags = param_count;
  type.dims[0] = (uint64_t)(uintptr_t)params;
  return type;
}

// Returns the string_id for the dialect type name.
static inline loom_string_id_t loom_type_dialect_name_id(loom_type_t type) {
  return (loom_string_id_t)type.encoding_id;
}

// Returns the number of type parameters.
static inline uint16_t loom_type_dialect_param_count(loom_type_t type) {
  return type.encoding_flags;
}

// Returns the type parameter array (may be NULL if param_count == 0).
static inline const loom_type_t* loom_type_dialect_params(loom_type_t type) {
  return (const loom_type_t*)(uintptr_t)type.dims[0];
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_TYPES_H_
