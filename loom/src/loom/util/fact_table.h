// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-value fact table: dense array of loom_value_facts_t keyed by
// loom_value_id_t. Arena-allocated. A zero-initialized table is valid (empty).
//
// Lookup always succeeds: returns unknown facts for undefined entries
// or out-of-range value IDs. Undefined entries are detected by
// known_divisor == 0 (valid facts always have known_divisor >= 1),
// allowing O(1) initialization via memset(0).
//
// Define stores facts for a value ID inside the caller-declared value
// capacity. Compute runs a forward pass over a function, calling each op's fact
// inference function to seed initial facts from constants and op semantics.
//
// The table is a reusable component: borrowed by the rewriter for
// canonicalization, owned by pass-scoped storage, and usable standalone for IPO
// or analysis tools.
//
// Typical lifecycle:
//
//   loom_value_fact_table_t table = {0};
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
//       &table, arena, value_count));
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(
//       &table, module, function));
//   loom_value_facts_t facts = loom_value_fact_table_lookup(&table, id);

#ifndef LOOM_UTIL_FACT_TABLE_H_
#define LOOM_UTIL_FACT_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Fact table
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_extension_entry_t
    loom_value_fact_extension_entry_t;
typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;
typedef struct loom_target_bundle_t loom_target_bundle_t;

typedef const loom_value_fact_domain_t* (
    *loom_value_fact_type_domain_resolver_fn_t)(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type);

typedef struct loom_value_fact_type_domain_resolver_callback_t {
  // Function that maps |type| to its type-owned fact domain.
  loom_value_fact_type_domain_resolver_fn_t fn;

  // Opaque callback payload passed to |fn|.
  void* user_data;
} loom_value_fact_type_domain_resolver_callback_t;

static inline loom_value_fact_type_domain_resolver_callback_t
loom_value_fact_type_domain_resolver_callback_empty(void) {
  return (loom_value_fact_type_domain_resolver_callback_t){
      .fn = NULL,
      .user_data = NULL,
  };
}

static inline loom_value_fact_type_domain_resolver_callback_t
loom_value_fact_type_domain_resolver_callback_make(
    loom_value_fact_type_domain_resolver_fn_t fn, void* user_data) {
  return (loom_value_fact_type_domain_resolver_callback_t){
      .fn = fn,
      .user_data = user_data,
  };
}

// Maximum lane facts stored in a small static vector extension. Larger static
// vectors degrade to unknown facts instead of allocating per-lane analysis
// payloads.
#define LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT 16

// Maximum raw payload bytes a type-owned fact domain may intern. Larger domain
// payloads degrade to no extension, which is the domain top/unknown value.
#define LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT 256

// Every logical element of an aggregate value shares the same facts. Vector
// values interpret the element as a lane; target-low register-range values
// interpret it as one register unit.
typedef struct loom_value_fact_uniform_element_t {
  // Scalar facts that apply to every lane.
  loom_value_facts_t element;
} loom_value_fact_uniform_element_t;

// Per-lane facts for a small all-static vector, in logical lane order.
typedef struct loom_value_fact_small_static_lanes_t {
  // Borrowed lane facts. Query results point into the owning fact table.
  const loom_value_facts_t* lanes;
  // Number of lane facts in the slice.
  iree_host_size_t count;
} loom_value_fact_small_static_lanes_t;

// Vector value is a lane-coordinate sequence: base + lane_ordinal * step.
typedef struct loom_value_fact_vector_iota_t {
  // Facts for the first produced coordinate.
  loom_value_facts_t base;
  // Facts for the logical lane-ordinal delta.
  loom_value_facts_t step;
} loom_value_fact_vector_iota_t;

// Vector value is a prefix mask produced by vector.mask.range.
typedef struct loom_value_fact_vector_prefix_mask_t {
  // Facts for the first tested coordinate.
  loom_value_facts_t lower_bound;
  // Facts for the exclusive coordinate bound.
  loom_value_facts_t upper_bound;
  // Facts for the coordinate delta between adjacent logical lanes.
  loom_value_facts_t step;
} loom_value_fact_vector_prefix_mask_t;

// Address-layout summary kind carried by encoding facts.
typedef enum loom_value_fact_address_layout_kind_e {
  // No usable address-layout summary is known.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN = 0,
  // Dense row-major layout. The consuming shaped type supplies rank/extents.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE = 1,
  // Explicit per-axis element strides.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED = 2,
} loom_value_fact_address_layout_kind_t;

// Summary of an address-layout encoding's logical-to-physical map.
typedef struct loom_value_fact_address_layout_t {
  // Known address-layout category.
  loom_value_fact_address_layout_kind_t kind;

  // Number of stride facts in strides for strided layouts. Dense layouts leave
  // this zero because rank comes from the consuming shaped type.
  uint8_t rank;

  // Borrowed per-axis element-stride facts for strided layouts. Query results
  // point into the owning fact table.
  const loom_value_facts_t* strides;
} loom_value_fact_address_layout_t;

// Logical low-bit matrix format carried by a packed matrix storage schema.
typedef enum loom_value_fact_matrix_format_e {
  // No usable matrix-format fact is known.
  LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN = 0,
  // AMD FP8 payload selected by hardware matrix-format immediates.
  LOOM_VALUE_FACT_MATRIX_FORMAT_FP8 = 1,
  // AMD BF8 payload selected by hardware matrix-format immediates.
  LOOM_VALUE_FACT_MATRIX_FORMAT_BF8 = 2,
  // AMD FP6 payload selected by hardware matrix-format immediates.
  LOOM_VALUE_FACT_MATRIX_FORMAT_FP6 = 3,
  // AMD BF6 payload selected by hardware matrix-format immediates.
  LOOM_VALUE_FACT_MATRIX_FORMAT_BF6 = 4,
  // AMD FP4 payload selected by hardware matrix-format immediates.
  LOOM_VALUE_FACT_MATRIX_FORMAT_FP4 = 5,
} loom_value_fact_matrix_format_t;

// Explicit scale operand shape used by a packed matrix storage schema.
typedef enum loom_value_fact_matrix_scale_kind_e {
  // No usable scale-kind fact is known.
  LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN = 0,
  // Matrix payload has no explicit scale operands.
  LOOM_VALUE_FACT_MATRIX_SCALE_NONE = 1,
  // Matrix payload uses 32-bit scale exponent operands.
  LOOM_VALUE_FACT_MATRIX_SCALE_32 = 2,
  // Matrix payload uses 16-bit scale exponent operands.
  LOOM_VALUE_FACT_MATRIX_SCALE_16 = 3,
} loom_value_fact_matrix_scale_kind_t;

// Hardware scale exponent format used by a packed matrix storage schema.
typedef enum loom_value_fact_matrix_scale_format_e {
  // No usable scale-format fact is known.
  LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN = 0,
  // No scale-format selector is used.
  LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE = 1,
  // AMD MATRIX_SCALE_FMT_E8 selector.
  LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E8 = 2,
  // AMD MATRIX_SCALE_FMT_E5M3 selector.
  LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E5M3 = 3,
  // AMD MATRIX_SCALE_FMT_E4M3 selector.
  LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E4M3 = 4,
} loom_value_fact_matrix_scale_format_t;

// Placement of scale exponent operands relative to a matrix fragment.
typedef enum loom_value_fact_matrix_scale_placement_e {
  // No usable scale-placement fact is known.
  LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_UNKNOWN = 0,
  // No scale exponent placement is used.
  LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_NONE = 1,
  // Scale exponents are supplied as explicit operands without row selection.
  LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_EXPLICIT = 2,
  // Scale exponents use hardware row-zero placement.
  LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_ROW0 = 3,
  // Scale exponents use hardware row-one placement.
  LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_ROW1 = 4,
} loom_value_fact_matrix_scale_placement_t;

// Scale conversion dependency shape for reference expansion.
typedef enum loom_value_fact_matrix_scale_conversion_e {
  // No usable scale-conversion fact is known.
  LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_UNKNOWN = 0,
  // No scale conversion is required.
  LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_NONE = 1,
  // Scale conversion is lane-local and can be scalarized independently.
  LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_LANE_LOCAL = 2,
  // Scale conversion is convergent and may observe values from other lanes.
  LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_CONVERGENT = 3,
} loom_value_fact_matrix_scale_conversion_t;

// Packed matrix operand facts for storage schemas that can map to native
// target matrix contracts or to an explicit reference expansion.
typedef struct loom_value_fact_matrix_storage_schema_t {
  // Logical matrix element format.
  loom_value_fact_matrix_format_t format;

  // Explicit scale operand shape.
  loom_value_fact_matrix_scale_kind_t scale_kind;

  // Hardware scale exponent format.
  loom_value_fact_matrix_scale_format_t scale_format;

  // Scale exponent operand placement.
  loom_value_fact_matrix_scale_placement_t scale_placement;

  // Whether reference scale conversion is lane-local or convergent.
  loom_value_fact_matrix_scale_conversion_t scale_conversion;

  // Number of 32-bit payload registers in the packed fragment.
  uint16_t packed_register_count;

  // Number of logical matrix elements represented by the packed fragment.
  uint16_t packed_element_count;

  // Whether a zero scale can refine to an unscaled target contract.
  bool zero_scale_fallback;
} loom_value_fact_matrix_storage_schema_t;

// Summary of a storage-schema encoding.
typedef struct loom_value_fact_storage_schema_t {
  // One-based static schema encoding ID when known. Zero means no exact nested
  // storage schema is known.
  uint16_t static_spec_encoding_id;

  // Packed matrix facts when the schema describes a native matrix fragment.
  loom_value_fact_matrix_storage_schema_t matrix;
} loom_value_fact_storage_schema_t;

// Summary of an SSA encoding value.
typedef struct loom_value_fact_encoding_summary_t {
  // Known semantic role from the value type or encoding family.
  loom_encoding_role_t role;

  // One-based static encoding spec ID when the value came from
  // encoding.define. Zero means no exact static spec is known.
  uint16_t static_spec_encoding_id;

  // Address-layout facts when this encoding directly is a layout or composes a
  // physical storage encoding with a known layout.
  loom_value_fact_address_layout_t address_layout;

  // Storage-schema facts when this encoding directly is a schema or composes a
  // physical storage encoding with a known schema.
  loom_value_fact_storage_schema_t storage_schema;
} loom_value_fact_encoding_summary_t;

// Known reference nullability for storage-like values.
typedef uint32_t loom_value_fact_reference_nullability_t;
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN \
  ((loom_value_fact_reference_nullability_t)0)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL \
  ((loom_value_fact_reference_nullability_t)1)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL \
  ((loom_value_fact_reference_nullability_t)2)

// Comparable alias scope for storage-like values. NONE means root_value_id is
// only provenance for addressing and same-root propagation; consumers must not
// use it to prove disjointness against another root.
typedef loom_value_id_t loom_value_fact_alias_scope_id_t;
#define LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE \
  ((loom_value_fact_alias_scope_id_t)LOOM_VALUE_ID_INVALID)

// Buffer value is an opaque storage root.
typedef struct loom_value_fact_buffer_reference_t {
  // Conservative byte extent facts for the root storage allocation.
  loom_value_facts_t maximum_byte_extent;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t minimum_alignment;

  // Target-independent memory space for the storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the storage root.
  loom_value_fact_reference_nullability_t nullability;
} loom_value_fact_buffer_reference_t;

// View value is a typed projection over a storage root.
typedef struct loom_value_fact_view_reference_t {
  // Byte offset facts for the view base relative to root_value_id.
  loom_value_facts_t base_byte_offset;

  // Conservative byte length facts for the whole-view footprint envelope.
  loom_value_facts_t footprint_byte_length;

  // Minimum provable alignment of base_byte_offset relative to root_value_id.
  // The root's own absolute pointer alignment is tracked separately.
  uint64_t minimum_alignment;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t root_minimum_alignment;

  // Static addressed element byte count, or -1 for sub-byte/unknown elements.
  int64_t static_element_byte_count;

  // Target-independent memory space for the underlying storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the underlying storage root.
  loom_value_fact_reference_nullability_t nullability;
} loom_value_fact_view_reference_t;

// Per-analysis context passed to op fact inference callbacks.
struct loom_fact_context_t {
  // Table that owns the dense facts and any extension payloads allocated by
  // inference helpers.
  loom_value_fact_table_t* table;

  // Function-like op whose body is currently being analyzed. Empty when facts
  // are computed for individual detached ops instead of a full function body.
  loom_func_like_t function;

  // Optional selected target bundle for target/profile-sensitive fact
  // inference. Generic analyses leave this NULL and receive source-level facts.
  const loom_target_bundle_t* target_bundle;

  // Optional type-domain resolver installed by layers that own registered type
  // descriptors. The fact table itself intentionally does not depend on the
  // generated type registry; callers that can map |type| to a descriptor can
  // return descriptor->fact_domain here.
  loom_value_fact_type_domain_resolver_callback_t resolve_type_domain;
};

// Type-owned operations for the optional extension slot in
// loom_value_facts_t.
//
// Scalar facts stay in the dense value record and use the generic transfer
// functions in loom/ir/facts.h. A fact domain only owns interpretation of the
// cold extension payload for values whose type declares the domain. Domains are
// immutable .rodata objects referenced from type descriptors; there is no
// process-global schema registry or schema ID stored in extension entries.
//
// Returning no extension is the domain top/unknown value. Domain callbacks must
// keep payloads bounded and degrade to no extension instead of allocating
// unbounded analysis state.
struct loom_value_fact_domain_t {
  // Returns true when |lhs| and |rhs| carry semantically equal extension facts
  // for |type|. Scalar range/divisor fields are compared by the generic caller.
  bool (*extensions_equal)(const loom_value_fact_domain_t* domain,
                           const loom_module_t* module, loom_type_t type,
                           const loom_value_fact_table_t* lhs_table,
                           loom_value_facts_t lhs,
                           const loom_value_fact_table_t* rhs_table,
                           loom_value_facts_t rhs);

  // Clones the extension payload carried by |facts| from |source| into
  // |target|. The generic caller copies scalar fields before invoking this.
  iree_status_t (*clone_extension)(const loom_value_fact_domain_t* domain,
                                   const loom_module_t* module,
                                   loom_type_t type,
                                   loom_value_fact_table_t* target,
                                   const loom_value_fact_table_t* source,
                                   loom_value_facts_t facts,
                                   loom_value_facts_t* inout_facts);

  // Computes the extension component of a join for two facts of |type|. The
  // generic caller has already joined scalar fields in |inout_facts|.
  iree_status_t (*meet_extension)(const loom_value_fact_domain_t* domain,
                                  const loom_module_t* module, loom_type_t type,
                                  loom_value_fact_table_t* target,
                                  const loom_value_fact_table_t* lhs_table,
                                  loom_value_facts_t lhs,
                                  const loom_value_fact_table_t* rhs_table,
                                  loom_value_facts_t rhs,
                                  loom_value_facts_t* inout_facts);

  // Widens a loop-carried extension fact. The default domain-free behavior is
  // conservative: stable extensions are preserved, changing extensions degrade
  // to no extension. |iteration| is zero-based for the loop summary solver.
  iree_status_t (*widen_extension)(
      const loom_value_fact_domain_t* domain, const loom_module_t* module,
      loom_type_t type, loom_value_fact_table_t* target,
      const loom_value_fact_table_t* previous_table,
      loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
      loom_value_facts_t next, uint32_t iteration,
      loom_value_facts_t* inout_facts);
};

struct loom_value_fact_table_t {
  // Arena for direct-address entries and touched-value storage.
  iree_arena_allocator_t* arena;
  // Arena for scope-local extension payloads and inference scratch buffers.
  iree_arena_allocator_t* transient_arena;

  // Dense fact entries indexed by value ID.
  loom_value_facts_t* entries;
  // Highest defined value ID plus one.
  iree_host_size_t count;
  // Allocated entry count.
  iree_host_size_t capacity;
  // Value IDs defined in the current populated scope.
  loom_value_id_t* touched_values;
  // Number of populated entries in touched_values.
  iree_host_size_t touched_count;
  // Allocated touched_values entry count.
  iree_host_size_t touched_capacity;
  // Context object passed to op-specific fact inference callbacks.
  loom_fact_context_t context;

  // Interned fact extension payloads. Extension IDs stored in
  // loom_value_facts_t are one-based indexes into entries and are only valid
  // for this table/context.
  struct {
    // Extension entries indexed by one-based extension ID minus one.
    loom_value_fact_extension_entry_t* entries;
    // Allocated extension entry count.
    iree_host_size_t capacity;
    // Defined extension entry count.
    iree_host_size_t count;
    // Hash buckets storing one-based extension IDs, or zero for empty buckets.
    loom_value_fact_extension_id_t* buckets;
    // Allocated hash bucket count.
    iree_host_size_t bucket_count;
  } extensions;

  // Reusable scratch buffers for fact inference calls. Allocated on first use,
  // grown only when an op needs more slots. Never shrinks. Old buffers are
  // abandoned in the arena and freed in bulk with the arena.
  struct {
    // Fact scratch buffer state.
    struct {
      // Scratch fact values used for operand and result fact arrays.
      loom_value_facts_t* values;
      // Allocated fact scratch entry count.
      iree_host_size_t capacity;
    } facts;
    // Value ID scratch buffer state.
    struct {
      // Scratch value IDs used by transforms that materialize replacements.
      loom_value_id_t* values;
      // Allocated value ID scratch entry count.
      iree_host_size_t capacity;
    } value_ids;
  } scratch;
};

// Initializes the table with the given arena and exactly pre-allocates
// |initial_capacity| value entries. All entries are zero-initialized
// (known_divisor == 0 means undefined). The arena is stored and used for
// subsequent extension payloads and scratch buffers. Value entries never grow
// after initialization; callers must declare the direct-address value domain up
// front.
iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity);

// Initializes the table using separate arenas for persistent direct-address
// entries and transient scope-local payloads. Use this when reusing one table
// across multiple populated scopes: clear touched entries, reset the transient
// arena, and keep the direct entry array live.
iree_status_t loom_value_fact_table_initialize_with_arenas(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* transient_arena, iree_host_size_t initial_capacity);

// Clears facts populated in the current scope and forgets transient extension
// and scratch state. Callers that provided a separate transient arena should
// reset that arena after this call. Direct-address entry storage and touched
// storage remain allocated for reuse.
void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table);

// Looks up facts for a value. Returns unknown facts if the value ID
// is out of range or the entry is undefined (known_divisor == 0).
static inline loom_value_facts_t loom_value_fact_table_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (value_id >= table->count || table->entries[value_id].known_divisor == 0) {
    return loom_value_facts_unknown();
  }
  return table->entries[value_id];
}

// Defines (or updates) facts for a value inside the initialized capacity.
iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts);

// Clones |facts| from |source| into |target|, re-interning any context-local
// extension payloads in the target table. The returned facts are valid for
// |target|'s fact context and preserve scalar range/divisibility fields.
iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts);

// Clones |facts| using the fact domain implied by |type|. Prefer this typed
// form when the value/type is available; extension payloads are type-owned.
iree_status_t loom_value_fact_table_clone_fact_for_type(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module, loom_type_t type, loom_value_facts_t facts,
    loom_value_facts_t* out_facts);

// Returns true when two fact values are semantically equal, including any
// extension payloads, even when the extension IDs were interned in different
// fact tables.
bool loom_value_fact_table_facts_equal(const loom_value_fact_table_t* lhs_table,
                                       loom_value_facts_t lhs,
                                       const loom_value_fact_table_t* rhs_table,
                                       loom_value_facts_t rhs);

// Typed semantic equality for facts whose values have |type|.
bool loom_value_fact_table_facts_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Returns true when both facts carry semantically equal extension payloads.
// Facts without extensions compare equal only when neither side has an
// extension.
bool loom_value_fact_table_extensions_equal(
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Typed semantic equality for extension payloads whose values have |type|.
bool loom_value_fact_table_extensions_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Computes a conservative join for two facts of |type| and stores the result in
// |target|. Extension payloads are preserved only when the type's fact domain
// proves they remain valid across the join.
iree_status_t loom_value_fact_table_meet_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* out_facts);

// Widens a loop-carried fact of |type| from |previous| to |next|. This is a
// convergence accelerator for summary solvers, not a scalar transfer function.
iree_status_t loom_value_fact_table_widen_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* previous_table,
    loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
    loom_value_facts_t next, uint32_t iteration, loom_value_facts_t* out_facts);

// Clones all defined source entries into |target|. When |module| is provided,
// extension payloads use the type-owned domain implied by each value ID.
// Undefined entries remain unset in |target| so normal block-argument and op
// fact seeding can fill them.
iree_status_t loom_value_fact_table_clone_defined_facts(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module);

// Computes facts for a single op by calling its vtable fact inference function.
// Gathers operand facts from the table, calls the callback, and defines result
// facts. No-op if the op has no inference function.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Computes facts for a single op and reports whether any result facts changed
// relative to the table's previous entries. No-op ops report false.
iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed);

// Seeds the table by running a forward pass over all ops in a
// function. For each op with an inference function, calls compute_op.
// Visits ops in dominance order so operand facts are available
// before use.
iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function);

// Returns a facts scratch buffer with at least |count| entries.
// The returned pointer is valid until the next call. Grows the
// allocation if needed; never shrinks.
iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out);

// Returns a value ID scratch buffer with at least |count| entries.
// Same lifetime and growth semantics as facts_scratch.
iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out);

// Creates facts for a vector whose every lane has |element| facts.
iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a uniform-element vector
// extension in |context|.
bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out);

// Creates facts for a small all-static vector with explicit per-lane facts.
// Vectors with more than LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT lanes degrade
// to unknown facts.
iree_status_t loom_value_facts_make_small_static_lanes(
    loom_fact_context_t* context, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a small-static-lanes vector
// extension in |context|.
bool loom_value_facts_query_small_static_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out);

// Returns true when |facts| represents a scalar value or aggregate whose every
// logical element has the same facts. Unknown scalar facts are not reported
// because carrying a uniform-unknown extension adds no useful information.
// |out_element| is required and receives the shared element facts.
bool loom_value_facts_query_all_equal_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element);

// Creates facts for a vector.iota-style lane-coordinate sequence.
iree_status_t loom_value_facts_make_vector_iota(
    loom_fact_context_t* context, loom_value_fact_vector_iota_t iota,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.iota extension in
// |context|.
bool loom_value_facts_query_vector_iota(const loom_fact_context_t* context,
                                        loom_value_facts_t facts,
                                        loom_value_fact_vector_iota_t* out);

// Creates facts for a vector.mask.range-style prefix mask.
iree_status_t loom_value_facts_make_vector_prefix_mask(
    loom_fact_context_t* context, loom_value_fact_vector_prefix_mask_t mask,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.mask.range
// extension in |context|.
bool loom_value_facts_query_vector_prefix_mask(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_vector_prefix_mask_t* out);

// Creates facts for an SSA encoding summary.
iree_status_t loom_value_facts_make_encoding_summary(
    loom_fact_context_t* context, loom_value_fact_encoding_summary_t summary,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is an encoding-summary
// extension in |context|.
bool loom_value_facts_query_encoding_summary(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_encoding_summary_t* out);

// Creates facts for a buffer storage root.
iree_status_t loom_value_facts_make_buffer_reference(
    loom_fact_context_t* context, loom_value_fact_buffer_reference_t reference,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a buffer-reference
// extension in |context|.
bool loom_value_facts_query_buffer_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_buffer_reference_t* out);

// Creates facts for a typed view projection.
iree_status_t loom_value_facts_make_view_reference(
    loom_fact_context_t* context, loom_value_fact_view_reference_t reference,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a view-reference extension
// in |context|.
bool loom_value_facts_query_view_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_view_reference_t* out);

// Creates a bounded raw extension payload for type-owned fact domains. The
// |payload_tag| namespace belongs to the value type domain; it is not a global
// schema ID. Payloads larger than LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT
// degrade to unknown facts.
iree_status_t loom_value_facts_make_extension_payload(
    loom_fact_context_t* context, uint8_t payload_tag, const void* payload,
    iree_host_size_t payload_length, loom_value_facts_t* out);

// Returns true and populates |out_payload|/|out_payload_length| when |facts|
// carries a raw payload with |payload_tag|. Returned bytes are borrowed from
// the fact table that owns |context| and remain valid for the table lifetime.
bool loom_value_facts_query_extension_payload(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    uint8_t payload_tag, const void** out_payload,
    iree_host_size_t* out_payload_length);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_TABLE_H_
