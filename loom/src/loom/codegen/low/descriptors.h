// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dense target-low descriptor table ABI.
//
// The low descriptor table is the compact runtime form consumed by target-low
// verification, scheduling, allocation feedback, and final target emitters.
// Generator inputs can be rich and target-specific; this ABI is intentionally
// flat, pointer-light, and shardable so selected target packages can link only
// the descriptor sets they need.

#ifndef LOOM_CODEGEN_LOW_DESCRIPTORS_H_
#define LOOM_CODEGEN_LOW_DESCRIPTORS_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/target/types.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

// ABI version for descriptor sets consumed by this header.
#define LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION 14u

// Sentinel for absent string-table offsets.
#define LOOM_LOW_STRING_OFFSET_NONE LOOM_BSTRING_TABLE_OFFSET_NONE

// Sentinel for absent target-family or descriptor-set stable IDs.
#define LOOM_LOW_STABLE_ID_NONE UINT64_C(0)

// Sentinel for absent 16-bit table identifiers.
#define LOOM_LOW_ID_NONE UINT16_MAX

// Sentinel for absent descriptor ordinals.
#define LOOM_LOW_DESCRIPTOR_ORDINAL_NONE UINT32_MAX

// Sentinel for absent stable descriptor IDs.
#define LOOM_LOW_DESCRIPTOR_ID_NONE UINT64_C(0)

// Sentinel for absent asm-form ordinals.
#define LOOM_LOW_ASM_FORM_ORDINAL_NONE UINT32_MAX

// Sentinel used before verification; verified descriptors must name a class.
#define LOOM_LOW_SCHEDULE_CLASS_NONE UINT16_MAX

// Sentinel for absent register classes.
#define LOOM_LOW_REG_CLASS_NONE UINT16_MAX

// Sentinel for absent enum immediate domains.
#define LOOM_LOW_ENUM_DOMAIN_NONE UINT16_MAX

// Sentinel for absent resources.
#define LOOM_LOW_RESOURCE_NONE UINT16_MAX

typedef enum loom_low_operand_role_e {
  // Unknown or uninitialized operand role.
  LOOM_LOW_OPERAND_ROLE_UNKNOWN = 0,
  // SSA result defined by the descriptor.
  LOOM_LOW_OPERAND_ROLE_RESULT = 1,
  // SSA operand consumed by the descriptor.
  LOOM_LOW_OPERAND_ROLE_OPERAND = 2,
  // Non-SSA read/write role; verified low packets use separate rows.
  LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT = 3,
  // Predicate or mask operand controlling execution.
  LOOM_LOW_OPERAND_ROLE_PREDICATE = 4,
  // Memory-address, resource, or descriptor operand.
  LOOM_LOW_OPERAND_ROLE_RESOURCE = 5,
  // Target-owned implicit architectural operand.
  LOOM_LOW_OPERAND_ROLE_IMPLICIT = 6,
} loom_low_operand_role_t;

// Bitset of descriptor operand flags.
typedef uint16_t loom_low_operand_flags_t;

// Operand is omitted from the target assembly spelling but participates in low
// semantics. Rows with role IMPLICIT are also omitted from low packet operands.
#define LOOM_LOW_OPERAND_FLAG_IMPLICIT ((uint16_t)1u << 0)
// Operand is tied to another operand by a constraint row.
#define LOOM_LOW_OPERAND_FLAG_TIED ((uint16_t)1u << 1)
// Operand must be considered early-clobbered by allocation.
#define LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER ((uint16_t)1u << 2)
// Operand is optional for some descriptor forms.
#define LOOM_LOW_OPERAND_FLAG_OPTIONAL ((uint16_t)1u << 3)

// Bitset of register-class alternative flags.
typedef uint16_t loom_low_reg_class_alt_flags_t;

// Register-class alternative is preferred by target lowering.
#define LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED ((uint16_t)1u << 0)
// Alternative represents an immediate or literal instead of a register class.
#define LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE ((uint16_t)1u << 1)
// Alternative is legal only after physical register assignment.
#define LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY ((uint16_t)1u << 2)

// Bitset of register-class flags.
typedef uint16_t loom_low_reg_class_flags_t;

// Register class is virtual-only and has no physical register inventory.
#define LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY ((uint16_t)1u << 0)
// Register class represents target-visible physical registers.
#define LOOM_LOW_REG_CLASS_FLAG_PHYSICAL ((uint16_t)1u << 1)
// Register class contains reference-counted or GC-visible references.
#define LOOM_LOW_REG_CLASS_FLAG_REFERENCE ((uint16_t)1u << 2)
// Register class cannot be represented in spill storage.
#define LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE ((uint16_t)1u << 3)

typedef enum loom_low_spill_slot_space_e {
  // Unknown or uninitialized spill storage space.
  LOOM_LOW_SPILL_SLOT_SPACE_UNKNOWN = 0,
  // CPU stack-frame storage.
  LOOM_LOW_SPILL_SLOT_SPACE_STACK = 1,
  // GPU per-lane scratch storage.
  LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH = 2,
  // Target-private per-invocation storage.
  LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE = 3,
  // GPU local data share or workgroup storage.
  LOOM_LOW_SPILL_SLOT_SPACE_LDS = 4,
} loom_low_spill_slot_space_t;

typedef enum loom_low_immediate_kind_e {
  // Unknown or uninitialized immediate kind.
  LOOM_LOW_IMMEDIATE_KIND_UNKNOWN = 0,
  // Signed integer immediate.
  LOOM_LOW_IMMEDIATE_KIND_SIGNED = 1,
  // Unsigned integer immediate.
  LOOM_LOW_IMMEDIATE_KIND_UNSIGNED = 2,
  // Symbol, function, block, or descriptor ordinal immediate.
  LOOM_LOW_IMMEDIATE_KIND_ORDINAL = 3,
  // Target-specific enum immediate.
  LOOM_LOW_IMMEDIATE_KIND_ENUM = 4,
} loom_low_immediate_kind_t;

// Bitset of immediate flags.
typedef uint16_t loom_low_immediate_flags_t;

// Immediate is resolved from a symbolic reference before final emission.
#define LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC ((uint16_t)1u << 0)
// Immediate value is encoded relative to the current packet or block.
#define LOOM_LOW_IMMEDIATE_FLAG_RELATIVE ((uint16_t)1u << 1)
// Immediate may be omitted from packet attributes and uses default_value.
#define LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE ((uint16_t)1u << 2)

typedef enum loom_low_effect_kind_e {
  // Unknown or uninitialized effect kind.
  LOOM_LOW_EFFECT_KIND_UNKNOWN = 0,
  // Descriptor reads memory or an external resource.
  LOOM_LOW_EFFECT_KIND_READ = 1,
  // Descriptor writes memory or an external resource.
  LOOM_LOW_EFFECT_KIND_WRITE = 2,
  // Descriptor may call outside the current low region.
  LOOM_LOW_EFFECT_KIND_CALL = 3,
  // Descriptor is a scheduling or memory barrier.
  LOOM_LOW_EFFECT_KIND_BARRIER = 4,
  // Descriptor observes or mutates a target counter.
  LOOM_LOW_EFFECT_KIND_COUNTER = 5,
  // Descriptor has convergent execution semantics.
  LOOM_LOW_EFFECT_KIND_CONVERGENT = 6,
  // Descriptor changes control flow.
  LOOM_LOW_EFFECT_KIND_CONTROL = 7,
} loom_low_effect_kind_t;

typedef enum loom_low_memory_space_e {
  // Effect has no memory-space attachment.
  LOOM_LOW_MEMORY_SPACE_NONE = 0,
  // Generic or target-default memory space.
  LOOM_LOW_MEMORY_SPACE_GENERIC = 1,
  // Device or process global memory.
  LOOM_LOW_MEMORY_SPACE_GLOBAL = 2,
  // Workgroup, shared, or LDS memory.
  LOOM_LOW_MEMORY_SPACE_WORKGROUP = 3,
  // Stack, frame, or spill memory.
  LOOM_LOW_MEMORY_SPACE_STACK = 4,
  // IREE VM reference table or reference state.
  LOOM_LOW_MEMORY_SPACE_VM_REF = 5,
  // Wasm linear memory.
  LOOM_LOW_MEMORY_SPACE_WASM_MEMORY = 6,
} loom_low_memory_space_t;

// Bitset of effect flags.
typedef uint16_t loom_low_effect_flags_t;

// Effect must be preserved during scheduling.
#define LOOM_LOW_EFFECT_FLAG_ORDERED ((uint16_t)1u << 0)
// Effect participates in alias-like dependency construction.
#define LOOM_LOW_EFFECT_FLAG_DEPENDENCY ((uint16_t)1u << 1)

typedef enum loom_low_constraint_kind_e {
  // Unknown or uninitialized constraint kind.
  LOOM_LOW_CONSTRAINT_KIND_UNKNOWN = 0,
  // Two descriptor operands must use the same assigned register.
  LOOM_LOW_CONSTRAINT_KIND_TIED = 1,
  // The descriptor may commute two operands.
  LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE = 2,
  // Operand is destructively updated.
  LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE = 3,
  // Operand or result has early-clobber allocation semantics.
  LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER = 4,
  // Descriptor may be rematerialized instead of spilled.
  LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE = 5,
  // Descriptor may participate in algebraic folding.
  LOOM_LOW_CONSTRAINT_KIND_FOLDABLE = 6,
} loom_low_constraint_kind_t;

// Bitset of descriptor constraint flags.
typedef uint16_t loom_low_constraint_flags_t;

typedef enum loom_low_latency_kind_e {
  // Unknown or uninitialized latency kind.
  LOOM_LOW_LATENCY_KIND_UNKNOWN = 0,
  // Latency is exact for this target model.
  LOOM_LOW_LATENCY_KIND_EXACT = 1,
  // Latency is calibrated but may vary by microarchitecture or operands.
  LOOM_LOW_LATENCY_KIND_ESTIMATE = 2,
  // Latency is intentionally variable or data dependent.
  LOOM_LOW_LATENCY_KIND_VARIABLE = 3,
} loom_low_latency_kind_t;

typedef enum loom_low_model_quality_e {
  // Unknown or uninitialized schedule model quality.
  LOOM_LOW_MODEL_QUALITY_UNKNOWN = 0,
  // Schedule model is exact enough to enforce.
  LOOM_LOW_MODEL_QUALITY_EXACT = 1,
  // Schedule model is calibrated from measurements.
  LOOM_LOW_MODEL_QUALITY_CALIBRATED = 2,
  // Schedule model is an estimate suitable for diagnostics and search.
  LOOM_LOW_MODEL_QUALITY_ESTIMATED = 3,
  // Schedule model is an explicit fallback.
  LOOM_LOW_MODEL_QUALITY_FALLBACK = 4,
} loom_low_model_quality_t;

// Bitset of schedule-class flags.
typedef uint16_t loom_low_schedule_class_flags_t;

// Schedule class may read memory.
#define LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD ((uint16_t)1u << 0)
// Schedule class may write memory.
#define LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE ((uint16_t)1u << 1)
// Schedule class may call out of the current low region.
#define LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL ((uint16_t)1u << 2)
// Schedule class changes control flow.
#define LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL ((uint16_t)1u << 3)

typedef enum loom_low_resource_kind_e {
  // Unknown or uninitialized resource kind.
  LOOM_LOW_RESOURCE_KIND_UNKNOWN = 0,
  // Scalar arithmetic or interpreter ALU resource.
  LOOM_LOW_RESOURCE_KIND_SCALAR_ALU = 1,
  // Vector arithmetic resource.
  LOOM_LOW_RESOURCE_KIND_VECTOR_ALU = 2,
  // Matrix or tensor-core-like resource.
  LOOM_LOW_RESOURCE_KIND_MATRIX = 3,
  // Load pipeline or address-generation resource.
  LOOM_LOW_RESOURCE_KIND_LOAD = 4,
  // Store pipeline or address-generation resource.
  LOOM_LOW_RESOURCE_KIND_STORE = 5,
  // Branch, call, or control resource.
  LOOM_LOW_RESOURCE_KIND_CONTROL = 6,
  // Address generation resource feeding load/store pipelines.
  LOOM_LOW_RESOURCE_KIND_ADDRESS = 7,
} loom_low_resource_kind_t;

// Bitset of resource flags.
typedef uint16_t loom_low_resource_flags_t;

typedef enum loom_low_hazard_kind_e {
  // Unknown or uninitialized hazard kind.
  LOOM_LOW_HAZARD_KIND_UNKNOWN = 0,
  // Consumer must be at least a fixed distance after producer.
  LOOM_LOW_HAZARD_KIND_MIN_DISTANCE = 1,
  // Consumer requires a target counter wait.
  LOOM_LOW_HAZARD_KIND_WAIT_COUNTER = 2,
  // Producer/consumer can use a bypass path.
  LOOM_LOW_HAZARD_KIND_BYPASS = 3,
  // Producer/consumer can or must fuse.
  LOOM_LOW_HAZARD_KIND_FUSION = 4,
} loom_low_hazard_kind_t;

typedef enum loom_low_hazard_reference_kind_e {
  // Unknown or uninitialized hazard reference kind.
  LOOM_LOW_HAZARD_REFERENCE_KIND_UNKNOWN = 0,
  // Hazard reference id names a resource table row.
  LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE = 1,
  // Hazard reference id names a target counter.
  LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER = 2,
  // Hazard reference id is target-owned and interpreted by an overlay.
  LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET = 3,
} loom_low_hazard_reference_kind_t;

// Bitset of hazard flags.
typedef uint16_t loom_low_hazard_flags_t;

// Bitset of descriptor flags.
typedef uint16_t loom_low_descriptor_flags_t;

// Descriptor has target-visible side effects.
#define LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING ((uint16_t)1u << 0)
// Descriptor is a terminator for the containing low block.
#define LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR ((uint16_t)1u << 1)
// Descriptor may be safely removed when all results are dead.
#define LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE ((uint16_t)1u << 2)
// Descriptor is a Loom pseudo packet that requires target lowering.
#define LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO ((uint16_t)1u << 3)

typedef struct loom_low_reg_class_t {
  // String-table offset for the stable register-class name.
  loom_bstring_table_offset_t name_string_offset;
  // Target bank identifier used by allocators and pressure reporting.
  uint16_t target_bank_id;
  // Register-class behavioral flags.
  loom_low_reg_class_flags_t flags;
  // Number of bits in one allocation unit.
  uint16_t alloc_unit_bits;
  // Number of physical registers, or zero when virtual/unbounded.
  uint16_t physical_count;
  // Alias-set identifier shared by overlapping register classes.
  uint16_t alias_set_id;
  // Register class used for spill/reload values, or LOOM_LOW_REG_CLASS_NONE.
  uint16_t spill_class_id;
  // Storage space used when values from this class are spilled.
  uint8_t spill_slot_space;
} loom_low_reg_class_t;

typedef struct loom_low_reg_class_alt_t {
  // Register-class table identifier, or LOOM_LOW_REG_CLASS_NONE for literals.
  uint16_t reg_class_id;
  // Alternative flags such as preferred, immediate, or physical-only.
  loom_low_reg_class_alt_flags_t flags;
} loom_low_reg_class_alt_t;

typedef struct loom_low_operand_t {
  // String-table offset for the descriptor field name.
  loom_bstring_table_offset_t field_name_string_offset;
  // Target-owned encoding field identifier, or zero when this operand does not
  // directly populate a binary encoding field.
  uint16_t encoding_field_id;
  // Semantic role this operand row plays.
  loom_low_operand_role_t role;
  // Operand flags used by verifier, allocator, and emitter.
  loom_low_operand_flags_t flags;
  // First register-class alternative row for this operand.
  uint16_t reg_class_alt_start;
  // Number of register-class alternatives accepted by this operand.
  uint16_t reg_class_alt_count;
  // Number of allocation units consumed or produced.
  uint16_t unit_count;
  // Target-owned data-format identifier.
  uint16_t data_format_id;
  // Scheduling stage where the operand is read.
  uint16_t read_stage;
  // Scheduling stage where the operand result becomes ready.
  uint16_t ready_stage;
} loom_low_operand_t;

typedef struct loom_low_immediate_t {
  // String-table offset for the immediate field name.
  loom_bstring_table_offset_t field_name_string_offset;
  // Target-owned encoding field identifier, or zero when this immediate does
  // not directly populate a binary encoding field.
  uint16_t encoding_field_id;
  // Immediate interpretation used by verifier and emitter.
  loom_low_immediate_kind_t kind;
  // Immediate flags such as symbolic or relative.
  loom_low_immediate_flags_t flags;
  // Encoded immediate width in bits.
  uint16_t bit_width;
  // Enum-domain table identifier for ENUM immediates.
  uint16_t enum_domain_id;
  // Reserved for generator-owned immediate encoding variants.
  uint16_t encoding_id;
  // Inclusive signed minimum when kind is signed.
  int64_t signed_min;
  // Inclusive unsigned maximum when kind is unsigned or ordinal.
  uint64_t unsigned_max;
  // Value used when a packet omits this immediate attribute.
  int64_t default_value;
} loom_low_immediate_t;

typedef struct loom_low_encoding_field_value_t {
  // Target-owned encoding field identifier.
  uint16_t encoding_field_id;
  // Reserved for future target-owned field flags.
  uint16_t reserved;
  // Constant value assigned to the encoding field for this descriptor.
  uint64_t value;
} loom_low_encoding_field_value_t;

typedef struct loom_low_enum_domain_t {
  // String-table offset for the stable enum-domain name.
  loom_bstring_table_offset_t name_string_offset;
  // First enum-value row for this domain.
  uint32_t value_start;
  // Number of enum-value rows for this domain.
  uint16_t value_count;
  // Reserved for future enum-domain flags.
  uint16_t reserved;
} loom_low_enum_domain_t;

typedef struct loom_low_enum_value_t {
  // String-table offset for the stable enum token.
  loom_bstring_table_offset_t token_string_offset;
  // Numeric value encoded for this token.
  int64_t value;
} loom_low_enum_value_t;

typedef struct loom_low_effect_t {
  // Effect kind used by dependency and legality construction.
  loom_low_effect_kind_t kind;
  // Memory space or external resource touched by the effect.
  loom_low_memory_space_t memory_space;
  // Target-owned scope identifier for ordering and visibility.
  uint16_t scope_id;
  // Effect flags used by scheduling and verification.
  loom_low_effect_flags_t flags;
  // Target counter identifier for counter effects.
  uint16_t counter_id;
  // Access width in bits, or zero when not width-specific.
  uint16_t width_bits;
} loom_low_effect_t;

typedef struct loom_low_constraint_t {
  // Constraint kind used by verification and allocation.
  loom_low_constraint_kind_t kind;
  // First descriptor operand index participating in the constraint.
  uint16_t lhs_operand_index;
  // Second descriptor operand index, or LOOM_LOW_ID_NONE when unary.
  uint16_t rhs_operand_index;
  // Constraint flags for target-owned refinements.
  loom_low_constraint_flags_t flags;
} loom_low_constraint_t;

typedef struct loom_low_issue_use_t {
  // Resource identifier consumed by this issue-use row.
  uint16_t resource_id;
  // Number of cycles the resource is occupied.
  uint16_t cycles;
  // Number of resource units consumed per cycle.
  uint16_t units;
  // Pipeline stage associated with this use.
  uint16_t stage;
} loom_low_issue_use_t;

typedef struct loom_low_pressure_delta_t {
  // Register class whose pressure changes.
  uint16_t reg_class_id;
  // Signed pressure delta in allocation units.
  int16_t delta;
} loom_low_pressure_delta_t;

typedef struct loom_low_resource_t {
  // String-table offset for the stable resource name.
  loom_bstring_table_offset_t name_string_offset;
  // Number of resource units available per cycle.
  uint16_t capacity_per_cycle;
  // Resource flags for target-owned refinements.
  loom_low_resource_flags_t flags;
  // Abstract resource kind used by generic diagnostics.
  loom_low_resource_kind_t kind;
  // Contention group identifier for related resources.
  uint16_t contention_group_id;
} loom_low_resource_t;

typedef struct loom_low_hazard_t {
  // Hazard kind used by schedule policy and verification.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Producer pipeline stage participating in the hazard.
  uint16_t producer_stage;
  // Consumer pipeline stage participating in the hazard.
  uint16_t consumer_stage;
  // Required distance or target-owned hazard value.
  uint16_t distance;
  // Hazard flags for target-owned refinements.
  loom_low_hazard_flags_t flags;
} loom_low_hazard_t;

typedef struct loom_low_schedule_class_t {
  // String-table offset for the stable schedule-class name.
  loom_bstring_table_offset_t name_string_offset;
  // Latency in cycles when latency_kind is exact or estimated.
  uint16_t latency_cycles;
  // Latency interpretation for scheduling and diagnostics.
  loom_low_latency_kind_t latency_kind;
  // First issue-use row for this schedule class.
  uint16_t issue_use_start;
  // Number of issue-use rows for this schedule class.
  uint16_t issue_use_count;
  // First hazard row for this schedule class.
  uint16_t hazard_start;
  // Number of hazard rows for this schedule class.
  uint16_t hazard_count;
  // Schedule-class flags such as load, store, call, or control.
  loom_low_schedule_class_flags_t flags;
  // Quality of the schedule model data.
  loom_low_model_quality_t model_quality;
  // First pressure-delta row for this schedule class.
  uint16_t pressure_delta_start;
  // Number of pressure-delta rows for this schedule class.
  uint16_t pressure_delta_count;
} loom_low_schedule_class_t;

typedef struct loom_low_descriptor_t {
  // String-table offset for the stable descriptor key.
  loom_bstring_table_offset_t key_string_offset;
  // Durable descriptor identity derived from the descriptor key. This is
  // stable across descriptor table reordering and unrelated descriptor
  // additions; descriptor-set ordinals are only transient row addresses.
  uint64_t stable_id;
  // String-table offset for the target mnemonic or packet name.
  loom_bstring_table_offset_t mnemonic_string_offset;
  // String-table offset for the primary semantic tag.
  loom_bstring_table_offset_t semantic_tag_string_offset;
  // First feature-mask word required by this descriptor.
  uint32_t feature_mask_word_start;
  // Number of feature-mask words required by this descriptor.
  uint16_t feature_mask_word_count;
  // First target-owned fixed encoding field value for this descriptor.
  uint32_t encoding_field_value_start;
  // Number of target-owned fixed encoding field values for this descriptor.
  uint16_t encoding_field_value_count;
  // Target-owned encoding format identifier. Zero means no target-specific
  // format selector is required by generic descriptor consumers.
  uint16_t encoding_format_id;
  // Target-owned encoding identifier.
  uint16_t encoding_id;
  // First operand/result row for this descriptor.
  uint32_t operand_start;
  // Total number of operand/result rows for this descriptor.
  uint16_t operand_count;
  // Number of leading operand rows that define results.
  uint16_t result_count;
  // First immediate row for this descriptor.
  uint32_t immediate_start;
  // Number of immediate rows for this descriptor.
  uint16_t immediate_count;
  // First effect row for this descriptor.
  uint32_t effect_start;
  // Number of effect rows for this descriptor.
  uint16_t effect_count;
  // First constraint row for this descriptor.
  uint32_t constraint_start;
  // Number of constraint rows for this descriptor.
  uint16_t constraint_count;
  // Required schedule-class identifier for this descriptor.
  uint16_t schedule_class_id;
  // Descriptor flags used by verifier, scheduler, and optimizer.
  loom_low_descriptor_flags_t flags;
  // Unique canonical asm form ordinal for descriptor-driven text emission, or
  // LOOM_LOW_ASM_FORM_ORDINAL_NONE when no unambiguous form exists.
  uint32_t canonical_asm_form_ordinal;
} loom_low_descriptor_t;

typedef struct loom_low_descriptor_ref_t {
  // String-table offset for the stable symbolic descriptor key.
  loom_bstring_table_offset_t key_string_offset;
  // Ordinal of the referenced descriptor row.
  uint32_t descriptor_ordinal;
} loom_low_descriptor_ref_t;

typedef struct loom_low_descriptor_id_ref_t {
  // Stable descriptor identity.
  uint64_t stable_id;
  // Ordinal of the referenced descriptor row.
  uint32_t descriptor_ordinal;
} loom_low_descriptor_id_ref_t;

typedef struct loom_low_asm_immediate_t {
  // Descriptor-local immediate index printed or parsed by this asm field.
  uint16_t immediate_index;
  // Optional string-table offset for a named immediate spelling.
  loom_bstring_table_offset_t name_string_offset;
} loom_low_asm_immediate_t;

typedef struct loom_low_asm_form_t {
  // String-table offset for the unqualified asm mnemonic.
  loom_bstring_table_offset_t mnemonic_string_offset;
  // Descriptor ordinal selected by this asm form.
  uint32_t descriptor_ordinal;
  // First descriptor-local result operand index in asm_operand_indices.
  uint32_t result_operand_index_start;
  // Number of result operand indices for this asm form.
  uint16_t result_operand_index_count;
  // First descriptor-local input operand index in asm_operand_indices.
  uint32_t operand_index_start;
  // Number of input operand indices for this asm form.
  uint16_t operand_index_count;
  // First immediate spelling row for this asm form.
  uint32_t immediate_start;
  // Number of immediate spelling rows for this asm form.
  uint16_t immediate_count;
} loom_low_asm_form_t;

typedef struct loom_low_descriptor_set_t {
  // Descriptor table ABI version.
  uint32_t abi_version;
  // Generator or hand-authored schema version.
  uint32_t generator_version;
  // Durable descriptor-set identity derived from the descriptor-set key.
  uint64_t stable_id;
  // Durable target-family identity derived from the target-family key, or NONE.
  uint64_t target_stable_id;
  // String-table offset for the descriptor-set key.
  loom_bstring_table_offset_t key_string_offset;
  // String-table offset for the target-family key.
  loom_bstring_table_offset_t target_key_string_offset;
  // String-table offset for the feature namespace key.
  loom_bstring_table_offset_t feature_key_string_offset;
  // Packed B-string table used by all string offsets.
  loom_bstring_table_t string_table;
  // Dense descriptor rows owned by this set.
  const loom_low_descriptor_t* descriptors;
  // Number of descriptor rows owned by this set.
  uint32_t descriptor_count;
  // Sorted symbolic descriptor-key reference rows.
  const loom_low_descriptor_ref_t* descriptor_refs;
  // Number of symbolic descriptor-key reference rows.
  uint32_t descriptor_ref_count;
  // Sorted stable descriptor-ID reference rows.
  const loom_low_descriptor_id_ref_t* descriptor_id_refs;
  // Number of stable descriptor-ID reference rows.
  uint32_t descriptor_id_ref_count;
  // Sorted asm forms keyed by unqualified mnemonic.
  const loom_low_asm_form_t* asm_forms;
  // Number of asm form rows owned by this set.
  uint32_t asm_form_count;
  // Packed descriptor-local operand indices referenced by asm forms.
  const uint16_t* asm_operand_indices;
  // Number of descriptor-local operand index rows owned by this set.
  uint32_t asm_operand_index_count;
  // Packed immediate spelling rows referenced by asm forms.
  const loom_low_asm_immediate_t* asm_immediates;
  // Number of immediate spelling rows owned by this set.
  uint32_t asm_immediate_count;
  // Dense operand/result rows referenced by descriptors.
  const loom_low_operand_t* operands;
  // Number of operand/result rows owned by this set.
  uint32_t operand_count;
  // Dense immediate rows referenced by descriptors.
  const loom_low_immediate_t* immediates;
  // Number of immediate rows owned by this set.
  uint32_t immediate_count;
  // Dense enum-domain rows referenced by ENUM immediates.
  const loom_low_enum_domain_t* enum_domains;
  // Number of enum-domain rows owned by this set.
  uint32_t enum_domain_count;
  // Dense enum-value rows referenced by enum domains.
  const loom_low_enum_value_t* enum_values;
  // Number of enum-value rows owned by this set.
  uint32_t enum_value_count;
  // Dense effect rows referenced by descriptors.
  const loom_low_effect_t* effects;
  // Number of effect rows owned by this set.
  uint32_t effect_count;
  // Dense constraint rows referenced by descriptors.
  const loom_low_constraint_t* constraints;
  // Number of constraint rows owned by this set.
  uint32_t constraint_count;
  // Dense register classes accepted by descriptor operands.
  const loom_low_reg_class_t* reg_classes;
  // Number of register classes owned by this set.
  uint32_t reg_class_count;
  // Dense register-class alternative rows referenced by operands.
  const loom_low_reg_class_alt_t* reg_class_alts;
  // Number of register-class alternative rows owned by this set.
  uint32_t reg_class_alt_count;
  // Dense schedule classes referenced by descriptors.
  const loom_low_schedule_class_t* schedule_classes;
  // Number of schedule classes owned by this set.
  uint32_t schedule_class_count;
  // Dense issue-use rows referenced by schedule classes.
  const loom_low_issue_use_t* issue_uses;
  // Number of issue-use rows owned by this set.
  uint32_t issue_use_count;
  // Dense target resources referenced by issue-use rows.
  const loom_low_resource_t* resources;
  // Number of resources owned by this set.
  uint32_t resource_count;
  // Dense hazard rows referenced by schedule classes.
  const loom_low_hazard_t* hazards;
  // Number of hazard rows owned by this set.
  uint32_t hazard_count;
  // Dense pressure-delta rows referenced by schedule classes.
  const loom_low_pressure_delta_t* pressure_deltas;
  // Number of pressure-delta rows owned by this set.
  uint32_t pressure_delta_count;
  // Dense feature-mask words referenced by descriptors.
  const uint64_t* feature_mask_words;
  // Number of feature-mask words owned by this set.
  uint32_t feature_mask_word_count;
  // Dense target-owned fixed encoding field values referenced by descriptors.
  const loom_low_encoding_field_value_t* encoding_field_values;
  // Number of fixed encoding field values owned by this set.
  uint32_t encoding_field_value_count;
} loom_low_descriptor_set_t;

// Returns a borrowed descriptor set linked into a target package. Providers
// must be stable and return the same non-NULL descriptor set for each call.
typedef const loom_low_descriptor_set_t* (*loom_low_descriptor_set_provider_t)(
    void);

// Borrowed descriptor-set registry available to low verification/emission runs.
// Entries may be in any order; lookups reject duplicate matching keys so
// callers do not need a separate normalization step for correctness. Registries
// may provide descriptor sets directly or through provider functions; provider
// tables let package-level registries stay static and grow without embedding a
// fixed-capacity materialization buffer.
typedef struct loom_low_descriptor_registry_t {
  // Borrowed descriptor-set pointers linked into the current compiler binary.
  const loom_low_descriptor_set_t* const* descriptor_sets;
  // Number of descriptor-set pointers in |descriptor_sets|.
  iree_host_size_t descriptor_set_count;
  // Borrowed descriptor-set provider functions linked into the current binary.
  const loom_low_descriptor_set_provider_t* descriptor_set_providers;
  // Number of provider functions in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Borrowed target-bundle presets linked with these descriptor sets.
  const loom_target_bundle_t* const* target_bundles;
  // Number of valid target-bundle presets in |target_bundles|.
  iree_host_size_t target_bundle_count;
} loom_low_descriptor_registry_t;

// Returns the total number of descriptor-set slots in |registry|, including
// direct descriptor-set pointers and provider-backed sets. NULL registries have
// zero slots.
iree_host_size_t loom_low_descriptor_registry_descriptor_set_count(
    const loom_low_descriptor_registry_t* registry);

// Returns the descriptor set at |index|, or NULL if |registry| is NULL, |index|
// is out of range, or a provider row is NULL/returns NULL.
const loom_low_descriptor_set_t* loom_low_descriptor_registry_descriptor_set_at(
    const loom_low_descriptor_registry_t* registry, iree_host_size_t index);

// Verifies structural integrity of a descriptor set. This checks table spans,
// string offsets, descriptor key uniqueness, and cross-table references; it
// does not perform full target legality.
iree_status_t loom_low_descriptor_set_verify(
    const loom_low_descriptor_set_t* descriptor_set);

// Verifies descriptor registry integrity. Registry validation is an
// infrastructure contract: malformed registries return status failures instead
// of user IR diagnostics.
iree_status_t loom_low_descriptor_registry_verify(
    const loom_low_descriptor_registry_t* registry);

// Looks up |key| in |registry|. Returns OK with NULL when no set matches.
// Duplicate matching keys return ALREADY_EXISTS because target binding and
// emission must be deterministic.
iree_status_t loom_low_descriptor_registry_lookup(
    const loom_low_descriptor_registry_t* registry, iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set);

// Looks up |stable_id| in |registry|. Returns OK with NULL when no set matches.
// Duplicate matching stable IDs return ALREADY_EXISTS because target binding
// and emission must be deterministic.
iree_status_t loom_low_descriptor_registry_lookup_by_id(
    const loom_low_descriptor_registry_t* registry, uint64_t stable_id,
    const loom_low_descriptor_set_t** out_descriptor_set);

// Returns the B-string view at |string_offset|. A NONE offset returns an empty
// string.
iree_status_t loom_low_descriptor_set_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string);

// Returns a descriptor row by ordinal, or NULL when |descriptor_ordinal| is out
// of bounds.
const loom_low_descriptor_t* loom_low_descriptor_set_descriptor_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal);

// Returns the durable descriptor identity derived from a descriptor key.
uint64_t loom_low_descriptor_stable_id_from_key(iree_string_view_t key);

// Returns an asm form row by ordinal, or NULL when |asm_form_ordinal| is out of
// bounds.
const loom_low_asm_form_t* loom_low_descriptor_set_asm_form_at(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t asm_form_ordinal);

// Resolves the unique canonical asm form for |descriptor_ordinal|. Descriptors
// with zero or multiple asm forms may intentionally have no canonical form and
// return FAILED_PRECONDITION.
iree_status_t loom_low_descriptor_set_lookup_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint32_t* out_asm_form_ordinal);

// Returns the stable diagnostic spelling for an operand role.
iree_string_view_t loom_low_operand_role_name(loom_low_operand_role_t role);

// Returns the stable diagnostic spelling for an immediate kind.
iree_string_view_t loom_low_immediate_kind_name(loom_low_immediate_kind_t kind);

// Returns the stable diagnostic spelling for an effect kind.
iree_string_view_t loom_low_effect_kind_name(loom_low_effect_kind_t kind);

// Returns the stable diagnostic spelling for a memory space.
iree_string_view_t loom_low_memory_space_name(
    loom_low_memory_space_t memory_space);

// Returns the stable diagnostic spelling for a spill slot space.
iree_string_view_t loom_low_spill_slot_space_name(
    loom_low_spill_slot_space_t space);

// Returns true if |space| is a known spill slot space.
bool loom_low_spill_slot_space_is_valid(loom_low_spill_slot_space_t space);

// Returns the stable diagnostic spelling for a constraint kind.
iree_string_view_t loom_low_constraint_kind_name(
    loom_low_constraint_kind_t kind);

// Returns the stable diagnostic spelling for a latency kind.
iree_string_view_t loom_low_latency_kind_name(loom_low_latency_kind_t kind);

// Returns the stable diagnostic spelling for a model-quality kind.
iree_string_view_t loom_low_model_quality_name(
    loom_low_model_quality_t quality);

// Returns the stable diagnostic spelling for a scheduler resource kind.
iree_string_view_t loom_low_resource_kind_name(loom_low_resource_kind_t kind);

// Returns the stable diagnostic spelling for a scheduler hazard kind.
iree_string_view_t loom_low_hazard_kind_name(loom_low_hazard_kind_t kind);

// Returns the stable diagnostic spelling for a scheduler hazard reference kind.
iree_string_view_t loom_low_hazard_reference_kind_name(
    loom_low_hazard_reference_kind_t kind);

// Resolves a symbolic descriptor key to an ordinal in |descriptor_set|.
iree_status_t loom_low_descriptor_set_lookup_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint32_t* out_descriptor_ordinal);

// Returns the descriptor ordinal for |stable_id| or
// LOOM_LOW_DESCRIPTOR_ORDINAL_NONE when the selected target has no descriptor
// with that identity. This is the compiled-path lookup after text or bytecode
// import has resolved descriptor spellings.
uint32_t loom_low_descriptor_set_lookup_descriptor_by_id(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t stable_id);

// Resolves an unqualified asm mnemonic to an asm form ordinal in
// |descriptor_set|. Descriptor sets verify that asm mnemonics are sorted and
// unambiguous.
iree_status_t loom_low_descriptor_set_lookup_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic, uint32_t* out_asm_form_ordinal);

// Appends a compact JSON manifest for |descriptor_set| to |builder|. The JSON
// is a diagnostic and test format, not the runtime representation.
iree_status_t loom_low_descriptor_set_format_manifest_json(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CODEGEN_LOW_DESCRIPTORS_H_
