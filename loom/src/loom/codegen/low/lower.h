// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-target-low lowering foundation.
//
// This layer owns the target-independent mechanics of lowering a verified
// source function into a low.func.def: target legality gating, function/block
// cloning, source-to-low SSA value mapping, source location preservation, and
// structural CFG terminators. Target packages provide descriptor choices and
// type mappings through callbacks so this library never links concrete backend
// descriptor tables.

#ifndef LOOM_CODEGEN_LOW_LOWER_H_
#define LOOM_CODEGEN_LOW_LOWER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/low_legality.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;
typedef struct loom_low_lower_rule_set_t loom_low_lower_rule_set_t;

typedef iree_status_t (*loom_low_lower_map_type_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
    loom_type_t* out_low_type);

typedef struct loom_low_lower_map_type_callback_t {
  // Callback invoked to map one source value type to a low register type.
  loom_low_lower_map_type_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_type_callback_t;

typedef iree_status_t (*loom_low_lower_map_value_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_type_t source_type, loom_type_t* out_low_type);

typedef struct loom_low_lower_map_value_callback_t {
  // Optional callback invoked to map one concrete source SSA value to a low
  // register type. Missing uses |map_type|.
  loom_low_lower_map_value_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_value_callback_t;

typedef enum loom_low_lower_abi_argument_kind_e {
  // Source argument is passed as a low function block argument.
  LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT = 0,
  // Source argument is imported through a function-local low.resource op.
  LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE = 1,
} loom_low_lower_abi_argument_kind_t;

typedef struct loom_low_lower_abi_argument_t {
  // ABI path used for the source argument.
  loom_low_lower_abi_argument_kind_t kind;
  // Register type used by the low argument or imported low.resource result.
  loom_type_t abi_type;
  // Resource import kind used when |kind| is RESOURCE.
  loom_low_resource_import_kind_t resource_import_kind;
  // Dense target ABI resource index used when |kind| is RESOURCE.
  int64_t resource_index;
  // Semantic source type recorded on low.resource. None defaults to the source
  // argument type.
  loom_type_t resource_semantic_type;
  // Optional low.resource builder flags for resource-specific metadata.
  loom_low_resource_build_flags_t resource_build_flags;
  // Valid byte extent for byte-addressable resources when the corresponding
  // builder flag is set.
  int64_t resource_valid_byte_count;
  // Resource-level cache swizzle byte stride when the corresponding builder
  // flag is set.
  int64_t resource_cache_swizzle_stride;
} loom_low_lower_abi_argument_t;

typedef iree_status_t (*loom_low_lower_map_argument_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

typedef struct loom_low_lower_map_argument_callback_t {
  // Optional callback invoked to map a source function argument to a direct low
  // argument or target ABI resource. Missing uses direct |map_type| behavior.
  loom_low_lower_map_argument_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_argument_callback_t;

typedef iree_status_t (*loom_low_lower_emit_preamble_fn_t)(
    void* user_data, loom_low_lower_context_t* context);

typedef struct loom_low_lower_emit_preamble_callback_t {
  // Optional callback invoked after the low function and entry block are
  // created, before ABI resources and source body packets are emitted.
  loom_low_lower_emit_preamble_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_preamble_callback_t;

typedef uint64_t loom_low_lower_plan_id_t;

#define LOOM_LOW_LOWER_PLAN_ID_NONE UINT64_MAX

typedef struct loom_low_lower_plan_t {
  // Target-owned dense plan id selected during planning.
  loom_low_lower_plan_id_t id;
  // Arena-owned target plan data selected during planning and consumed by
  // emission during the same lowering run. Core lowering never interprets it.
  const void* target_data;
} loom_low_lower_plan_t;

// Returns a selected lowering plan. |target_data| points at immutable
// target-owned storage allocated from the current lowering arena, or NULL when
// |id| fully describes the plan.
static inline loom_low_lower_plan_t loom_low_lower_plan_make(
    loom_low_lower_plan_id_t id, const void* target_data) {
  return (loom_low_lower_plan_t){
      .id = id,
      .target_data = target_data,
  };
}

static inline loom_low_lower_plan_t loom_low_lower_plan_empty(void) {
  return loom_low_lower_plan_make(LOOM_LOW_LOWER_PLAN_ID_NONE, NULL);
}

static inline bool loom_low_lower_plan_is_empty(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_LOW_LOWER_PLAN_ID_NONE;
}

typedef iree_status_t (*loom_low_lower_select_op_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan);

typedef struct loom_low_lower_select_op_callback_t {
  // Callback invoked during planning to select the exact lowering plan for
  // one non-structural source op. Missing or unsupported plans must return
  // loom_low_lower_plan_empty().
  loom_low_lower_select_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_select_op_callback_t;

typedef iree_status_t (*loom_low_lower_emit_op_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan);

typedef struct loom_low_lower_emit_op_callback_t {
  // Callback invoked during emission to execute a plan selected during
  // planning for one non-structural source op.
  loom_low_lower_emit_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_op_callback_t;

typedef struct loom_low_lower_policy_t {
  // Stable policy name used in diagnostics and status messages.
  iree_string_view_t name;
  // Maps source semantic types to target-low register types.
  loom_low_lower_map_type_callback_t map_type;
  // Optionally maps concrete source SSA values to target-low register types
  // when type alone does not determine the target register class.
  loom_low_lower_map_value_callback_t map_value;
  // Optionally maps source function arguments to non-direct ABI imports.
  loom_low_lower_map_argument_callback_t map_argument;
  // Optionally emits target live-ins or other structural preamble packets.
  loom_low_lower_emit_preamble_callback_t emit_preamble;
  // Optional table-driven source-op lowering rules.
  const loom_low_lower_rule_set_t* rule_set;
  // Optional target-owned selector used before a target has table rules.
  loom_low_lower_select_op_callback_t select_op;
  // Optional target-owned emitter for plans selected by |select_op|.
  loom_low_lower_emit_op_callback_t emit_op;
} loom_low_lower_policy_t;

// Verifies that |policy| is complete enough to lower source ops. Policy
// validation is an infrastructure contract: malformed policies return status
// failures instead of user IR diagnostics.
iree_status_t loom_low_lower_policy_verify(
    const loom_low_lower_policy_t* policy);

typedef struct loom_low_lower_policy_registry_entry_t {
  // Target contract-set key that selects |policy|.
  iree_string_view_t contract_set_key;
  // Borrowed lowering policy used for bundles naming |contract_set_key|.
  const loom_low_lower_policy_t* policy;
} loom_low_lower_policy_registry_entry_t;

typedef struct loom_low_lower_policy_registry_t {
  // Borrowed contract-set key to policy table linked into the target package.
  const loom_low_lower_policy_registry_entry_t* entries;
  // Number of rows in |entries|.
  iree_host_size_t entry_count;
} loom_low_lower_policy_registry_t;

// Initializes |out_registry| from a target-owned entry table. The table is
// borrowed and must outlive |out_registry|.
void loom_low_lower_policy_registry_initialize_from_entries(
    loom_low_lower_policy_registry_t* out_registry,
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count);

// Verifies that |registry| is a well-formed contract-key to policy table.
// Registry validation is an infrastructure contract: malformed registries
// return status failures instead of user IR diagnostics.
iree_status_t loom_low_lower_policy_registry_verify(
    const loom_low_lower_policy_registry_t* registry);

// Looks up the lowering policy for |contract_set_key|. Empty keys are rejected
// and missing keys return NOT_FOUND so target package omissions fail loud.
iree_status_t loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key,
    const loom_low_lower_policy_t** out_policy);

// Looks up the lowering policy for |bundle|'s target-contract key.
iree_status_t loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_lower_policy_t** out_policy);

// Returns true when |registry| has a lowering policy for |bundle|'s target
// contract set. This is a selection predicate, not full registry validation.
bool loom_low_lower_policy_registry_has_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle);

typedef struct loom_low_lower_options_t {
  // Module-local target.profile symbol used by the emitted low.func.def.
  loom_symbol_ref_t target_ref;
  // Target bundle selected for this lowering attempt.
  const loom_target_bundle_t* bundle;
  // Low descriptor registry linked into the current compiler binary.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Descriptor payload requirements needed by the consumer after lowering.
  loom_low_descriptor_requirement_flags_t descriptor_requirements;
  // Optional target-specific legality providers forwarded to source legality.
  loom_target_low_legality_provider_list_t legality_provider_list;
  // Optional source legality feedback diagnostics forwarded to providers.
  loom_target_low_legality_diagnostic_flags_t legality_diagnostic_flags;
  // Target lowering policy for descriptor and type choices.
  const loom_low_lower_policy_t* policy;
  // Optional suffix appended to the source function symbol. Empty uses "__low".
  iree_string_view_t low_function_suffix;
  // Structured diagnostic emitter for user legality and lowering failures.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before aborting. Zero means no limit.
  uint32_t max_errors;
} loom_low_lower_options_t;

typedef struct loom_low_lower_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of remark diagnostics emitted by legality providers.
  uint32_t remark_count;
  // Descriptor set selected by |options.bundle|.
  const loom_low_descriptor_set_t* descriptor_set;
  // Emitted low.func.def op, or NULL when user diagnostics prevented emission.
  loom_op_t* low_func_op;
  // Module-local symbol reference for |low_func_op|.
  loom_symbol_ref_t low_func_ref;
} loom_low_lower_result_t;

// Lowers one func.def-like source function into a sibling low.func.def.
//
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|. The function returns OK in that case and does not emit a
// low.func.def. Infrastructure failures such as malformed options, invalid
// target symbols, or a policy that violates the lowering contract are returned
// as status failures.
iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result);

// Returns the module being mutated by the current lowering.
loom_module_t* loom_low_lower_context_module(loom_low_lower_context_t* context);

// Returns the builder positioned in the current low block. Only valid while
// emit_preamble or emit_op callback code is emitting; select_op callbacks must
// not mutate IR.
loom_builder_t* loom_low_lower_context_builder(
    loom_low_lower_context_t* context);

// Returns the source function being lowered.
loom_func_like_t loom_low_lower_context_source_function(
    const loom_low_lower_context_t* context);

// Returns the emitted low.func.def op, or NULL during planning.
loom_op_t* loom_low_lower_context_low_function(
    const loom_low_lower_context_t* context);

// Returns the selected target bundle.
const loom_target_bundle_t* loom_low_lower_context_bundle(
    const loom_low_lower_context_t* context);

// Returns the selected descriptor set.
const loom_low_descriptor_set_t* loom_low_lower_context_descriptor_set(
    const loom_low_lower_context_t* context);

// Returns source value facts computed before planning. The table describes
// the source function being lowered and remains valid only during callbacks.
const loom_value_fact_table_t* loom_low_lower_context_fact_table(
    const loom_low_lower_context_t* context);

// Returns the number of non-structural source-op lowering plans selected during
// planning. Preamble callbacks may inspect these plans before body emission.
iree_host_size_t loom_low_lower_context_selected_plan_count(
    const loom_low_lower_context_t* context);

// Returns the source op associated with one selected lowering plan.
const loom_op_t* loom_low_lower_context_selected_plan_source_op(
    const loom_low_lower_context_t* context, iree_host_size_t index);

// Returns one selected lowering plan. Table-driven rule rows return an empty
// plan because their rule data is owned by core lowering.
loom_low_lower_plan_t loom_low_lower_context_selected_plan(
    const loom_low_lower_context_t* context, iree_host_size_t index);

// Allocates transient storage from the current lowering arena. The allocation
// remains valid until the current loom_low_lower_function call returns.
iree_status_t loom_low_lower_allocate_scratch_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr);

// Allocates one target-owned selected-plan payload from the current lowering
// arena. The returned storage remains valid until the current
// loom_low_lower_function call returns. Targets should populate the payload in
// place during planning and treat it as immutable during emission.
iree_status_t loom_low_lower_allocate_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data);

// Creates a module-local symbol derived from the emitted low function symbol.
// The result is suitable for target-owned function records such as low.slot
// declarations. |suffix| is appended to the low function name; |index| is then
// appended in decimal form when |append_index| is true.
iree_status_t loom_low_lower_create_function_symbol(
    loom_low_lower_context_t* context, iree_string_view_t suffix,
    bool append_index, uint32_t index, loom_symbol_ref_t* out_symbol_ref);

// Maps |source_type| through the active policy. A policy that rejects a user
// type emits a diagnostic and returns loom_type_none() in |out_low_type|.
iree_status_t loom_low_lower_map_type(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type);

// Maps |source_value_id|'s type through the active policy. A policy that
// rejects a user value emits a diagnostic and returns loom_type_none() in
// |out_low_type|.
iree_status_t loom_low_lower_map_value(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_value_id_t source_value_id,
                                       loom_type_t* out_low_type);

// Looks up the low SSA value already bound to |source_value_id|.
iree_status_t loom_low_lower_lookup_value(loom_low_lower_context_t* context,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t* out_low_value_id);

// Binds one source SSA value to the corresponding low SSA value. The source
// value's display name is copied when available.
iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id);

// Interns a descriptor-set register-class spelling in the low module.
iree_status_t loom_low_lower_register_class_string_id(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    loom_string_id_t* out_string_id);

// Creates a target-low register type from a descriptor-set register-class ID.
iree_status_t loom_low_lower_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type);

// Emits a descriptor-backed low.op selected by stable descriptor ID.
//
// The selected descriptor set on |context| is the authority for converting the
// durable ID into the packet's textual descriptor spelling. Source lowerings
// should pass generated descriptor ID constants or target selector results here
// instead of interning descriptor key strings and relying on builders to hash
// them back into IDs.
iree_status_t loom_low_lower_emit_descriptor_op(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op);

// Emits a descriptor-backed low.const selected by stable descriptor ID.
iree_status_t loom_low_lower_emit_descriptor_const(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// Emits ERR_BACKEND_001 for an unsupported source-to-low lowering subject.
iree_status_t loom_low_lower_emit_reject(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         iree_string_view_t subject_kind,
                                         iree_string_view_t subject_name,
                                         iree_string_view_t reason);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_H_
