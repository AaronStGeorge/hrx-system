// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pipeline.h"

#include "loom/codegen/low/pipeline.h"
#include "loom/pass/builder.h"

typedef struct loom_target_pipeline_build_context_t {
  // Target providers linked into the current compile driver.
  const loom_target_environment_t* target_environment;
  // Pass capabilities required by produced pass IR.
  loom_pass_environment_t pass_environment;
  // Caller-selected options for the pipeline.
  const loom_target_pipeline_options_t* options;
} loom_target_pipeline_build_context_t;

static iree_status_t loom_target_pipeline_build_run(loom_builder_t* builder,
                                                    iree_string_view_t key) {
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, key, loom_named_attr_slice_empty(),
                                &run_op);
}

static iree_status_t loom_target_pipeline_build_source_to_low(
    loom_builder_t* builder, const loom_target_pipeline_options_t* options) {
  loom_named_attr_t option_attrs[1] = {0};
  loom_named_attr_slice_t option_slice = loom_named_attr_slice_empty();
  if (options != NULL && options->source_to_low_max_errors != 0) {
    loom_string_id_t option_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        builder->module, IREE_SV("max-errors"), &option_name_id));
    option_attrs[0] = (loom_named_attr_t){
        .name_id = option_name_id,
        .value = loom_attr_i64(options->source_to_low_max_errors),
    };
    option_slice = loom_make_named_attr_slice(option_attrs, 1);
  }

  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, IREE_SV("source-to-low"), option_slice,
                                &run_op);
}

static iree_status_t loom_target_pipeline_contribute_phase(
    loom_builder_t* builder,
    const loom_target_pipeline_build_context_t* context,
    loom_target_pipeline_phase_t phase) {
  return loom_target_environment_contribute_pipeline(
      context->target_environment, phase, context->pass_environment, builder);
}

static iree_status_t loom_target_pipeline_build_cleanup(
    loom_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("canonicalize")));
  return loom_target_pipeline_build_run(builder, IREE_SV("cse"));
}

static iree_status_t loom_target_pipeline_build_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  return loom_target_pipeline_build_cleanup(builder);
}

static iree_status_t loom_target_pipeline_build_low_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("cfg-simplify")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  return loom_target_pipeline_build_run(builder, IREE_SV("low-dce"));
}

static iree_status_t loom_target_pipeline_build_source_normalization(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_SOURCE_NORMALIZATION));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("legalize-math")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("normalize-kernel-resources")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("promote-private-fragments")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("vector-reduce-axes-to-scalar")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("vector-gather-to-scalar")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("linearize-view-accesses")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("unroll-scf-for")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("scf-to-cfg")));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("cfg-simplify")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  return loom_target_pipeline_build_run(builder, IREE_SV("branch-sink"));
}

static iree_status_t loom_target_pipeline_build_low_preparation(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION));
  return loom_low_pipeline_build_packetization_preparation(builder);
}

static iree_status_t loom_target_pipeline_build_source_low_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_for(
      builder, LOOM_PASS_ANCHOR_FUNC,
      loom_target_pipeline_build_source_normalization, user_data, &for_op));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_to_low(builder, context->options));
  return loom_pass_ir_build_for(builder, LOOM_PASS_ANCHOR_FUNC,
                                loom_target_pipeline_build_low_cleanup_body,
                                NULL, &for_op);
}

static iree_status_t loom_target_pipeline_build_prepared_low_body(
    loom_builder_t* builder, void* user_data) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_low_body(builder, user_data));
  loom_op_t* for_op = NULL;
  return loom_pass_ir_build_for(builder, LOOM_PASS_ANCHOR_FUNC,
                                loom_target_pipeline_build_low_preparation,
                                user_data, &for_op);
}

iree_status_t loom_target_pipeline_build_to_source_low(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
  };
  return loom_pass_ir_build_pipeline(pipeline_module, name,
                                     LOOM_PASS_ANCHOR_MODULE,
                                     loom_target_pipeline_build_source_low_body,
                                     (void*)&context, out_pipeline_op);
}

iree_status_t loom_target_pipeline_build_to_prepared_low(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
  };
  return loom_pass_ir_build_pipeline(
      pipeline_module, name, LOOM_PASS_ANCHOR_MODULE,
      loom_target_pipeline_build_prepared_low_body, (void*)&context,
      out_pipeline_op);
}
