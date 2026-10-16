// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/executables.h"

#include "loom/serve/runtime/program/parameter_layout.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loomc/iree.h"

typedef struct loom_serve_finite_generation_t {
  // Management, preferred selection, captured batch, and bound runner owners.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this generation and its execution tables.
  iree_allocator_t host_allocator;
  // Retained immutable source snapshot and exact-input native reuse.
  loom_serve_jit_t* jit;
  // Original model evidence or owned trailing storage for a new generation.
  loom_serve_compilation_report_t* report;
  // Cold declaration count, including partial construction on failure.
  iree_host_size_t count;
  // Owned replacement stages; NULL for the model-owned original stages.
  loom_serve_jit_stage_t** compiled;
  // Original model table or owned descriptors paired with compiled stages.
  const loom_serve_stage_t* bindings;
} loom_serve_finite_generation_t;

struct loom_serve_finite_executables_t {
  // Allocator owning this finite model's code policy.
  iree_allocator_t host_allocator;
  // Immutable borrowed model resources, live through all compiler callbacks.
  loom_serve_finite_executable_options_t options;
  // Original source snapshot and descriptors, available for restoration.
  loom_serve_finite_generation_t* original;
  // Retained source generation selected for subsequent batches.
  loom_serve_finite_generation_t* selected;
  // Retained source generation captured by the current batch.
  loom_serve_finite_generation_t* captured;
  // Retained owner of the runner's current copied execution table.
  loom_serve_finite_generation_t* bound;
};

static void loom_serve_finite_generation_retain(
    loom_serve_finite_generation_t* generation) {
  if (generation) {
    iree_atomic_ref_count_inc(&generation->ref_count);
  }
}

static void loom_serve_finite_generation_release(void* value) {
  loom_serve_finite_generation_t* generation = value;
  if (!generation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&generation->ref_count) != 1) {
    return;
  }
  if (generation->compiled) {
    for (iree_host_size_t i = 0; i < generation->count; ++i) {
      if (generation->bindings) {
        iree_hal_command_buffer_release(generation->bindings[i].command_buffer);
        loom_serve_dispatch_release(generation->bindings[i].dispatch);
      }
      loom_serve_jit_stage_free(generation->compiled[i]);
    }
    iree_allocator_free(generation->host_allocator,
                        (void*)generation->bindings);
  }
  loom_serve_jit_release(generation->jit);
  iree_allocator_free(generation->host_allocator, generation->compiled);
  iree_allocator_free(generation->host_allocator, generation);
}

static iree_status_t loom_serve_finite_generation_create(
    loom_serve_jit_t* original, iree_allocator_t host_allocator,
    loom_serve_finite_generation_t** out_generation) {
  *out_generation = NULL;
  loom_serve_finite_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      iree_sizeof_struct(*generation) +
          (original ? 0 : sizeof(loom_serve_compilation_report_t)),
      (void**)&generation));
  iree_atomic_ref_count_init(&generation->ref_count);
  generation->host_allocator = host_allocator;
  generation->jit = original;
  loom_serve_jit_retain(original);
  generation->report =
      original
          ? loom_serve_jit_report(original)
          : (loom_serve_compilation_report_t*)((uint8_t*)generation +
                                               iree_sizeof_struct(*generation));
  *out_generation = generation;
  return iree_ok_status();
}

iree_status_t loom_serve_finite_executables_allocate(
    const loom_serve_finite_executable_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_finite_executables_t** out_executables) {
  *out_executables = NULL;
  loom_serve_finite_executables_t* executables = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executables), (void**)&executables));
  executables->host_allocator = host_allocator;
  executables->options = *options;
  iree_status_t status = loom_serve_finite_generation_create(
      options->jit, host_allocator, &executables->original);
  if (iree_status_is_ok(status)) {
    loom_serve_finite_generation_t* original = executables->original;
    original->count = loom_serve_preparation_stage_count(options->preparation);
    original->bindings = options->stages;
    executables->selected = original;
    loom_serve_finite_generation_retain(original);
    executables->bound = original;
    loom_serve_finite_generation_retain(original);
  }
  if (iree_status_is_ok(status)) {
    *out_executables = executables;
  } else {
    loom_serve_finite_executables_free(executables);
  }
  return status;
}

void loom_serve_finite_executables_free(
    loom_serve_finite_executables_t* executables) {
  if (!executables) {
    return;
  }
  loom_serve_finite_generation_release(executables->captured);
  loom_serve_finite_generation_release(executables->bound);
  loom_serve_finite_generation_release(executables->selected);
  loom_serve_finite_generation_release(executables->original);
  iree_allocator_free(executables->host_allocator, executables);
}

void loom_serve_finite_executables_capture(
    loom_serve_finite_executables_t* executables) {
  executables->captured = executables->selected;
  loom_serve_finite_generation_retain(executables->captured);
}

iree_status_t loom_serve_finite_executables_bind(
    loom_serve_finite_executables_t* executables) {
  loom_serve_finite_generation_t* generation = executables->captured;
  if (executables->bound == generation) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_module_bind_stages(
      executables->options.runner, generation->count, generation->bindings));
  loom_serve_finite_generation_retain(generation);
  loom_serve_finite_generation_release(executables->bound);
  executables->bound = generation;
  return iree_ok_status();
}

void loom_serve_finite_executables_release(
    loom_serve_finite_executables_t* executables) {
  loom_serve_finite_generation_release(executables->captured);
  executables->captured = NULL;
}

typedef struct loom_serve_finite_generation_job_t {
  // Finite source-indexing task, retired before asynchronous stage compilation.
  loomc_task_t task;
  // Borrowed immutable model resources through terminal completion.
  loom_serve_finite_executables_t* executables;
  // Owned generation transferred at terminal completion, including failure.
  loom_serve_finite_generation_t* generation;
  // Provider namespace borrowed through source indexing only.
  loom_serve_sources_t sources;
  // Indexing outcome consumed after the task relinquishes compiler access.
  iree_status_t status;
  // Terminal ownership transfer after all model and source accesses retire.
  loom_serve_variant_completion_fn_t completion;
  // Borrowed terminal consumer state.
  void* user_data;
} loom_serve_finite_generation_job_t;

static iree_status_t loom_serve_finite_generation_record(
    loom_serve_finite_generation_job_t* job) {
  const loom_serve_finite_executable_options_t* options =
      &job->executables->options;
  loom_serve_finite_generation_t* generation = job->generation;
  loom_serve_stage_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(generation->host_allocator, generation->count,
                                  sizeof(*bindings), (void**)&bindings));
  generation->bindings = bindings;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < generation->count && iree_status_is_ok(status); ++i) {
    loom_serve_jit_stage_t* compiled = generation->compiled[i];
    const loom_cmd_program_t* program = loom_serve_jit_stage_program(compiled);
    if (program) {
      const bool has_workspace =
          program->requirements.transient.required_byte_length != 0;
      if (program->requirements.rebindable_binding_count != 2 + has_workspace ||
          (has_workspace &&
           program->requirements.transient.binding_index != 2) ||
          program->requirements.launch_counts.binding_index != UINT32_MAX) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "replacement changes finite stage bindings");
      } else if (options->layout) {
        status = loom_serve_parameter_layout_check(options->layout, program);
      } else if (program->requirements.fixed_buffer_count) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "replacement adds checkpoint parameters");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_jit_stage_record(
            compiled,
            iree_hal_queue_family(
                loom_serve_device_dispatch_queue(options->device)),
            options->command_mode, NULL, options->parameters,
            &bindings[i].command_buffer);
        bindings[i].transient = program->requirements.transient;
      }
    } else {
      bindings[i].dispatch = loom_serve_jit_stage_dispatch(compiled);
      loom_serve_dispatch_retain(bindings[i].dispatch);
      if (loom_serve_dispatch_binding_count(bindings[i].dispatch) != 2) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "replacement changes direct stage bindings");
      }
    }
    bindings[i].binding_count = 2;
  }
  return status;
}

static void loom_serve_finite_generation_complete(
    loom_serve_finite_generation_job_t* job, iree_status_t status) {
  loom_serve_finite_generation_t* generation = job->generation;
  const loom_serve_variant_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(generation->host_allocator, job);
  completion(user_data, status, generation, generation->report);
}

static void loom_serve_finite_generation_compiled(
    void* user_data, iree_status_t status, loom_serve_stage_set_t* stages) {
  loom_serve_finite_generation_job_t* job = user_data;
  job->generation->count = stages->count;
  job->generation->compiled = stages->values;
  stages->count = 0;
  stages->values = NULL;
  loom_serve_stage_set_free(stages);
  if (iree_status_is_ok(status)) {
    status = loom_serve_finite_generation_record(job);
  }
  loom_serve_finite_generation_complete(job, status);
}

static void loom_serve_finite_generation_index(loomc_task_t* task,
                                               loomc_host_size_t worker) {
  loom_serve_finite_generation_job_t* job =
      (loom_serve_finite_generation_job_t*)((uint8_t*)task -
                                            offsetof(
                                                loom_serve_finite_generation_job_t,
                                                task));
  loom_serve_finite_generation_t* generation = job->generation;
  loom_serve_device_t* device = job->executables->options.device;
  job->status = loom_serve_jit_create(
      loom_serve_device_handle(device),
      loom_serve_device_dispatch_queue(device), loom_serve_device_host(device),
      &job->sources, NULL, generation->report, generation->host_allocator,
      &generation->jit);
}

static void loom_serve_finite_generation_indexed(loomc_task_t* task) {
  loom_serve_finite_generation_job_t* job =
      (loom_serve_finite_generation_job_t*)((uint8_t*)task -
                                            offsetof(
                                                loom_serve_finite_generation_job_t,
                                                task));
  iree_status_t status = job->status;
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_set_compile_catalog_async(
        job->generation->jit, job->executables->options.preparation,
        loom_serve_finite_generation_compiled, job,
        job->generation->host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_finite_generation_complete(job, status);
  }
}

static const loomc_task_vtable_t finite_generation_task_vtable = {
    .execute = loom_serve_finite_generation_index,
    .destroy = loom_serve_finite_generation_indexed,
};

static iree_status_t loom_serve_finite_generation_compile(
    void* self, const loom_serve_sources_t* sources,
    loom_serve_variant_completion_fn_t completion, void* user_data) {
  loom_serve_finite_executables_t* executables = self;
  loom_serve_finite_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_finite_generation_create(
      NULL, executables->host_allocator, &generation));
  loom_serve_finite_generation_job_t* job = NULL;
  iree_status_t status = iree_allocator_malloc(executables->host_allocator,
                                               sizeof(*job), (void**)&job);
  if (iree_status_is_ok(status)) {
    job->executables = executables;
    job->generation = generation;
    job->sources = *sources;
    job->completion = completion;
    job->user_data = user_data;
    loomc_task_initialize(&finite_generation_task_vtable, &job->task);
    status = iree_status_from_loomc(loomc_task_sink_submit(
        loom_serve_host_compiler_sink(
            loom_serve_device_host(executables->options.device)),
        &job->task));
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executables->host_allocator, job);
    loom_serve_finite_generation_release(generation);
  }
  return status;
}

static iree_status_t loom_serve_finite_generation_select(void* self,
                                                         void* value) {
  loom_serve_finite_executables_t* executables = self;
  loom_serve_finite_generation_t* generation =
      value ? value : executables->original;
  loom_serve_finite_generation_retain(generation);
  loom_serve_finite_generation_release(executables->selected);
  executables->selected = generation;
  return iree_ok_status();
}

loom_serve_execution_update_t loom_serve_finite_executables_update(
    loom_serve_finite_executables_t* executables) {
  return (loom_serve_execution_update_t){
      .self = executables,
      .compile = loom_serve_finite_generation_compile,
      .select = loom_serve_finite_generation_select,
      .dispose = loom_serve_finite_generation_release};
}
