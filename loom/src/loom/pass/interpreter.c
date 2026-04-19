// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/interpreter.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"

typedef struct loom_pass_interpreter_epoch_t {
  // Number of module symbols at the observation point.
  iree_host_size_t symbol_count;
  // Number of module-level operations at the observation point.
  iree_host_size_t module_op_count;
  // Number of SSA values at the observation point.
  iree_host_size_t value_count;
  // Number of interned types at the observation point.
  iree_host_size_t type_count;
  // Number of encoding instances at the observation point.
  iree_host_size_t encoding_count;
} loom_pass_interpreter_epoch_t;

typedef struct loom_pass_interpreter_frame_t {
  // Active anchor kind.
  loom_pass_kind_t kind;
  // Current symbol when kind is LOOM_PASS_FUNCTION.
  loom_symbol_t* symbol;
  // Current function when kind is LOOM_PASS_FUNCTION.
  loom_func_like_t function;
} loom_pass_interpreter_frame_t;

typedef struct loom_pass_interpreter_state_t {
  // Program being executed.
  const loom_pass_program_t* program;
  // Mutable module receiving pass effects.
  loom_module_t* module;
  // Caller-supplied execution options.
  const loom_pass_interpreter_options_t* options;
} loom_pass_interpreter_state_t;

typedef struct loom_pass_interpreter_symbol_snapshot_entry_t {
  // Symbol entry captured before entering a pass.for body.
  loom_symbol_t* symbol;
  // Function-like view of symbol->defining_op.
  loom_func_like_t function;
} loom_pass_interpreter_symbol_snapshot_entry_t;

static const char* loom_pass_interpreter_anchor_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "func";
    default:
      return "unknown";
  }
}

static loom_pass_interpreter_epoch_t loom_pass_interpreter_epoch(
    const loom_module_t* module) {
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  return (loom_pass_interpreter_epoch_t){
      .symbol_count = module->symbols.count,
      .module_op_count = module_block ? module_block->op_count : 0,
      .value_count = module->values.count,
      .type_count = module->types.count,
      .encoding_count = module->encodings.count,
  };
}

static bool loom_pass_interpreter_epoch_changed(
    loom_pass_interpreter_epoch_t lhs, loom_pass_interpreter_epoch_t rhs) {
  return lhs.symbol_count != rhs.symbol_count ||
         lhs.module_op_count != rhs.module_op_count ||
         lhs.value_count != rhs.value_count ||
         lhs.type_count != rhs.type_count ||
         lhs.encoding_count != rhs.encoding_count;
}

static iree_string_view_t loom_pass_interpreter_symbol_name(
    const loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame) {
  if (!frame->symbol ||
      frame->symbol->name_id >= state->module->strings.count) {
    return IREE_SV("<none>");
  }
  return state->module->strings.entries[frame->symbol->name_id];
}

static iree_status_t loom_pass_interpreter_make_pass(
    loom_pass_interpreter_state_t* state,
    const loom_pass_program_instruction_t* instruction,
    iree_arena_allocator_t* instance_arena, loom_pass_t* out_pass) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  void* pass_user_data = NULL;
  if (state->options->configure.fn) {
    IREE_RETURN_IF_ERROR(state->options->configure.fn(
        state->options->configure.user_data, instruction, &pass_user_data));
  }
  *out_pass = (loom_pass_t){
      .info = invoke->info,
      .module_run = invoke->module_run,
      .instance_arena = instance_arena,
      .arena = instance_arena,
      .decoded_options = invoke->decoded_options,
      .diagnostic_emitter = state->options->diagnostic_emitter,
      .user_data = pass_user_data,
  };
  if (invoke->info->statistic_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t statistics_size =
      (iree_host_size_t)invoke->info->statistic_count * sizeof(int64_t);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(instance_arena, statistics_size,
                                           (void**)&out_pass->statistics));
  memset(out_pass->statistics, 0, statistics_size);
  return iree_ok_status();
}

static bool loom_pass_interpreter_statistics_changed(const loom_pass_t* pass) {
  if (!pass->statistics || !pass->info->statistic_defs) {
    return false;
  }
  for (uint16_t i = 0; i < pass->info->statistic_count; ++i) {
    if (pass->statistics[i] == 0) {
      continue;
    }
    iree_string_view_t name = pass->info->statistic_defs[i].name;
    if (iree_string_view_equal(name, IREE_SV("changes")) ||
        iree_string_view_equal(name, IREE_SV("changed"))) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_interpreter_invoke_module(
    loom_pass_interpreter_state_t* state,
    const loom_pass_program_instruction_t* instruction, loom_pass_t* pass,
    bool* out_changed) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  if (!invoke->module_run) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module pass callback is required");
  }
  loom_pass_interpreter_epoch_t before =
      loom_pass_interpreter_epoch(state->module);
  iree_status_t status = invoke->module_run(pass, state->module);
  loom_pass_interpreter_epoch_t after =
      loom_pass_interpreter_epoch(state->module);
  if (iree_status_is_ok(status)) {
    *out_changed |= loom_pass_interpreter_epoch_changed(before, after);
    *out_changed |= loom_pass_interpreter_statistics_changed(pass);
  }
  return status;
}

static iree_status_t loom_pass_interpreter_invoke_function(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, loom_pass_t* pass,
    bool* out_changed) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  if (!invoke->function_run) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function pass callback is required");
  }
  if (!loom_func_like_isa(frame->function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function pass requires current function");
  }

  iree_arena_allocator_t function_arena;
  iree_arena_initialize(state->options->block_pool, &function_arena);
  pass->function_run = invoke->function_run;
  pass->arena = &function_arena;
  iree_arena_reset(&function_arena);

  loom_pass_interpreter_epoch_t before =
      loom_pass_interpreter_epoch(state->module);
  iree_status_t status =
      invoke->function_run(pass, state->module, frame->function);
  loom_pass_interpreter_epoch_t after =
      loom_pass_interpreter_epoch(state->module);

  pass->arena = pass->instance_arena;
  iree_arena_deinitialize(&function_arena);
  if (iree_status_is_ok(status)) {
    *out_changed |= loom_pass_interpreter_epoch_changed(before, after);
    *out_changed |= loom_pass_interpreter_statistics_changed(pass);
  }
  return status;
}

static iree_status_t loom_pass_interpreter_execute_range(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame, iree_host_size_t start,
    iree_host_size_t end, bool* out_changed);

static iree_status_t loom_pass_interpreter_invoke(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  const loom_pass_program_invoke_t* invoke = &instruction->invoke;
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(state->options->block_pool, &instance_arena);

  loom_pass_t pass = {0};
  iree_status_t status = loom_pass_interpreter_make_pass(
      state, instruction, &instance_arena, &pass);

  bool created = false;
  if (iree_status_is_ok(status) && invoke->descriptor->create) {
    status = invoke->descriptor->create(&pass, iree_string_view_empty());
    created = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status)) {
    if (invoke->info->kind == LOOM_PASS_MODULE) {
      status = loom_pass_interpreter_invoke_module(state, instruction, &pass,
                                                   out_changed);
    } else if (invoke->info->kind == LOOM_PASS_FUNCTION) {
      status = loom_pass_interpreter_invoke_function(state, frame, instruction,
                                                     &pass, out_changed);
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported pass kind");
    }
  }

  if (created && invoke->descriptor->destroy) {
    invoke->descriptor->destroy(&pass);
  }
  iree_arena_deinitialize(&instance_arena);
  if (!iree_status_is_ok(status)) {
    iree_string_view_t symbol_name =
        loom_pass_interpreter_symbol_name(state, frame);
    return iree_status_annotate_f(
        status, "while executing pass '%.*s' at %s anchor on @%.*s",
        (int)invoke->descriptor->key.size, invoke->descriptor->key.data,
        loom_pass_interpreter_anchor_name(frame->kind), (int)symbol_name.size,
        symbol_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_build_function_snapshot(
    loom_pass_interpreter_state_t* state, iree_arena_allocator_t* arena,
    loom_pass_interpreter_symbol_snapshot_entry_t** out_entries,
    iree_host_size_t* out_count) {
  *out_entries = NULL;
  *out_count = 0;
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(state->module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }
    ++count;
  }
  if (count == 0) {
    return iree_ok_status();
  }

  loom_pass_interpreter_symbol_snapshot_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*entries),
                                                 (void**)&entries));
  iree_host_size_t entry_index = 0;
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(state->module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }
    entries[entry_index++] = (loom_pass_interpreter_symbol_snapshot_entry_t){
        .symbol = symbol,
        .function = function,
    };
  }
  *out_entries = entries;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_execute_for(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  if (frame->kind != LOOM_PASS_MODULE ||
      instruction->for_each_symbol.symbol_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.for execution anchor");
  }
  if (instruction->for_each_symbol.snapshot_kind !=
      LOOM_PASS_PROGRAM_SYMBOL_SNAPSHOT_FUNCTIONS_BY_SYMBOL_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.for snapshot kind");
  }

  iree_arena_allocator_t snapshot_arena;
  iree_arena_initialize(state->options->block_pool, &snapshot_arena);
  loom_pass_interpreter_symbol_snapshot_entry_t* entries = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = loom_pass_interpreter_build_function_snapshot(
      state, &snapshot_arena, &entries, &count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    loom_pass_interpreter_frame_t body_frame = {
        .kind = LOOM_PASS_FUNCTION,
        .symbol = entries[i].symbol,
        .function = entries[i].function,
    };
    bool body_changed = false;
    status = loom_pass_interpreter_execute_range(
        state, &body_frame, instruction->for_each_symbol.body_start,
        instruction->for_each_symbol.body_end, &body_changed);
    *out_changed |= body_changed;
  }
  iree_arena_deinitialize(&snapshot_arena);
  return status;
}

static iree_status_t loom_pass_interpreter_find_attr(
    loom_pass_program_attr_list_t attrs, iree_string_view_t name,
    const loom_pass_program_attr_t** out_attr) {
  *out_attr = NULL;
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    if (iree_string_view_equal(attrs.attrs[i].name, name)) {
      *out_attr = &attrs.attrs[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "missing pass predicate attr '%.*s'", (int)name.size,
                          name.data);
}

static iree_status_t loom_pass_interpreter_evaluate_name_predicate(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_where_t* where, bool* out_match) {
  *out_match = false;
  const loom_pass_program_attr_t* value_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_find_attr(
      where->attrs, IREE_SV("value"), &value_attr));
  if (value_attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where name predicate requires string 'value' attr");
  }
  iree_string_view_t symbol_name =
      loom_pass_interpreter_symbol_name(state, frame);
  *out_match =
      iree_string_view_equal(symbol_name, value_attr->value.string_value);
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_execute_where(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  const loom_pass_program_where_t* where = &instruction->where;
  bool match = false;
  if (iree_string_view_equal(where->predicate, IREE_SV("name"))) {
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_evaluate_name_predicate(
        state, frame, where, &match));
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported pass.where predicate '%.*s'",
                            (int)where->predicate.size, where->predicate.data);
  }
  if (!match) {
    return iree_ok_status();
  }
  return loom_pass_interpreter_execute_range(state, frame, where->body_start,
                                             where->body_end, out_changed);
}

static iree_status_t loom_pass_interpreter_execute_repeat(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame,
    const loom_pass_program_instruction_t* instruction, bool* out_changed) {
  const loom_pass_program_repeat_t* repeat = &instruction->repeat;
  if (repeat->mode == LOOM_PASS_REPEAT_MODE_FIXED) {
    for (int64_t i = 0; i < repeat->count; ++i) {
      bool body_changed = false;
      IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_range(
          state, frame, repeat->body_start, repeat->body_end, &body_changed));
      *out_changed |= body_changed;
    }
    return iree_ok_status();
  }
  if (repeat->mode != LOOM_PASS_REPEAT_MODE_UNTIL_CONVERGED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass.repeat mode");
  }

  bool ever_changed = false;
  for (int64_t i = 0; i < repeat->max_iterations; ++i) {
    bool body_changed = false;
    IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_range(
        state, frame, repeat->body_start, repeat->body_end, &body_changed));
    ever_changed |= body_changed;
    if (!body_changed) {
      *out_changed |= ever_changed;
      return iree_ok_status();
    }
  }
  *out_changed |= ever_changed;
  return iree_make_status(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      "pass.repeat<until_converged> exceeded max_iterations=%" PRId64,
      repeat->max_iterations);
}

static iree_status_t loom_pass_interpreter_execute_range(
    loom_pass_interpreter_state_t* state,
    const loom_pass_interpreter_frame_t* frame, iree_host_size_t start,
    iree_host_size_t end, bool* out_changed) {
  for (iree_host_size_t pc = start; pc < end; ++pc) {
    const loom_pass_program_instruction_t* instruction =
        &state->program->instructions[pc];
    if (instruction->anchor_kind != frame->kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass program anchor mismatch at instruction %" PRIhsz, pc);
    }
    switch (instruction->kind) {
      case LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_invoke(
            state, frame, instruction, out_changed));
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_for(
            state, frame, instruction, out_changed));
        pc = instruction->for_each_symbol.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_WHERE: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_where(
            state, frame, instruction, out_changed));
        pc = instruction->where.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT: {
        IREE_RETURN_IF_ERROR(loom_pass_interpreter_execute_repeat(
            state, frame, instruction, out_changed));
        pc = instruction->repeat.body_end - 1;
        break;
      }
      case LOOM_PASS_PROGRAM_INSTRUCTION_FAIL:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s",
                                (int)instruction->message.message.size,
                                instruction->message.message.data);
      case LOOM_PASS_PROGRAM_INSTRUCTION_HALT:
        return iree_make_status(IREE_STATUS_ABORTED, "%.*s",
                                (int)instruction->message.message.size,
                                instruction->message.message.data);
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported pass program instruction");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_interpreter_options_verify(
    const loom_pass_interpreter_options_t* options) {
  if (!options || !options->block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass interpreter options with block pool are "
                            "required");
  }
  return iree_ok_status();
}

iree_status_t loom_pass_interpreter_run_module(
    const loom_pass_program_t* program, loom_module_t* module,
    const loom_pass_interpreter_options_t* options) {
  if (!program || !module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass program and module are required");
  }
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_options_verify(options));
  if (program->root_kind != LOOM_PASS_MODULE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected module-root pass program");
  }
  loom_pass_interpreter_state_t state = {
      .program = program,
      .module = module,
      .options = options,
  };
  loom_pass_interpreter_frame_t frame = {
      .kind = LOOM_PASS_MODULE,
  };
  bool changed = false;
  return loom_pass_interpreter_execute_range(
      &state, &frame, 0, program->instruction_count, &changed);
}

iree_status_t loom_pass_interpreter_run_function(
    const loom_pass_program_t* program, loom_module_t* module,
    loom_func_like_t function, const loom_pass_interpreter_options_t* options) {
  if (!program || !module || !loom_func_like_isa(function)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass program, module, and current function are required");
  }
  IREE_RETURN_IF_ERROR(loom_pass_interpreter_options_verify(options));
  if (program->root_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected function-root pass program");
  }
  loom_pass_interpreter_state_t state = {
      .program = program,
      .module = module,
      .options = options,
  };
  loom_symbol_ref_t callee = loom_func_like_callee(function);
  loom_symbol_t* symbol = NULL;
  if (loom_symbol_ref_is_valid(callee) && callee.module_id == 0 &&
      callee.symbol_id < module->symbols.count) {
    symbol = &module->symbols.entries[callee.symbol_id];
  }
  loom_pass_interpreter_frame_t frame = {
      .kind = LOOM_PASS_FUNCTION,
      .symbol = symbol,
      .function = function,
  };
  bool changed = false;
  return loom_pass_interpreter_execute_range(
      &state, &frame, 0, program->instruction_count, &changed);
}
