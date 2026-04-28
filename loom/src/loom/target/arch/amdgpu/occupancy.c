// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/occupancy.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/types.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct loom_amdgpu_occupancy_register_class_model_t {
  // Stable register-class name.
  iree_string_view_t register_class;
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Occupancy register-file pool shared by resident waves.
  uint32_t pool_units;
  // Allocation granularity used by occupancy calculations.
  uint32_t allocation_granularity;
} loom_amdgpu_occupancy_register_class_model_t;

typedef struct loom_amdgpu_occupancy_model_t {
  // Stable descriptor-set identity selected by the target bundle.
  uint64_t descriptor_set_stable_id;
  // AMDGPU wave size used by this model.
  uint32_t wave_size;
  // Maximum resident waves per SIMD.
  uint32_t max_waves_per_simd;
  // Register-class occupancy models in diagnostic order.
  const loom_amdgpu_occupancy_register_class_model_t* register_classes;
  // Number of entries in |register_classes|.
  iree_host_size_t register_class_count;
} loom_amdgpu_occupancy_model_t;

static const loom_amdgpu_occupancy_register_class_model_t
    kAmdgpuGfx950RegisterClasses[] = {
        {IREE_SVL("amdgpu.sgpr"), AMDGPU_GFX950_CORE_REG_CLASS_ID_AMDGPU_SGPR,
         800, 16},
        {IREE_SVL("amdgpu.vgpr"), AMDGPU_GFX950_CORE_REG_CLASS_ID_AMDGPU_VGPR,
         1024, 4},
        {IREE_SVL("amdgpu.agpr"), AMDGPU_GFX950_CORE_REG_CLASS_ID_AMDGPU_AGPR,
         256, 4},
};

static const loom_amdgpu_occupancy_register_class_model_t
    kAmdgpuGfx11RegisterClasses[] = {
        {IREE_SVL("amdgpu.sgpr"), AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_SGPR,
         800, 16},
        {IREE_SVL("amdgpu.vgpr"), AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_VGPR,
         1024, 4},
};

static const loom_amdgpu_occupancy_register_class_model_t
    kAmdgpuGfx12RegisterClasses[] = {
        {IREE_SVL("amdgpu.sgpr"), AMDGPU_GFX12_CORE_REG_CLASS_ID_AMDGPU_SGPR,
         800, 16},
        {IREE_SVL("amdgpu.vgpr"), AMDGPU_GFX12_CORE_REG_CLASS_ID_AMDGPU_VGPR,
         1024, 4},
};

static const loom_amdgpu_occupancy_register_class_model_t
    kAmdgpuGfx1250RegisterClasses[] = {
        {IREE_SVL("amdgpu.sgpr"), AMDGPU_GFX1250_CORE_REG_CLASS_ID_AMDGPU_SGPR,
         800, 16},
        {IREE_SVL("amdgpu.vgpr"), AMDGPU_GFX1250_CORE_REG_CLASS_ID_AMDGPU_VGPR,
         1024, 4},
};

static const loom_amdgpu_occupancy_model_t kAmdgpuOccupancyModels[] = {
    {
        .descriptor_set_stable_id = AMDGPU_GFX950_CORE_DESCRIPTOR_SET_ID,
        .wave_size = 64,
        .max_waves_per_simd = 16,
        .register_classes = kAmdgpuGfx950RegisterClasses,
        .register_class_count = IREE_ARRAYSIZE(kAmdgpuGfx950RegisterClasses),
    },
    {
        .descriptor_set_stable_id = AMDGPU_GFX11_CORE_DESCRIPTOR_SET_ID,
        .wave_size = 64,
        .max_waves_per_simd = 16,
        .register_classes = kAmdgpuGfx11RegisterClasses,
        .register_class_count = IREE_ARRAYSIZE(kAmdgpuGfx11RegisterClasses),
    },
    {
        .descriptor_set_stable_id = AMDGPU_GFX12_CORE_DESCRIPTOR_SET_ID,
        .wave_size = 64,
        .max_waves_per_simd = 16,
        .register_classes = kAmdgpuGfx12RegisterClasses,
        .register_class_count = IREE_ARRAYSIZE(kAmdgpuGfx12RegisterClasses),
    },
    {
        .descriptor_set_stable_id = AMDGPU_GFX1250_CORE_DESCRIPTOR_SET_ID,
        .wave_size = 64,
        .max_waves_per_simd = 16,
        .register_classes = kAmdgpuGfx1250RegisterClasses,
        .register_class_count = IREE_ARRAYSIZE(kAmdgpuGfx1250RegisterClasses),
    },
};

static iree_status_t loom_amdgpu_occupancy_round_up_u32(
    uint32_t value, uint32_t multiple, uint32_t* out_rounded_value) {
  IREE_ASSERT_ARGUMENT(out_rounded_value);
  if (value == 0 || multiple <= 1) {
    *out_rounded_value = value;
    return iree_ok_status();
  }
  uint32_t remainder = value % multiple;
  if (remainder == 0) {
    *out_rounded_value = value;
    return iree_ok_status();
  }
  const uint32_t delta = multiple - remainder;
  if (value > UINT32_MAX - delta) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU occupancy rounded allocation unit count overflows");
  }
  *out_rounded_value = value + delta;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_wave_limit(
    const loom_amdgpu_occupancy_register_class_model_t* model,
    uint32_t max_waves_per_simd, uint32_t allocated_units,
    uint32_t* out_rounded_units, uint32_t* out_wave_limit) {
  IREE_ASSERT_ARGUMENT(model);
  IREE_ASSERT_ARGUMENT(out_rounded_units);
  IREE_ASSERT_ARGUMENT(out_wave_limit);

  if (allocated_units == 0) {
    *out_rounded_units = 0;
    *out_wave_limit = max_waves_per_simd;
    return iree_ok_status();
  }
  uint32_t rounded_units = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_round_up_u32(
      allocated_units, model->allocation_granularity, &rounded_units));
  *out_rounded_units = rounded_units;
  if (rounded_units == 0) {
    *out_wave_limit = 0;
    return iree_ok_status();
  }
  uint32_t wave_limit = model->pool_units / rounded_units;
  if (wave_limit > max_waves_per_simd) {
    wave_limit = max_waves_per_simd;
  }
  *out_wave_limit = wave_limit;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_next_cliff_units(
    const loom_amdgpu_occupancy_register_class_model_t* model,
    uint32_t max_waves_per_simd, uint32_t allocated_units,
    uint32_t current_wave_limit, uint32_t* out_next_cliff_units) {
  IREE_ASSERT_ARGUMENT(out_next_cliff_units);
  *out_next_cliff_units = 0;
  if (current_wave_limit == 0) {
    return iree_ok_status();
  }
  const uint64_t start_candidate = (uint64_t)allocated_units + 1;
  uint64_t stop_candidate =
      (uint64_t)model->pool_units + model->allocation_granularity;
  if (stop_candidate > UINT32_MAX) {
    stop_candidate = UINT32_MAX;
  }
  for (uint64_t candidate = start_candidate; candidate <= stop_candidate;
       ++candidate) {
    uint32_t ignored_rounded_units = 0;
    uint32_t candidate_wave_limit = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
        model, max_waves_per_simd, (uint32_t)candidate, &ignored_rounded_units,
        &candidate_wave_limit));
    if (candidate_wave_limit < current_wave_limit) {
      *out_next_cliff_units = (uint32_t)candidate;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_select_model(
    uint64_t descriptor_set_stable_id, iree_string_view_t descriptor_set_key,
    const loom_amdgpu_occupancy_model_t** out_model) {
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuOccupancyModels);
       ++i) {
    const loom_amdgpu_occupancy_model_t* model = &kAmdgpuOccupancyModels[i];
    if (model->descriptor_set_stable_id == descriptor_set_stable_id) {
      *out_model = model;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU occupancy model is not defined for descriptor set '%.*s' "
      "(stable ID 0x%016" PRIx64 ")",
      (int)descriptor_set_key.size, descriptor_set_key.data,
      descriptor_set_stable_id);
}

iree_status_t loom_amdgpu_occupancy_build_schedule_pressure_cliffs(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_pressure_cliff_list_t* out_pressure_cliffs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_pressure_cliffs);
  *out_pressure_cliffs = loom_low_schedule_pressure_cliff_list_empty();

  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &descriptor_set_key));
  const loom_amdgpu_occupancy_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_select_model(
      descriptor_set->stable_id, descriptor_set_key, &model));
  if (model->register_class_count == 0 || model->max_waves_per_simd == 0) {
    return iree_ok_status();
  }

  iree_host_size_t max_cliff_count = 0;
  if (!iree_host_size_checked_mul(model->register_class_count,
                                  model->max_waves_per_simd,
                                  &max_cliff_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU schedule pressure cliff capacity exceeds host size");
  }
  loom_low_schedule_pressure_cliff_t* cliffs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, max_cliff_count, sizeof(*cliffs), (void**)&cliffs));
  iree_host_size_t cliff_count = 0;

  for (iree_host_size_t class_index = 0;
       class_index < model->register_class_count; ++class_index) {
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[class_index];
    uint32_t previous_wave_limit = model->max_waves_per_simd;
    uint64_t stop_candidate =
        (uint64_t)class_model->pool_units + class_model->allocation_granularity;
    if (stop_candidate > UINT32_MAX) {
      stop_candidate = UINT32_MAX;
    }
    for (uint64_t candidate = 1; candidate <= stop_candidate; ++candidate) {
      uint32_t ignored_rounded_units = 0;
      uint32_t candidate_wave_limit = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
          class_model, model->max_waves_per_simd, (uint32_t)candidate,
          &ignored_rounded_units, &candidate_wave_limit));
      if (candidate_wave_limit >= previous_wave_limit) {
        continue;
      }
      if (cliff_count >= max_cliff_count) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU schedule pressure cliff capacity was underestimated");
      }
      cliffs[cliff_count++] = (loom_low_schedule_pressure_cliff_t){
          .descriptor_reg_class_id = class_model->descriptor_reg_class_id,
          .cliff_units = (uint32_t)candidate,
          .tier_before = previous_wave_limit,
          .tier_after = candidate_wave_limit,
      };
      previous_wave_limit = candidate_wave_limit;
      if (candidate_wave_limit == 0) {
        break;
      }
    }
  }

  *out_pressure_cliffs = (loom_low_schedule_pressure_cliff_list_t){
      .values = cliffs,
      .count = cliff_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_find_register_class(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_occupancy_register_class_t* register_classes,
    iree_host_size_t register_class_count, uint16_t descriptor_reg_class_id,
    uint32_t* out_index) {
  IREE_ASSERT_ARGUMENT(out_index);
  for (iree_host_size_t i = 0; i < register_class_count; ++i) {
    if (register_classes[i].descriptor_reg_class_id ==
        descriptor_reg_class_id) {
      if (i > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU occupancy class index overflows");
      }
      *out_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  if (descriptor_reg_class_id >=
      allocation->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU occupancy assignment references invalid descriptor register "
        "class ID %" PRIu16,
        descriptor_reg_class_id);
  }
  const loom_low_reg_class_t* descriptor_reg_class =
      &allocation->target.descriptor_set->reg_classes[descriptor_reg_class_id];
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      allocation->target.descriptor_set,
      descriptor_reg_class->name_string_offset, &register_class));
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU occupancy model for descriptor set '%.*s' does not cover "
      "register class '%.*s'",
      (int)allocation->target.descriptor_set_key.size,
      allocation->target.descriptor_set_key.data, (int)register_class.size,
      register_class.data);
}

static iree_status_t loom_amdgpu_occupancy_collect_allocations(
    const loom_low_allocation_sidecar_t* allocation,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    iree_host_size_t class_summary_count) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    uint32_t class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_find_register_class(
        allocation, class_summaries, class_summary_count,
        assignment->descriptor_reg_class_id, &class_index));
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    uint64_t allocated_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (allocated_end > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU occupancy allocated register range overflows");
    }
    if ((uint32_t)allocated_end >
        class_summaries[class_index].allocated_units) {
      class_summaries[class_index].allocated_units = (uint32_t)allocated_end;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_add_u32(uint32_t lhs, uint32_t rhs,
                                                   uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  uint64_t value = (uint64_t)lhs + rhs;
  if (value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU occupancy summary overflows uint32_t");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_collect_spills(
    const loom_low_allocation_sidecar_t* allocation,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    iree_host_size_t class_summary_count,
    loom_amdgpu_occupancy_sidecar_t* sidecar) {
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    if (spill_plan->assignment_index >= allocation->assignment_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU occupancy spill plan references "
                              "assignment %" PRIu32
                              " but allocation has %zu "
                              "assignment(s)",
                              spill_plan->assignment_index,
                              allocation->assignment_count);
    }
    if (spill_plan->slot_space != LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU occupancy expected scratch spill slots, got '%.*s'",
          (int)loom_low_spill_slot_space_name(spill_plan->slot_space).size,
          loom_low_spill_slot_space_name(spill_plan->slot_space).data);
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[spill_plan->assignment_index];
    uint32_t class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_find_register_class(
        allocation, class_summaries, class_summary_count,
        assignment->descriptor_reg_class_id, &class_index));

    loom_amdgpu_occupancy_register_class_t* class_summary =
        &class_summaries[class_index];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_count, 1, &class_summary->spill_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_bytes, spill_plan->byte_size,
        &class_summary->spill_bytes));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_store_count, spill_plan->store_count,
        &class_summary->spill_store_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        class_summary->spill_reload_count, spill_plan->reload_count,
        &class_summary->spill_reload_count));

    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(sidecar->spill_count, 1,
                                                       &sidecar->spill_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        sidecar->scratch_spill_bytes, spill_plan->byte_size,
        &sidecar->scratch_spill_bytes));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        sidecar->spill_store_count, spill_plan->store_count,
        &sidecar->spill_store_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_add_u32(
        sidecar->spill_reload_count, spill_plan->reload_count,
        &sidecar->spill_reload_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_flat_workgroup_size(
    const loom_target_export_plan_t* export_plan, uint32_t* out_flat_size) {
  IREE_ASSERT_ARGUMENT(out_flat_size);
  *out_flat_size = 0;
  if (!export_plan || export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_ok_status();
  }
  const loom_target_workgroup_size_t size =
      export_plan->hal_kernel.required_workgroup_size;
  uint64_t flat_size = (uint64_t)size.x * size.y * size.z;
  if (flat_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU workgroup size overflows uint32_t");
  }
  *out_flat_size = (uint32_t)flat_size;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_finalize_limits(
    const loom_amdgpu_occupancy_model_t* model,
    loom_amdgpu_occupancy_register_class_t* class_summaries,
    loom_amdgpu_occupancy_sidecar_t* sidecar) {
  sidecar->resident_waves_per_simd = sidecar->max_waves_per_simd;
  sidecar->limiting_register_class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE;
  for (iree_host_size_t i = 0; i < sidecar->register_class_count; ++i) {
    loom_amdgpu_occupancy_register_class_t* class_summary = &class_summaries[i];
    const loom_amdgpu_occupancy_register_class_model_t* class_model =
        &model->register_classes[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_wave_limit(
        class_model, sidecar->max_waves_per_simd,
        class_summary->allocated_units, &class_summary->rounded_units,
        &class_summary->wave_limit));
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_next_cliff_units(
        class_model, sidecar->max_waves_per_simd,
        class_summary->allocated_units, class_summary->wave_limit,
        &class_summary->next_cliff_units));
    class_summary->units_until_next_cliff =
        class_summary->next_cliff_units == 0
            ? UINT32_MAX
            : class_summary->next_cliff_units - class_summary->allocated_units;
    if (class_summary->wave_limit < sidecar->resident_waves_per_simd) {
      sidecar->resident_waves_per_simd = class_summary->wave_limit;
      sidecar->limiting_register_class_index = (uint32_t)i;
    }
  }
  if (sidecar->max_waves_per_simd == 0) {
    sidecar->occupancy_percent = 0;
  } else {
    sidecar->occupancy_percent =
        (sidecar->resident_waves_per_simd * 100u) / sidecar->max_waves_per_simd;
  }
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_occupancy_limiting_resource_name(
    const loom_amdgpu_occupancy_sidecar_t* sidecar) {
  if (sidecar->limiting_register_class_index ==
      LOOM_AMDGPU_OCCUPANCY_CLASS_NONE) {
    return IREE_SV("max_waves");
  }
  if (sidecar->limiting_register_class_index >= sidecar->register_class_count) {
    return IREE_SV("unknown");
  }
  return sidecar->register_classes[sidecar->limiting_register_class_index]
      .register_class;
}

static iree_status_t loom_amdgpu_occupancy_emit_summary(
    const loom_amdgpu_occupancy_sidecar_t* sidecar,
    iree_diagnostic_emitter_t emitter) {
  iree_string_view_t limiting_resource =
      loom_amdgpu_occupancy_limiting_resource_name(sidecar);
  for (iree_host_size_t i = 0; i < sidecar->register_class_count; ++i) {
    const loom_amdgpu_occupancy_register_class_t* class_summary =
        &sidecar->register_classes[i];
    loom_diagnostic_param_t params[] = {
        loom_param_string(
            loom_low_diagnostic_target_key(&sidecar->allocation->target)),
        loom_param_string(
            loom_low_diagnostic_export_name(&sidecar->allocation->target)),
        loom_param_string(
            loom_low_diagnostic_config_key(&sidecar->allocation->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            sidecar->allocation->module, sidecar->allocation->function_op)),
        loom_param_string(class_summary->register_class),
        loom_param_u32(class_summary->pool_units),
        loom_param_u32(class_summary->allocated_units),
        loom_param_u32(sidecar->occupancy_percent),
        loom_param_string(limiting_resource),
    };
    loom_diagnostic_emission_t emission = {
        .op = sidecar->allocation->function_op,
        .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 10),
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_occupancy_build(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_occupancy_options_t* options,
    iree_arena_allocator_t* arena,
    loom_amdgpu_occupancy_sidecar_t* out_sidecar) {
  IREE_ASSERT_ARGUMENT(out_sidecar);
  if (!allocation || !arena) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation sidecar and arena are required for AMDGPU occupancy");
  }
  *out_sidecar = (loom_amdgpu_occupancy_sidecar_t){0};

  const loom_amdgpu_occupancy_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_select_model(
      allocation->target.descriptor_set->stable_id,
      allocation->target.descriptor_set_key, &model));

  loom_amdgpu_occupancy_register_class_t* register_classes = NULL;
  if (model->register_class_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, model->register_class_count, sizeof(*register_classes),
        (void**)&register_classes));
    memset(register_classes, 0,
           model->register_class_count * sizeof(*register_classes));
  }

  loom_amdgpu_occupancy_sidecar_t sidecar = {
      .allocation = allocation,
      .target_cpu = allocation->target.bundle_storage.snapshot.target_cpu,
      .wave_size = model->wave_size,
      .max_waves_per_simd = model->max_waves_per_simd,
      .resident_waves_per_simd = model->max_waves_per_simd,
      .limiting_register_class_index = LOOM_AMDGPU_OCCUPANCY_CLASS_NONE,
      .register_classes = register_classes,
      .register_class_count = model->register_class_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_flat_workgroup_size(
      &allocation->target.bundle_storage.export_plan,
      &sidecar.flat_workgroup_size));
  if (sidecar.flat_workgroup_size != 0 && sidecar.wave_size != 0) {
    sidecar.waves_per_workgroup =
        (sidecar.flat_workgroup_size + sidecar.wave_size - 1) /
        sidecar.wave_size;
  }
  for (iree_host_size_t i = 0; i < model->register_class_count; ++i) {
    register_classes[i] = (loom_amdgpu_occupancy_register_class_t){
        .register_class = model->register_classes[i].register_class,
        .descriptor_reg_class_id =
            model->register_classes[i].descriptor_reg_class_id,
        .pool_units = model->register_classes[i].pool_units,
        .allocation_granularity =
            model->register_classes[i].allocation_granularity,
        .units_until_next_cliff = UINT32_MAX,
    };
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_collect_allocations(
      allocation, register_classes, model->register_class_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_collect_spills(
      allocation, register_classes, model->register_class_count, &sidecar));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_occupancy_finalize_limits(model, register_classes, &sidecar));

  if (options && iree_any_bit_set(options->diagnostic_flags,
                                  LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_occupancy_emit_summary(&sidecar, options->emitter));
  }
  *out_sidecar = sidecar;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_occupancy_write_cliff_u32_or_null(
    uint32_t value, loom_output_stream_t* stream) {
  if (value == UINT32_MAX || value == 0) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_amdgpu_occupancy_write_register_class(
    const loom_amdgpu_occupancy_register_class_t* register_class,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"register_class\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, register_class->register_class));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"allocated_units\":%" PRIu32 ",\"rounded_units\":%" PRIu32
      ",\"pool_units\":%" PRIu32 ",\"allocation_granularity\":%" PRIu32
      ",\"wave_limit\":%" PRIu32 ",\"next_cliff_units\":",
      register_class->allocated_units, register_class->rounded_units,
      register_class->pool_units, register_class->allocation_granularity,
      register_class->wave_limit));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      register_class->next_cliff_units, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"units_until_next_cliff\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_cliff_u32_or_null(
      register_class->units_until_next_cliff, stream));
  return loom_output_stream_write_format(
      stream,
      ",\"spill_count\":%" PRIu32 ",\"spill_bytes\":%" PRIu32
      ",\"spill_store_count\":%" PRIu32 ",\"spill_reload_count\":%" PRIu32 "}",
      register_class->spill_count, register_class->spill_bytes,
      register_class->spill_store_count, register_class->spill_reload_count);
}

iree_status_t loom_amdgpu_occupancy_format_json(
    const loom_amdgpu_occupancy_sidecar_t* sidecar,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  if (!sidecar || !sidecar->allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU occupancy sidecar is required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.occupancy.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream,
      loom_low_diagnostic_function_name(sidecar->allocation->module,
                                        sidecar->allocation->function_op)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, sidecar->allocation->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, sidecar->allocation->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target_cpu\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, sidecar->target_cpu));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"wave_size\":%" PRIu32 ",\"max_waves_per_simd\":%" PRIu32
      ",\"flat_workgroup_size\":%" PRIu32 ",\"waves_per_workgroup\":%" PRIu32
      ",\"resident_waves_per_simd\":%" PRIu32 ",\"occupancy_percent\":%" PRIu32
      ",\"limiting_resource\":",
      sidecar->wave_size, sidecar->max_waves_per_simd,
      sidecar->flat_workgroup_size, sidecar->waves_per_workgroup,
      sidecar->resident_waves_per_simd, sidecar->occupancy_percent));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_amdgpu_occupancy_limiting_resource_name(sidecar)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"spill_count\":%" PRIu32 ",\"scratch_spill_bytes\":%" PRIu32
      ",\"spill_store_count\":%" PRIu32 ",\"spill_reload_count\":%" PRIu32
      ",\"register_classes\":[",
      sidecar->spill_count, sidecar->scratch_spill_bytes,
      sidecar->spill_store_count, sidecar->spill_reload_count));
  for (iree_host_size_t i = 0; i < sidecar->register_class_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_write_register_class(
        &sidecar->register_classes[i], &stream));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
