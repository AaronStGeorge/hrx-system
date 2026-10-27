// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include "iree/base/internal/math.h"

// Allocation move membership and register-part projections are resolved from
// the owned machine tables at generation time.
#include "loom/target/arch/amd/xdna/aie2p/descriptors/register_tables.inl"

#define LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT 16u

uint32_t loom_aie2p_descriptor_register_part_units(uint16_t register_part_id) {
  return register_part_id == LOOM_LOW_REGISTER_PART_NONE
             ? UINT32_MAX
             : kRegisterPartAtomicUnitMasks[register_part_id];
}

uint32_t loom_aie2p_descriptor_select_move(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination) {
  const uint32_t candidates =
      kMoveSourceMasks[source] & kMoveDestinationMasks[destination];
  return candidates != 0
             ? kMoveDescriptorOrdinals[iree_math_count_trailing_zeros_u32(
                   candidates)]
             : LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

uint8_t loom_aie2p_descriptor_move_parts(
    loom_aie2p_register_move_t move, loom_aie2p_register_move_t out_parts[2]) {
  const uint8_t source_pair = kMoveScalarPairIndices[move.source];
  const uint8_t destination_pair = kMoveScalarPairIndices[move.destination];
  if (source_pair != 0 && destination_pair != 0) {
    for (uint8_t i = 0; i < 2; ++i) {
      out_parts[i] = (loom_aie2p_register_move_t){
          .source = kMoveScalarPairs[source_pair][i],
          .destination = kMoveScalarPairs[destination_pair][i],
      };
    }
    return 2;
  }
  out_parts[0] = move;
  return 1;
}

loom_aie2p_move_route_id_t loom_aie2p_descriptor_resolve_move_route(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination) {
  const uint32_t direct_candidates =
      kMoveSourceMasks[source] & kMoveDestinationMasks[destination];
  if (direct_candidates != 0) {
    return (loom_aie2p_move_route_id_t)(kMoveRouteDirectFirst +
                                        iree_math_count_trailing_zeros_u32(
                                            direct_candidates));
  }

  const uint8_t source_category = kMoveScalarSourceCategories[source];
  const uint8_t destination_category =
      kMoveScalarDestinationCategories[destination];
  if (source_category != UINT8_MAX && destination_category != UINT8_MAX) {
    return (loom_aie2p_move_route_id_t)(kMoveRouteScalarFirst +
                                        destination_category *
                                            kMoveRouteScalarSourceCount +
                                        source_category);
  }

  const loom_aie2p_move_projection_t* source_projection =
      &kMoveProjections[kMoveProjectionIndices[source]];
  const loom_aie2p_move_projection_t* destination_projection =
      &kMoveProjections[kMoveProjectionIndices[destination]];
  return source_projection->route_id != 0 &&
                 source_projection->route_id == destination_projection->route_id
             ? source_projection->route_id
             : LOOM_AIE2P_MOVE_ROUTE_ID_INVALID;
}

uint8_t loom_aie2p_descriptor_move_route_step_count(
    loom_aie2p_move_route_id_t route_id) {
  IREE_ASSERT(route_id != LOOM_AIE2P_MOVE_ROUTE_ID_INVALID &&
              route_id < kMoveRouteLimit);
  return route_id < kMoveRouteAggregateFirst
             ? 1
             : kMoveAggregateStepCounts[route_id - kMoveRouteAggregateFirst];
}

static void loom_aie2p_descriptor_append_move_resource_class(
    uint16_t schedule_class_id, loom_aie2p_move_route_step_t* step) {
  if (schedule_class_id == UINT16_MAX) {
    return;
  }
  IREE_ASSERT_LT(step->resource_schedule_class_count,
                 LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT);
  step->resource_schedule_class_ids[step->resource_schedule_class_count++] =
      schedule_class_id;
}

static void loom_aie2p_descriptor_set_scalar_move_resources(
    uint8_t source_category, uint8_t destination_category,
    loom_aie2p_move_route_step_t* step) {
  if (source_category + 1u == kMoveRouteScalarSourceCount) {
    loom_aie2p_descriptor_append_move_resource_class(
        kMoveScalarCoreIdScheduleClassId, step);
    return;
  }
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarCommonScheduleClassId, step);
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarDestinationScheduleClassIds[destination_category], step);
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarSourceScheduleClassIds[source_category], step);
}

loom_aie2p_move_route_step_t loom_aie2p_descriptor_move_route_step(
    loom_aie2p_move_route_id_t route_id, loom_aie2p_register_move_t move,
    uint8_t step_index) {
  IREE_ASSERT(route_id != LOOM_AIE2P_MOVE_ROUTE_ID_INVALID &&
              route_id < kMoveRouteLimit);
  IREE_ASSERT_LT(step_index,
                 loom_aie2p_descriptor_move_route_step_count(route_id));
  loom_aie2p_move_route_step_t step = {
      .move = move,
      .encoding = LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR,
  };
  if (route_id < kMoveRouteScalarFirst) {
    step.descriptor_ordinal =
        kMoveDescriptorOrdinals[route_id - kMoveRouteDirectFirst];
    return step;
  }
  if (route_id < kMoveRouteAggregateFirst) {
    const uint8_t scalar_route = route_id - kMoveRouteScalarFirst;
    step.descriptor_ordinal = kMoveScalarDescriptorOrdinal;
    loom_aie2p_descriptor_set_scalar_move_resources(
        scalar_route % kMoveRouteScalarSourceCount,
        scalar_route / kMoveRouteScalarSourceCount, &step);
    return step;
  }

  const uint8_t aggregate_route = route_id - kMoveRouteAggregateFirst;
  const loom_aie2p_move_projection_t* source_projection =
      &kMoveProjections[kMoveProjectionIndices[move.source]];
  const loom_aie2p_move_projection_t* destination_projection =
      &kMoveProjections[kMoveProjectionIndices[move.destination]];
  IREE_ASSERT_EQ(source_projection->route_id, route_id);
  IREE_ASSERT_EQ(destination_projection->route_id, route_id);
  step.move = (loom_aie2p_register_move_t){
      .source = kMoveProjectionComponents[source_projection->component_start +
                                          step_index],
      .destination =
          kMoveProjectionComponents[destination_projection->component_start +
                                    step_index],
  };
  step.encoding = kMoveAggregateEncodingKinds[aggregate_route];
  if (step.encoding == LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR) {
    step.descriptor_ordinal = kMoveScalarDescriptorOrdinal;
    const uint8_t source_category =
        kMoveScalarSourceCategories[step.move.source];
    const uint8_t destination_category =
        kMoveScalarDestinationCategories[step.move.destination];
    IREE_ASSERT(source_category != UINT8_MAX &&
                destination_category != UINT8_MAX);
    loom_aie2p_descriptor_set_scalar_move_resources(
        source_category, destination_category, &step);
  } else {
    step.descriptor_ordinal = kMoveEventDescriptorOrdinal;
    loom_aie2p_descriptor_append_move_resource_class(
        kMoveAggregateScheduleClassIds[aggregate_route], &step);
  }
  return step;
}

static void loom_aie2p_descriptor_append_field(
    loom_aie2p_encoding_field_id_t field_id, uint64_t value,
    iree_host_size_t* field_count,
    loom_aie2p_encoding_field_value_t* field_values) {
  IREE_ASSERT(*field_count < LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT &&
              "verified AIE2P descriptor exceeds field storage");
  field_values[*field_count] = (loom_aie2p_encoding_field_value_t){
      .field_id = field_id,
      .value = value,
  };
  ++*field_count;
}

loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal,
    const loom_low_allocation_assignment_t* const* operand_assignments,
    const int64_t* immediate_values) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];

  loom_aie2p_encoding_field_value_t
      field_values[LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT];
  iree_host_size_t field_count = 0;
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor_layout->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field =
        &descriptor_set->encoding_field_values
             [descriptor_layout->encoding_field_value_start + i];
    loom_aie2p_descriptor_append_field(field->encoding_field_id, field->value,
                                       &field_count, field_values);
  }

  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->encoding_field_id == 0) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment = operand_assignments[i];
    const uint8_t encoded_register =
        loom_aie2p_machine_adapt_allocated_register(
            operand->encoding_adapter_id,
            (loom_aie2p_physical_register_id_t)assignment->location_base);
    loom_aie2p_descriptor_append_field(operand->encoding_field_id,
                                       encoded_register, &field_count,
                                       field_values);
  }

  for (uint16_t i = 0; i < descriptor_layout->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor_layout->immediate_start + i];
    if (immediate->encoding_field_id == 0) {
      continue;
    }
    const uint64_t encoded_immediate =
        loom_aie2p_machine_encode_verified_immediate(immediate->encoding_id,
                                                     immediate_values[i]);
    loom_aie2p_descriptor_append_field(immediate->encoding_field_id,
                                       encoded_immediate, &field_count,
                                       field_values);
  }

  return loom_aie2p_encoding_pack_verified_instruction(
      descriptor->encoding_id, field_values, field_count);
}
