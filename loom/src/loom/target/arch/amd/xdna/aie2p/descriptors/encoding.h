// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final adaptation from allocated AIE2P Low descriptors to instruction slots.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_

#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encodes one compiler-verified descriptor after register allocation.
// Assignment pointers correspond positionally to descriptor operand rows,
// including results. Immediate values correspond positionally to descriptor
// immediates and symbolic values have already been resolved. Low verification,
// allocation, and generated target tables prove every join consumed here.
loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal,
    const loom_low_allocation_assignment_t* const* operand_assignments,
    const int64_t* immediate_values);

// Physical operands of one native register-move instruction.
typedef struct loom_aie2p_register_move_t {
  // Physical register read by the instruction.
  loom_aie2p_physical_register_id_t source;
  // Physical register written by the instruction.
  loom_aie2p_physical_register_id_t destination;
} loom_aie2p_register_move_t;

// Selects the native descriptor for one allocation move between valid AIE2P
// physical registers. Returns LOOM_LOW_DESCRIPTOR_ORDINAL_NONE when no direct
// move is declared. When several moves apply, descriptor order determines the
// selected encoding. The result also identifies the move's operand timing and
// resource requirements for physical scheduling.
uint32_t loom_aie2p_descriptor_select_move(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination);

// Decomposes an allocated move into native register operands. L registers
// require two scalar moves in low-to-high order; other registers retain one
// move. The caller preserves allocation's sequential move order and selects
// each instruction with loom_aie2p_descriptor_select_move.
uint8_t loom_aie2p_descriptor_move_parts(
    loom_aie2p_register_move_t move, loom_aie2p_register_move_t out_parts[2]);

// Dense generated route selected for one complete physical allocation move.
typedef uint8_t loom_aie2p_move_route_id_t;
#define LOOM_AIE2P_MOVE_ROUTE_ID_INVALID ((loom_aie2p_move_route_id_t)0)

enum loom_aie2p_move_step_encoding_e {
  // Encode through the selected Low descriptor.
  LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR = 0,
  // Encode one native 1024-bit accumulator component.
  LOOM_AIE2P_MOVE_STEP_ENCODING_ACCUMULATOR_1024 = 1,
  // Encode one native 512-bit FIFO component.
  LOOM_AIE2P_MOVE_STEP_ENCODING_FIFO_512 = 2,
};
typedef uint8_t loom_aie2p_move_step_encoding_t;

#define LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT 3u

// One native instruction in a generated physical move route.
typedef struct loom_aie2p_move_route_step_t {
  // Concrete native source and destination registers.
  loom_aie2p_register_move_t move;
  // Descriptor providing register-event behavior for physical issue.
  uint32_t descriptor_ordinal;
  // Canonical schedule classes whose union is the complete resource demand.
  uint16_t resource_schedule_class_ids
      [LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT];
  // Number of valid entries in |resource_schedule_class_ids|; zero selects the
  // descriptor's own schedule class.
  uint8_t resource_schedule_class_count;
  // Native instruction encoding selected for the step.
  loom_aie2p_move_step_encoding_t encoding;
} loom_aie2p_move_route_step_t;

// Resolves a complete native route for valid final physical registers. Direct
// allocation descriptors win over generated scalar and aggregate routes.
// Returns INVALID when the target has no value-preserving route.
loom_aie2p_move_route_id_t loom_aie2p_descriptor_resolve_move_route(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination);

// Returns the native instruction count for a valid generated move route.
uint8_t loom_aie2p_descriptor_move_route_step_count(
    loom_aie2p_move_route_id_t route_id);

// Projects one step from a valid route. The route, original physical move, and
// step index are analysis-owned trusted state.
loom_aie2p_move_route_step_t loom_aie2p_descriptor_move_route_step(
    loom_aie2p_move_route_id_t route_id, loom_aie2p_register_move_t move,
    uint8_t step_index);

// Returns the mask of atomic-unit ordinals touched by a register part within
// any physical register in its declared class. NONE selects all units. The
// generator proves this projection against every candidate's subregisters.
uint32_t loom_aie2p_descriptor_register_part_units(uint16_t register_part_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ENCODING_H_
