// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector storage shape helpers.
//
// These helpers describe source-level vector storage properties without naming
// a target register file. Targets can apply their own lane/register caps on top
// of these facts while sharing the same type-shape interpretation.

#ifndef LOOM_OPS_VECTOR_STORAGE_H_
#define LOOM_OPS_VECTOR_STORAGE_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed integer storage shape for a rank-1 vector type.
typedef struct loom_vector_packed_integer_storage_shape_t {
  // Source vector type being described.
  loom_type_t type;
  // Integer element type carried by each logical source lane.
  loom_scalar_type_t element_type;
  // Number of logical lanes in the static rank-1 vector.
  uint32_t lane_count;
  // Bit count of each logical source lane.
  uint32_t element_bit_count;
  // Total payload bits occupied by all logical lanes.
  uint32_t payload_bit_count;
  // Bit count of one target-selected storage unit.
  uint32_t storage_unit_bit_count;
  // Number of storage units needed to hold payload_bit_count bits.
  uint32_t storage_unit_count;
} loom_vector_packed_integer_storage_shape_t;

// Returns the rank-1 static lane count for |type| when it has |element_type|
// and is within |maximum_lane_count|. Returns zero for dynamic, non-rank-1,
// wrong-element, empty, or over-limit vectors.
uint32_t loom_vector_static_rank1_lane_count(loom_type_t type,
                                             loom_scalar_type_t element_type,
                                             uint32_t maximum_lane_count);

// Describes integer bitstream storage for a rank-1 static vector in units of
// |storage_unit_bit_count| bits. Returns false for non-integer vectors,
// dynamic/non-rank-1 vectors, arithmetic overflow, empty payloads, or shapes
// requiring more than |maximum_storage_unit_count| units.
bool loom_vector_packed_integer_storage_shape(
    loom_type_t type, uint32_t storage_unit_bit_count,
    uint32_t maximum_storage_unit_count,
    loom_vector_packed_integer_storage_shape_t* out_shape);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_VECTOR_STORAGE_H_
