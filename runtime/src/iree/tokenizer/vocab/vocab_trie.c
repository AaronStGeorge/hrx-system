// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Double-Array Aho-Corasick (DAAC) trie implementation.
//
// The Double-Array technique provides O(1) state transitions using:
//   next_state = BASE[current_state] + byte
//   valid = (CHECK[next_state] == current_state)
//
// This is the classic algorithm from Aoe (1989) with modern refinements.
// Memory usage is approximately 12 bytes per state (3 x int32_t arrays).
//
// Construction uses a two-phase approach:
//   Phase 1: Build a temporary trie with intrusive sorted child lists
//   Phase 2: Convert to double-array format via BFS
//
// Phase 1 is O(N * L) where N is token count and L is average length.
// Phase 2 uses a first-fit algorithm that scans for valid BASE values.

#include "iree/tokenizer/vocab/vocab_trie.h"

#include <string.h>

#define IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY 512
static_assert((IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY &
               (IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY - 1)) == 0,
              "child cache capacity must be a power of two");

//===----------------------------------------------------------------------===//
// Internal Types
//===----------------------------------------------------------------------===//

// Temporary node used during construction.
// Child edges are intrusive sibling links, avoiding separate storage per node.
typedef struct iree_tokenizer_daac_build_node_t {
  // Phase-specific payload: construction records token IDs, then conversion
  // replaces each consumed token ID with its assigned double-array state.
  union {
    // Token ID if a word ends here, or -1 otherwise.
    int32_t token_id;
    // Assigned double-array state once queued for conversion.
    int32_t array_state;
  } value;
  // First child node index, or UINT32_MAX when this node is a leaf.
  uint32_t first_child;
  // Next sibling during construction, then the BFS queue link after the
  // parent has consumed its sorted child list.
  uint32_t next_sibling;
  // Byte labeling the incoming edge from this node's parent.
  uint8_t edge_byte;
} iree_tokenizer_daac_build_node_t;

// Builder state for constructing DAAC from tokens.
typedef struct iree_tokenizer_daac_builder_t {
  // Allocator for growable arrays and the final immutable trie.
  iree_allocator_t allocator;

  // Temporary trie nodes, independently reallocated to release old copies.
  iree_tokenizer_daac_build_node_t* nodes;
  // Number of initialized nodes.
  iree_host_size_t node_count;
  // Allocated node capacity.
  iree_host_size_t node_capacity;
  // Direct lookup for the root's frequently traversed byte fanout.
  uint32_t root_children[256];
  // Packed (parent node index, edge byte) keys for cached edge traversals.
  uint64_t child_cache_keys[IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY];
  // Child node indices corresponding to child_cache_keys.
  uint32_t child_cache_values[IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY];

  // Transition base offsets under construction.
  int32_t* base;
  // Parent state for each occupied slot, or -1 for free slots.
  int32_t* check;
  // Token ID at each state, or -1 if no token ends there.
  int32_t* output;
  // Number of used slots, including gaps.
  iree_host_size_t array_size;
  // Allocated capacity of each construction array.
  iree_host_size_t array_capacity;
  // Hint for where to start searching.
  iree_host_size_t next_check_pos;

  // Maximum inserted token length in bytes.
  iree_host_size_t max_depth;
} iree_tokenizer_daac_builder_t;

static iree_host_size_t iree_tokenizer_daac_child_cache_index(
    uint32_t parent_index, uint8_t byte) {
  // Multiplicative mixing keeps sequential node indices from aliasing before
  // the edge byte participates in the direct-mapped index.
  return ((parent_index * 0x9E3779B9u) ^ byte) &
         (IREE_TOKENIZER_DAAC_CHILD_CACHE_CAPACITY - 1);
}

//===----------------------------------------------------------------------===//
// Builder - Initialization and Cleanup
//===----------------------------------------------------------------------===//

static void iree_tokenizer_daac_builder_deinitialize(
    iree_tokenizer_daac_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(builder->allocator, builder->nodes);
  iree_allocator_free(builder->allocator, builder->base);
  iree_allocator_free(builder->allocator, builder->check);
  iree_allocator_free(builder->allocator, builder->output);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Builder - Node Allocation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_daac_builder_ensure_node_capacity(
    iree_tokenizer_daac_builder_t* builder) {
  if (builder->node_count < builder->node_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      builder->allocator, /*min_capacity=*/256, sizeof(*builder->nodes),
      &builder->node_capacity, (void**)&builder->nodes);
}

static iree_status_t iree_tokenizer_daac_builder_allocate_node(
    iree_tokenizer_daac_builder_t* builder, uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_daac_builder_ensure_node_capacity(builder));

  // Use >= to prevent collision with UINT32_MAX sentinel in find_child.
  if (builder->node_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "node count %" PRIhsz " exceeds uint32_t limit",
                            builder->node_count);
  }

  uint32_t index = (uint32_t)builder->node_count++;
  iree_tokenizer_daac_build_node_t* node = &builder->nodes[index];
  node->value.token_id = -1;
  node->first_child = UINT32_MAX;
  node->next_sibling = UINT32_MAX;
  node->edge_byte = 0;

  *out_index = index;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Builder - Child Management
//===----------------------------------------------------------------------===//

// Finds a child with the given byte in the parent's sorted sibling list.
// Returns the child index if found, or UINT32_MAX if absent. out_previous is
// the sibling after which a new child belongs, or UINT32_MAX at the front.
static uint32_t iree_tokenizer_daac_find_child(
    iree_tokenizer_daac_builder_t* builder, uint32_t parent_index, uint8_t byte,
    uint32_t* out_previous) {
  if (parent_index == 0 && builder->root_children[byte] != UINT32_MAX) {
    *out_previous = UINT32_MAX;
    return builder->root_children[byte];
  }
  const uint64_t cache_key = ((uint64_t)parent_index << 8) | byte;
  const iree_host_size_t cache_index =
      iree_tokenizer_daac_child_cache_index(parent_index, byte);
  if (builder->child_cache_keys[cache_index] == cache_key) {
    *out_previous = UINT32_MAX;
    return builder->child_cache_values[cache_index];
  }
  uint32_t previous = UINT32_MAX;
  uint32_t child = builder->nodes[parent_index].first_child;
  while (child != UINT32_MAX) {
    const iree_tokenizer_daac_build_node_t* child_node = &builder->nodes[child];
    if (child_node->edge_byte >= byte) {
      *out_previous = previous;
      if (child_node->edge_byte == byte) {
        builder->child_cache_keys[cache_index] = cache_key;
        builder->child_cache_values[cache_index] = child;
        return child;
      }
      return UINT32_MAX;
    }
    previous = child;
    child = child_node->next_sibling;
  }
  *out_previous = previous;
  return UINT32_MAX;
}

// Adds a child to a node, maintaining sorted order by byte.
// Returns the index of the new child node.
static iree_status_t iree_tokenizer_daac_add_child(
    iree_tokenizer_daac_builder_t* builder, uint32_t parent_index, uint8_t byte,
    uint32_t previous_sibling, uint32_t* out_child_index) {
  // Allocate the new child node first.
  uint32_t child_index = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_daac_builder_allocate_node(builder, &child_index));

  iree_tokenizer_daac_build_node_t* child = &builder->nodes[child_index];
  child->edge_byte = byte;
  if (previous_sibling == UINT32_MAX) {
    iree_tokenizer_daac_build_node_t* parent = &builder->nodes[parent_index];
    child->next_sibling = parent->first_child;
    parent->first_child = child_index;
  } else {
    iree_tokenizer_daac_build_node_t* previous =
        &builder->nodes[previous_sibling];
    child->next_sibling = previous->next_sibling;
    previous->next_sibling = child_index;
  }
  if (parent_index == 0) {
    builder->root_children[byte] = child_index;
  } else {
    const iree_host_size_t cache_index =
        iree_tokenizer_daac_child_cache_index(parent_index, byte);
    builder->child_cache_keys[cache_index] =
        ((uint64_t)parent_index << 8) | byte;
    builder->child_cache_values[cache_index] = child_index;
  }

  *out_child_index = child_index;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Builder - Temporary Trie Construction
//===----------------------------------------------------------------------===//

// Inserts a token string into the temporary trie.
static iree_status_t iree_tokenizer_daac_builder_insert(
    iree_tokenizer_daac_builder_t* builder, const uint8_t* string,
    iree_host_size_t length, int32_t token_id) {
  uint32_t node_index = 0;

  for (iree_host_size_t i = 0; i < length; ++i) {
    uint8_t byte = string[i];

    // Look for existing child with this byte.
    uint32_t previous_sibling = UINT32_MAX;
    uint32_t child_index = iree_tokenizer_daac_find_child(
        builder, node_index, byte, &previous_sibling);

    if (child_index != UINT32_MAX) {
      // Child exists, follow it.
      node_index = child_index;
    } else {
      // Create new child.
      uint32_t new_child = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_daac_add_child(
          builder, node_index, byte, previous_sibling, &new_child));
      node_index = new_child;
    }
  }

  // Mark token at final node.
  builder->nodes[node_index].value.token_id = token_id;
  if (length > builder->max_depth) {
    builder->max_depth = length;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Builder - Double-Array Construction
//===----------------------------------------------------------------------===//

// Grows the double-array capacity.
static iree_status_t iree_tokenizer_daac_builder_grow_arrays(
    iree_tokenizer_daac_builder_t* builder, iree_host_size_t needed) {
  if (needed <= builder->array_capacity) {
    return iree_ok_status();
  }

  iree_host_size_t old_capacity = builder->array_capacity;
  iree_host_size_t new_capacity =
      iree_max(old_capacity * 2, iree_max(1024, needed));

  // Grow all three arrays together.
  int32_t* new_base = NULL;
  int32_t* new_check = NULL;
  int32_t* new_output = NULL;

  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      builder->allocator, new_capacity, sizeof(*new_base), (void**)&new_base));
  iree_status_t status = iree_allocator_malloc_array(
      builder->allocator, new_capacity, sizeof(*new_check), (void**)&new_check);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(builder->allocator, new_base);
    return status;
  }
  status =
      iree_allocator_malloc_array(builder->allocator, new_capacity,
                                  sizeof(*new_output), (void**)&new_output);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(builder->allocator, new_base);
    iree_allocator_free(builder->allocator, new_check);
    return status;
  }

  // Copy old data.
  if (old_capacity > 0) {
    memcpy(new_base, builder->base, old_capacity * sizeof(*new_base));
    memcpy(new_check, builder->check, old_capacity * sizeof(*new_check));
    memcpy(new_output, builder->output, old_capacity * sizeof(*new_output));
  }

  // Initialize new slots as free.
  for (iree_host_size_t i = old_capacity; i < new_capacity; ++i) {
    new_base[i] = 0;
    new_check[i] = -1;  // Free slot marker.
    new_output[i] = -1;
  }

  iree_allocator_free(builder->allocator, builder->base);
  iree_allocator_free(builder->allocator, builder->check);
  iree_allocator_free(builder->allocator, builder->output);

  builder->base = new_base;
  builder->check = new_check;
  builder->output = new_output;
  builder->array_capacity = new_capacity;

  return iree_ok_status();
}

// Finds a BASE value where all children fit without collision.
// Uses first-fit with a search hint for efficiency.
static iree_status_t iree_tokenizer_daac_builder_find_base(
    iree_tokenizer_daac_builder_t* builder,
    const iree_tokenizer_daac_build_node_t* node, int32_t* out_base) {
  if (node->first_child == UINT32_MAX) {
    *out_base = 0;
    return iree_ok_status();
  }

  const uint8_t first_byte = builder->nodes[node->first_child].edge_byte;
  uint32_t last_child = node->first_child;
  while (builder->nodes[last_child].next_sibling != UINT32_MAX) {
    last_child = builder->nodes[last_child].next_sibling;
  }
  const uint8_t last_byte = builder->nodes[last_child].edge_byte;

  // Start searching from next_check_pos, adjusted for first child byte.
  // base_candidate must be >= 1.
  iree_host_size_t start_pos = builder->next_check_pos;
  if (start_pos < first_byte + 1) {
    start_pos = first_byte + 1;
  }
  iree_host_size_t base_candidate = start_pos - first_byte;
  if (base_candidate == 0) {
    base_candidate = 1;
  }

  while (true) {
    // Ensure arrays are large enough for base_candidate + last_byte.
    iree_host_size_t max_slot = base_candidate + last_byte;
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_daac_builder_grow_arrays(builder, max_slot + 1));

    // Check if all children fit without collision.
    bool fits = true;
    uint32_t child = node->first_child;
    while (child != UINT32_MAX) {
      iree_host_size_t slot = base_candidate + builder->nodes[child].edge_byte;
      if (builder->check[slot] != -1) {
        fits = false;
        // Update next_check_pos hint: we know slots up to here might be full.
        if (slot >= builder->next_check_pos) {
          builder->next_check_pos = slot + 1;
        }
        break;
      }
      child = builder->nodes[child].next_sibling;
    }

    if (fits) {
      if (base_candidate > INT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "base value %" PRIhsz " exceeds int32_t limit",
                                base_candidate);
      }
      *out_base = (int32_t)base_candidate;
      return iree_ok_status();
    }

    ++base_candidate;

    // Safety check: don't loop forever.
    if (base_candidate > builder->array_capacity + builder->node_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "DAAC construction failed: excessive search");
    }
  }
}

// Converts temporary trie to double-array format using BFS.
static iree_status_t iree_tokenizer_daac_builder_to_array(
    iree_tokenizer_daac_builder_t* builder) {
  // Initialize with generous capacity.
  iree_host_size_t initial_capacity = 0;
  if (!iree_host_size_checked_mul(builder->node_count, 2, &initial_capacity) ||
      !iree_host_size_checked_add(initial_capacity, 512, &initial_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "initial array capacity overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_daac_builder_grow_arrays(builder, initial_capacity));

  builder->next_check_pos = 1;  // Start searching from slot 1.
  builder->array_size = 1;

  // Initialize root.
  builder->base[0] = 0;
  builder->check[0] = 0;  // Root's check is itself (marks as occupied).
  builder->output[0] = builder->nodes[0].value.token_id;
  builder->nodes[0].value.array_state = 0;

  iree_status_t status = iree_ok_status();

  // Breadth-first conversion reuses consumed sibling links as queue links.
  // Appended nodes retain their child lists until they themselves are visited.
  uint32_t queue_head = 0;
  uint32_t queue_tail = 0;
  while (queue_head != UINT32_MAX && iree_status_is_ok(status)) {
    uint32_t build_node = queue_head;
    const iree_tokenizer_daac_build_node_t* node = &builder->nodes[build_node];
    if (node->first_child == UINT32_MAX) {
      queue_head = node->next_sibling;
      continue;
    }

    // Find BASE value for this node.
    int32_t base_value = 0;
    status = iree_tokenizer_daac_builder_find_base(builder, node, &base_value);
    if (!iree_status_is_ok(status)) {
      break;
    }

    builder->base[node->value.array_state] = base_value;

    // Allocate children.
    uint32_t child_build_node = node->first_child;
    while (child_build_node != UINT32_MAX) {
      iree_tokenizer_daac_build_node_t* child =
          &builder->nodes[child_build_node];
      const uint32_t next_child = child->next_sibling;
      const uint8_t byte = child->edge_byte;
      iree_host_size_t child_slot = (iree_host_size_t)base_value + byte;

      // Mark as occupied.
      builder->check[child_slot] = node->value.array_state;
      builder->output[child_slot] = child->value.token_id;

      if (child_slot >= builder->array_size) {
        builder->array_size = child_slot + 1;
      }

      // Enqueue child for processing.
      if (child_slot > INT32_MAX) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "array state %" PRIhsz " exceeds int32_t limit", child_slot);
        break;
      }
      child->value.array_state = (int32_t)child_slot;
      child->next_sibling = UINT32_MAX;
      builder->nodes[queue_tail].next_sibling = child_build_node;
      queue_tail = child_build_node;
      child_build_node = next_child;
    }
    queue_head = builder->nodes[build_node].next_sibling;
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Public API - Build
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_trie_build(
    const iree_tokenizer_token_t* tokens, iree_host_size_t token_count,
    iree_const_byte_span_t string_table, iree_allocator_t allocator,
    iree_tokenizer_vocab_trie_t** out_trie) {
  IREE_ASSERT_ARGUMENT(out_trie);
  *out_trie = NULL;

  // Validate token count fits in int32_t (used for token IDs).
  if (token_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token count %" PRIhsz " exceeds int32_t limit %d",
                            token_count, INT32_MAX);
  }

  // Handle empty vocab - create a minimal DAAC with just the root state.
  if (token_count == 0) {
    iree_host_size_t total_size = 0;
    iree_host_size_t base_offset = 0;
    iree_host_size_t check_offset = 0;
    iree_host_size_t output_offset = 0;
    IREE_RETURN_IF_ERROR(
        IREE_STRUCT_LAYOUT(sizeof(iree_tokenizer_vocab_trie_t), &total_size,
                           IREE_STRUCT_FIELD(1, int32_t, &base_offset),
                           IREE_STRUCT_FIELD(1, int32_t, &check_offset),
                           IREE_STRUCT_FIELD(1, int32_t, &output_offset)));

    uint8_t* slab = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&slab));

    iree_tokenizer_vocab_trie_t* trie = (iree_tokenizer_vocab_trie_t*)slab;
    trie->allocator = allocator;
    trie->max_depth = 0;
    trie->node_count = 1;  // Just the root.
    trie->array_size = 1;
    trie->base = (int32_t*)(slab + base_offset);
    trie->check = (int32_t*)(slab + check_offset);
    trie->output = (int32_t*)(slab + output_offset);
    trie->base[0] = 0;
    trie->check[0] = 0;
    trie->output[0] = -1;  // No token at root.

    *out_trie = trie;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize builder (all pointers start NULL from memset).
  iree_tokenizer_daac_builder_t builder;
  memset(&builder, 0, sizeof(builder));
  builder.allocator = allocator;
  memset(builder.root_children, 0xFF, sizeof(builder.root_children));
  memset(builder.child_cache_keys, 0xFF, sizeof(builder.child_cache_keys));

  // Allocate root node.
  uint32_t root_index = 0;
  iree_status_t status =
      iree_tokenizer_daac_builder_allocate_node(&builder, &root_index);

  // Insert all non-UNUSED tokens into temporary trie.
  for (iree_host_size_t i = 0; i < token_count && iree_status_is_ok(status);
       ++i) {
    const iree_tokenizer_token_t* token = &tokens[i];
    // Skip UNUSED tokens (gap tokens, reserved slots).
    if (iree_any_bit_set(token->attributes, IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      continue;
    }
    // Validate string bounds before access.
    iree_host_size_t string_end = 0;
    if (!iree_host_size_checked_add(token->string_offset, token->string_length,
                                    &string_end) ||
        string_end > string_table.data_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "token %" PRIhsz " string bounds [%u, %" PRIhsz
                                ") exceed table size %" PRIhsz,
                                i, token->string_offset, string_end,
                                string_table.data_length);
      break;
    }
    const uint8_t* string = string_table.data + token->string_offset;
    status = iree_tokenizer_daac_builder_insert(
        &builder, string, token->string_length, (int32_t)i);
  }

  // Convert to double-array format.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_daac_builder_to_array(&builder);
    iree_allocator_free(builder.allocator, builder.nodes);
    builder.nodes = NULL;
  }

  // Allocate final slab and copy data.
  iree_tokenizer_vocab_trie_t* trie = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t total_size = 0;
    iree_host_size_t base_offset = 0;
    iree_host_size_t check_offset = 0;
    iree_host_size_t output_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_tokenizer_vocab_trie_t), &total_size,
        IREE_STRUCT_FIELD(builder.array_size, int32_t, &base_offset),
        IREE_STRUCT_FIELD(builder.array_size, int32_t, &check_offset),
        IREE_STRUCT_FIELD(builder.array_size, int32_t, &output_offset));

    if (iree_status_is_ok(status)) {
      uint8_t* slab = NULL;
      status = iree_allocator_malloc(allocator, total_size, (void**)&slab);
      if (iree_status_is_ok(status)) {
        trie = (iree_tokenizer_vocab_trie_t*)slab;
        trie->allocator = allocator;
        trie->max_depth = builder.max_depth;
        trie->node_count = builder.node_count;
        trie->array_size = builder.array_size;

        trie->base = (int32_t*)(slab + base_offset);
        trie->check = (int32_t*)(slab + check_offset);
        trie->output = (int32_t*)(slab + output_offset);

        memcpy(trie->base, builder.base, builder.array_size * sizeof(int32_t));
        memcpy(trie->check, builder.check,
               builder.array_size * sizeof(int32_t));
        memcpy(trie->output, builder.output,
               builder.array_size * sizeof(int32_t));
      }
    }
  }

  iree_tokenizer_daac_builder_deinitialize(&builder);

  if (iree_status_is_ok(status)) {
    *out_trie = trie;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Public API - Free
//===----------------------------------------------------------------------===//

void iree_tokenizer_vocab_trie_free(iree_tokenizer_vocab_trie_t* trie) {
  if (!trie) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  // Single slab allocation: all arrays are within the trie allocation.
  iree_allocator_free(trie->allocator, trie);
  IREE_TRACE_ZONE_END(z0);
}

// Cursor operations are inline in vocab_trie.h.
