// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// Axis table
//===----------------------------------------------------------------------===//
//
// The axis table maps each registered axis to its current epoch, optional
// bridge semaphore, and permanent failure status. It is an implementation
// detail of the frontier tracker: the tracker owns the storage, publishes new
// entries via iree_async_frontier_tracker_register_axis, and queries it during
// advance/wait/fail/retire. Nothing outside this file should touch it directly.
//
// Lookup is an open-addressed hash table instead of an append-only vector:
//
//   ┌──────────────┐
//   │ axis hash    │
//   └──────┬───────┘
//          │
//          ▼
//   ┌──────┬──────┬──────┬──────┬──────┐
//   │ slot │ slot │ slot │ slot │ slot │  power-of-two slot table
//   └──────┴──────┴──────┴──────┴──────┘
//             │
//             ▼
//      {axis, epoch, semaphore, failure}
//
// The hot advance/query paths only hash and probe the axis table. They do not
// linearly scan all registered axes.

// An entry in the axis table mapping an axis to its current epoch and optional
// semaphore. The semaphore field enables bridging: when the axis advances, the
// corresponding semaphore is signaled, connecting the frontier system to the
// proactor's semaphore-based wait infrastructure.
//
// For directly-owned axes, the semaphore is the participant's local timeline
// semaphore signaled when that participant advances.
//
// For remotely-owned axes, the semaphore is a proxy signaled by the transport
// or protocol layer when it receives a frontier update from the remote owner.
typedef struct iree_alignas(64) iree_async_axis_table_entry_t {
  // Set to 1 with release ordering after the entry is fully initialized.
  // Readers acquire this flag before accessing the entry. Never cleared.
  iree_atomic_int32_t occupied;

  // The full 64-bit axis identifier.
  iree_async_axis_t axis;

  // Current epoch for this axis. Updated atomically (acquire/release) when
  // the axis advances. Reads on the hot path use acquire semantics.
  iree_atomic_int64_t current_epoch;

  // Optional semaphore backing this axis. If non-NULL, advancing the epoch
  // also signals this semaphore. This enables proactor wait operations to
  // trigger on axis advancement without polling.
  //
  // This is a borrowed bridge pointer. Participants must quiesce all
  // advancement on the axis and retire the axis before destroying a borrowed
  // semaphore or any owner state it references.
  iree_async_semaphore_t* semaphore;

  // Permanent failure status for this axis. OK means the axis is healthy or
  // retired after satisfying all relevant work. Non-OK statuses are owned by
  // the table entry and freed when the tracker is destroyed.
  iree_status_t failure_status;

  // Root of the intrusive pending-epoch index, guarded by waiters_mutex.
  iree_async_frontier_waiter_t* waiters_root;

  // Earliest indexed waiter, guarded by waiters_mutex.
  iree_async_frontier_waiter_t* waiters_minimum;

  // Earliest pending epoch, or UINT64_MAX when empty. Published after indexing
  // and before rechecking progress to close the opposing advance handshake.
  iree_atomic_int64_t next_wait_epoch;
} iree_async_axis_table_entry_t;

static_assert(sizeof(iree_async_axis_table_entry_t) == 64,
              "axis lookup and readiness guard fit one cache line");

// A fixed-capacity hash table mapping axes to their current state.
//
// Hot-path access pattern:
//   entry = lookup(table, axis);
//   if (entry) { CAS(entry->current_epoch); }
//
// Thread safety:
//   - Entry insertion is serialized by the tracker's waiters_mutex.
//   - Readers acquire each entry's occupied flag before accessing its payload.
//   - current_epoch is updated atomically during steady-state.
//   - The table itself (entries pointer, capacity, slot_count) is immutable
//     after creation. Published entries are never removed or reused.
typedef struct iree_async_axis_table_t {
  // Storage for all hash table slots. The table does not own this
  // pointer; the enclosing tracker holds the underlying FAM storage.
  iree_async_axis_table_entry_t* entries;

  // Number of registered axis entries, protected by the tracker's mutex.
  uint32_t count;

  // Maximum number of axes that may be registered.
  uint32_t capacity;

  // Number of hash table slots in |entries|. Always zero or a power of two.
  uint32_t slot_count;

  // Mask used to wrap hash probes. Meaningful only when |slot_count| is
  // non-zero.
  uint32_t slot_mask;
} iree_async_axis_table_t;

// Returns the number of hash slots required to support |capacity| registered
// axes while keeping expected probe lengths short.
static iree_status_t iree_async_axis_table_calculate_slot_count(
    uint32_t capacity, uint32_t* out_slot_count) {
  *out_slot_count = 0;
  if (capacity == 0) {
    return iree_ok_status();
  }
  if (capacity > (UINT32_MAX >> 2)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "axis table capacity %u is too large", capacity);
  }
  iree_host_size_t slot_count =
      iree_host_size_next_power_of_two((iree_host_size_t)capacity * 2);
  if (slot_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "axis table capacity %u is too large", capacity);
  }
  *out_slot_count = (uint32_t)slot_count;
  return iree_ok_status();
}

// Initializes an axis table with the given pre-allocated slot storage.
static inline void iree_async_axis_table_initialize(
    iree_async_axis_table_t* table, iree_async_axis_table_entry_t* entries,
    uint32_t capacity, uint32_t slot_count) {
  table->entries = entries;
  table->count = 0;
  table->capacity = capacity;
  table->slot_count = slot_count;
  table->slot_mask = slot_count == 0 ? 0 : slot_count - 1;
}

// Mixes an axis value for hash-table slot selection.
static inline uint32_t iree_async_axis_table_hash(iree_async_axis_t axis) {
  uint64_t value = axis;
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return (uint32_t)value;
}

// Looks up |axis| and returns the matching entry, or NULL if not registered.
static inline iree_async_axis_table_entry_t* iree_async_axis_table_lookup(
    const iree_async_axis_table_t* table, iree_async_axis_t axis) {
  if (table->slot_count == 0) {
    return NULL;
  }
  uint32_t slot = iree_async_axis_table_hash(axis) & table->slot_mask;
  for (uint32_t probe = 0; probe < table->slot_count; ++probe) {
    iree_async_axis_table_entry_t* entry = &table->entries[slot];
    if (!iree_atomic_load(&entry->occupied, iree_memory_order_acquire)) {
      return NULL;
    }
    if (entry->axis == axis) {
      return entry;
    }
    slot = (slot + 1) & table->slot_mask;
  }
  return NULL;
}

// Registers |axis| in the table under the tracker's waiters_mutex.
static iree_status_t iree_async_axis_table_register(
    iree_async_axis_table_t* table, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore) {
  if (iree_async_axis_table_lookup(table, axis) != NULL) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "frontier tracker axis 0x%016" PRIX64 " already registered", axis);
  }
  if (table->count >= table->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "frontier tracker axis table capacity %u "
                            "exhausted registering axis 0x%016" PRIX64,
                            table->capacity, axis);
  }

  uint32_t slot = iree_async_axis_table_hash(axis) & table->slot_mask;
  for (uint32_t probe = 0; probe < table->slot_count; ++probe) {
    iree_async_axis_table_entry_t* entry = &table->entries[slot];
    if (!iree_atomic_load(&entry->occupied, iree_memory_order_relaxed)) {
      entry->axis = axis;
      iree_atomic_store(&entry->current_epoch, 0, iree_memory_order_relaxed);
      entry->semaphore = semaphore;
      entry->failure_status = iree_ok_status();
      iree_atomic_store(&entry->next_wait_epoch, (int64_t)UINT64_MAX,
                        iree_memory_order_relaxed);
      ++table->count;
      iree_atomic_store(&entry->occupied, 1, iree_memory_order_release);
      return iree_ok_status();
    }
    slot = (slot + 1) & table->slot_mask;
  }

  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "frontier tracker axis table has no free slots");
}

//===----------------------------------------------------------------------===//
// Tracker storage
//===----------------------------------------------------------------------===//

struct iree_async_frontier_tracker_t {
  // Reference count controlling the tracker lifetime.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the tracker slab allocation.
  iree_allocator_t allocator;

  // Generation value embedded into locally constructed axes to avoid ABA across
  // execution contexts that reuse the same machine/domain/ordinal values.
  uint8_t session_epoch;

  // Local machine ordinal embedded into locally constructed axes.
  uint8_t machine_index;

  // Axis-to-progress table with fixed storage and independently published
  // entries.
  iree_async_axis_table_t axis_table;

  // Mutex protecting axis registration, the waiter list and per-axis failures.
  iree_slim_mutex_t waiters_mutex;

  // All pending waiters for terminal failure and destruction. Normal progress
  // uses the per-axis index. All accesses hold waiters_mutex until destruction.
  iree_async_frontier_waiter_t* waiters_head;

  // Flexible array member containing tracker-owned axis table slots.
  iree_async_axis_table_entry_t axis_table_entries[];
};

//===----------------------------------------------------------------------===//
// Pending epoch index
//===----------------------------------------------------------------------===//

static uint8_t iree_async_frontier_waiter_height(
    const iree_async_frontier_waiter_t* waiter) {
  return waiter ? waiter->index.height : 0;
}

static void iree_async_frontier_waiter_update_height(
    iree_async_frontier_waiter_t* waiter) {
  waiter->index.height =
      1 + iree_max(iree_async_frontier_waiter_height(waiter->index.left),
                   iree_async_frontier_waiter_height(waiter->index.right));
}

static void iree_async_frontier_waiter_replace(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* old_waiter,
    iree_async_frontier_waiter_t* new_waiter) {
  if (!old_waiter->index.parent) {
    entry->waiters_root = new_waiter;
  } else if (old_waiter == old_waiter->index.parent->index.left) {
    old_waiter->index.parent->index.left = new_waiter;
  } else {
    old_waiter->index.parent->index.right = new_waiter;
  }
  if (new_waiter) {
    new_waiter->index.parent = old_waiter->index.parent;
  }
}

static iree_async_frontier_waiter_t* iree_async_frontier_waiter_rotate_left(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* waiter) {
  iree_async_frontier_waiter_t* pivot = waiter->index.right;
  iree_async_frontier_waiter_replace(entry, waiter, pivot);
  waiter->index.right = pivot->index.left;
  if (waiter->index.right) {
    waiter->index.right->index.parent = waiter;
  }
  pivot->index.left = waiter;
  waiter->index.parent = pivot;
  iree_async_frontier_waiter_update_height(waiter);
  iree_async_frontier_waiter_update_height(pivot);
  return pivot;
}

static iree_async_frontier_waiter_t* iree_async_frontier_waiter_rotate_right(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* waiter) {
  iree_async_frontier_waiter_t* pivot = waiter->index.left;
  iree_async_frontier_waiter_replace(entry, waiter, pivot);
  waiter->index.left = pivot->index.right;
  if (waiter->index.left) {
    waiter->index.left->index.parent = waiter;
  }
  pivot->index.right = waiter;
  waiter->index.parent = pivot;
  iree_async_frontier_waiter_update_height(waiter);
  iree_async_frontier_waiter_update_height(pivot);
  return pivot;
}

static void iree_async_frontier_waiter_rebalance(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* waiter) {
  while (waiter) {
    iree_async_frontier_waiter_update_height(waiter);
    const int balance =
        (int)iree_async_frontier_waiter_height(waiter->index.left) -
        (int)iree_async_frontier_waiter_height(waiter->index.right);
    if (balance > 1) {
      if (iree_async_frontier_waiter_height(waiter->index.left->index.left) <
          iree_async_frontier_waiter_height(waiter->index.left->index.right)) {
        iree_async_frontier_waiter_rotate_left(entry, waiter->index.left);
      }
      waiter = iree_async_frontier_waiter_rotate_right(entry, waiter);
    } else if (balance < -1) {
      if (iree_async_frontier_waiter_height(waiter->index.right->index.right) <
          iree_async_frontier_waiter_height(waiter->index.right->index.left)) {
        iree_async_frontier_waiter_rotate_right(entry, waiter->index.right);
      }
      waiter = iree_async_frontier_waiter_rotate_left(entry, waiter);
    }
    waiter = waiter->index.parent;
  }
}

static iree_async_frontier_waiter_t* iree_async_frontier_waiter_successor(
    iree_async_frontier_waiter_t* waiter) {
  if (waiter->index.right) {
    waiter = waiter->index.right;
    while (waiter->index.left) {
      waiter = waiter->index.left;
    }
    return waiter;
  }
  while (waiter->index.parent && waiter == waiter->index.parent->index.right) {
    waiter = waiter->index.parent;
  }
  return waiter->index.parent;
}

static void iree_async_frontier_waiter_index_insert(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* waiter) {
  waiter->index.left = NULL;
  waiter->index.right = NULL;
  waiter->index.parent = NULL;
  waiter->index.peers = NULL;
  waiter->index.height = 1;
  if (!entry->waiters_root) {
    entry->waiters_root = waiter;
  } else {
    iree_async_frontier_waiter_t* parent = entry->waiters_root;
    for (;;) {
      if (waiter->index.epoch == parent->index.epoch) {
        waiter->index.height = UINT8_MAX;
        waiter->index.left = parent;
        waiter->index.right = parent->index.peers;
        if (waiter->index.right) {
          waiter->index.right->index.left = waiter;
        }
        parent->index.peers = waiter;
        break;
      }
      iree_async_frontier_waiter_t** child =
          waiter->index.epoch < parent->index.epoch ? &parent->index.left
                                                    : &parent->index.right;
      if (!*child) {
        *child = waiter;
        waiter->index.parent = parent;
        break;
      }
      parent = *child;
    }
    if (waiter->index.height != UINT8_MAX) {
      iree_async_frontier_waiter_rebalance(entry, parent);
    }
  }
  if (!entry->waiters_minimum ||
      waiter->index.epoch < entry->waiters_minimum->index.epoch) {
    entry->waiters_minimum = waiter;
  }
  iree_atomic_store(&entry->next_wait_epoch,
                    (int64_t)entry->waiters_minimum->index.epoch,
                    iree_memory_order_seq_cst);
}

static void iree_async_frontier_waiter_index_erase(
    iree_async_axis_table_entry_t* entry,
    iree_async_frontier_waiter_t* waiter) {
  // Equal epochs share one ordered entry. Cancelling any peer or promoting
  // one when its representative leaves requires no tree traversal/rebalance.
  if (waiter->index.height == UINT8_MAX) {
    iree_async_frontier_waiter_t* previous = waiter->index.left;
    if (previous->index.height == UINT8_MAX) {
      previous->index.right = waiter->index.right;
    } else {
      previous->index.peers = waiter->index.right;
    }
    if (waiter->index.right) {
      waiter->index.right->index.left = previous;
    }
    waiter->index.height = 0;
    return;
  }
  if (waiter->index.peers) {
    iree_async_frontier_waiter_t* peer = waiter->index.peers;
    peer->index.peers = peer->index.right;
    if (peer->index.peers) {
      peer->index.peers->index.left = peer;
    }
    peer->index.left = waiter->index.left;
    peer->index.right = waiter->index.right;
    peer->index.height = waiter->index.height;
    if (peer->index.left) {
      peer->index.left->index.parent = peer;
    }
    if (peer->index.right) {
      peer->index.right->index.parent = peer;
    }
    iree_async_frontier_waiter_replace(entry, waiter, peer);
    if (entry->waiters_minimum == waiter) {
      entry->waiters_minimum = peer;
    }
    waiter->index.height = 0;
    return;
  }
  if (entry->waiters_minimum == waiter) {
    entry->waiters_minimum = iree_async_frontier_waiter_successor(waiter);
  }
  iree_async_frontier_waiter_t* rebalance_from = waiter->index.parent;
  if (!waiter->index.left) {
    iree_async_frontier_waiter_replace(entry, waiter, waiter->index.right);
  } else if (!waiter->index.right) {
    iree_async_frontier_waiter_replace(entry, waiter, waiter->index.left);
  } else {
    iree_async_frontier_waiter_t* successor = waiter->index.right;
    while (successor->index.left) {
      successor = successor->index.left;
    }
    if (successor->index.parent != waiter) {
      rebalance_from = successor->index.parent;
      iree_async_frontier_waiter_replace(entry, successor,
                                         successor->index.right);
      successor->index.right = waiter->index.right;
      successor->index.right->index.parent = successor;
    } else {
      rebalance_from = successor;
    }
    iree_async_frontier_waiter_replace(entry, waiter, successor);
    successor->index.left = waiter->index.left;
    successor->index.left->index.parent = successor;
    iree_async_frontier_waiter_update_height(successor);
  }
  waiter->index.left = NULL;
  waiter->index.right = NULL;
  waiter->index.parent = NULL;
  waiter->index.height = 0;
  iree_async_frontier_waiter_rebalance(entry, rebalance_from);
  iree_atomic_store(&entry->next_wait_epoch,
                    entry->waiters_minimum
                        ? (int64_t)entry->waiters_minimum->index.epoch
                        : (int64_t)UINT64_MAX,
                    iree_memory_order_release);
}

// Removes a pending waiter from the terminal-path inventory. The caller owns
// any corresponding index removal and subsequent callback dispatch.
static void iree_async_frontier_tracker_unlink_waiter(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter) {
  if (waiter->previous) {
    waiter->previous->next = waiter->next;
  } else {
    tracker->waiters_head = waiter->next;
  }
  if (waiter->next) {
    waiter->next->previous = waiter->previous;
  }
  waiter->previous = NULL;
  waiter->next = NULL;
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the index of |axis| in |frontier|->entries, or -1 if not found.
static int32_t iree_async_frontier_find_axis(
    const iree_async_frontier_t* frontier, iree_async_axis_t axis) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].axis == axis) {
      return (int32_t)i;
    }
  }
  return -1;
}

// Public wait/query boundaries validate axis registration before consuming the
// immutable axis entries. Registered entries remain present after retirement.
static iree_status_t iree_async_frontier_tracker_validate_axes(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (iree_async_axis_table_lookup(&tracker->axis_table,
                                     frontier->entries[i].axis) == NULL) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "frontier entry %" PRIu8
                              " references unknown axis 0x%016" PRIX64,
                              i, frontier->entries[i].axis);
    }
  }
  return iree_ok_status();
}

// Result of checking a waiter's satisfaction state.
typedef enum iree_async_waiter_check_result_e {
  IREE_ASYNC_WAITER_CHECK_PENDING = 0,    // Not yet satisfied.
  IREE_ASYNC_WAITER_CHECK_SATISFIED = 1,  // All entries satisfied.
  IREE_ASYNC_WAITER_CHECK_FAILED = 2,     // At least one axis failed.
} iree_async_waiter_check_result_t;

// Checks whether a validated frontier is satisfied or failed.
// If FAILED, |out_failure_status| is set to the cloned failure status (caller
// owns). Must be called under the waiters_mutex.
static iree_async_waiter_check_result_t
iree_async_frontier_tracker_check_frontier(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier, iree_status_t* out_failure_status,
    uint8_t* out_pending_ordinal) {
  bool satisfied = true;
  if (out_pending_ordinal) {
    *out_pending_ordinal = UINT8_MAX;
  }
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    iree_async_axis_table_entry_t* entry = iree_async_axis_table_lookup(
        &tracker->axis_table, frontier->entries[i].axis);
    // Check for axis failure.
    iree_status_t failure = entry->failure_status;
    if (!iree_status_is_ok(failure)) {
      *out_failure_status = iree_status_clone(failure);
      return IREE_ASYNC_WAITER_CHECK_FAILED;
    }

    // Sequential consistency participates in wait()'s publication handshake
    // with lock-free advance(). The post-insertion recheck must observe any
    // advancement that missed the newly published waiter.
    int64_t current_epoch =
        iree_atomic_load(&entry->current_epoch, iree_memory_order_seq_cst);
    if ((uint64_t)current_epoch < frontier->entries[i].epoch) {
      // A later axis may already have failed. Pending work must not mask that
      // terminal result or leave a newly registered waiter stranded.
      if (satisfied && out_pending_ordinal) {
        *out_pending_ordinal = i;
      }
      satisfied = false;
    }
  }
  return satisfied ? IREE_ASYNC_WAITER_CHECK_SATISFIED
                   : IREE_ASYNC_WAITER_CHECK_PENDING;
}

// Indexes the next unresolved dependency, preserving the completed prefix.
// The caller holds waiters_mutex. Initial admission checks all axis failures;
// subsequent failure publication removes every referencing waiter under this
// same lock, including waiters currently indexed on a different axis.
static bool iree_async_frontier_tracker_index_next(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter, uint8_t start_ordinal) {
  for (uint8_t i = start_ordinal; i < waiter->frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t point = waiter->frontier->entries[i];
    iree_async_axis_table_entry_t* entry =
        iree_async_axis_table_lookup(&tracker->axis_table, point.axis);
    if ((uint64_t)iree_atomic_load(&entry->current_epoch,
                                   iree_memory_order_seq_cst) >= point.epoch) {
      continue;
    }
    waiter->index.axis_slot = (uint32_t)(entry - tracker->axis_table.entries);
    waiter->index.frontier_ordinal = i;
    waiter->index.epoch = point.epoch;
    iree_async_frontier_waiter_index_insert(entry, waiter);
    // Either advance observes the published minimum, or this opposing read
    // observes its epoch. Acquire/release on distinct atomics is insufficient.
    if ((uint64_t)iree_atomic_load(&entry->current_epoch,
                                   iree_memory_order_seq_cst) < point.epoch) {
      return true;
    }
    iree_async_frontier_waiter_index_erase(entry, waiter);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_frontier_tracker_create(
    iree_async_frontier_tracker_options_t options, iree_allocator_t allocator,
    iree_async_frontier_tracker_t** out_tracker) {
  IREE_ASSERT_ARGUMENT(out_tracker);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options.axis_table_capacity);
  *out_tracker = NULL;

  uint32_t axis_table_slot_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_axis_table_calculate_slot_count(
              options.axis_table_capacity, &axis_table_slot_count));

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(sizeof(iree_async_frontier_tracker_t), &total_size,
                         IREE_STRUCT_FIELD_FAM(axis_table_slot_count,
                                               iree_async_axis_table_entry_t)));

  iree_async_frontier_tracker_t* tracker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_aligned(allocator, total_size, 64, 0,
                                        (void**)&tracker));
  memset(tracker, 0, total_size);

  iree_atomic_ref_count_init(&tracker->ref_count);
  tracker->allocator = allocator;
  tracker->session_epoch = options.session_epoch;
  tracker->machine_index = options.machine_index;
  iree_async_axis_table_initialize(
      &tracker->axis_table, tracker->axis_table_entries,
      options.axis_table_capacity, axis_table_slot_count);
  iree_slim_mutex_initialize(&tracker->waiters_mutex);

  *out_tracker = tracker;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_frontier_tracker_destroy(
    iree_async_frontier_tracker_t* tracker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Final resource release has exclusive access. Detach before callbacks,
  // which may publish work on other trackers without taking this one's lock.
  iree_async_frontier_waiter_t* dispatch_head = tracker->waiters_head;
  tracker->waiters_head = NULL;
  for (iree_async_frontier_waiter_t* waiter = dispatch_head; waiter;
       waiter = waiter->next) {
    waiter->index.height = 0;
  }

  iree_async_frontier_waiter_t* waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, iree_make_status(IREE_STATUS_CANCELLED,
                                                         "tracker destroyed"));
    waiter = next;
  }

  // Deinitialize mutex.
  iree_slim_mutex_deinitialize(&tracker->waiters_mutex);

  // Free failure statuses.
  for (uint32_t i = 0; i < tracker->axis_table.slot_count; ++i) {
    iree_async_axis_table_entry_t* entry = &tracker->axis_table.entries[i];
    if (iree_atomic_load(&entry->occupied, iree_memory_order_relaxed) &&
        !iree_status_is_ok(entry->failure_status)) {
      iree_status_free(entry->failure_status);
    }
  }

  iree_allocator_t allocator = tracker->allocator;
  iree_allocator_free_aligned(allocator, tracker);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_frontier_tracker_retain(
    iree_async_frontier_tracker_t* tracker) {
  if (tracker) {
    iree_atomic_ref_count_inc(&tracker->ref_count);
  }
}

IREE_API_EXPORT void iree_async_frontier_tracker_release(
    iree_async_frontier_tracker_t* tracker) {
  if (tracker && iree_atomic_ref_count_dec(&tracker->ref_count) == 1) {
    iree_async_frontier_tracker_destroy(tracker);
  }
}

IREE_API_EXPORT uint8_t iree_async_frontier_tracker_session_epoch(
    const iree_async_frontier_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  return tracker->session_epoch;
}

IREE_API_EXPORT uint8_t iree_async_frontier_tracker_machine_index(
    const iree_async_frontier_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  return tracker->machine_index;
}

IREE_API_EXPORT iree_status_t iree_async_frontier_tracker_register_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(tracker);
  iree_slim_mutex_lock(&tracker->waiters_mutex);
  iree_status_t status =
      iree_async_axis_table_register(&tracker->axis_table, axis, semaphore);
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  return status;
}

IREE_API_EXPORT bool iree_async_frontier_tracker_query_epoch(
    const iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch) {
  IREE_ASSERT_ARGUMENT(tracker);
  iree_async_axis_table_entry_t* entry =
      iree_async_axis_table_lookup(&tracker->axis_table, axis);
  if (entry == NULL) {
    return false;
  }
  int64_t current_epoch =
      iree_atomic_load(&entry->current_epoch, iree_memory_order_acquire);
  return (uint64_t)current_epoch >= epoch;
}

IREE_API_EXPORT iree_status_t iree_async_frontier_tracker_query(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier, bool* out_satisfied) {
  IREE_RETURN_IF_ERROR(
      iree_async_frontier_tracker_validate_axes(tracker, frontier));
  iree_slim_mutex_lock(&tracker->waiters_mutex);
  iree_status_t status = iree_ok_status();
  iree_async_waiter_check_result_t result =
      iree_async_frontier_tracker_check_frontier(tracker, frontier, &status,
                                                 NULL);
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  if (iree_status_is_ok(status)) {
    *out_satisfied = result == IREE_ASYNC_WAITER_CHECK_SATISFIED;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Tracker operations
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_frontier_tracker_advance(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch) {
  // Phase 1: Find axis and update epoch without taking the waiter lock.
  iree_async_axis_table_entry_t* entry =
      iree_async_axis_table_lookup(&tracker->axis_table, axis);
  if (entry == NULL) {
    return 0;
  }

  // Publish monotonically advancing completion in the same total order as
  // waiter publication and the opposing observations in wait() and below.
  int64_t current_epoch;
  do {
    current_epoch =
        iree_atomic_load(&entry->current_epoch, iree_memory_order_acquire);
    if (epoch <= (uint64_t)current_epoch) {
      return 0;  // Monotonic: epoch not advancing.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &entry->current_epoch, &current_epoch, (int64_t)epoch,
      iree_memory_order_seq_cst, iree_memory_order_relaxed));

  // Phase 2: Signal semaphore if present.
  if (entry->semaphore != NULL) {
    // Each timeline value must be signaled exactly once. If the signal fails
    // (e.g. semaphore already past this epoch), it indicates a structural
    // error. Fail the semaphore so waiters get a proper diagnostic.
    iree_status_t signal_status =
        iree_async_semaphore_signal(entry->semaphore, epoch, NULL);
    if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
      iree_async_semaphore_fail(entry->semaphore, signal_status);
    }
  }

  // The minimum belongs to this axis: unrelated waits and later epochs do not
  // touch the metadata mutex or any waiter storage.
  if (epoch < (uint64_t)iree_atomic_load(&entry->next_wait_epoch,
                                         iree_memory_order_seq_cst)) {
    return 0;
  }

  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;
  iree_host_size_t dispatched_count = 0;
  iree_slim_mutex_lock(&tracker->waiters_mutex);
  while (entry->waiters_minimum &&
         entry->waiters_minimum->index.epoch <= epoch) {
    iree_async_frontier_waiter_t* waiter = entry->waiters_minimum;
    iree_async_frontier_waiter_t* peers = waiter->index.peers;
    waiter->index.peers = NULL;
    iree_async_frontier_waiter_index_erase(entry, waiter);
    // One tree removal covers this whole ready epoch. Each peer retains its
    // own next frontier dependency and terminal callback ownership.
    while (waiter) {
      iree_async_frontier_waiter_t* next = peers;
      peers = peers ? peers->index.right : NULL;
      waiter->index.height = 0;
      if (!iree_async_frontier_tracker_index_next(
              tracker, waiter, waiter->index.frontier_ordinal + 1)) {
        iree_async_frontier_tracker_unlink_waiter(tracker, waiter);
        waiter->dispatch_status = iree_ok_status();
        *dispatch_tail = waiter;
        dispatch_tail = &waiter->next;
        ++dispatched_count;
      }
      waiter = next;
    }
  }
  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  while (dispatch_head) {
    iree_async_frontier_waiter_t* waiter = dispatch_head;
    dispatch_head = waiter->next;
    waiter->callback(waiter->user_data, waiter->dispatch_status);
  }
  return dispatched_count;
}

iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter) {
  IREE_RETURN_IF_ERROR(
      iree_async_frontier_tracker_validate_axes(tracker, frontier));

  waiter->frontier = frontier;
  waiter->callback = callback;
  waiter->user_data = user_data;
  waiter->next = NULL;
  waiter->previous = NULL;
  memset(&waiter->index, 0, sizeof(waiter->index));

  iree_slim_mutex_lock(&tracker->waiters_mutex);
  iree_status_t failure_status = iree_ok_status();
  uint8_t pending_ordinal = UINT8_MAX;
  iree_async_waiter_check_result_t result =
      iree_async_frontier_tracker_check_frontier(
          tracker, frontier, &failure_status, &pending_ordinal);
  if (result != IREE_ASYNC_WAITER_CHECK_PENDING) {
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, failure_status);
    return iree_ok_status();
  }

  waiter->next = tracker->waiters_head;
  if (waiter->next) {
    waiter->next->previous = waiter;
  }
  tracker->waiters_head = waiter;
  const bool pending =
      iree_async_frontier_tracker_index_next(tracker, waiter, pending_ordinal);
  if (!pending) {
    iree_async_frontier_tracker_unlink_waiter(tracker, waiter);
  }
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  if (!pending) {
    callback(user_data, iree_ok_status());
  }
  return iree_ok_status();
}

bool iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter) {
  iree_slim_mutex_lock(&tracker->waiters_mutex);
  const bool pending = waiter->index.height != 0;
  if (pending) {
    iree_async_frontier_waiter_index_erase(
        &tracker->axis_table.entries[waiter->index.axis_slot], waiter);
    iree_async_frontier_tracker_unlink_waiter(tracker, waiter);
  }
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  return pending;
}

static void iree_async_frontier_tracker_fail_axis_impl(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status, bool retire) {
  // fail_axis/retire_axis are the error-path and clean-teardown entry points,
  // respectively. Both require a real status from the caller. Silently
  // synthesizing one hides caller bugs and produces misleading diagnostics
  // when an axis turns up "cancelled" without context.
  IREE_ASSERT(!iree_status_is_ok(status),
              "fail/retire require a non-OK status from the caller");

  // Find axis.
  iree_async_axis_table_entry_t* entry =
      iree_async_axis_table_lookup(&tracker->axis_table, axis);
  if (entry == NULL) {
    // Unknown axis; free status and return.
    iree_status_free(status);
    return;
  }

  // Take lock before checking/storing failure status.
  // This protects against:
  //   - Write-write race: two concurrent fail_axis() calls for the same axis
  //   - Read-write race: check_waiter() reads failure_status under lock
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  iree_async_semaphore_t* semaphore = entry->semaphore;
  if (retire) {
    entry->semaphore = NULL;
    semaphore = NULL;
  }

  // Check for existing failure (first-failure-wins).
  if (!iree_status_is_ok(entry->failure_status)) {
    // Already failed; ignore this failure.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    iree_status_free(status);
    return;
  }

  // Store failure status (take ownership).
  entry->failure_status = status;

  // Preserve the borrowed bridge while dispatching outside the lock. A
  // concurrent retirement or a failure callback may clear the bridge and
  // release its owner's reference.
  if (semaphore != NULL) {
    iree_async_semaphore_retain(semaphore);
  }

  // Failure is terminal even when another dependency is currently indexed.
  // This cold path visits the full pending inventory; normal progress never
  // does.
  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;
  iree_async_frontier_waiter_t* waiter = tracker->waiters_head;
  while (waiter) {
    iree_async_frontier_waiter_t* next = waiter->next;
    if (iree_async_frontier_find_axis(waiter->frontier, axis) >= 0) {
      iree_async_frontier_waiter_index_erase(
          &tracker->axis_table.entries[waiter->index.axis_slot], waiter);
      iree_async_frontier_tracker_unlink_waiter(tracker, waiter);
      *dispatch_tail = waiter;
      dispatch_tail = &waiter->next;
    }
    waiter = next;
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Semaphore failure invokes timepoint callbacks synchronously. Like frontier
  // callbacks, these may reenter the tracker, including to register a new axis.
  if (semaphore != NULL) {
    iree_async_semaphore_fail(semaphore, iree_status_clone(status));
    iree_async_semaphore_release(semaphore);
  }

  // Dispatch all collected waiters outside the lock.
  waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, iree_status_clone(status));
    waiter = next;
  }
}

void iree_async_frontier_tracker_fail_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status) {
  iree_async_frontier_tracker_fail_axis_impl(tracker, axis, status,
                                             /*retire=*/false);
}

IREE_API_EXPORT void iree_async_frontier_tracker_retire_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status) {
  iree_async_frontier_tracker_fail_axis_impl(tracker, axis, status,
                                             /*retire=*/true);
}
