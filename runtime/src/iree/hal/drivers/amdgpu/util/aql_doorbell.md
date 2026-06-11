# The AQL queue doorbell (and why it has to be kind-aware)

This note explains the doorbell mechanism the amdgpu HAL uses to submit AQL
packets, the failure mode nightly ROCm on MI300X exposed, and the fix in
`aql_ring.h`. Every claim cites the code it comes from. Line numbers are as of
the `users/zjgarvey/fix/aql-doorbell-kind-aware` branch and may drift.

## What a doorbell is

An AQL (Architected Queuing Language) queue is a ring buffer of 64-byte packets
shared between the host (producer) and the GPU command processor, or "CP"
(consumer). Submitting work is three steps:

1. Reserve a slot by atomically bumping the queue's `write_dispatch_id`.
2. Write the packet into `ring.base[id & mask]` and publish its header.
3. **Ring the doorbell**: tell the CP "there are packets up to ID *N* ready."

Without step 3 the CP may never look at the new packet — the doorbell is the
wakeup. In this driver the ring and these three steps live in
`runtime/src/iree/hal/drivers/amdgpu/util/aql_ring.h`
(`iree_hal_amdgpu_aql_ring_reserve` / `_commit` / `_doorbell`).

## How the doorbell is represented: one signal, two kinds

Each hardware queue owns a doorbell **signal**, `hsa_queue.doorbell_signal`
(`runtime/src/iree/hal/drivers/amdgpu/abi/queue.h:61`). An
`iree_amd_signal_t` has a `kind` field and then a **union** whose meaning
depends on that kind
(`runtime/src/iree/hal/drivers/amdgpu/abi/signal.h:134-137`):

```c
iree_amd_signal_kind64_t kind;
union {
  volatile iree_hsa_signal_value_t value;       // kind == USER
  volatile uint64_t* hardware_doorbell_ptr;     // kind == DOORBELL
};
```

The kinds are defined in `signal.h:65-74`:

- `IREE_AMD_SIGNAL_KIND_DOORBELL = -1` — a firmware/hardware doorbell. The union
  slot is a **memory-mapped register pointer** (`hardware_doorbell_ptr`).
  Writing a packet ID straight to that address wakes the CP with zero runtime
  indirection. The header comment notes these are "reserved for hardware" and
  must only be *written* (never read/waited on) by anyone other than the origin.
- `IREE_AMD_SIGNAL_KIND_USER = 1` — an ordinary user signal (busy-wait or
  interrupt-backed). The same union slot is a plain signal **value**, not an
  address. You must go through the HSA signal-store API to update it; the
  hardware/runtime translates that into the actual CP wakeup.

The fast path the ring wants is the DOORBELL case: a single relaxed-release
atomic store to MMIO, no function-pointer hop through libhsa.

## The bug: assuming every doorbell is DOORBELL-kind

The original ring resolved the doorbell unconditionally as an MMIO pointer:

```c
// pre-fix aql_ring_initialize
out_ring->doorbell = (volatile int64_t*)doorbell_signal->hardware_doorbell_ptr;
```

It then rang it with a raw atomic store
(`aql_ring_doorbell`, `aql_ring.h:250`).

On nightly ROCm / MI300X the queue's `doorbell_signal` comes back as
**USER-kind**, not DOORBELL-kind. For a USER-kind signal the union slot holds a
signal *value*, so reading it as `hardware_doorbell_ptr` yields a small integer
reinterpreted as a pointer. The "raw atomic store to the doorbell" then writes
through that bogus address and **faults** — every H2D copy and every kernel
dispatch rings the doorbell, so the first submission after queue creation
SIGSEGVs. (This is the crash the paired hip-cts `aql_doorbell` test guards
against: 32 H2D/D2H roundtrips, each of which rings the doorbell.)

Why does ROCm hand back a USER-kind doorbell here? `signal.h` explains the two
host-visible signal implementations (busy-wait vs interrupt; `signal.h:97-118`):
interrupt-backed USER signals carry a platform event handle and are updated via
the HSA API rather than a bare MMIO poke. Some ROCm builds back the AQL queue's
doorbell with exactly that, instead of the classic hardware doorbell register.

## The fix: ring according to the signal kind

`aql_ring.h` now groups the two representations and chooses at init
(`aql_ring.h:81-92`, `:110-145`):

```c
struct {
  volatile int64_t* ptr;        // non-NULL only for DOORBELL-kind
  iree_hsa_signal_t signal;     // used when ptr == NULL
} doorbell;
```

`iree_hal_amdgpu_aql_ring_initialize` (now taking a `libhsa` handle, passed in
from `host_queue.c:619`) inspects the signal kind (`aql_ring.h:131-138`):

- `kind == IREE_AMD_SIGNAL_KIND_DOORBELL` → cache `hardware_doorbell_ptr` in
  `doorbell.ptr` (the fast path is preserved byte-for-byte).
- otherwise (USER/interrupt-kind, or no signal) → leave `doorbell.ptr == NULL`
  and keep the signal handle in `doorbell.signal`.

`iree_hal_amdgpu_aql_ring_doorbell` (`aql_ring.h:250-261`) then branches:

```c
if (ring->doorbell.ptr != NULL) {
  // Fast path: DOORBELL-kind — direct release store to the MMIO register.
  iree_atomic_store(ring->doorbell.ptr, packet_id, release);
} else {
  // Fallback: USER/interrupt-kind — ring through the HSA signal-store API.
  iree_hsa_signal_store_screlease(ring->libhsa, ring->doorbell.signal, packet_id);
}
```

`hsa_signal_store_screlease` (system-scope release) matches the ordering of the
fast path's release store, so the CP observes the packet writes before it sees
the doorbell update on either path.

## Why `libhsa` is now injected into the ring

The fallback path is the only place the ring needs libhsa
(`aql_ring.h:67-70` documents the field as unretained and required to outlive
the ring). `iree_hal_amdgpu_aql_ring_initialize` gained a `const
iree_hal_amdgpu_libhsa_t* libhsa` parameter; the production caller threads it
from the host queue (`host_queue.c:619`) and the live PM4 test was updated to
match (`util/pm4_dispatch_live_test.cc`).

## Cost

The fast path is unchanged: one `iree_atomic_store` to MMIO, no libhsa hop, when
the doorbell is DOORBELL-kind. The branch on `doorbell.ptr` is a single
predictable compare. Only USER/interrupt-kind doorbells (the case that
previously crashed) pay the libhsa indirection — and on those, a function call
is unavoidable because there is no writable MMIO register to poke.

## Code map

| Concern | Location |
| --- | --- |
| Signal kind enum (USER / DOORBELL) | `runtime/src/iree/hal/drivers/amdgpu/abi/signal.h:65-74` |
| Signal struct: `kind` + value/doorbell union | `runtime/src/iree/hal/drivers/amdgpu/abi/signal.h:134-137` |
| Queue's doorbell signal | `runtime/src/iree/hal/drivers/amdgpu/abi/queue.h:61` |
| Ring struct: grouped doorbell fields | `runtime/src/iree/hal/drivers/amdgpu/util/aql_ring.h:81-92` |
| Kind-aware init | `runtime/src/iree/hal/drivers/amdgpu/util/aql_ring.h:110-145` |
| Kind-aware ring (fast path + fallback) | `runtime/src/iree/hal/drivers/amdgpu/util/aql_ring.h:250-261` |
| libhsa threaded from host queue | `runtime/src/iree/hal/drivers/amdgpu/host_queue.c:619` |
