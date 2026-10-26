# SDMA queue context

A KFD SDMA queue binds a command ring and its control words to a process
address space, an SDMA engine and a hardware queue slot. The process supplies
mapped ring storage; KFD builds the memory queue descriptor (MQD), allocates
the doorbell and installs the context through its selected scheduler. The
MQD and scheduler state remain driver-owned throughout queue execution.
[Native creation][create] [Scheduler creation][create-hws]

[Engine selection](engine-selection.md) describes which engine receives the
queue. [Publication](publication.md) describes how producers expose complete
commands once the queue exists. The context representation below belongs to
KFD user queues; Linux's [scheduled IB transport](command-buffers.md) has its
own ring and context-storage owner.

## Native inputs and retained mappings

`kfd_ioctl_create_queue_args` carries the ring base, ring byte length,
read-pointer address, write-pointer address, GPU node, queue type, percentage,
priority and optional engine ID. KFD returns a numeric queue ID and the
doorbell mmap offset. The input pointer addresses identify control **storage**;
they are not initial pointer values. [Create ABI][uapi] [Input translation][inputs]

| Input | Meaning and admission in the cited KFD implementation |
| --- | --- |
| `ring_base_address` | Process virtual byte address. MQD builders encode it in 256-byte units. |
| `ring_size` | A 32-bit byte count. Create/update require a power of two or zero, then clamp values below `KFD_MIN_QUEUE_RING_SIZE` to 1024 bytes. A zero size is therefore not preserved as zero by these entry points. |
| `read_pointer_address` | Process address of SDMA's read-pointer writeback word. KFD retains the containing native BO and its VM association. |
| `write_pointer_address` | Process address of the published write frontier. Queue loaders and firmware can consume it independently of the producer's private reservation state. |
| `queue_type`, `sdma_engine_id` | Ordinary SDMA, xGMI SDMA or an explicit engine request; the native scheduler resolves the engine and slot. |
| `queue_percentage`, `queue_priority` | Validated scheduling inputs. The low eight percentage bits become `queue_percent`; the SDMA MQD initializers below do not translate priority into a ring-control priority field. |

[ABI and minimum][uapi] [Create admission][inputs] [Update admission][update-ioctl]
[SDMA initializers][mqd9] [MES scheduling inputs][mes-input]

After clamping, the nonzero power-of-two input domain is 1 KiB through
2 GiB, before allocation and mapping checks. The active-queue predicate
also requires a nonzero ring address, positive percentage, and neither
evicted nor suspended state. Successful construction and active execution
are separate states. [Activity predicate][active]

For SDMA, `kfd_queue_acquire_buffers` acquires a page-rounded ring extent
and a `PAGE_SIZE` mapping for each pointer. `kfd_queue_buffer_get` compares
the address's GPU-page number with the mapping start and the expected page
extent with its end. It increments the BO reference and the VM association's
`queue_refcount`. Those page comparisons are not a separate check of the
low address bits required by a ring or pointer register. The compute-only
EOP/CWSR branch is not part of this SDMA acquisition path.
[Buffer admission][buffers]

The [native queue-storage contract](../architectures.md#kfd-queue-storage)
therefore differs from ordinary pageable payload allocation. MES also needs
the WPTR BO on the queue's AMDGPU device and maps that BO into GART so that
firmware can inspect work on an unmapped queue. Its address in the MES request
can differ from the process virtual address recorded in the MQD.
[MES storage owner][user-queue] [MES address construction][mes-input]

ROCr's thunk allocates a separate page-aligned, nonpaged, host-accessible,
uncached queue object containing `uint64_t` RPTR and WPTR words. Its public
`HSA_QUEUEID` is a pointer-valued handle to that object; the object contains
the distinct numeric KFD queue ID. ROCr's `SharedQueue` wrapper is yet another
object and is not the SDMA firmware descriptor. The allocated word width
does not override the narrower legacy readers described below.
[Thunk storage][thunk-storage] [Thunk creation][thunk-create]
[Runtime wrapper][runtime-wrapper]

## Descriptor family selection

KFD selects an ASIC queue-manager family using the ASIC identity and **GC IP**.
These predicates name the source's actual selector; they are not SDMA IP
versions or compiler-target comparisons. Each family supplies an SDMA MQD
manager alongside its compute managers. [Family selector][selector]

| Selection, in the source's order | SDMA MQD family |
| --- | --- |
| Kaveri, Hawaii | CIK, `cik_sdma_rlc_registers` |
| Carrizo, Tonga, Fiji, Polaris10/11/12, VegaM | VI, `vi_sdma_mqd` |
| Other ASIC, GC at least 12.1.0 | v12_1 manager, `v12_sdma_mqd` |
| Otherwise GC at least 12.0.0 | v12 manager, `v12_sdma_mqd` |
| Otherwise GC at least 11.0.0 | v11, `v11_sdma_mqd` |
| Otherwise GC at least 10.1.1 | v10, `v10_sdma_mqd` |
| Otherwise GC at least 9.0.1 | v9, `v9_sdma_mqd` |

The selector rejects the remaining case. Native device discovery and the
scheduler's callbacks still determine whether a particular transport can
install that descriptor. In particular, v12/v12_1 manager callback assignments
do not supply a manual SDMA loader: their `gfx_v12_kfd2kgd` and
`gfx_v12_1_kfd2kgd` tables omit the SDMA load/destroy/occupied callbacks.
Their ordinary MES route is described under [installation](#installation-and-write-pointer-consumers).
[GC12 interface][interface12] [GC12.1 interface][interface121]

### DWORD layout

Offsets below are zero-based DWORD offsets in the cited C MQD structures,
not MMIO register offsets. CIK uses the member prefix `sdma_rlc_`; the other
structures use `sdmax_rlcx_`. `sdma_engine_id` and `sdma_queue_id` are full
member names. The v10 structure has **130 DWORDs**; the other listed
structures have 128. v12_1 uses the v12 structure.
[CIK layout][layout-cik] [VI layout][layout-vi] [v9 layout][layout9]
[v10 layout][layout10] [v11 layout][layout11] [v12 layout][layout12]

| Member suffix | CIK / VI | v9 | v10 | v11 | v12 / v12_1 |
| --- | --- | --- | --- | --- | --- |
| `rb_cntl` | 0 | 0 | 0 | 0 | 0 |
| `rb_base`, `rb_base_hi` | 1, 2 | 1, 2 | 1, 2 | 1, 2 | 1, 2 |
| `rb_rptr`, `rb_rptr_hi` | 3, absent | 3, 4 | 3, 4 | 3, 4 | 3, 4 |
| `rb_wptr`, `rb_wptr_hi` | 4, absent | 5, 6 | 5, 6 | 5, 6 | 5, 6 |
| `rb_wptr_poll_cntl` | 5 | 7 | 7 | absent | absent |
| `rb_rptr_addr_lo`, `rb_rptr_addr_hi` | 9, 8 | 9, 8 | 9, 8 | 8, 7 | 7, 8 |
| `ib_cntl` | 10 | 10 | 10 | 9 | 9 |
| `doorbell` | 18 | 18 | 18 | 17 | 15 |
| `virtual_addr` | 19 | absent | absent | absent | absent |
| `doorbell_offset` | absent | 22 | 22 | 19 | 17 |
| `sched_cntl` | absent | absent | absent | 22 | 20 |
| `dummy_reg` | absent | 27 | 27 | 25 | 23 |
| `rb_wptr_poll_addr_lo`, `rb_wptr_poll_addr_hi` | 7, 6 | 29, 28 | 29, 28 | 27, 26 | 24, 25 |
| `sdma_engine_id`, `sdma_queue_id` | 126, 127 | 126, 127 | 128, 129 | 126, 127 | 126, 127 |

These are the control/address members relevant to ordinary initialization
and loading. The full structures also contain saved IB, execution, status
and reserved words. Initial creation clears the structure before filling the
selected fields. v11 clears a full page when MES is enabled, and only the
structure otherwise; v12 clears the structure; v12_1 clears a full page.
The allocation extent and the structure size are separate quantities.
[CIK initialization][init-cik] [VI initialization][mqd-vi]
[v9 initialization][mqd9] [v10 initialization][mqd10]
[v11 initialization][mqd11] [v12 initialization][mqd12]
[v12_1 initialization][mqd121]

An update writes the selected fields again without clearing the whole MQD,
so saved pointer/execution state can survive it. CIK/VI/v9/v10/v11 also
provide checkpoint/restore callbacks that copy the saved descriptor, patch
its new doorbell selection and leave the queue inactive; v12/v12_1 do not
install those callbacks. [CIK restore][restore-cik] [VI restore][restore-legacy]
[v9 restore][restore9] [v10 restore][restore10] [v11 restore][restore11]
[v12 manager][manager12] [v12_1 manager][manager121]

### Fields written during creation and update

All seven builders set ring size, VMID, RPTR writeback enable/timer, ring
base, RPTR writeback address, doorbell selection and engine/slot identity.
The following table gives their additional policy. On initial creation,
members not written by these builders retain their cleared values.
[CIK builder][mqd-cik] [VI builder][mqd-vi] [v9 builder][mqd9]
[v10 builder][mqd10] [v11 builder][mqd11] [v12 builder][mqd12]
[v12_1 builder][mqd121]

| Family | Ring exponent calculation | Additional initialized state |
| --- | --- | --- |
| CIK / VI | `order_base_2(queue_size / 4)` | `virtual_addr` from native `sdma_vm_addr`; doorbell offset is part of `doorbell`. |
| v9 | `order_base_2(queue_size / 4)` | Separate doorbell-offset word, `dummy_reg = 0xf`, `IB_CNTL.SWITCH_INSIDE_IB = 1`. |
| v10 | `ffs(queue_size / sizeof(unsigned int)) - 1` | Separate doorbell offset and `dummy_reg = 0xf`; no IB-switch bit added here. |
| v11 | Same `ffs - 1` expression | `F32_WPTR_POLL_ENABLE = 1`, WPTR poll address, phase quantum and `dummy_reg = 0xf`; no IB-switch bit added here. |
| v12 / v12_1 | Same `ffs - 1` expression | `MCU_WPTR_POLL_ENABLE = 1`, WPTR poll address, phase quantum, `dummy_reg = 0xf` and `SWITCH_INSIDE_IB = 1`. |

For an admitted power-of-two ring of `N` bytes, both expressions encode
`log2(N / 4)`. This differs from the compute queue's size-minus-one exponent.
For example, an 8 MiB ring has 2,097,152 DWORDs and exponent 21. The builders
shift that exponent directly; they do not mask it with `RB_SIZE_MASK`.
The native input domain and field width both matter.

Register spellings are `SDMA0_RLC0_*` in the legacy/v9/v10 manager headers,
`SDMA0_QUEUE0_*` for v11/v12, and `SDMA0_SDMA_QUEUE0_*` for v12_1. The
GC9.4.3 loader's SDMA4.4.2 header instead spells them `SDMA_RLC0_*`.

| Register field | Bits / representation | KFD builder value |
| --- | --- | --- |
| `SDMA0_RLC0_RB_CNTL.RB_ENABLE` and queue-name equivalents | Bit 0 | Initially zero; the installation owner enables the ring. |
| `RB_CNTL.RB_SIZE` | Bits 5:1 in OSS2/3, SDMA4.2.2/4.4.2 and GC10/11/12 headers; bits **6:1** in SDMA4.0 | The DWORD exponent above. v9's manager includes SDMA4.0, while some v9 loaders include the narrower revisions. |
| `RB_CNTL.RB_SWAP_ENABLE` | Bit 9 | Zero. |
| `RB_CNTL.RPTR_WRITEBACK_ENABLE` | Bit 12 | One. |
| `RB_CNTL.RPTR_WRITEBACK_SWAP_ENABLE` | Bit 13 | Zero. |
| `RB_CNTL.RPTR_WRITEBACK_TIMER` | Bits 20:16 | Encoding 6. The masks and builder do not define a wall-clock period. |
| `RB_CNTL.RB_PRIV` | Bit 23 | Zero. |
| `RB_CNTL.RB_VMID` | Bits 27:24 | Native queue VMID. |
| `RB_CNTL.WPTR_POLL_ENABLE` | Bit 8 in GC11/12 headers | Zero in the listed builders. This is distinct from the next field. |
| `RB_CNTL.F32_WPTR_POLL_ENABLE` / `MCU_WPTR_POLL_ENABLE` | Bit 11 in GC11 / GC12 headers | One in v11 / v12 and v12_1. |
| `RB_CNTL.WPTR_POLL_SWAP_ENABLE` | Bit 10 in GC11/12 headers | Zero. |
| `RB_BASE`, `RB_BASE_HI` | Low 32 and high 24 bits of `ring_address >> 8` | Ring base in 256-byte units. |
| `RB_RPTR_ADDR_LO.ADDR`, `RB_RPTR_ADDR_HI.ADDR` | Low-address bits 31:2, high 32 bits | Unshifted byte-address halves of the RPTR word. |
| `RB_WPTR_POLL_ADDR_LO.ADDR`, `RB_WPTR_POLL_ADDR_HI.ADDR` | Low-address bits 31:2, high 32 bits | Unshifted WPTR address halves in v11/v12/v12_1; zero in older builders. |
| Legacy `DOORBELL.OFFSET` | Bits 20:0 | Native DWORD doorbell offset, shift 0. |
| Modern `DOORBELL_OFFSET.OFFSET` | Bits 27:2 | Native DWORD doorbell offset shifted left by two. |
| `DOORBELL.ENABLE`, `DOORBELL.CAPTURED` | Bits 28, 30 | Initially zero. The manual loader sets enable. |
| `IB_CNTL.SWITCH_INSIDE_IB` | Bit 8 | Set in v9/v12/v12_1 as listed above. Other IB-control fields start zero. |
| `SCHEDULE_CNTL.CONTEXT_QUANTUM` | Bits 15:8 | v11/v12/v12_1 store `(amdgpu_sdma_phase_quantum << 8) & 0xff00`; other schedule fields start zero. |

[OSS2 fields][mask-oss2] [OSS2.4 fields][mask-oss24]
[OSS3 fields][mask-oss3] [SDMA4.0 fields][mask4]
[SDMA4.2.2 fields][mask422] [SDMA4.4.2 fields][mask442]
[GC10 fields][mask10] [GC10.3 fields][mask103]
[GC11 fields][mask11] [GC12 fields][mask12] [GC12.1 fields][mask121]

CIK's manager includes OSS2.4 definitions and its native loader includes
OSS2.0; the assigned fields agree in those headers. VI uses OSS3.0.
Legacy `RB_RPTR.OFFSET` and `RB_WPTR.OFFSET` occupy bits 31:2; modern
low/high pointer-register fields each span 32 bits. The low two bits remain
zero for DWORD-granular command positions. [Legacy fields][mask-oss3]
[Modern fields][mask11]

The address masks establish four-byte address granularity for the pointer
fields, not that an arbitrary four-byte allocation suffices for a modern
64-bit pointer or satisfies native BO admission. Likewise, the ring's
256-byte encoding is separate from the 4096-byte allocation alignment in
ROCr's `BlitSdma::Initialize`. [Runtime ring allocation][runtime-allocation]
CIK/VI's `VIRTUAL_ADDR.SHARED_BASE` occupies bits 10:8; their VM initializer
derives it from the process aperture and masks it into that field. It does
not set the header's ATC, PTR32 or VM-hole bits.
[CIK VM policy][vm-cik] [VI VM policy][vm-vi]

The native doorbell DWORD index is
`bo_byte_offset / 4 + doorbell_id * ceil(doorbell_size / 4)`. For SOC15
SDMA, KFD chooses the process-relative doorbell ID from the node/physical
engine routing table and queue slot; it does not derive it from the ring
address. Thus access width, process-relative ID, BAR DWORD index and encoded
byte offset are distinct quantities. [Doorbell selection][doorbell-owner]
[Index conversion][doorbell-index]

On GC 9.4.3, 9.4.4, and 9.5.0, shader-originated notifications additionally
require the assigned index's 32-byte routing block to belong to the intended
engine. KFD's corrected allocator aligns the engine base before assigning
queue slots. The [shader-routing description](device-publication.md#shader-routing-on-gc-943-944-and-950)
distinguishes this requirement from CPU notification and GPUVM accessibility.

Linux describes `sdma_phase_quantum` in units of 1K GPU clock cycles, with
default 32 and zero meaning no change. The MQD builder itself always stores
the masked field value, including zero. This SDMA context-switch phase
quantum is separate from MES's process/gang time quanta and from the RPTR
writeback timer. Neither setting measures command completion latency.
[Phase-quantum parameter][quantum] [MES quanta][mes-input]

## Installation and write-pointer consumers

The default KFD policy is HWS with oversubscription. The source also offers
HWS without oversubscription and a no-HWS policy that statically assigns
queues to hardware contexts. Hawaii and Tonga select no-HWS explicitly.
Within HWS creation, `enable_mes` selects MES; otherwise the CP scheduler
receives a run list. [Scheduling policy][sched-policy]
[Policy and ASIC selection][selector] [HWS creation][create-hws]

There is a concrete explicit-engine distinction at this boundary. HWS creation
passes `SDMA_BY_ENG_ID` through `allocate_sdma_queue`, normalizing it to SDMA
or SDMA_XGMI before choosing the MQD manager. The no-HWS creator chooses its
manager first and its SDMA allocation branch recognizes only SDMA and
SDMA_XGMI; the shared type-to-manager helper also recognizes only those two
types. The public enum and engine-range admission therefore do not establish
the same explicit-engine setup path under no-HWS.
[Type-to-manager helper][manager-type] [No-HWS creation][create-manual]
[HWS normalization][create-hws] [Engine admission][create]

### Manual register installation

The manual path calls the selected MQD manager's `load_mqd`. Its SDMA wrapper
passes the process WPTR address and memory context to the native
`hqd_sdma_load` callback. The examined loaders disable `RB_ENABLE`, wait for
`CONTEXT_STATUS.IDLE`, install the saved RPTR and current WPTR, program ring
base/writeback/doorbell state, and enable the ring. Their 2000-ms idle-wait
limit bounds this **installation wait**, not the runtime of each SDMA command.
[SDMA wrapper][mqd-wrapper] [CIK loader][load7] [VI loader][load8]
[GFX11 loader][load11]

| Native loader family | WPTR consumed from process memory | Additional distinction |
| --- | --- | --- |
| CIK / VI | One `uint32_t` | Single RPTR/WPTR registers and combined doorbell offset/control; programs `VIRTUAL_ADDR`. |
| GFX9, Arcturus/Aldebaran, GC9.4.3 | One `uint64_t` | Programs pointer low/high halves and separate doorbell offset. |
| GFX10 / GFX10.3 / GFX11 | One `uint64_t` | Programs pointer low/high halves and separate doorbell offset. |
| GC12 / GC12.1 interface tables | No manual SDMA callback in these tables | Follow the actual MES installation path rather than inferring a loader from the manager's generic wrapper. |

[GFX9 loader][load9] [Arcturus/Aldebaran loader][load-arcturus]
[GC9.4.3 loader][load943] [GFX10 loader][load10]
[GFX10.3 loader][load103] [GFX11 loader][load11]
[GC12 interface][interface12] [GC12.1 interface][interface121]

If the user-WPTR read fails, these loaders program the saved RPTR value as
WPTR. The modern loaders bracket the low/high WPTR update with
`MINOR_PTR_UPDATE = 1` then zero. This native restoration behavior is
independent of a producer ringing its doorbell. The GFX11 loader writes
`RB_CNTL` from its MQD but does not explicitly program the MQD's WPTR poll
addresses, schedule-control word or IB-control word. A populated MQD field
alone cannot specify every manual-loading path's register state.
[GFX11 complete loader][load11] [v11 MQD policy][mqd11]

### CP scheduler and MES

The CP-scheduled path constructs kernel-owned process/queue run-list entries
that point at each queue's MQD. Its map/unmap/query commands and completion
fence belong to the scheduler, not to the application's SDMA ring. The
separate [command-buffer chapter](command-buffers.md) explains SDMA IB
execution; an application IB does not perform this native context installation.
[Run-list owner][runlist] [Unmap and completion][unmap-hws]

Linux discovery enables MES for GC11.0.0–11.0.4, GC11.5.0–11.5.4,
GC11.5.6, GC11.7.0/11.7.1, GC12.0.0/12.0.1 and GC12.1.0 in the cited
switch. KFD's shared resources carry that selected `enable_mes` value.
The MES add request contains the MQD GPU address, process page-table base,
doorbell offset, queue type, process/gang context and scheduling parameters,
plus queue size in DWORDs. `convert_to_amdgpu_ring_type` accepts the
normalized ordinary SDMA type; it does not accept SDMA_XGMI in this path.
[MES discovery][mes-selection] [Shared resources][mes-resources]
[MES request construction][mes-input]

KFD supplies both the process WPTR virtual address and an MC address formed
from the WPTR BO's GPU offset plus the word's offset within its page. The
MES11 packet builder selects the MC address when the firmware API version is
at least 2 and the process address otherwise. MES12 and MES12.1 always use
the MC address. Their add operations submit the scheduler request and poll
its API completion. These writers establish which addresses firmware
receives; they do not supply a universal firmware polling interval or make
doorbell notification the sole way it can discover work.
[MES input addresses][mes-input] [MES11 packet][mes11]
[MES12 packet][mes12] [MES12.1 packet][mes121]

The publication consequence is the same across these distinctions: complete,
visible command bytes precede the canonical WPTR update. Private reservation
state stays separate. Delaying a doorbell cannot make an early public WPTR
store a portable reservation operation.
[Publication and consumers](publication.md#visibility-write-pointer-and-doorbell)

## Update, removal and final storage use

MQD storage has a native owner. Without MES, KFD allocates a shared GTT trunk
for the HIQ and all SDMA MQDs, selecting a descriptor offset by engine and
slot. Freeing an individual SDMA MQD wrapper does not free that trunk.
MES-selected managers instead allocate their queue MQDs through their
native allocation path. v11 with MES and v12/v12_1 use the GTT suballocator;
`AMDGPU_MQD_SIZE_ALIGN` adds 32 bytes of firmware fence space and rounds to
a GPU page. Structure size alone is not the required allocation extent.
These allocations are distinct from the process's
command ring, pointer words and payload buffers.
[Shared MQD allocation][mqd-allocation] [Trunk allocation][mqd-trunk]
[Trunk retirement][mqd-trunk-retire] [v11 MES selection][mes-allocation]
[v12 selection][manager12] [v12_1 selection][manager121]
[v11 allocation][allocate11] [v12 allocation][allocate12]
[v12_1 allocation][allocate121] [MQD allocation padding][mqd-padding]

A normal lifetime connects each separate owner:

1. The process maps the ring and pointer storage to the selected GPU and
   initializes control values. KFD acquires those BOs and VM associations,
   chooses an engine/slot, creates the MQD and installs the queue.
2. The producer retains every command extent until the read frontier permits
   ring reuse. Transfer destinations, sources and completion words retain
   their separate final users under the [completion protocol](fence.md).
3. Before replacing or removing the queue, its owner stops new publication
   and establishes the required completion of already published work.
   Removing a context is not itself an application payload-success signal.
4. Native update/removal unloads the direct context or communicates with the
   selected scheduler. The caller retains old ring/control mappings across
   that operation and checks its result before releasing them.
5. On successful destruction, KFD drops its retained BOs and queue state;
   the thunk frees its pointer-storage object only after a successful
   DESTROY_QUEUE ioctl. The runtime can then release its ring and wrapper.
   The thunk's per-node doorbell mapping is shared across queues and has a
   process-level teardown owner.

[Buffer acquisition][buffers] [Queue creation][create-hws]
[Queue update][update-native] [Queue destruction][destroy-native]
[Context update and reload][update-context]
[Thunk destruction][thunk-destroy] [Runtime destruction][runtime-destroy]
[Shared doorbell mapping][thunk-doorbell]

The native reference counts have precise ordering. `pqm_destroy_queue`
decrements the VM associations' queue counts before calling DQM removal,
and drops BO references afterward. When replacing a non-null ring,
`pqm_update_queue_properties` acquires the new ring and drops its reference
to the old ring before invoking DQM update. Thus the caller's retained
mapping/allocation is part of the lifetime; those kernel counts alone do
not establish that a concurrently freed old ring remains usable until
the operation finishes. [Update ownership][update-native]
[Destroy ownership][destroy-native] [Reference release][release-buffers]

The cited failure branches also differ from successful removal. PQM continues
its cleanup after DQM returns `-ETIME` or `-EIO`, while returning that error.
The manual SDMA destroy callbacks disable the ring, wait for idle, clear the
doorbell, re-enable the ring control and save RPTR; a timeout returns before
the successful sequence completes. HWS and MES have their own failure and
reset paths. These branches do not establish that every failed removal
retains every object, nor that it completed the application's transfers.
[PQM errors][destroy-native] [Direct removal][destroy11]
[HWS removal][destroy-hws] [MES removal][remove-mes]

[uapi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/linux/kfd_ioctl.h#L58-L117
[inputs]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L209-L336
[create]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L338-L447
[update-ioctl]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L467-L523
[buffers]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L196-L348
[release-buffers]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L350-L406
[user-queue]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_process_queue_manager.c#L246-L322
[update-native]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_process_queue_manager.c#L589-L646
[update-context]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L1070-L1177
[destroy-native]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_process_queue_manager.c#L518-L587
[selector]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L3215-L3336
[manager-type]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L81-L87
[create-manual]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L736-L860
[create-hws]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L2221-L2332
[layout-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/cik_structs.h#L158-L288
[layout-vi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/vi_structs.h#L27-L157
[layout9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/v9_structs.h#L27-L157
[layout10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/v10_structs.h#L542-L673
[layout11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/v11_structs.h#L542-L672
[layout12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/v12_structs.h#L542-L672
[init-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_cik.c#L142-L157
[mqd-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_cik.c#L223-L249
[mqd-vi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_vi.c#L352-L395
[mqd9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v9.c#L526-L574
[mqd10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v10.c#L356-L400
[mqd11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L426-L482
[mqd12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12.c#L305-L363
[mqd121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12_1.c#L380-L438
[mask-oss2]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/oss/oss_2_0_sh_mask.h#L1157-L1242
[mask-oss24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/oss/oss_2_4_sh_mask.h#L1277-L1368
[mask-oss3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/oss/oss_3_0_sh_mask.h#L2041-L2134
[mask4]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/sdma0/sdma0_4_0_sh_mask.h#L1480-L1618
[mask422]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/sdma0/sdma0_4_2_2_sh_mask.h#L1498-L1638
[mask442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/sdma/sdma_4_4_2_sh_mask.h#L1732-L1874
[mask10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/gc/gc_10_1_0_sh_mask.h#L1270-L1410
[mask103]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/gc/gc_10_3_0_sh_mask.h#L1299-L1439
[mask11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h#L977-L1113
[mask12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/gc/gc_12_0_0_sh_mask.h#L897-L1009
[mask121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/include/asic_reg/gc/gc_12_1_0_sh_mask.h#L985-L1099
[vm-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager_cik.c#L159-L170
[vm-vi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager_vi.c#L161-L172
[quantum]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c#L542-L547
[sched-policy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c#L735-L745
[mqd-wrapper]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager.c#L245-L271
[load7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v7.c#L239-L292
[load8]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v8.c#L263-L315
[load9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v9.c#L384-L451
[load-arcturus]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_arcturus.c#L123-L190
[load943]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gc_9_4_3.c#L59-L126
[load10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v10.c#L373-L440
[load103]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v10_3.c#L359-L426
[load11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v11.c#L345-L412
[destroy11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v11.c#L536-L573
[interface12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v12.c#L518-L533
[interface121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v12_1.c#L521-L536
[mes-selection]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_discovery.c#L2972-L3011
[mes-resources]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c#L172-L237
[mes-input]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L187-L282
[mes11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/mes_v11_0.c#L321-L379
[mes12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/mes_v12_0.c#L306-L361
[mes121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/mes_v12_1.c#L289-L348
[remove-mes]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L284-L329
[runlist]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_packet_manager.c#L99-L287
[unmap-hws]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L2650-L2726
[destroy-hws]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L2793-L2896
[mqd-allocation]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager.c#L65-L98
[mqd-trunk]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L3187-L3213
[mqd-trunk-retire]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L3354-L3361
[mes-allocation]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L502-L598
[allocate11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L101-L112
[allocate12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12.c#L82-L93
[allocate121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12_1.c#L140-L154
[thunk-storage]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L411-L514
[thunk-create]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L692-L851
[thunk-destroy]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L920-L942
[thunk-doorbell]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L206-L351
[runtime-wrapper]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L23-L51
[runtime-allocation]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L220-L222
[runtime-destroy]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L71-L84
[active]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_priv.h#L557-L561
[restore-legacy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_vi.c#L397-L431
[restore-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_cik.c#L286-L320
[restore9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v9.c#L576-L610
[restore10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v10.c#L402-L437
[restore11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L355-L390
[manager12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12.c#L383-L467
[manager121]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v12_1.c#L644-L728
[doorbell-owner]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L544-L616
[doorbell-index]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_doorbell_mgr.c#L125-L139
[mqd-padding]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_mes.h#L497-L503
