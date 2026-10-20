# AMD heterogeneous conformance

This suite composes CPU, GPU, and NPU execution and memory services. It owns tests
that join engines; `gpu/cts/` and `xdna/cts/` own engine-specific behavior, and
`fabric/cts/` owns shared native memory contracts independently of GPU HAL
execution.

| Target | Participants | Contract |
| --- | --- | --- |
| `cpu_xdna_test` | Task and XDNA HAL | Consumer-first CPU→NPU→CPU arithmetic through one backing, changing-input replay, default and explicit transitions, recorded direct and indirect bindings, deferred allocation, host registration, and public mapping permissions. |
| `virtual_buffer_test` | Task and XDNA HAL, native GPU SDMA | Stable addresses across backing replacement, independently resolved mapping groups, host aliases and device-only reservations, and native process/instance lifetimes. |

The virtual-buffer suite submits GPU work through libamdf. It isolates the
native memory integration contract and does not qualify RDNA HAL queue execution.
Tests that exercise complete heterogeneous HAL pipelines belong here alongside
it, using real HAL devices for every participant.

Each target declares its required engines and hardware separately. CPU/XDNA
tests need an NPU; virtual-buffer tests need both a GPU and an NPU. Ordinary
runtime capability checks identify unsupported native profiles.

```sh
iree-bazel-test --config=asan \
  --//runtime/config/hal:drivers=task,amd \
  //runtime/src/iree/hal/drivers/amd/cts:cpu_xdna_test \
  //runtime/src/iree/hal/drivers/amd/cts:virtual_buffer_test
```
