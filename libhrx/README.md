# libhrx Runtime

`libhrx` is the HRX runtime compatibility layer built inside the top-level HRX
source tree. It exports the public `hrx_*` C ABI, the C++ RAII helpers, HIP-facing
streaming support, and optional passthrough/interception tools.

This directory is no longer a standalone build root in this repository. The
top-level `CMakeLists.txt` configures the reduced IREE runtime first, then adds
`libhrx/` as a subproject.

## Build

Configure from the repository root:

```bash
cmake -S . -B build/hrx \
  -DCMAKE_PREFIX_PATH=/srv/vm-shared/projects/pyre-workspace/rocm \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build/hrx --target hrx hrx-info
```

Useful options:

| Option | Default | Purpose |
|--------|---------|---------|
| `LIBHRX_BUILD` | ON | Build libhrx and compatibility targets. |
| `LIBHRX_BUILD_CTS` | `${IREE_BUILD_TESTS}` | Build Catch2 CTS binaries. |
| `LIBHRX_BUILD_PASSTHROUGH` | ON | Build HIP passthrough/interception libraries. |
| `LIBHRX_BUILD_CUDA_BINDING` | OFF | Build the CUDA compatibility binding. |
| `HRX_HERMETIC_BUILD` | OFF | Forbid FetchContent fallbacks when package discovery fails. |
| `IREE_HAL_DRIVER_AMDGPU` | ON | Build GPU support through IREE's AMDGPU HAL driver. |

The imported compiler API and compiler-produced VMFB test artifacts are
intentionally excluded from this reduced runtime bring-up. VM bytecode modules
can still be loaded through `hrx_module_load_vmfb` when callers provide their own
module bytes.

## Runtime

- CPU devices use IREE local-task/local-sync runtime drivers.
- GPU devices use the IREE AMDGPU runtime driver when it is enabled.
- `src/streaming/` and `src/binding/hip/` map HIP-facing streams, events,
  memory, modules, graph capture, and graph execution onto the public runtime.
- `src/passthrough/` provides optional HIP passthrough, logging, and buffer
  tracing shared libraries for comparison and debugging work.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HRX_GPU_DRIVER` | Selects the GPU HAL driver name. Currently expects `amdgpu`. |
| `HRX_GPU_DEBUG` | Prints IREE status details from GPU init/profiling when nonzero. |
| `HRX_PROFILE_FILE` | Enables GPU HAL profile capture to the given path. |
| `HRX_PROFILE_MODE` | Profile families: `queue`, `dispatch`, `executable`, or `all`. |
| `HRX_LIBRARY` | CTS dlopen path for `libhrx.so`; normally set by CTest. |
| `HIP_PASSTHROUGH_BACKEND_LIB` | Real `libamdhip64.so` used by HIP passthrough libraries. |
| `HIP_PASSTHROUGH_INTERCEPTOR` | Interceptor library for `src/passthrough/passthrough.c`. |
| `HIP_INTERCEPTION_LIBRARY` | Legacy interceptor variable used by `hip_intercept.c`. |
| `HIP_LOG_FILE` | HIP passthrough/logging output path. |
| `HIP_LOG_LEVEL` | Logging level: `0` off, `1` errors, `2` calls, `3` verbose. |
| `HIP_TRACE_FILE` | Buffer tracer output path. |
| `HIP_TRACE_LEVEL` | Trace level: `0` off, `1` errors, `2` calls, `3` buffers, `4` verbose. |
| `HIP_TRACE_SYNC` | Synchronize before/after kernel launches when set to `1`. |
| `HIP_TRACE_DUMP` | Buffer dump mode: `1` hex bytes, `2` FNV-1a hash. |
| `HIP_TRACE_DUMP_MAX` | Maximum bytes dumped per buffer. |
| `HIP_TRACE_KERNEL_FILTER` | Only trace kernels whose name contains this substring. |
| `HIP_TRACE_KERNEL_COUNT` | Limit traced kernel launches; `0` means unlimited. |
| `HIP_TRACE_KERNEL_FULL_DUMP` | Colon-separated kernel-name filters for unlimited buffer dumps. |

## Key Code Locations

| Path | Notes |
|------|-------|
| `include/hrx_runtime.h` | Public C runtime ABI. |
| `include/hrx_runtime_cxx.h` | C++ status/RAII helpers over retained HRX handles. |
| `src/libhrx/` | `libhrx.so` implementation. |
| `src/streaming/` | HIP-oriented streaming substrate. |
| `src/binding/hip/` | HIP API surface implemented on the streaming substrate. |
| `src/passthrough/` | HIP passthrough, logging, and buffer tracing libraries. |
| `tools/hrx_info.c` | CLI for enumeration, smoke tests, and profile checks. |
| `cts/` | Catch2 CTS and `find_package(hrx)` smoke consumer. |
