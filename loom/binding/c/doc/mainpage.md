# Loom C API

The `loomc` library is the public C ABI for embedding the Loom compiler in
native language drivers, JITs, autotuners, packaging tools, and runtime
executable caches. The API is shaped around in-memory sources, reusable
compiler/linker/index handles, caller-owned workspaces, structured diagnostics,
and in-memory artifacts.

Core `loomc` headers expose C types defined by the Loom API. Optional
integration headers can adapt to host ecosystems such as IREE without making
those types part of the core ABI.

## Contract Summary

- `loomc_status_t` reports API and infrastructure failures that prevent an
  operation result from being produced.
- `loomc_result_t` and operation-specific result handles report compiler,
  linker, configuration, and source-program outcomes through state,
  diagnostics, reports, and artifacts.
- `loomc_source_t` is an immutable source handle with explicit borrowed, copied,
  or externally owned storage.
- `loomc_workspace_t` is mutable scratch for one worker at a time and is reset
  between operations or phases.
- Prepared pass programs, compiler, linker, and frozen link-index handles are
  intended to be immutable and reusable across many invocations.

The public headers are the primary API documentation. Contributor conventions
for those comments live in `loom/binding/c/doc/STYLE_GUIDE.md`.
