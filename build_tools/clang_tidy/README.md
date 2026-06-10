# IREE Clang-Tidy Checks

This directory contains the out-of-tree clang-tidy plugin for IREE C/C++
contract and style checks.

The plugin builds against the LLVM installation that provides the `clang-tidy`
binary that loads it. Normal runtime, Loom, and libhrx builds do not require
LLVM development packages.

## Bazel

```bash
python dev.py bazel clang-tidy //runtime/src/iree/base:all
python dev.py bazel clang-tidy --base origin/main
python dev.py bazel clang-tidy --all --profile ci
```

Bazel exposes the matching LLVM install through the optional
`@iree_clang_tidy_llvm` repository. The repository is a stub unless explicitly
enabled. `dev.py bazel clang-tidy` enables it with
`--repo_env=IREE_CLANG_TIDY_LLVM=auto`.

Discovery checks `IREE_CLANG_TIDY_LLVM_CONFIG`, `LLVM_CONFIG`,
`IREE_CLANG_TIDY_LLVM_ROOT`, `IREE_LLVM_ROOT`, `LLVM_ROOT`, and then `PATH`.
Use `IREE_CLANG_TIDY_BINARY` or `IREE_CLANG_TIDY_CLANGXX_BINARY` only when the
tools are not next to the discovered `llvm-config`.

The Bazel action runner uses the same configured C/C++ compile arguments that
feed `dev.py bazel compile-commands`, builds one cacheable action per source
file, and writes a per-source report:

```bash
iree-bazel-test --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  //build_tools/clang_tidy:status_checks_test \
  //build_tools/clang_tidy:trace_checks_test
```

## CMake

```bash
python dev.py cmake configure
python dev.py cmake clang-tidy runtime/src/iree/base/status.c
python dev.py cmake clang-tidy --base origin/main
```

The CMake command builds the plugin in `.tmp/iree-clang-tidy-plugin`,
materializes generated C/C++ compile inputs in the configured CMake build tree,
and runs `run-clang-tidy` against source files using that build tree's
`compile_commands.json`. Select the CMake build tree with `--cmake-build-dir`
or `IREE_CMAKE_BUILD_DIR`. The runner defaults to a capped parallel job count
and can be tuned with `IREE_CLANG_TIDY_JOBS`.

Plugin-only CMake validation is also available:

```bash
cmake -S build_tools/clang_tidy -B .tmp/iree-clang-tidy-plugin \
  -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build .tmp/iree-clang-tidy-plugin
ctest --test-dir .tmp/iree-clang-tidy-plugin --output-on-failure
```

## Checks

### `iree-status-discarded`

`iree-status-discarded` diagnoses calls returning `iree_status_t` when the call
is used as a bare expression statement, including `(void)` casts:

```c
some_status_returning_function();
(void)some_status_returning_function();
```

The status result must be returned, stored for later consumption, or explicitly
consumed. The check intentionally does not reason about whether a callee is
infallible; if the function returns `iree_status_t`, the caller owns that result.

### `iree-status-borrowed-parameter`

`iree-status-borrowed-parameter` diagnoses functions that take an
`iree_status_t` parameter by value but only observe it:

```c
static bool is_unavailable(iree_status_t status) {
  return iree_status_is_unavailable(status);
}
```

A by-value `iree_status_t` parameter means the callee participates in ownership.
The callee must return it, store it into an owning destination, consume it,
clone it before fanout, or transfer it to another owning status API. Observer
helpers should expose the non-owning representation they actually need:

```c
static bool is_unavailable(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_UNAVAILABLE;
}

static iree_status_t append_status(iree_string_builder_t* builder,
                                   const iree_status_t* status) {
  return iree_status_format_to(*status, append_chunk, builder)
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to format status");
}
```

The check intentionally keeps exceptions narrow. Status primitives that define
the observer API, known status sinks, C++ `iree::Status` formatting internals,
and documented borrowed callback boundaries are modeled explicitly. Ordinary
debug/reporting helpers should use `iree_status_code_t`, `const iree_status_t*`,
or `const iree_status_t&` instead of accepting a by-value status they do not
own.

### `iree-status-lifetime`

`iree-status-lifetime` treats local `iree_status_t` variables as owned linear
values. The check diagnoses mechanically provable local lifetime errors:

```c
iree_status_t status = do_work();
(void)status;  // Status leaves scope unconsumed.

iree_status_t status = do_work();
status = do_cleanup();  // Overwrites an owned status.

iree_status_t status = do_work();
iree_status_ignore(status);
return status;  // Uses a consumed status value.
```

The accepted terminal actions mirror `runtime/src/iree/base/status.h`: return
the status, store it into an owning destination, consume it with
`iree_status_free`/`iree_status_ignore`/`iree_status_consume_code`, or transfer
it through helpers such as `iree_status_join`, `iree_status_annotate`, and
`iree_status_freeze` while continuing to own the returned status.

The lifetime model focuses on local ownership states that can be proven from
the AST with high confidence. Straight-line code, block scope, local transfers,
returns, `if` branches, status helper calls, C++ status wrappers, and known
status-owning callback boundaries are checked directly. Loop-carried status
variables and functions with `goto` cleanup paths are treated conservatively so
diagnostics remain high signal.

### `iree-status-transfer-order`

`iree-status-transfer-order` diagnoses full expressions where the same local
`iree_status_t` appears more than once and at least one occurrence transfers
ownership:

```c
return iree_status_join(status, status);
return iree_status_join(status, notify_failure(status));
```

C does not sequence function argument evaluation. In the second example, either
argument can be evaluated first, so one path can consume or replace the status
while the other path still tries to use the old owned value. The reliable shape
is explicit sequencing through owned temporaries:

```c
iree_status_t notify_status = notify_failure(status);
return iree_status_join(status, notify_status);
```

When two independent consumers intentionally need the same failure payload, clone
first and give each owner one status value:

```c
iree_status_t cloned_status = iree_status_clone(status);
iree_status_t notify_status = notify_failure(cloned_status);
return iree_status_join(status, notify_status);
```

The check is local and syntactic by design. It models known status consumers,
status transfer helpers, C++ status wrapper constructors, and status observer
functions. Pure observer expressions such as multiple `iree_status_is_*` checks
do not transfer ownership and are accepted.

### `iree-trace-zone-balance`

`iree-trace-zone-balance` treats trace zones as scoped C resources. A terminal
path inside an active zone must end the zone before leaving the function. A
status-returning helper used inside an active zone must use the
trace-zone-aware form so the failure path ends the zone:

```c
IREE_TRACE_ZONE_BEGIN(z0);
IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, do_work());
IREE_TRACE_ZONE_END(z0);
return iree_ok_status();
```

Plain early-return helpers inside an active zone are diagnosed because their
failure path skips the end macro:

```c
IREE_TRACE_ZONE_BEGIN(z0);
IREE_RETURN_IF_ERROR(do_work());  // Use IREE_RETURN_AND_END_ZONE_IF_ERROR.
IREE_TRACE_ZONE_END(z0);
return iree_ok_status();
```

Plain source returns and known return macros such as `HIP_RETURN_ERROR` are
also diagnosed when they leave an active zone:

```c
IREE_TRACE_ZONE_BEGIN(z0);
return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad input");
```

Falling off the end of a function with an active zone is diagnosed for the same
reason: a zone is a scoped C resource and the function must close it on every
terminal path.

The reliable shapes are to use `IREE_RETURN_AND_END_ZONE` for immediate return
expressions, `IREE_RETURN_AND_END_ZONE_IF_ERROR` for conditional status helper
returns, or an explicit `IREE_TRACE_ZONE_END` immediately before returning a
status that is already carried in a local variable.

End macros using statically named zones are checked against the active zone
stack. Ending a zone after this function has already ended it, ending an outer
zone while an inner zone is active, or passing the wrong static zone ID to a
return-and-end helper is diagnosed. Dynamic zone ID carriers such as
`iree_zone_id_t zone_id` are treated conservatively because they often model an
explicit ownership transfer across helper boundaries.

The check uses the preprocessor's macro expansion stream so disabled tracing,
HRX wrapper macros, and multi-line helper invocations are modeled by the macro
the developer wrote rather than by the helper's implementation detail. Return
statements generated inside macro bodies are not diagnosed independently; known
return macros are checked at the macro expansion site. Ambiguous block-local,
loop, and switch zone balance is handled conservatively to keep this check high
signal.
