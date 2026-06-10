# IREE Clang-Tidy Checks

This directory contains the out-of-tree clang-tidy plugin for IREE C/C++
contract and style checks.

The plugin is intentionally separate from the product CMake build. Normal
runtime, Loom, and libhrx builds should not require LLVM development packages.
Build the plugin directly against the same LLVM installation that provides the
`clang-tidy` binary that will load it:

```bash
cmake -S build_tools/clang_tidy -B .tmp/iree-clang-tidy-plugin \
  -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build .tmp/iree-clang-tidy-plugin
ctest --test-dir .tmp/iree-clang-tidy-plugin --output-on-failure
```

Bazel exposes the matching LLVM install through the optional
`@iree_clang_tidy_llvm` repository. The repository is a stub unless explicitly
enabled, so normal Bazel commands do not require LLVM tools or development
headers:

```bash
bazel build --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  @iree_clang_tidy_llvm//:llvm_identity \
  @iree_clang_tidy_llvm//:clang-tidy
```

Discovery checks `IREE_CLANG_TIDY_LLVM_CONFIG`, `LLVM_CONFIG`,
`IREE_CLANG_TIDY_LLVM_ROOT`, `IREE_LLVM_ROOT`, `LLVM_ROOT`, and then `PATH`.
Use `IREE_CLANG_TIDY_BINARY` or `IREE_CLANG_TIDY_CLANGXX_BINARY` only when the
tools are not next to the discovered `llvm-config`.

With LLVM enabled, Bazel can build and test the plugin as a host tool:

```bash
bazel build --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  //build_tools/clang_tidy:IREEClangTidyPlugin.so
bazel test --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  //build_tools/clang_tidy:plugin_smoke_test
```

Those targets are tagged `manual` so normal wildcard builds stay independent of
the local LLVM installation.

The Bazel action runner uses the same configured C/C++ compile arguments that
feed `dev.py bazel compile-commands`, builds one cacheable action per source
file, and writes a per-source report:

```bash
bazel build --repo_env=IREE_CLANG_TIDY_LLVM=auto \
  //build_tools/clang_tidy:action_smoke
```

Build files can load `iree_clang_tidy` from `:clang_tidy.bzl` to define
additional clang-tidy gates over configured C/C++ targets.

The `iree-smoke` check only diagnoses the deliberately named test function
`iree_clang_tidy_smoke_bad`, and exists to prove that the plugin builds, loads,
registers checks, and emits diagnostics.

The first real contract check is `iree-status-discarded`. It diagnoses a call
returning `iree_status_t` when that call is used as a bare expression statement,
including `(void)` casts. Returning, assigning, checking in another expression,
or passing the status to the explicit consumer `iree_status_ignore` is accepted.
