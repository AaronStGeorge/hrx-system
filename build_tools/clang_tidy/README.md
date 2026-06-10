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
  //build_tools/clang_tidy:status_checks_test
```

## CMake

```bash
python dev.py cmake configure
python dev.py cmake clang-tidy runtime/src/iree/base/status.c
python dev.py cmake clang-tidy --base origin/main
```

The CMake command builds the plugin in `.tmp/iree-clang-tidy-plugin` and runs
`clang-tidy` against source files using the configured CMake
`compile_commands.json`. Select the CMake build tree with
`--cmake-build-dir` or `IREE_CMAKE_BUILD_DIR`.

Plugin-only CMake validation is also available:

```bash
cmake -S build_tools/clang_tidy -B .tmp/iree-clang-tidy-plugin \
  -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build .tmp/iree-clang-tidy-plugin
ctest --test-dir .tmp/iree-clang-tidy-plugin --output-on-failure
```

## Checks

`iree-status-discarded` diagnoses calls returning `iree_status_t` when the call
is used as a bare expression statement, including `(void)` casts:

```c
some_status_returning_function();
(void)some_status_returning_function();
```

The status result must be returned, stored for later consumption, or explicitly
consumed. The check intentionally does not reason about whether a callee is
infallible; if the function returns `iree_status_t`, the caller owns that result.
