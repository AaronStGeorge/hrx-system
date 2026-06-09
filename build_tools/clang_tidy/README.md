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

The first checked-in check is `iree-smoke`. It only diagnoses the deliberately
named test function `iree_clang_tidy_smoke_bad`, and exists to prove that the
plugin builds, loads, registers checks, and emits diagnostics before real IREE
contracts are added.
