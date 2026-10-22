# C++ Compiler Benchmark Inputs

This directory owns the controlled source inputs consumed by the public C API
compiler microbenchmarks. The files contain kernels, configuration contracts,
and required providers only. Correctness scenarios, generators, numerical
oracles, and target execution policy live elsewhere and are deliberately
absent from the timed translation units.

These files are benchmark inputs rather than a shared kernel library. Update
one only when the benchmark is intentionally adopting a new compiler workload,
and review the resulting source-size and timing change as part of that update.
Authoring examples and importer tests must not depend on this directory.

The snapshots are licensed under Apache-2.0 with LLVM Exceptions.
