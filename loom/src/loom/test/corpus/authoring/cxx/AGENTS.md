# C++ Authoring Corpus

This package contains a small, curated set of complete Loom C++ programs. Each
program must demonstrate a distinct authoring workflow that is useful to a
kernel or library author: a substantial algorithm, target/config
specialization, or composition across source modules. Parser features,
diagnostics, lowering cells, and small language semantics belong in the C++
importer or target unit tests instead.

Keep the source inventory explicit in `BUILD.bazel`. Adding a file is an
admission decision, not automatic test discovery. A new example must document
the authoring concept it owns in `README.md`, carry an independent correctness
oracle, and avoid duplicating an existing example for another target or shape.
Target packages own execution profiles, architecture matrices, variants, and
expected failures; this package publishes reusable test modules without
selecting a device.

Every module target is incompatible unless the C++ importer is enabled. The
programs remain semantically target-neutral: generic target queries such as
subgroup size are valid authoring contracts, while concrete target symbols and
provider bindings belong to target-owned tests.

These sources are examples and correctness programs, not benchmark inputs.
Compiler microbenchmarks own private source snapshots so edits to examples,
checks, or documentation cannot silently change the bytes being timed.
