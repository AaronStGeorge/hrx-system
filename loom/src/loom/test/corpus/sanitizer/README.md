# Sanitizer Execution Corpus

This directory contains target-neutral Loom programs that exercise sanitizer
contracts through production execution paths. The manifest is the exhaustive
source inventory for access, race, capacity, value, layout, and operation
reporting semantics.

Each fixture keeps its subject kernels, independent oracle functions, and
`check.scenario` trials together. Structured `check.expect.event<device>`
expectations identify the sanitizer mode and report fields that regressed
without matching textual runtime output.

Target packages own instrumentation modes, runtime feedback configuration,
capacity limits, compiler profiles, execution devices, and support
qualification. The semantic sources contain no target declarations or physical
architecture names. Target-independent IR formation, folding, and diagnostics
remain beside the sanitizer pass or dialect that owns each transform.
