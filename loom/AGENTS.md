# Loom Agent Guide

This guide applies to work under `loom/`. More specific subtree guides can add
target rules, but they inherit the ownership split here.

## Responsibility Boundaries

Python owns source-of-truth data that already lives naturally in Python:
dialect declarations, descriptor schemas, target facts, vendor machine-readable
ISA data, generated contract rows, spelling tables, and build-time validation.
Invariants known when the compiler is generated belong in Python and fail the
build or generator test there.

C owns the shipping compiler surface: structs, enums, public and private
headers, semantic comments, function declarations, hot-path interpreters,
lowering mechanics, value-fact queries, diagnostics, and policy that depends on
user IR. Checked-in C headers are design documents and should be the place a
human or agent finds the representation contract by search.

Generated files are not design documents. A generator that emits a public struct
or enum for a backend table is usually hiding the contract in the wrong place.
Move the type to a checked-in header and have Python emit rows that initialize
that type.

## Generated Tables

Generated backend tables should be data. The normal shape is a checked-in
header and checked-in C interpreter or lookup helper consuming generated rows.
Large target tables usually fit as `.inl` fragments included by checked-in `.c`
files, or as generated `.c` files whose checked-in header owns all types and
declarations. The table-generator contract excludes bespoke control flow,
callbacks, switch trees, lookup functions, and new semantic APIs. Schema-level
surfaces such as dialect op bindings are the exception: there the generated
artifact is itself the canonical API.

Generators earn their place when they cover data sets large enough or volatile
enough that hand maintenance is worse: vendor ISA records, many descriptor rows,
many source-to-low rule rows, cross-product feature matrices, or tables whose
invalid combinations can be exhaustively rejected during generation. The
payoff is meaningful handwritten duplication deleted, stronger build-time
validation, or hot-path runtime work removed.

A tiny stable table belongs in C. A singleton generated row, an unconditional
extern selected by address, or a few enum constants wrapped in hundreds of lines
of Python and build/test wiring is code bloat unless it lands as part of the
same coherent change that expands to a real family and removes enough C
complexity to make the trade obvious.

## Runtime Cost

Loom's JIT goal makes target lowering hot-path code. Generated metadata should
let C do less work at runtime, not add indirection around constants that were
already known. Descriptor consistency, row count limits, opcode-family
compatibility, feature coverage, and impossible operand forms are generation
time checks. Runtime code evaluates user-dependent facts, resolves descriptor
availability, emits diagnostics, and walks compact trusted data.

## Cleaning Existing Code

Existing generators are in scope for cleanup when they violate these boundaries.
Finding a generated header that owns backend table structs, generated lookup
code that could be a checked-in generic helper over rows, or a generator whose
only output is a tiny stable table is a design finding, not precedent. Repair
the ownership split before building new work on top of it.

## Review Questions

Before adding or extending a generator, answer:

- What checked-in C contract consumes the generated data?
- Which handwritten duplication or runtime work disappears?
- Which invalid combinations become build-time failures?
- How many rows or families are expected now, and what concrete pressure makes
  that grow?
- Would this be clearer and faster as a normal C constant table?

If those answers are weak, the durable change is probably a small checked-in C
table, a better fact query, a structured diagnostic, or no change at all.
