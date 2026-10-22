# Native dialect construction

Each dialect package exposes three dependency surfaces:

| Target | Consumer | Contents |
| --- | --- | --- |
| `:op_defs` | Inspect existing operations | Generated declarations, accessors, and their type/enum vocabulary |
| `:builders` | Construct operations and attributes | Declarations plus generated constructors and the dialect's required handwritten/type constructors |
| The package's combined library | Implement or register compiler behavior | Construction plus verification, inference, canonicalization, and semantic tables |

The same surfaces are available for common and target-owned dialects. A native
caller depends on `//loom/src/loom/ir:builders` for module construction and on
the dialect builders it uses, for example `//loom/src/loom/ops/scalar:builders`.
Common parameterized types, including channels and views, are provided by
`//loom/src/loom/ops:type_builders`; dialect-owned parameterized types are
included in that dialect's `:builders` surface.

Construction uses the caller's module and its context-owned metadata. These
libraries do not register dialects, define another descriptor owner, or supply
verification and optimization pipelines. The caller obtains its configured
module from the compiler it uses. This is an exact-build native C interface,
not an extension of the `loomc` exported ABI.

Low construction includes `loom/codegen/low/builder.h` alongside the generated
structural constructors. Descriptor-backed Low operations require resolved
descriptors; their handwritten constructors preserve that contract. The combined
Low semantic library keeps its original generated constructors but does not
bring descriptor-aware construction into read-only emitter dependencies.

Each dialect still has one batched generator action. Its existing `builders.c`
output is compiled once into the construction library and reused by the combined
library. The generated `ops.h`/`ops.inc` declaration surface is shared rather
than regenerated for each consumer. Semantic callback declarations remain in
that surface; depending on it does not link their implementations. Enum headers
likewise expose vocabulary without requiring target record tables.

`construction_dependency_boundary` checks the complete configured dialect
family together. It keeps compiler registries, semantic dialect libraries,
analysis, rewriting, and format implementations out of the construction closure;
the value-remapping primitive used by region builders remains permitted.
