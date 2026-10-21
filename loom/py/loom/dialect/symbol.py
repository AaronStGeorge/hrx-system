# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared module-symbol vocabulary."""

from loom.dsl import EnumCase, EnumDef

Visibility = EnumDef(
    "Visibility",
    [
        # Value 0 is reserved for "absent" (private) in optional enum attrs.
        EnumCase("public", 1, doc="Visible outside the module."),
    ],
    doc="Symbol visibility. Absent (0) means private (module-internal).",
)

Retain = EnumDef(
    "Retain",
    [
        # Value 0 is reserved for "absent" (ordinary DCE may erase).
        EnumCase("retain", 1, doc="Preserve the symbol across ordinary DCE."),
    ],
    doc="Private symbol retention policy. Absent (0) permits ordinary DCE.",
)

Linkage = EnumDef(
    "Linkage",
    [
        EnumCase(
            "linkonce",
            1,
            doc="Coalesce definitions with producer-promised equivalent semantics.",
        ),
    ],
    doc=("Definition linkage. Absent (0) is an ordinary strong definition. Linkers validate symbol contracts, not link-once bodies."),
    c_type="loom_symbol_linkage_t",
    c_const_prefix="LOOM_SYMBOL_LINKAGE",
    c_include="loom/ir/ir.h",
)
