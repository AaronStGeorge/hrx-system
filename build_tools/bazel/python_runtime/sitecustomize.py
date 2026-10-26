# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keeps the hermetic runtime immutable while Python actions execute."""

import sys

sys.dont_write_bytecode = True
