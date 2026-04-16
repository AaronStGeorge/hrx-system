# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the shared cache-policy vocabulary."""

from loom.dialect.cache import CacheScope, CacheTemporal


class TestCacheVocabulary:
    def test_cache_temporal_values(self) -> None:
        assert [(case.keyword, case.value) for case in CacheTemporal.cases] == [
            ("regular", 0),
            ("non_temporal", 1),
            ("high_temporal", 2),
            ("last_use", 3),
            ("writeback", 4),
            ("non_temporal_regular", 5),
            ("regular_non_temporal", 6),
            ("non_temporal_high_temporal", 7),
            ("non_temporal_writeback", 8),
            ("bypass", 9),
        ]

    def test_cache_scope_values(self) -> None:
        assert [(case.keyword, case.value) for case in CacheScope.cases] == [
            ("cu", 0),
            ("se", 1),
            ("device", 2),
            ("system", 3),
        ]
