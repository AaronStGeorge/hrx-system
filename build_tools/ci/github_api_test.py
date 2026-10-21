# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest
import urllib.request

from build_tools.ci import github_api


class GitHubAPITest(unittest.TestCase):
    def test_authorization_does_not_cross_redirect_origins(self):
        handler = github_api.GitHubRedirectHandler()
        request = urllib.request.Request(
            "https://api.github.com/repos/ROCm/hrx-system/actions/jobs/1/logs",
            headers={"Authorization": "Bearer secret"},
        )

        cross_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://results.example.com/job.log",
        )
        same_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://API.GITHUB.COM:443/job.log",
        )

        self.assertIsNotNone(cross_origin_request)
        self.assertFalse(cross_origin_request.has_header("Authorization"))
        self.assertIsNotNone(same_origin_request)
        self.assertTrue(same_origin_request.has_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
