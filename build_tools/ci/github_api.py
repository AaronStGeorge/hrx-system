# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small authenticated client for GitHub's JSON and download APIs."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

API_VERSION = "2022-11-28"


def _url_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.casefold()
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    hostname = parsed.hostname.casefold() if parsed.hostname else None
    return scheme, hostname, port


class GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keeps GitHub API credentials within the request's origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        redirected_request = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected_request is not None and _url_origin(
            request.full_url
        ) != _url_origin(new_url):
            redirected_request.remove_header("Authorization")
        return redirected_request


def _request(url: str, token: str, accept: str) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def get_bytes(
    url: str,
    token: str,
    *,
    accept: str = "application/vnd.github+json",
    timeout_seconds: int = 60,
) -> bytes:
    opener = urllib.request.build_opener(GitHubRedirectHandler())
    with opener.open(_request(url, token, accept), timeout=timeout_seconds) as response:
        return response.read()


def get_json(
    url: str,
    token: str,
    *,
    timeout_seconds: int = 60,
) -> Any:
    return json.loads(get_bytes(url, token, timeout_seconds=timeout_seconds))
