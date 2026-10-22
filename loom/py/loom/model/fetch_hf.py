# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional Hugging Face transport, revision ownership, and access coordination."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .fetch_source import FetchError
from .manifest import _origin_object, _url

_INSTALL = (
    "install the optional provider with: "
    "python -m pip install 'huggingface_hub>=2.2,<3'"
)


@dataclass
class AccessRequired(FetchError):
    # Actual repository supplying selected files, independent of training origin.
    repository: str
    # Provider-established action: gate, credentials, permissions, or rate_limit.
    reason: str
    # Repository browser page; never a signed file redirect or administrative API.
    url: str
    # Effective authenticated account, absent when unavailable or anonymous.
    account: str | None
    # Credential source name only; credential values never leave the provider.
    credential_source: str
    # Provider's minimum retry delay in seconds, when supplied.
    retry_after: float = 0

    def __post_init__(self):
        messages = {
            "gate": f"access approval is required at {self.url}",
            "credentials": "run hf auth login with an authorized account",
            "permissions": "the credential lacks required repository permissions",
            "rate_limit": "Hugging Face rate-limited the access request",
        }
        FetchError.__init__(self, f"{self.repository}: {messages[self.reason]}")

    def document(self) -> dict:
        return {
            "kind": "action_required",
            "provider": "hf",
            "repository": self.repository,
            "reason": self.reason,
            "url": self.url,
            "account": self.account,
            "credential_source": self.credential_source,
            "retry_after": self.retry_after,
            "message": str(self),
        }


AccessWait = Callable[[AccessRequired, float, threading.Event], None]


def hf_url_source(
    location: str, *, endpoint: str | None = None
) -> tuple[dict, str] | None:
    """Recognizes model file URLs without importing the optional SDK."""
    parsed = urlsplit(location)
    endpoints = (endpoint or os.getenv("HF_ENDPOINT", ""), "https://huggingface.co")
    for candidate in endpoints:
        if not candidate:
            continue
        candidate = candidate.rstrip("/")
        base = urlsplit(candidate)
        prefix = base.path + "/"
        if (parsed.scheme, parsed.netloc) != (
            base.scheme,
            base.netloc,
        ) or not parsed.path.startswith(prefix):
            continue
        match = re.fullmatch(
            r"([^/]+(?:/[^/]+)?)/(?:blob|resolve)/([^/]+)/(.+)",
            parsed.path[len(prefix) :],
        )
        if match is None:
            continue
        _url(location, "Hugging Face file URL")
        if set(parse_qs(parsed.query, keep_blank_values=True)) - {"download"}:
            raise FetchError("Hugging Face file URL has unsupported query parameters")
        repository, revision, path = map(unquote, match.groups())
        source = {
            "kind": "hf",
            "repository": repository,
            "revision": revision,
            "path": path,
        }
        _origin_object(source, "Hugging Face file URL", root=False)
        return source, candidate
    return None


class HuggingFace:
    """One operation's optional SDK client; no process-global credential mutation."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        cache_dir: Path | None = None,
        wait_for_access: AccessWait | None = None,
    ):
        try:
            import httpx2
            import huggingface_hub as hub
            import huggingface_hub.constants as constants
            import huggingface_hub.errors as errors
            from huggingface_hub.utils import validate_repo_id
        except ImportError as error:
            raise FetchError(
                "Hugging Face support is unavailable; " + _INSTALL
            ) from error
        version = re.match(r"^(\d+)\.(\d+)", hub.__version__)
        if not version or not (2, 2) <= tuple(map(int, version.groups())) < (3, 0):
            raise FetchError("unsupported huggingface_hub version; " + _INSTALL)
        # SDK module and errors are loaded only when this provider is constructed.
        self.hub = hub
        # The actual SDK exception family, including its gated/not-found hierarchy.
        self.errors = errors
        # The provider owns repository spelling validation at its public boundary.
        self.validate_repository = validate_repo_id
        # SDK environment-derived authentication and cache policy.
        self.constants = constants
        # Transport failures are reported without SDK request URLs or credentials.
        self.transport_error = httpx2.HTTPError
        # SDK-owned cache root; downloaded payloads stay here without a Loom copy.
        self.cache_dir = cache_dir
        # Endpoint and user agent follow the SDK's normal environment defaults.
        endpoint = (endpoint or constants.ENDPOINT).rstrip("/")
        _url(endpoint, "Hugging Face endpoint")
        self.api = hub.HfApi(endpoint=endpoint, library_name="loom-fetch")
        # Caller-supplied interaction policy; None fails immediately with remediation.
        self.wait_for_access = wait_for_access
        # Cancellation wakes approval waits shared by all selected repositories.
        self.cancelled = threading.Event()
        # One coordinator owns credential checks and human interaction across workers.
        self.access_lock = threading.RLock()
        # Positive checks apply only to the same effective credential in this operation.
        self.admitted: dict[str, str | None] = {}
        # A later successful check supersedes failures from an older shard attempt.
        self.access_epochs: dict[str, int] = {}
        # Account names are queried once per effective credential, never per shard/poll.
        self.accounts: dict[str, str] = {}
        # Floating revisions resolve once and survive access waits and retries.
        self.revisions: dict[tuple[str, str], str] = {}

    def _credential(self) -> str | None:
        if self.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN:
            return None
        return self.hub.get_token()

    def _repository(self, repository: str):
        try:
            self.validate_repository(repository)
        except self.errors.HFValidationError as error:
            raise FetchError("invalid Hugging Face repository identity") from error

    @property
    def cache_directory(self) -> Path:
        return Path(self.cache_dir or self.constants.HF_HUB_CACHE)

    def _credential_source(self) -> str:
        if self.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN:
            return "HF_HUB_DISABLE_IMPLICIT_TOKEN"
        if os.environ.get("HF_TOKEN"):
            return "HF_TOKEN"
        if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return "HUGGING_FACE_HUB_TOKEN"
        return "hf auth login"

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise FetchError("Hugging Face acquisition cancelled")

    def _action(self, repository: str, error, token: str | None) -> AccessRequired:
        status = error.response.status_code
        if isinstance(error, self.errors.RemoteEntryNotFoundError):
            raise FetchError(
                f"selected Hugging Face file does not exist in {repository}"
            ) from error
        if isinstance(error, self.errors.RevisionNotFoundError):
            raise FetchError(
                f"selected Hugging Face revision does not exist in {repository}"
            ) from error
        account = None
        if token and (isinstance(error, self.errors.GatedRepoError) or status == 401):
            if token not in self.accounts:
                try:
                    result = self.api.whoami(token=token)
                except self.errors.HfHubHTTPError as account_error:
                    if account_error.response.status_code != 401:
                        raise FetchError(
                            "Hugging Face account lookup failed"
                        ) from account_error
                    return AccessRequired(
                        repository,
                        "credentials",
                        self.api.endpoint + "/" + repository,
                        None,
                        self._credential_source(),
                    )
                except self.transport_error as account_error:
                    raise FetchError(
                        "Hugging Face account lookup failed"
                    ) from account_error
                name = result.get("name")
                if not isinstance(name, str) or not re.fullmatch(
                    r"[A-Za-z0-9_.-]{1,256}", name
                ):
                    raise FetchError("Hugging Face returned an invalid account name")
                self.accounts[token] = name
            account = self.accounts[token]
        if isinstance(error, self.errors.GatedRepoError):
            reason = "gate"
        elif status == 401:
            reason = "credentials"
        elif status == 403:
            reason = "permissions"
        elif status == 429:
            reason = "rate_limit"
        elif status == 404:
            raise FetchError(
                f"Hugging Face repository {repository} is missing or inaccessible"
            ) from error
        else:
            raise FetchError(
                f"Hugging Face returned HTTP {status} for {repository}"
            ) from error
        delay = error.response.headers.get("Retry-After", "")
        if delay:
            try:
                delay = (
                    float(int(delay))
                    if delay.isdecimal()
                    else max(0, parsedate_to_datetime(delay).timestamp() - time.time())
                )
            except (ValueError, TypeError, OverflowError) as delay_error:
                raise FetchError(
                    "Hugging Face returned invalid retry guidance"
                ) from delay_error
        return AccessRequired(
            repository,
            reason,
            self.api.endpoint + "/" + repository,
            account,
            self._credential_source(),
            delay or 0,
        )

    def _wait(self, action: AccessRequired, delay: float):
        self._check_cancelled()
        if self.wait_for_access is None:
            raise action
        self.wait_for_access(action, max(delay, action.retry_after), self.cancelled)
        self._check_cancelled()

    def ensure_access(self, repository: str, *, denied=None):
        """Checks read access once per credential; only the coordinator waits."""
        self._repository(repository)
        with self.access_lock:
            self._check_cancelled()
            token = self._credential()
            if (
                denied is None
                and repository in self.admitted
                and self.admitted[repository] == token
            ):
                return
            delay = 5.0
            if denied is not None:
                self.admitted.pop(repository, None)
                self._wait(self._action(repository, denied, token), delay)
            while True:
                self._check_cancelled()
                token = self._credential()
                try:
                    self.api.auth_check(repository, token=token or False)
                except self.errors.HfHubHTTPError as error:
                    self._wait(self._action(repository, error, token), delay)
                    delay = min(delay * 2, 60)
                    continue
                except self.transport_error as error:
                    raise FetchError(
                        f"Hugging Face access check failed for {repository}"
                    ) from error
                self.admitted[repository] = token
                self.access_epochs[repository] = (
                    self.access_epochs.get(repository, 0) + 1
                )
                return

    def _request(self, repository: str, operation):
        while True:
            self._check_cancelled()
            with self.access_lock:
                credential = self._credential()
                epoch = self.access_epochs.get(repository, 0)
            try:
                return operation(credential or False)
            except self.errors.HfHubHTTPError as error:
                with self.access_lock:
                    # Another shard may have completed the same approval wait.
                    # Its successful check admits this retry without another prompt.
                    if (
                        credential != self._credential()
                        or epoch != self.access_epochs.get(repository, 0)
                    ):
                        self.ensure_access(repository)
                    else:
                        self.ensure_access(repository, denied=error)
            except self.errors.LocalEntryNotFoundError as error:
                raise FetchError(
                    "Hugging Face could not retrieve the selected file for "
                    f"{repository}"
                ) from error
            except self.transport_error as error:
                raise FetchError(
                    f"Hugging Face transport failed for {repository}"
                ) from error

    def resolve_revision(self, requested: dict, *, offline: bool) -> str:
        repository, revision = (
            requested["repository"],
            requested.get("revision", "main"),
        )
        self._repository(repository)
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
        key = repository, revision
        with self.access_lock:
            if key in self.revisions:
                return self.revisions[key]
            if offline:
                try:
                    cache = self.hub.scan_cache_dir(self.cache_dir)
                except self.errors.CacheNotFound as error:
                    raise FetchError(
                        "--offline Hugging Face revision is not cached"
                    ) from error
                resolved = next(
                    (
                        repo.refs[revision].commit_hash
                        for repo in cache.repos
                        if repo.repo_type == "model"
                        and repo.repo_id == repository
                        and revision in repo.refs
                    ),
                    None,
                )
                if resolved is None:
                    raise FetchError(
                        "--offline has no recorded revision for "
                        f"{repository}@{revision}"
                    )
            else:
                self.ensure_access(repository)
                resolved = self._request(
                    repository,
                    lambda token: self.api.model_info(
                        repository, revision=revision, expand=["sha"], token=token
                    ),
                ).sha
            if not isinstance(resolved, str) or not re.fullmatch(
                r"[0-9a-f]{40}", resolved
            ):
                raise FetchError("Hugging Face returned an invalid immutable revision")
            self.revisions[key] = resolved
            return resolved

    def cached_file(self, source: dict) -> Path | None:
        self._repository(source["repository"])
        cached = self.hub.try_to_load_from_cache(
            source["repository"],
            source["path"],
            revision=source["revision"],
            cache_dir=self.cache_dir,
        )
        return Path(cached) if isinstance(cached, str) else None

    def metadata(self, source: dict):
        self.ensure_access(source["repository"])
        url = self.hub.hf_hub_url(
            source["repository"],
            source["path"],
            revision=source["revision"],
            endpoint=self.api.endpoint,
        )
        return self._request(
            source["repository"],
            lambda token: self.hub.get_hf_file_metadata(
                url, token=token, endpoint=self.api.endpoint, library_name="loom-fetch"
            ),
        )

    def download(self, source: dict, *, offline: bool) -> Path:
        cached = self.cached_file(source)
        if cached is not None:
            return cached
        if offline:
            raise FetchError(
                "--offline Hugging Face file is not cached: "
                f"{source['repository']}/{source['path']}"
            )
        self.ensure_access(source["repository"])
        return Path(
            self._request(
                source["repository"],
                lambda token: self.hub.hf_hub_download(
                    source["repository"],
                    source["path"],
                    revision=source["revision"],
                    cache_dir=self.cache_dir,
                    endpoint=self.api.endpoint,
                    token=token,
                    library_name="loom-fetch",
                ),
            )
        )

    def file_url(self, source: dict, *, endpoint: str | None = None) -> str:
        return (
            (endpoint or self.api.endpoint)
            + "/"
            + source["repository"]
            + "/resolve/"
            + quote(source["revision"], safe="")
            + "/"
            + quote(source["path"], safe="/")
        )
