"""GitHub issue scraper.

Iterates framework repos and yields `RawIssue` objects. Supports incremental
scrapes via `since` (GitHub's `updated_at` filter). Authenticated requests
are strongly preferred — set `GITHUB_TOKEN` in the environment to raise the
rate limit from 60/hr to 5000/hr.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentfail.frameworks import FRAMEWORKS
from agentfail.schema import Framework, RawIssue

log = structlog.get_logger(__name__)

GITHUB_API = "https://api.github.com"
PAGE_SIZE = 100  # GitHub max

# Keyword heuristic used to avoid classifying obvious feature requests.
# Intentionally permissive — classifier is authoritative, this just trims
# API cost on the long tail of non-failure issues.
_FAILURE_HINT_KEYWORDS = (
    "bug",
    "error",
    "fail",
    "crash",
    "broken",
    "exception",
    "traceback",
    "not working",
    "doesn't work",
    "hang",
    "freeze",
    "stuck",
    "loop",
    "infinite",
    "regression",
    "typeerror",
    "valueerror",
    "keyerror",
    "attributeerror",
    "runtimeerror",
)

_FAILURE_LABEL_FRAGMENTS = (
    "bug",
    "error",
    "crash",
    "broken",
    "regression",
    "defect",
)

_NON_FAILURE_LABEL_FRAGMENTS = (
    "enhancement",
    "feature",
    "feat:",
    "docs",
    "documentation",
    "question",
    "discussion",
    "rfc",
    "proposal",
)


class GitHubClient:
    """Thin authenticated wrapper around the GitHub REST API."""

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agentfail-scraper/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        resp = self._client.get(url, params=params)

        # Handle primary rate limit: wait until reset + 1s, then retry via tenacity
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait_s = max(0, reset - int(time.time())) + 1
            log.warning("github.rate_limited", wait_seconds=wait_s)
            time.sleep(wait_s)
            raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)

        resp.raise_for_status()
        return resp

    def list_issues(
        self,
        repo: str,
        *,
        state: Literal["open", "closed", "all"] = "all",
        since: datetime | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Paginate through issues in a repo. Yields raw dicts from GitHub.

        Note: GitHub's "issues" endpoint returns both issues and pull requests;
        we filter PRs out in `scrape_framework`.
        """
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params: dict[str, Any] = {
            "state": state,
            "per_page": PAGE_SIZE,
            "sort": "updated",
            "direction": "desc",
        }
        if since is not None:
            params["since"] = since.isoformat().replace("+00:00", "Z")

        while url:
            resp = self._get(url, params=params)
            batch = resp.json()
            if not isinstance(batch, list):
                log.error("github.unexpected_payload", repo=repo, payload=batch)
                return
            yield from batch

            # Follow pagination via Link header; stop when no `next` link.
            next_link = _parse_next_link(resp.headers.get("Link", ""))
            if next_link is None:
                return
            url = next_link
            params = None  # subsequent pages include params in the URL


def _parse_next_link(link_header: str) -> str | None:
    """Parse the GitHub `Link` header and return the `rel=next` URL, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' in section:
            start = section.find("<")
            end = section.find(">")
            if start != -1 and end > start:
                return section[start + 1 : end]
    return None


def _looks_like_failure(issue: dict[str, Any]) -> bool:
    """Heuristic pre-filter: is this issue plausibly a failure report?

    Intentionally over-inclusive — the LLM classifier is authoritative. This
    just drops obvious feature requests and discussion threads so we don't
    pay API cost to classify them.
    """
    # Exclude PRs (the /issues endpoint returns both)
    if "pull_request" in issue:
        return False

    label_names = [lbl["name"].lower() for lbl in issue.get("labels", [])]

    # Exclude if labeled only as non-failure kinds and nothing else signals failure
    if (
        label_names
        and any(any(frag in name for frag in _NON_FAILURE_LABEL_FRAGMENTS) for name in label_names)
        and not any(any(frag in name for frag in _FAILURE_LABEL_FRAGMENTS) for name in label_names)
    ):
        return False

    # Include if any label suggests failure
    if any(any(frag in name for frag in _FAILURE_LABEL_FRAGMENTS) for name in label_names):
        return True

    # Fall back to title keywords (body is often empty at scrape time)
    title = (issue.get("title") or "").lower()
    if any(kw in title for kw in _FAILURE_HINT_KEYWORDS):
        return True

    # Include unlabeled issues — many frameworks triage lazily; classifier decides
    return not label_names


def _to_raw_issue(framework: Framework, payload: dict[str, Any]) -> RawIssue:
    return RawIssue(
        framework_slug=framework.slug,
        issue_number=payload["number"],
        node_id=payload["node_id"],
        title=payload["title"],
        body=payload.get("body"),
        url=payload["html_url"],
        labels=tuple(lbl["name"] for lbl in payload.get("labels", [])),
        state=payload["state"],
        author_association=payload.get("author_association"),
        is_pull_request="pull_request" in payload,
        created_at=datetime.fromisoformat(payload["created_at"].replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(payload["updated_at"].replace("Z", "+00:00")),
        closed_at=(
            datetime.fromisoformat(payload["closed_at"].replace("Z", "+00:00"))
            if payload.get("closed_at")
            else None
        ),
        comment_count=payload.get("comments", 0),
    )


def scrape_framework(
    framework: Framework,
    client: GitHubClient,
    *,
    since: datetime | None = None,
) -> Iterator[RawIssue]:
    """Yield failure-plausible issues for one framework."""
    log.info("scrape.start", framework=framework.slug, since=since)
    count = 0
    kept = 0
    for payload in client.list_issues(framework.repo, since=since):
        count += 1
        if not _looks_like_failure(payload):
            continue
        try:
            yield _to_raw_issue(framework, payload)
            kept += 1
        except Exception as exc:  # noqa: BLE001 — log and continue, don't abort whole framework
            log.exception("scrape.parse_error", framework=framework.slug, error=str(exc))
    log.info("scrape.done", framework=framework.slug, scanned=count, kept=kept)


def scrape_all(
    client: GitHubClient,
    *,
    since: datetime | None = None,
    frameworks: tuple[Framework, ...] = FRAMEWORKS,
) -> Iterator[RawIssue]:
    """Scrape every framework sequentially, yielding one `RawIssue` at a time."""
    for framework in frameworks:
        yield from scrape_framework(framework, client, since=since)
