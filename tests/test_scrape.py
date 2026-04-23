"""Scraper tests — use respx to mock the GitHub REST API."""

from __future__ import annotations

import httpx
import respx

from agentfail.frameworks import FRAMEWORKS
from agentfail.scrape import (
    GitHubClient,
    _looks_like_failure,
    _parse_next_link,
    _to_raw_issue,
    scrape_framework,
)


def test_parse_next_link_handles_absent_header():
    assert _parse_next_link("") is None


def test_parse_next_link_finds_next():
    hdr = (
        '<https://api.github.com/repositories/1/issues?page=2>; rel="next", '
        '<https://api.github.com/repositories/1/issues?page=10>; rel="last"'
    )
    assert _parse_next_link(hdr) == "https://api.github.com/repositories/1/issues?page=2"


def test_parse_next_link_returns_none_when_no_next():
    hdr = '<https://api.github.com/repositories/1/issues?page=10>; rel="last"'
    assert _parse_next_link(hdr) is None


class TestLooksLikeFailure:
    def test_excludes_pull_request(self, pr_payload):
        assert not _looks_like_failure(pr_payload)

    def test_excludes_feature_request(self, feature_request_payload):
        assert not _looks_like_failure(feature_request_payload)

    def test_includes_bug_labeled(self, github_issue_payload):
        assert _looks_like_failure(github_issue_payload)

    def test_includes_keyword_title(self):
        payload = {"title": "crash on startup", "labels": []}
        assert _looks_like_failure(payload)

    def test_includes_unlabeled(self):
        payload = {"title": "something odd", "labels": []}
        assert _looks_like_failure(payload)

    def test_bug_label_overrides_feature_label(self):
        # When an issue has both a bug-shaped and feature-shaped label,
        # we keep it — the bug signal wins.
        payload = {
            "title": "odd",
            "labels": [{"name": "enhancement"}, {"name": "type:bug"}],
        }
        assert _looks_like_failure(payload)


def test_to_raw_issue_parses_full_payload(github_issue_payload):
    fw = FRAMEWORKS[0]
    issue = _to_raw_issue(fw, github_issue_payload)
    assert issue.issue_number == 1234
    assert issue.framework_slug == fw.slug
    assert issue.labels == ("type:bug", "priority:high")
    assert issue.is_pull_request is False
    assert issue.state == "open"
    assert issue.closed_at is None


def test_to_raw_issue_flags_pull_request(pr_payload):
    fw = FRAMEWORKS[0]
    issue = _to_raw_issue(fw, pr_payload)
    assert issue.is_pull_request is True


@respx.mock
def test_scrape_framework_yields_filtered_issues(
    github_issue_payload, feature_request_payload, pr_payload
):
    fw = FRAMEWORKS[0]
    # URL prefix match so the scraper's query params don't miss the mock.
    respx.route(
        method="GET",
        url__startswith=f"https://api.github.com/repos/{fw.repo}/issues",
    ).mock(
        return_value=httpx.Response(
            200,
            json=[github_issue_payload, feature_request_payload, pr_payload],
            headers={"Link": ""},  # no next page
        )
    )
    client = GitHubClient(token="fake-token")
    issues = list(scrape_framework(fw, client))
    # Only the bug-labeled payload survives; feature + PR are filtered.
    assert len(issues) == 1
    assert issues[0].issue_number == 1234


@respx.mock
def test_scrape_framework_follows_pagination(github_issue_payload):
    import re

    fw = FRAMEWORKS[0]
    page2 = {**github_issue_payload, "number": 9991, "node_id": "I_kwDOAB12_5M6F7gZ"}
    base = f"https://api.github.com/repos/{fw.repo}/issues"

    # respx routes are matched in registration order — register page=2 first
    # so the more specific pattern wins before the catch-all first-page route.
    respx.route(
        method="GET",
        url__regex=re.compile(rf"{re.escape(base)}.*page=2"),
    ).mock(return_value=httpx.Response(200, json=[page2], headers={"Link": ""}))
    respx.route(
        method="GET",
        url__startswith=base,
    ).mock(
        return_value=httpx.Response(
            200,
            json=[github_issue_payload],
            headers={"Link": f'<{base}?page=2>; rel="next"'},
        )
    )

    client = GitHubClient(token="fake-token")
    issues = list(scrape_framework(fw, client))
    assert {i.issue_number for i in issues} == {1234, 9991}
