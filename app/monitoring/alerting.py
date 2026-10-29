# Opens or updates a GitHub Issue when drift is detected, so the alert lives somewhere visible
# with zero new external services/credentials - just the workflow's own GITHUB_TOKEN.

from __future__ import annotations

import requests

from monitoring.drift_report import DriftResult

GITHUB_API = "https://api.github.com"
DRIFT_LABEL = "model-drift"


def format_alert_body(result: DriftResult, run_url: str | None = None) -> str:
    lines = [
        f"**Dataset drift detected** on {result.sample_size} recently logged predictions.",
        "",
        f"- Drifted column share: {result.drifted_share:.2f}",
        f"- Drifted columns: {', '.join(result.drifted_columns) or 'none'}",
    ]
    if run_url:
        lines += [
            "",
            f"[Workflow run]({run_url}) - full Evidently report is attached there as an artifact.",
        ]
    lines += ["", "A retrain has been triggered automatically."]
    return "\n".join(lines)


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def alert_on_drift(repo: str, result: DriftResult, token: str, run_url: str | None = None) -> None:
    """
    Open a new GitHub Issue labeled `model-drift`, or comment on an existing open one rather
    than creating duplicates on every scheduled run while drift persists. No-op if
    result.dataset_drift is False.
    """
    if not result.dataset_drift:
        return

    headers = _headers(token)
    body = format_alert_body(result, run_url=run_url)

    existing = requests.get(
        f"{GITHUB_API}/repos/{repo}/issues",
        headers=headers,
        params={"state": "open", "labels": DRIFT_LABEL},
        timeout=30,
    )
    existing.raise_for_status()
    open_issues = existing.json()

    if open_issues:
        issue_number = open_issues[0]["number"]
        response = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments",
            headers=headers,
            json={"body": body},
            timeout=30,
        )
    else:
        response = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            headers=headers,
            json={"title": "Model drift detected", "body": body, "labels": [DRIFT_LABEL]},
            timeout=30,
        )
    response.raise_for_status()
