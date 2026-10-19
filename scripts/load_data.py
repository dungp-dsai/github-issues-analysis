from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_ISSUE_DATES = [
    "created_at",
    "closed_at",
    "api_created_at",
    "api_updated_at",
    "api_closed_at",
]


def load_issues() -> pd.DataFrame:
    """Load GitHub issues."""
    return pd.read_csv(DATA_DIR / "issues.csv", parse_dates=_ISSUE_DATES)


def load_issue_comments() -> pd.DataFrame:
    """Load comments on GitHub issues."""
    return pd.read_csv(DATA_DIR / "issue_comments.csv", parse_dates=["created_at"])


def load_issue_assignees() -> pd.DataFrame:
    """Load assignees of GitHub issues."""
    return pd.read_csv(DATA_DIR / "issue_assignees.csv")


def load_issue_labels() -> pd.DataFrame:
    """Load labels of GitHub issues."""
    return pd.read_csv(DATA_DIR / "issue_labels.csv")
