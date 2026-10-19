"""RQ1: How do issues opened by bot accounts differ from those opened by users?

Run from the repository root:

    uv run python scripts/rq1/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (  # noqa: E402
    BOT_COLOR,
    RESULTS,
    USER_COLOR,
    apply_fdr,
    exclude_dominant_bot,
    load_issue_labels,
    one_issue_per_repo,
    prepare_issues,
    split_groups,
    test_binary,
    test_categorical,
    test_numeric,
)

OUT = RESULTS / "rq1"


def _row(family: str, outcome: str, tested: dict) -> dict:
    tested = {k: v for k, v in tested.items() if k != "table"}
    tested["family"] = family
    tested["outcome"] = outcome
    return tested


def numeric_rows(family: str, frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[dict]:
    bot, user = split_groups(frame)
    rows = []
    for outcome, column in columns:
        rows.append(_row(family, outcome, test_numeric(bot[column], user[column])))
    return rows


def binary_rows(family: str, frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[dict]:
    bot, user = split_groups(frame)
    rows = []
    for outcome, column in columns:
        rows.append(
            _row(
                family,
                outcome,
                test_binary(
                    int(bot[column].sum()),
                    int(len(bot)),
                    int(user[column].sum()),
                    int(len(user)),
                ),
            )
        )
    return rows


def primary_tests(frame: pd.DataFrame, family: str = "primary") -> list[dict]:
    rows = numeric_rows(
        family,
        frame,
        [
            ("body_characters", "body_chars"),
            ("title_characters", "title_chars"),
            ("n_labels", "n_labels"),
            ("n_gharchive_events", "n_gharchive_events"),
        ],
    )
    closed = frame.loc[frame["api_closed"] & frame["resolution_hours"].notna()]
    bot, user = split_groups(closed)
    rows.append(_row(family, "resolution_hours_if_api_closed", test_numeric(bot["resolution_hours"], user["resolution_hours"])))
    rows.extend(
        binary_rows(
            family,
            frame,
            [
                ("api_state_closed", "api_closed"),
                ("has_label", "has_label"),
                ("has_milestone", "has_milestone"),
                ("locked", "locked"),
                ("body_empty", "body_empty"),
            ],
        )
    )
    bot_all, user_all = split_groups(frame.loc[frame["api_closed"]])
    reason = test_categorical(bot_all["state_reason"].fillna("(missing)"), user_all["state_reason"].fillna("(missing)"))
    reason.pop("table", None)
    rows.append(_row(family, "state_reason_if_api_closed", reason))
    return rows


def secondary_tests(frame: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    rows = binary_rows(
        "secondary_text_and_timing",
        frame,
        [
            ("body_has_code_fence", "has_code_fence"),
            ("body_has_url", "has_url"),
            ("body_has_markdown_heading", "has_heading"),
            ("body_has_checklist", "has_checklist"),
            ("created_on_weekend_utc", "created_weekend"),
            ("archive_state_equals_api_state", "archive_matches_api"),
        ],
    )
    bot, user = split_groups(frame)
    hour = test_categorical(bot["created_hour"].astype(int).astype(str), user["created_hour"].astype(int).astype(str))
    hour.pop("table", None)
    rows.append(_row("secondary_text_and_timing", "api_created_hour_utc", hour))

    linked = labels.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    counts = linked.groupby("label")["issue_id"].nunique()
    frequent = counts[counts >= 30].index.tolist()
    issue_ids = frame[["issue_id", "author_type"]]
    for label in frequent:
        holders = set(linked.loc[linked["label"].eq(label), "issue_id"])
        flagged = issue_ids.assign(has_label=issue_ids["issue_id"].isin(holders))
        b = flagged.loc[flagged["author_type"].eq("Bot"), "has_label"]
        u = flagged.loc[flagged["author_type"].eq("User"), "has_label"]
        rows.append(
            _row(
                "secondary_labels",
                f"label::{label}",
                test_binary(int(b.sum()), int(len(b)), int(u.sum()), int(len(u))),
            )
        )
    return rows


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_binary_rates(frame: pd.DataFrame, path: Path) -> None:
    specs = [
        ("api_closed", "API state closed"),
        ("has_label", "Has a label"),
        ("has_milestone", "Has a milestone"),
        ("locked", "Conversation locked"),
        ("body_empty", "Empty body"),
        ("has_code_fence", "Body has code fence"),
        ("has_heading", "Body has heading"),
        ("has_url", "Body has URL"),
    ]
    bot, user = split_groups(frame)
    labels = [label for _, label in specs]
    bot_rates = [100 * bot[col].mean() for col, _ in specs]
    user_rates = [100 * user[col].mean() for col, _ in specs]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(y + 0.18, bot_rates, height=0.36, color=BOT_COLOR, label="Bot")
    ax.barh(y - 0.18, user_rates, height=0.36, color=USER_COLOR, label="User")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Percent of issues")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Issue attributes by author account type")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_lengths(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, column, title, xlabel in [
        (axes[0], "body_chars", "Issue body length", "Characters (log scale)"),
        (axes[1], "title_chars", "Issue title length", "Characters"),
    ]:
        data = [
            frame.loc[frame["author_type"].eq("Bot"), column].clip(lower=1 if column == "body_chars" else 0),
            frame.loc[frame["author_type"].eq("User"), column].clip(lower=1 if column == "body_chars" else 0),
        ]
        box = ax.boxplot(
            data,
            tick_labels=["Bot", "User"],
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for patch, color in zip(box["boxes"], [BOT_COLOR, USER_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        if column == "body_chars":
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_resolution(frame: pd.DataFrame, path: Path) -> None:
    closed = frame.loc[frame["api_closed"] & frame["resolution_hours"].notna()]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    data = [
        closed.loc[closed["author_type"].eq("Bot"), "resolution_hours"],
        closed.loc[closed["author_type"].eq("User"), "resolution_hours"],
    ]
    box = ax.boxplot(
        data,
        tick_labels=["Bot", "User"],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch, color in zip(box["boxes"], [BOT_COLOR, USER_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Hours from API created to API closed")
    ax.set_title("Resolution time when api_state is closed")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_state_reason(frame: pd.DataFrame, path: Path) -> None:
    closed = frame.loc[frame["api_closed"]].copy()
    closed["reason"] = closed["state_reason"].fillna("(missing)")
    rates = pd.crosstab(closed["author_type"], closed["reason"], normalize="index").reindex(["Bot", "User"]) * 100
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(rates.columns))
    ax.bar(x - 0.18, rates.loc["Bot"], width=0.36, color=BOT_COLOR, label="Bot")
    ax.bar(x + 0.18, rates.loc["User"], width=0.36, color=USER_COLOR, label="User")
    ax.set_xticks(x, rates.columns, rotation=20, ha="right")
    ax.set_ylabel("Percent of API-closed issues")
    ax.set_title("GitHub close reason (state_reason)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_hours(frame: pd.DataFrame, path: Path) -> None:
    hours = list(range(24))
    bot, user = split_groups(frame)
    bot_share = bot["created_hour"].value_counts(normalize=True).reindex(hours, fill_value=0) * 100
    user_share = user["created_hour"].value_counts(normalize=True).reindex(hours, fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(hours, bot_share, color=BOT_COLOR, marker="o", ms=3, label="Bot")
    ax.plot(hours, user_share, color=USER_COLOR, marker="o", ms=3, label="User")
    ax.set_xlabel("Hour of api_created_at (UTC)")
    ax.set_ylabel("Percent of issues")
    ax.set_xticks(hours)
    ax.set_ylim(0, max(bot_share.max(), user_share.max()) * 1.25)
    ax.set_title("Issue creation hour")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def describe_population(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for author, part in frame.groupby("author_type"):
        rows.append(
            {
                "author_type": author,
                "n_issues": int(len(part)),
                "n_repos": int(part["repo_name"].nunique()),
                "n_author_logins": int(part["author_login"].nunique()),
                "top_author_login": part["author_login"].value_counts().index[0],
                "top_author_n": int(part["author_login"].value_counts().iloc[0]),
                "api_created_min": str(part["api_created_at"].min()),
                "api_created_max": str(part["api_created_at"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    issues = prepare_issues()
    labels = load_issue_labels()

    rows = primary_tests(issues, "primary")
    rows.extend(secondary_tests(issues, labels))
    rows.extend(primary_tests(exclude_dominant_bot(issues), "robustness_without_coderabbitai"))
    rows.extend(primary_tests(one_issue_per_repo(issues), "robustness_one_issue_per_repo"))
    tested = apply_fdr(pd.DataFrame(rows))
    tested.to_csv(OUT / "tests.csv", index=False)

    describe_population(issues).to_csv(OUT / "population.csv", index=False)
    agent = pd.crosstab(issues["author_type"], issues["agent_type"], margins=True)
    agent.to_csv(OUT / "agent_type_by_author_type.csv")
    authors = (
        issues.groupby(["author_type", "author_login"])
        .size()
        .reset_index(name="n_issues")
        .sort_values(["author_type", "n_issues"], ascending=[True, False])
    )
    authors.groupby("author_type").head(15).to_csv(OUT / "top_authors.csv", index=False)

    plot_binary_rates(issues, OUT / "attribute_rates.pdf")
    plot_lengths(issues, OUT / "text_length.pdf")
    plot_resolution(issues, OUT / "resolution_hours.pdf")
    plot_state_reason(issues, OUT / "state_reason.pdf")
    plot_hours(issues, OUT / "created_hour.pdf")

    primary = tested.loc[tested["family"].eq("primary") & tested["significant_fdr_0.05"]]
    print(f"Wrote {OUT}")
    print(f"Issues after dropping pull requests: {len(issues)}")
    print(issues["author_type"].value_counts().to_string())
    print("Primary results with FDR q < 0.05:")
    cols = ["outcome", "bot_stat", "user_stat", "p_value", "p_fdr", "effect", "effect_interp"]
    print(primary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
