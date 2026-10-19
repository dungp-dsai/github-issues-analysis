"""RQ2: How does engagement differ between bot-authored and user-authored issues?

Engagement uses the schema fields comments_api (GitHub API comment count),
the reactions_* counts, and n_assignees. comments_count_reported is empty
in this export and is not used. n_comments_actual counts rows in the partial
issue_comments.csv export and is analyzed only as a secondary check.

Run from the repository root:

    uv run python scripts/rq2/run.py
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
    REACTION_COLUMNS,
    RESULTS,
    USER_COLOR,
    apply_fdr,
    exclude_dominant_bot,
    one_issue_per_repo,
    prepare_issues,
    split_groups,
    test_binary,
    test_numeric,
)
from scripts.load_data import load_issue_assignees, load_issue_comments  # noqa: E402

OUT = RESULTS / "rq2"

REACTION_LABELS = {
    "reactions_plus_one": "+1",
    "reactions_minus_one": "-1",
    "reactions_laugh": "laugh",
    "reactions_hooray": "hooray",
    "reactions_confused": "confused",
    "reactions_heart": "heart",
    "reactions_rocket": "rocket",
    "reactions_eyes": "eyes",
}


def _row(family: str, outcome: str, tested: dict) -> dict:
    tested = {k: v for k, v in tested.items() if k != "table"}
    tested["family"] = family
    tested["outcome"] = outcome
    return tested


def primary_tests(frame: pd.DataFrame, family: str = "primary") -> list[dict]:
    bot, user = split_groups(frame)
    rows = []
    for outcome, column in [
        ("comments_api", "comments_api"),
        ("reactions_total", "reactions_total"),
        ("n_assignees", "n_assignees"),
    ]:
        rows.append(_row(family, outcome, test_numeric(bot[column], user[column])))
    for outcome, column in [
        ("any_comment_api", "has_comment_api"),
        ("any_reaction", "has_reaction"),
        ("any_assignee", "has_assignee"),
    ]:
        rows.append(
            _row(
                family,
                outcome,
                test_binary(int(bot[column].sum()), len(bot), int(user[column].sum()), len(user)),
            )
        )
    return rows


def secondary_tests(frame: pd.DataFrame, comments: pd.DataFrame, assignees: pd.DataFrame) -> list[dict]:
    bot, user = split_groups(frame)
    rows = [_row("secondary_export", "n_comments_actual", test_numeric(bot["n_comments_actual"], user["n_comments_actual"]))]
    rows.append(
        _row(
            "secondary_export",
            "any_exported_comment",
            test_binary(
                int(bot["n_comments_actual"].gt(0).sum()),
                len(bot),
                int(user["n_comments_actual"].gt(0).sum()),
                len(user),
            ),
        )
    )
    for column, label in REACTION_LABELS.items():
        b_flag = bot[column].gt(0)
        u_flag = user[column].gt(0)
        rows.append(
            _row(
                "secondary_reaction_types",
                f"any_{label}",
                test_binary(int(b_flag.sum()), len(bot), int(u_flag.sum()), len(user)),
            )
        )

    exported = comments.merge(
        frame[["issue_id", "author_type", "author_login"]],
        on="issue_id",
        how="inner",
        suffixes=("_comment", "_issue"),
    )
    # load_data columns are author_login on both tables; merge suffixes apply
    # only to overlapping names. issue author_login was passed explicitly.
    comment_login = "author_login_comment" if "author_login_comment" in exported.columns else "author_login"
    issue_login = "author_login_issue" if "author_login_issue" in exported.columns else None
    if issue_login is not None:
        exported["self_comment"] = exported[comment_login].eq(exported[issue_login])
        # Among issues with at least one exported comment, did the issue author
        # write any of those exported comments?
        per_issue = exported.groupby(["issue_id", "author_type"], as_index=False)["self_comment"].any()
        b = per_issue.loc[per_issue["author_type"].eq("Bot"), "self_comment"]
        u = per_issue.loc[per_issue["author_type"].eq("User"), "self_comment"]
        rows.append(
            _row(
                "secondary_export",
                "exported_comment_by_issue_author",
                test_binary(int(b.sum()), len(b), int(u.sum()), len(u)),
            )
        )
        summary = (
            exported.groupby("author_type")
            .agg(
                n_exported_comments=(comment_login, "size"),
                n_distinct_comment_authors=(comment_login, "nunique"),
                share_rows_by_issue_author=("self_comment", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT / "exported_comment_summary.csv", index=False)

    assigned = assignees.merge(frame[["issue_id", "author_type", "author_login"]], on="issue_id", how="inner")
    assigned["self_assigned"] = assigned["assignee_login"].eq(assigned["author_login"])
    per_issue = assigned.groupby(["issue_id", "author_type"], as_index=False)["self_assigned"].any()
    # Compare self-assignment only among issues that have an assignee, so the
    # test is not a restatement of the any-assignee rate.
    b = per_issue.loc[per_issue["author_type"].eq("Bot"), "self_assigned"]
    u = per_issue.loc[per_issue["author_type"].eq("User"), "self_assigned"]
    rows.append(
        _row(
            "secondary_assignees",
            "self_assigned_if_any_assignee",
            test_binary(int(b.sum()), len(b), int(u.sum()), len(u)),
        )
    )
    top = (
        assigned.groupby(["author_type", "assignee_login"])
        .size()
        .reset_index(name="n_issue_links")
        .sort_values(["author_type", "n_issue_links"], ascending=[True, False])
    )
    top.groupby("author_type").head(10).to_csv(OUT / "top_assignees.csv", index=False)
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


def plot_rates(frame: pd.DataFrame, path: Path) -> None:
    specs = [
        ("has_comment_api", "Any API comment"),
        ("has_reaction", "Any reaction"),
        ("has_assignee", "Any assignee"),
    ]
    bot, user = split_groups(frame)
    y = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.barh(y + 0.18, [100 * bot[c].mean() for c, _ in specs], height=0.36, color=BOT_COLOR, label="Bot")
    ax.barh(y - 0.18, [100 * user[c].mean() for c, _ in specs], height=0.36, color=USER_COLOR, label="User")
    ax.set_yticks(y, [label for _, label in specs])
    ax.set_xlabel("Percent of issues")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Engagement presence by author account type")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_counts(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    for ax, column, title in [
        (axes[0], "comments_api", "API comment count"),
        (axes[1], "reactions_total", "Reaction count"),
        (axes[2], "n_assignees", "Assignee count"),
    ]:
        data = [
            frame.loc[frame["author_type"].eq("Bot"), column],
            frame.loc[frame["author_type"].eq("User"), column],
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
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def plot_reactions(frame: pd.DataFrame, path: Path) -> None:
    bot, user = split_groups(frame)
    labels = [REACTION_LABELS[c] for c in REACTION_COLUMNS]
    bot_rates = [100 * bot[c].gt(0).mean() for c in REACTION_COLUMNS]
    user_rates = [100 * user[c].gt(0).mean() for c in REACTION_COLUMNS]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.bar(x - 0.18, bot_rates, width=0.36, color=BOT_COLOR, label="Bot")
    ax.bar(x + 0.18, user_rates, width=0.36, color=USER_COLOR, label="User")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Percent of issues with at least one")
    ax.set_title("Reaction types")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    issues = prepare_issues()
    comments = load_issue_comments()
    assignees = load_issue_assignees()

    rows = primary_tests(issues, "primary")
    rows.extend(secondary_tests(issues, comments, assignees))
    rows.extend(primary_tests(exclude_dominant_bot(issues), "robustness_without_coderabbitai"))
    rows.extend(primary_tests(one_issue_per_repo(issues), "robustness_one_issue_per_repo"))
    tested = apply_fdr(pd.DataFrame(rows))
    tested.to_csv(OUT / "tests.csv", index=False)

    plot_rates(issues, OUT / "engagement_rates.pdf")
    plot_counts(issues, OUT / "engagement_counts.pdf")
    plot_reactions(issues, OUT / "reaction_types.pdf")

    coverage = pd.DataFrame(
        [
            {
                "n_issues": int(len(issues)),
                "sum_comments_api": int(issues["comments_api"].sum()),
                "n_exported_comment_rows": int(len(comments)),
                "n_issues_with_exported_comment": int(issues["n_comments_actual"].gt(0).sum()),
                "comments_count_reported_non_null": int(issues["comments_count_reported"].notna().sum()),
            }
        ]
    )
    coverage.to_csv(OUT / "comment_coverage.csv", index=False)

    primary = tested.loc[tested["family"].eq("primary") & tested["significant_fdr_0.05"]]
    print(f"Wrote {OUT}")
    print("Primary results with FDR q < 0.05:")
    cols = ["outcome", "bot_stat", "user_stat", "p_value", "p_fdr", "effect", "effect_interp"]
    print(primary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
