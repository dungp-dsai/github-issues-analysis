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
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (  # noqa: E402
    RESULTS,
    load_issue_labels,
    prepare_issues,
    split_groups,
    test_categorical,
    test_numeric,
    test_two_proportion,
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


def proportion_rows(family: str, frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[dict]:
    """Two-proportion Z-tests for binary outcomes."""
    bot, user = split_groups(frame)
    rows = []
    for outcome, column in columns:
        rows.append(
            _row(
                family,
                outcome,
                test_two_proportion(
                    int(bot[column].sum()),
                    int(len(bot)),
                    int(user[column].sum()),
                    int(len(user)),
                ),
            )
        )
    return rows


def frequent_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Issue-label rows whose label appears on at least 30 issues in the analysis set."""
    linked = labels.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    counts = linked.groupby("label")["issue_id"].nunique()
    frequent = counts[counts >= 30].index
    return linked.loc[linked["label"].isin(frequent)].copy()


def approach_tests(frame: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    """Tests named in the RQ1 approach.

    Numeric outcomes use a two-sided Mann-Whitney U test. Binary outcomes use a
    two-proportion Z-test. Categorical outcomes use a chi-square test of
    independence. Significance is p < 0.05.
    """
    rows: list[dict] = []
    closed = frame.loc[frame["api_closed"] & frame["resolution_hours"].notna()]
    bot_closed, user_closed = split_groups(closed)
    rows.append(
        _row(
            "lifecycle",
            "resolution_hours_if_api_closed",
            test_numeric(bot_closed["resolution_hours"], user_closed["resolution_hours"]),
        )
    )
    rows.extend(proportion_rows("lifecycle", frame, [("api_state_closed", "api_closed")]))
    bot_reason, user_reason = split_groups(frame.loc[frame["api_closed"]])
    reason = test_categorical(
        bot_reason["state_reason"].fillna("(missing)"),
        user_reason["state_reason"].fillna("(missing)"),
    )
    reason.pop("table", None)
    rows.append(_row("lifecycle", "state_reason_if_api_closed", reason))

    rows.extend(
        numeric_rows(
            "metadata",
            frame,
            [
                ("n_labels", "n_labels"),
                ("n_gharchive_events", "n_gharchive_events"),
            ],
        )
    )
    rows.extend(
        proportion_rows(
            "metadata",
            frame,
            [
                ("has_label", "has_label"),
                ("has_milestone", "has_milestone"),
            ],
        )
    )
    linked = frequent_labels(frame, labels)
    categories = test_categorical(
        linked.loc[linked["author_type"].eq("Bot"), "label"],
        linked.loc[linked["author_type"].eq("User"), "label"],
    )
    categories.pop("table", None)
    rows.append(_row("metadata", "label_categories", categories))

    issue_ids = frame[["issue_id", "author_type"]]
    for label in sorted(linked["label"].unique()):
        holders = set(linked.loc[linked["label"].eq(label), "issue_id"])
        flagged = issue_ids.assign(present=issue_ids["issue_id"].isin(holders))
        bot_flag = flagged.loc[flagged["author_type"].eq("Bot"), "present"]
        user_flag = flagged.loc[flagged["author_type"].eq("User"), "present"]
        rows.append(
            _row(
                "label_presence",
                f"label::{label}",
                test_two_proportion(int(bot_flag.sum()), int(len(bot_flag)), int(user_flag.sum()), int(len(user_flag))),
            )
        )

    rows.extend(
        numeric_rows(
            "content",
            frame,
            [
                ("title_characters", "title_chars"),
                ("body_characters", "body_chars"),
            ],
        )
    )
    rows.extend(
        proportion_rows(
            "content",
            frame,
            [
                ("body_empty", "body_empty"),
                ("body_has_url", "has_url"),
                ("body_has_markdown_heading", "has_heading"),
                ("body_has_code_fence", "has_code_fence"),
                ("body_has_checklist", "has_checklist"),
            ],
        )
    )
    return rows


ACCOUNT_ORDER = ["Bot", "User"]


def _style():
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "medium",
            "axes.labelcolor": "#2c2c2c",
            "text.color": "#2c2c2c",
            "xtick.color": "#3a3a3a",
            "ytick.color": "#3a3a3a",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )


def _human_label(value: str) -> str:
    if value == "User":
        return "Human"
    return value.replace("user issues", "human issues")


def _save(fig: plt.Figure, path: Path) -> None:
    for ax in fig.axes:
        ax.grid(False)
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = [tick.get_text() for tick in axis.get_ticklabels()]
            if any(label == "User" or "user issues" in label for label in labels):
                axis.set_ticks(axis.get_ticklocs())
                axis.set_ticklabels([_human_label(label) for label in labels])
    for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text)):
        updated = _human_label(text.get_text())
        if updated != text.get_text():
            text.set_text(updated)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _account_frame(categories: list[str], bot_values: list[float], user_values: list[float], value_name: str) -> pd.DataFrame:
    rows = []
    for category, bot_value, user_value in zip(categories, bot_values, user_values):
        rows.append({"Metric": category, "Account": "Bot", value_name: bot_value})
        rows.append({"Metric": category, "Account": "User", value_name: user_value})
    return pd.DataFrame(rows)


def _polish_bars(ax: plt.Axes) -> None:
    for container in ax.containers:
        for bar in container:
            bar.set_linewidth(0.7)
            bar.set_edgecolor("white")


def _label_bars(ax: plt.Axes, fmt: str, fontsize: float = 7.5) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2, fontsize=fontsize, color="#333333")


def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.2,
        columnspacing=1.4,
    )


def _pretty_reason(name: str) -> str:
    if name in {"(missing)", "nan", "None"}:
        return "Missing"
    return str(name).replace("_", " ")


def plot_lifecycle(frame: pd.DataFrame, path: Path) -> None:
    """One figure, three panels: state, resolution time, close reason."""
    bot, user = split_groups(frame)
    state = _account_frame(
        ["Closed", "Open"],
        [100 * bot["api_closed"].mean(), 100 * (1 - bot["api_closed"].mean())],
        [100 * user["api_closed"].mean(), 100 * (1 - user["api_closed"].mean())],
        "Percent",
    )
    closed = frame.loc[frame["api_closed"] & frame["resolution_hours"].notna()].copy()
    closed["Account"] = closed["author_type"]
    reasons = closed["state_reason"].fillna("Missing").astype(str)
    preferred = ["completed", "not_planned"]
    present = list(reasons.unique())
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(name for name in present if name not in preferred and name != "Missing"))
    if "Missing" in present:
        ordered.append("Missing")
    reason_frame = closed.assign(reason=reasons)
    rates = (
        pd.crosstab(reason_frame["Account"], reason_frame["reason"], normalize="index")
        .reindex(index=ACCOUNT_ORDER, columns=ordered)
        .fillna(0)
        * 100
    )
    reason_plot = _account_frame(
        [_pretty_reason(name) for name in ordered],
        [float(rates.loc["Bot", name]) for name in ordered],
        [float(rates.loc["User", name]) for name in ordered],
        "Percent",
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1.05, 1.0, 1.2]})

    sns.barplot(
        data=state,
        x="Metric",
        y="Percent",
        hue="Account",
        hue_order=ACCOUNT_ORDER,
        order=["Closed", "Open"],
        palette="pastel",
        saturation=1,
        width=0.72,
        ax=axes[0],
    )
    _polish_bars(axes[0])
    _label_bars(axes[0], "%.1f")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Percent of issues")
    axes[0].set_title("(a) Issue state")

    sns.boxplot(
        data=closed,
        x="Account",
        y="resolution_hours",
        order=ACCOUNT_ORDER,
        hue="Account",
        hue_order=ACCOUNT_ORDER,
        palette="pastel",
        saturation=1,
        width=0.55,
        showfliers=False,
        legend=False,
        linewidth=0.8,
        medianprops={"color": "#2b2b2b", "linewidth": 1.5},
        whiskerprops={"color": "#6d6d6d", "linewidth": 0.8},
        capprops={"color": "#6d6d6d", "linewidth": 0.8},
        boxprops={"edgecolor": "#8a8a8a", "linewidth": 0.7},
        ax=axes[1],
    )
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(0.01, 8000)
    axes[1].set_yticks([0.1, 1, 10, 100, 1000])
    axes[1].set_yticklabels(["0.1", "1", "10", "100", "1,000"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Hours (symlog)")
    axes[1].set_title("(b) Resolution time")
    for index, account in enumerate(ACCOUNT_ORDER):
        median = float(closed.loc[closed["Account"].eq(account), "resolution_hours"].median())
        axes[1].annotate(
            f"{median:.0f} h",
            xy=(index, median),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

    sns.barplot(
        data=reason_plot,
        x="Metric",
        y="Percent",
        hue="Account",
        hue_order=ACCOUNT_ORDER,
        order=[_pretty_reason(name) for name in ordered],
        palette="pastel",
        saturation=1,
        width=0.72,
        ax=axes[2],
    )
    _polish_bars(axes[2])
    _label_bars(axes[2], "%.1f")
    axes[2].set_ylim(0, 118)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Percent of closed issues")
    axes[2].set_title("(c) Close reason")
    axes[2].tick_params(axis="x", labelsize=7.5, labelrotation=18)
    for tick in axes[2].get_xticklabels():
        tick.set_ha("right")

    _legend(fig, axes[0])
    if axes[2].legend_ is not None:
        axes[2].legend_.remove()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_metadata_counts(frame: pd.DataFrame, path: Path) -> None:
    """Grouped columns for metadata summaries. The top-left panel is the share of issues with a label."""
    bot, user = split_groups(frame)
    plot_df = _account_frame(
        ["Issues with a label", "Label count", "Archive events", "Milestone"],
        [
            100 * float(bot["has_label"].mean()),
            float(bot["n_labels"].mean()),
            float(bot["n_gharchive_events"].mean()),
            100 * float(bot["has_milestone"].mean()),
        ],
        [
            100 * float(user["has_label"].mean()),
            float(user["n_labels"].mean()),
            float(user["n_gharchive_events"].mean()),
            100 * float(user["has_milestone"].mean()),
        ],
        "Value",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.35))
    panels = [
        (axes[0, 0], "Issues with a label", "Percent of issues", "{:.1f}", 0, 100),
        (axes[0, 1], "Label count", "Mean labels per issue", "{:.2f}", 0, 1.7),
        (axes[1, 0], "Archive events", "Mean events per issue", "{:.2f}", 0, 2.05),
        (axes[1, 1], "Milestone", "Percent of issues", "{:.1f}", 0, 6),
    ]
    panels = [
        (ax, plot_df.loc[plot_df["Metric"].eq(title)], title, ylabel, fmt, ymin, ymax)
        for ax, title, ylabel, fmt, ymin, ymax in panels
    ]
    for ax, data, title, ylabel, fmt, ymin, ymax in panels:
        sns.barplot(
            data=data,
            x="Account",
            y="Value",
            order=ACCOUNT_ORDER,
            hue="Account",
            hue_order=ACCOUNT_ORDER,
            palette="pastel",
            saturation=1,
            width=0.62,
            legend=False,
            ax=ax,
        )
        _polish_bars(ax)
        _label_bars(ax, fmt)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=sns.color_palette("pastel")[0], edgecolor="white", linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=sns.color_palette("pastel")[1], edgecolor="white", linewidth=0.7),
    ]
    fig.legend(handles, ACCOUNT_ORDER, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


def plot_metadata_labels(frame: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    """Side-by-side horizontal bars: top 5 labels within bot issues and within user issues."""
    linked = labels.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    bot, user = split_groups(frame)
    rows = []
    for label in linked["label"].dropna().unique():
        holders = set(linked.loc[linked["label"].eq(label), "issue_id"])
        rows.append(
            {
                "Label": label,
                "Bot": 100 * float(bot["issue_id"].isin(holders).mean()),
                "User": 100 * float(user["issue_id"].isin(holders).mean()),
            }
        )
    rates = pd.DataFrame(rows)
    bot_top = rates.sort_values(["Bot", "Label"], ascending=[False, True]).head(5)
    user_top = rates.sort_values(["User", "Label"], ascending=[False, True]).head(5)
    peak = float(max(bot_top["Bot"].max(), user_top["User"].max()))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), sharex=True)
    panels = [
        (axes[0], bot_top, "Bot", "(a) Top 5 labels on bot issues"),
        (axes[1], user_top, "User", "(b) Top 5 labels on user issues"),
    ]
    colors = {"Bot": sns.color_palette("pastel")[0], "User": sns.color_palette("pastel")[1]}
    for ax, selected, account, title in panels:
        # Seaborn draws the first category at the top, so the largest share comes first.
        ordered = selected.sort_values([account, "Label"], ascending=[False, True])
        sns.barplot(
            data=ordered,
            y="Label",
            x=account,
            order=ordered["Label"].tolist(),
            color=colors[account],
            saturation=1,
            width=0.72,
            orient="y",
            ax=ax,
        )
        _polish_bars(ax)
        texts = ["<0.1" if value < 0.05 else f"{value:.1f}" for value in ordered[account]]
        ax.bar_label(ax.containers[0], labels=texts, padding=3, fontsize=8, color="#333333")
        ax.set_xlim(0, max(peak * 1.22, 5))
        ax.set_ylabel("")
        ax.set_xlabel("Percent of issues")
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


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

    tested = pd.DataFrame(approach_tests(issues, labels))
    tested["significant_0.05"] = tested["p_value"] < 0.05
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

    plot_lifecycle(issues, OUT / "lifecycle_resolution.pdf")
    plot_metadata_counts(issues, OUT / "metadata_counts.pdf")
    plot_metadata_labels(issues, labels, OUT / "metadata_labels.pdf")

    print(f"Wrote {OUT}")
    print(f"Issues after dropping pull requests: {len(issues)}")
    print(issues["author_type"].value_counts().to_string())
    print("Tests with p < 0.05:")
    cols = ["family", "outcome", "test", "bot_stat", "user_stat", "statistic", "p_value"]
    print(tested.loc[tested["significant_0.05"], cols].to_string(index=False))


if __name__ == "__main__":
    main()
