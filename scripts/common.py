"""Shared data preparation and significance tests for the bot-versus-user analyses.

Column meanings follow scripts/dataset_schema.pdf. Group membership is
author_type, the GitHub account type of the issue author (User or Bot).
agent_type is an automatic guess and is never used to assign groups.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.load_data import (
    load_issue_assignees,
    load_issue_comments,
    load_issue_labels,
    load_issues,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 20261019

# Calculator.net 95% row. The page's unlimited-population example rounds
# z=1.96, p=0.5, margin=0.05 up to 385.
Z_95 = 1.96
MARGIN = 0.05
PROPORTION = 0.5

BOT_COLOR = "#0072B2"
USER_COLOR = "#D55E00"

REACTION_COLUMNS = [
    "reactions_plus_one",
    "reactions_minus_one",
    "reactions_laugh",
    "reactions_hooray",
    "reactions_confused",
    "reactions_heart",
    "reactions_rocket",
    "reactions_eyes",
]


def sample_size_for_proportion(
    population: int,
    z: float = Z_95,
    margin: float = MARGIN,
    proportion: float = PROPORTION,
) -> dict:
    """Finite-population sample size for a proportion.

    Matches the sample-size formula on
    https://www.calculator.net/sample-size-calculator.html :
    n0 = z^2 * p * (1-p) / e^2, then
    n = n0 / (1 + (n0 - 1) / N), rounded up.
    """
    n0 = (z**2) * proportion * (1 - proportion) / (margin**2)
    if population <= 0:
        raise ValueError("population must be positive")
    n = n0 / (1 + (n0 - 1) / population)
    return {
        "population": int(population),
        "z": z,
        "margin": margin,
        "proportion": proportion,
        "n0": n0,
        "n": int(math.ceil(n - 1e-9)),
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for x versus y, from the Mann-Whitney U of x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto").statistic
    return float((2 * u) / (len(x) * len(y)) - 1)


def interpret_cliffs_delta(delta: float) -> str:
    magnitude = abs(delta)
    if math.isnan(magnitude):
        return "undefined"
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    denom = n * (min(n_rows, n_cols) - 1)
    if denom <= 0 or n <= 0:
        return float("nan")
    return float(math.sqrt(max(chi2, 0) / denom))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values, clipped to [0, 1]."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def _median_diff_ci(
    bot: np.ndarray,
    user: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bot = np.asarray(bot, dtype=float)
    user = np.asarray(user, dtype=float)
    point = float(np.median(bot) - np.median(user))
    if len(bot) == 0 or len(user) == 0:
        return point, float("nan"), float("nan")
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b = rng.choice(bot, size=len(bot), replace=True)
        u = rng.choice(user, size=len(user), replace=True)
        diffs[i] = np.median(b) - np.median(u)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return point, float(low), float(high)


def test_numeric(bot: pd.Series, user: pd.Series, seed: int = SEED) -> dict:
    b = pd.to_numeric(bot, errors="coerce").dropna().to_numpy(dtype=float)
    u = pd.to_numeric(user, errors="coerce").dropna().to_numpy(dtype=float)
    result = {
        "test": "Mann-Whitney U",
        "n_bot": int(len(b)),
        "n_user": int(len(u)),
        "bot_stat": float(np.median(b)) if len(b) else float("nan"),
        "user_stat": float(np.median(u)) if len(u) else float("nan"),
        "bot_q1": float(np.quantile(b, 0.25)) if len(b) else float("nan"),
        "bot_q3": float(np.quantile(b, 0.75)) if len(b) else float("nan"),
        "user_q1": float(np.quantile(u, 0.25)) if len(u) else float("nan"),
        "user_q3": float(np.quantile(u, 0.75)) if len(u) else float("nan"),
        "bot_mean": float(np.mean(b)) if len(b) else float("nan"),
        "user_mean": float(np.mean(u)) if len(u) else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "effect_name": "Cliff's delta (bot - user direction via U)",
        "effect": float("nan"),
        "effect_interp": "undefined",
        "estimate_name": "median bot minus median user",
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if len(b) == 0 or len(u) == 0 or (np.all(b == b[0]) and np.all(u == u[0]) and b[0] == u[0]):
        result["p_value"] = 1.0
        result["effect"] = 0.0
        result["effect_interp"] = "negligible"
        result["estimate"] = 0.0
        result["ci_low"] = 0.0
        result["ci_high"] = 0.0
        return result
    mw = stats.mannwhitneyu(b, u, alternative="two-sided", method="auto")
    delta = cliffs_delta(b, u)
    estimate, low, high = _median_diff_ci(b, u, seed=seed)
    result.update(
        {
            "statistic": float(mw.statistic),
            "p_value": float(mw.pvalue),
            "effect": delta,
            "effect_interp": interpret_cliffs_delta(delta),
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
        }
    )
    return result


def _odds_ratio_bot_over_user(a: int, b: int, c: int, d: int) -> float:
    """Odds of the event for bots divided by odds for users.

    Table layout is [[bot event, bot non-event], [user event, user non-event]].
    Haldane-Anscombe 0.5 correction is applied when any cell is zero.
    """
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float((a / b) / (c / d))


def test_two_proportion(bot_true: int, bot_n: int, user_true: int, user_n: int) -> dict:
    """Two-sided two-proportion Z-test with the pooled standard error under H0."""
    result = {
        "test": "two-proportion Z",
        "n_bot": int(bot_n),
        "n_user": int(user_n),
        "bot_stat": float(bot_true / bot_n) if bot_n else float("nan"),
        "user_stat": float(user_true / user_n) if user_n else float("nan"),
        "bot_q1": float("nan"),
        "bot_q3": float("nan"),
        "user_q1": float("nan"),
        "user_q3": float("nan"),
        "bot_mean": float("nan"),
        "user_mean": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "effect_name": "rate bot minus rate user",
        "effect": float("nan"),
        "effect_interp": "",
        "estimate_name": "rate bot minus rate user",
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if bot_n == 0 or user_n == 0:
        result["p_value"] = 1.0
        result["statistic"] = 0.0
        return result
    p1 = result["bot_stat"]
    p2 = result["user_stat"]
    pooled = (bot_true + user_true) / (bot_n + user_n)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / bot_n + 1.0 / user_n))
    if se == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = (p1 - p2) / se
        p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    result["statistic"] = float(z_stat)
    result["p_value"] = p_value
    result["effect"] = float(p1 - p2)
    result["estimate"] = float(p1 - p2)
    return result


def test_binary(bot_true: int, bot_n: int, user_true: int, user_n: int) -> dict:
    table = np.array(
        [
            [bot_true, bot_n - bot_true],
            [user_true, user_n - user_true],
        ]
    )
    result = {
        "test": "chi-square",
        "n_bot": int(bot_n),
        "n_user": int(user_n),
        "bot_stat": float(bot_true / bot_n) if bot_n else float("nan"),
        "user_stat": float(user_true / user_n) if user_n else float("nan"),
        "bot_q1": float("nan"),
        "bot_q3": float("nan"),
        "user_q1": float("nan"),
        "user_q3": float("nan"),
        "bot_mean": float("nan"),
        "user_mean": float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
        "effect_name": "odds ratio (bot / user)",
        "effect": float("nan"),
        "effect_interp": "",
        "estimate_name": "rate bot minus rate user",
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if bot_n == 0 or user_n == 0 or table.sum() == 0:
        result["p_value"] = 1.0
        return result
    expected = stats.chi2_contingency(table, correction=False).expected_freq
    if np.any(expected < 5):
        oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")
        result["test"] = "Fisher exact"
        result["statistic"] = float(oddsratio)
        result["p_value"] = float(p_value)
    else:
        chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
        result["statistic"] = float(chi2)
        result["p_value"] = float(p_value)
    result["effect"] = _odds_ratio_bot_over_user(*table.ravel())
    result["estimate"] = result["bot_stat"] - result["user_stat"]
    # Wald interval on the difference of two independent proportions.
    p1, p2 = result["bot_stat"], result["user_stat"]
    se = math.sqrt(p1 * (1 - p1) / bot_n + p2 * (1 - p2) / user_n)
    result["ci_low"] = float(result["estimate"] - Z_95 * se)
    result["ci_high"] = float(result["estimate"] + Z_95 * se)
    return result


def test_categorical(bot_labels: pd.Series, user_labels: pd.Series) -> dict:
    combined = pd.concat(
        [
            pd.DataFrame({"group": "Bot", "label": bot_labels.astype(str)}),
            pd.DataFrame({"group": "User", "label": user_labels.astype(str)}),
        ],
        ignore_index=True,
    )
    table = pd.crosstab(combined["group"], combined["label"])
    table = table.reindex(index=["Bot", "User"]).fillna(0).astype(int)
    table = table.loc[:, table.sum(axis=0) > 0]
    chi2, p_value, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    return {
        "test": "chi-square",
        "n_bot": int(table.loc["Bot"].sum()) if "Bot" in table.index else 0,
        "n_user": int(table.loc["User"].sum()) if "User" in table.index else 0,
        "bot_stat": float("nan"),
        "user_stat": float("nan"),
        "bot_q1": float("nan"),
        "bot_q3": float("nan"),
        "user_q1": float("nan"),
        "user_q3": float("nan"),
        "bot_mean": float("nan"),
        "user_mean": float("nan"),
        "statistic": float(chi2),
        "p_value": float(p_value),
        "effect_name": "Cramer's V",
        "effect": cramers_v(float(chi2), n, table.shape[0], table.shape[1]),
        "effect_interp": f"dof={int(dof)}; min expected={float(np.min(expected)):.2f}",
        "estimate_name": "",
        "estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "table": table,
    }


def apply_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_fdr"] = np.nan
    for family, idx in out.groupby("family").groups.items():
        values = out.loc[idx, "p_value"].tolist()
        out.loc[idx, "p_fdr"] = benjamini_hochberg(values)
    out["significant_fdr_0.05"] = out["p_fdr"] < 0.05
    return out


def split_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bot = frame.loc[frame["author_type"].eq("Bot")]
    user = frame.loc[frame["author_type"].eq("User")]
    return bot, user


def prepare_issues() -> pd.DataFrame:
    """Issue table restricted to rows the schema identifies as issues.

    is_pull_request is true when the GitHub record is a pull request.
    Resolution hours use the live API timestamps because the schema states
    that state_final and the archive stream can be outdated, and because
    created_at equals closed_at on every archive-closed row in this export
    (so time_to_close_hours is uniformly 0).
    """
    issues = load_issues()
    issues = issues.loc[~issues["is_pull_request"].astype(bool)].copy()
    issues["author_type"] = issues["author_type"].astype(str)
    unknown = sorted(set(issues["author_type"]) - {"User", "Bot"})
    if unknown:
        raise ValueError(f"Unexpected author_type values: {unknown}")

    issues["body_text"] = issues["body"].fillna("")
    issues["title_text"] = issues["title"].fillna("")
    issues["body_empty"] = issues["body_text"].str.strip().eq("")
    issues["body_chars"] = issues["body_text"].str.len()
    issues["title_chars"] = issues["title_text"].str.len()
    issues["api_closed"] = issues["api_state"].eq("closed")
    issues["resolution_hours"] = (
        issues["api_closed_at"] - issues["api_created_at"]
    ).dt.total_seconds() / 3600
    issues["has_label"] = issues["n_labels"].gt(0)
    issues["has_milestone"] = issues["milestone_title"].notna()
    issues["has_comment_api"] = issues["comments_api"].gt(0)
    issues["has_reaction"] = issues["reactions_total"].gt(0)
    issues["has_assignee"] = issues["n_assignees"].gt(0)
    issues["has_code_fence"] = issues["body_text"].str.contains(r"```", regex=True)
    issues["has_url"] = issues["body_text"].str.contains(r"https?://", regex=True)
    issues["has_heading"] = issues["body_text"].str.contains(r"(?m)^#{1,6}\s", regex=True)
    issues["has_checklist"] = issues["body_text"].str.contains(
        r"(?m)^\s*[-*]\s+\[[ xX]\]", regex=True
    )
    issues["created_hour"] = issues["api_created_at"].dt.hour
    issues["created_weekend"] = issues["api_created_at"].dt.dayofweek.ge(5)
    issues["archive_matches_api"] = issues["state_final"].eq(issues["api_state"])
    return issues


def one_issue_per_repo(issues: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    return (
        issues.groupby("repo_name", group_keys=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )


def exclude_dominant_bot(issues: pd.DataFrame, login: str = "coderabbitai[bot]") -> pd.DataFrame:
    """Drop the most frequent bot author so results are not only that account."""
    keep = ~((issues["author_type"].eq("Bot")) & (issues["author_login"].eq(login)))
    return issues.loc[keep].copy()
