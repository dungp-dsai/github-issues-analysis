"""RQ2: How does engagement differ between bot-authored and user-authored issues?

Engagement uses the schema fields comments_api (GitHub API comment count),
the reactions_* counts, and n_assignees. comments_count_reported is empty
in this export and is not used. n_comments_actual counts rows in the partial
issue_comments.csv export and is analyzed only as a secondary check.

Run from the repository root:

    uv run python scripts/rq2/run.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from html.parser import HTMLParser
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
    plt.close(fig)


def plot_comments(frame: pd.DataFrame, comments: pd.DataFrame, path: Path) -> None:
    """Two panels: share of issues with a comment, and exported comment-body length."""
    sns.set_theme(style="whitegrid", palette="pastel")
    bot, user = split_groups(frame)
    colors = sns.color_palette("pastel", 2)
    linked = comments.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    linked["body_chars"] = linked["body"].fillna("").str.len()
    grouped = [
        linked.loc[linked["author_type"].eq("Bot"), "body_chars"].to_numpy(dtype=float),
        linked.loc[linked["author_type"].eq("User"), "body_chars"].to_numpy(dtype=float),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25))

    rates = [100 * float(bot["has_comment_api"].mean()), 100 * float(user["has_comment_api"].mean())]
    bars = axes[0].bar(
        ["Bot", "User"],
        rates,
        color=colors,
        width=0.62,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color="#333333")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent of issues")
    axes[0].set_xlabel("")
    axes[0].set_title("(a) Issues with a comment")

    # A few exported comments run past 70,000 characters and would flatten the
    # violin. The axis shows lengths through 15,000 characters.
    y_max = 15000
    shown = [group[group <= y_max] for group in grouped]
    violin = axes[1].violinplot(shown, showmeans=False, showmedians=True)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.95)
        body.set_zorder(2)
    violin["cmedians"].set_color("#2b2b2b")
    violin["cmedians"].set_linewidth(1.2)
    for part in ("cbars", "cmins", "cmaxes"):
        violin[part].remove()
    axes[1].boxplot(
        shown,
        positions=[1, 2],
        widths=0.1,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 0.8, "color": "#333333"},
        whiskerprops={"linewidth": 0.8, "color": "#333333"},
        capprops={"linewidth": 0.8, "color": "#333333"},
        zorder=3,
    )
    for index, group in enumerate(grouped, start=1):
        median = float(np.median(group))
        axes[1].annotate(
            f"{median:,.0f}",
            xy=(index, median),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color="#222222",
            zorder=5,
        )
    axes[1].set_xticks([1, 2], ["Bot", "User"])
    axes[1].set_xlim(0.4, 2.6)
    axes[1].set_ylim(-400, y_max)
    axes[1].set_ylabel("Characters")
    axes[1].set_xlabel("")
    axes[1].set_title("(b) Comment length")
    for ax in axes:
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`]+`")
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_NONWORD = re.compile(r"[^a-z\s]")
_EXTRA_STOPS = {
    "let", "make", "made", "seem", "still", "back", "even", "way", "thing",
    "much", "many", "well", "per", "via", "also", "using", "use", "used",
    "please", "thanks", "thank", "hello", "hey", "would", "could", "should",
    "just", "get", "got", "one", "two", "etc", "yes", "true", "false", "none",
    "null", "http", "https", "www", "com", "org", "html", "png", "jpg", "jpeg",
    "gif", "svg", "href", "src", "img", "div", "span", "class", "style",
    "width", "height", "align", "center", "margin", "alt", "nbsp", "amp",
    "quot", "like", "currently", "look", "really", "already", "maybe",
    "actually", "something", "anything",
}


def _ensure_nltk() -> None:
    import nltk

    path = ROOT / ".cache" / "nltk"
    path.mkdir(parents=True, exist_ok=True)
    if str(path) not in nltk.data.path:
        nltk.data.path.insert(0, str(path))
    resources = {
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "stopwords": "corpora/stopwords",
    }
    for resource, location in resources.items():
        found = False
        for candidate in (location, f"{location}.zip"):
            try:
                nltk.data.find(candidate)
                found = True
                break
            except LookupError:
                continue
        if not found:
            nltk.download(resource, download_dir=str(path), quiet=True)


def comment_keywords(texts: pd.Series) -> Counter:
    """Keywords in comment bodies after markup removal, stopword removal, and lemmatization."""
    _ensure_nltk()
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    stops = set(stopwords.words("english")) | _EXTRA_STOPS
    lemmatizer = WordNetLemmatizer()
    counts: Counter = Counter()
    for raw in texts.fillna(""):
        parser = _HtmlText()
        try:
            parser.feed(str(raw))
            parser.close()
            text = " ".join(parser.parts)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", str(raw))
        text = _FENCE.sub(" ", text)
        text = _INLINE.sub(" ", text)
        text = _MD_IMG.sub(" ", text)
        text = _MD_LINK.sub(r" \1 ", text)
        text = _URL.sub(" ", text)
        text = _EMAIL.sub(" ", text)
        text = _NONWORD.sub(" ", text.lower())
        words = [word for word in word_tokenize(text) if word.isalpha() and len(word) >= 3]
        for word, tag in pos_tag(words):
            if tag.startswith("V"):
                lemma = lemmatizer.lemmatize(word, wordnet.VERB)
            elif tag.startswith("N"):
                lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
            else:
                lemma = word
            if len(lemma) >= 3 and lemma.isalpha() and lemma not in stops:
                counts[lemma] += 1
    return counts


def plot_comment_keywords(frame: pd.DataFrame, comments: pd.DataFrame, path: Path) -> None:
    """Side-by-side word clouds of keywords in comments on bot issues and on user issues."""
    from wordcloud import WordCloud

    sns.set_theme(style="white", palette="pastel")
    linked = comments.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    palette = sns.color_palette("pastel", 2)
    # Pastel fills are too light for small words. Keep the same hues, darkened.
    anchors = [tuple(int(channel * 0.62 * 255) for channel in color) for color in palette]
    font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.15))
    rows = []
    for ax, account, anchor, title in (
        (axes[0], "Bot", anchors[0], "(a) Bot"),
        (axes[1], "User", anchors[1], "(b) User"),
    ):
        counts = comment_keywords(linked.loc[linked["author_type"].eq(account), "body"])
        for rank, (word, count) in enumerate(counts.most_common(40), start=1):
            rows.append({"author_type": account, "rank": rank, "keyword": word, "count": int(count)})
        rng = np.random.default_rng(20261019)

        def _color(word, font_size, position, orientation, random_state=None, _anchor=anchor, _rng=rng, **_kwargs):
            del word, font_size, position, orientation, random_state
            scale = float(_rng.uniform(0.72, 1.0))
            return tuple(int(channel * scale) for channel in _anchor)

        cloud = WordCloud(
            width=980,
            height=640,
            background_color="white",
            max_words=55,
            prefer_horizontal=0.92,
            relative_scaling=0.4,
            min_font_size=11,
            margin=6,
            random_state=20261019,
            font_path=str(font),
            color_func=_color,
            collocations=False,
        ).generate_from_frequencies(counts)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
    pd.DataFrame(rows).to_csv(OUT / "comment_keywords.csv", index=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


REACTION_EMOJI = {
    "reactions_plus_one": "👍",
    "reactions_minus_one": "👎",
    "reactions_laugh": "😄",
    "reactions_hooray": "🎉",
    "reactions_confused": "😕",
    "reactions_heart": "❤️",
    "reactions_rocket": "🚀",
    "reactions_eyes": "👀",
}


def _reaction_icon(character: str) -> np.ndarray:
    """Rasterize one Apple color emoji so the PDF can show it as a tick label."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 64)
    image = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((6, 4), character, font=font, embedded_color=True)
    box = image.getbbox()
    if box:
        image = image.crop(box)
    return np.asarray(image)


def plot_reactions(frame: pd.DataFrame, path: Path) -> None:
    """Two ranked horizontal bars, bot and user, with the reaction icon as the tick."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    sns.set_theme(style="white", palette="pastel")
    bot, user = split_groups(frame)
    rows = []
    for column in REACTION_COLUMNS:
        rows.append(
            {
                "reaction": column,
                "Bot": 100 * float(bot[column].gt(0).mean()),
                "User": 100 * float(user[column].gt(0).mean()),
            }
        )
    rates = pd.DataFrame(rows)
    peak = float(rates[["Bot", "User"]].to_numpy().max())
    icons = {column: _reaction_icon(REACTION_EMOJI[column]) for column in REACTION_COLUMNS}
    colors = {"Bot": sns.color_palette("pastel")[0], "User": sns.color_palette("pastel")[1]}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.7), sharex=True)
    groups = {"Bot": bot, "User": user}
    panels = (
        (axes[0], "Bot", "(a) Bot"),
        (axes[1], "User", "(b) User"),
    )
    for ax, account, title in panels:
        ordered = rates.sort_values([account, "reaction"], ascending=[False, True])
        sns.barplot(
            data=ordered,
            y="reaction",
            x=account,
            order=ordered["reaction"].tolist(),
            color=colors[account],
            saturation=1,
            width=0.72,
            orient="y",
            ax=ax,
        )
        for bar in ax.containers:
            for patch in bar:
                patch.set_linewidth(0.7)
                patch.set_edgecolor("white")
        texts = ["<0.1" if value < 0.05 else f"{value:.1f}" for value in ordered[account]]
        ax.bar_label(ax.containers[0], labels=texts, padding=3, fontsize=8, color="#333333")
        tick_names = [tick.get_text() for tick in ax.get_yticklabels()]
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        for y_pos, name in zip(ax.get_yticks(), tick_names):
            icon = OffsetImage(icons[name], zoom=0.30)
            mark = AnnotationBbox(
                icon,
                (0, y_pos),
                xybox=(-8, 0),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
                box_alignment=(1, 0.5),
                pad=0,
            )
            mark.set_clip_on(False)
            ax.add_artist(mark)
        ax.set_xlim(0, max(peak * 1.22, 5))
        ax.set_ylabel("")
        ax.set_xlabel("Percent of issues")
        ax.set_title(title)
        ax.grid(False)
        part = groups[account]
        n_with = int(part["has_reaction"].sum())
        n_total = int(len(part))
        ax.text(
            0.97,
            0.06,
            f"{n_with:,} / {n_total:,}\n{100 * n_with / n_total:.1f}% with a reaction",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#222222",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d5d5d5",
                "linewidth": 0.6,
            },
            zorder=5,
        )
    fig.tight_layout(w_pad=3.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def _sankey_ribbon(ax: plt.Axes, x0: float, x1: float, y0_lo: float, y0_hi: float, y1_lo: float, y1_hi: float, color) -> None:
    t = np.linspace(0, 1, 48)
    s = _smoothstep(t)
    x = x0 + (x1 - x0) * s
    y_hi = y0_hi + (y1_hi - y0_hi) * s
    y_lo = y0_lo + (y1_lo - y0_lo) * s
    ax.fill(np.concatenate([x, x[::-1]]), np.concatenate([y_hi, y_lo[::-1]]), color=color, alpha=0.45, linewidth=0, zorder=1)


def plot_assignee_sankey(frame: pd.DataFrame, assignees: pd.DataFrame, path: Path) -> None:
    """Assignee links from the issue author type to four named logins and a User group."""
    linked = assignees.merge(frame[["issue_id", "author_type"]], on="issue_id", how="inner")
    top = linked["assignee_login"].value_counts().head(4).index.tolist()
    linked["destination"] = linked["assignee_login"].where(linked["assignee_login"].isin(top), "User")
    flows = linked.groupby(["author_type", "destination"]).size().to_dict()

    source_order = ["Bot", "User"]
    source_totals = {name: int(sum(value for (src, _), value in flows.items() if src == name)) for name in source_order}
    dest_totals = {name: int(sum(value for (_, dst), value in flows.items() if dst == name)) for name in set(linked["destination"])}

    def _stack(order: list[str], totals: dict[str, int], gap: float) -> tuple[dict[str, tuple[float, float]], float]:
        height = sum(totals[name] for name in order) + gap * (len(order) - 1)
        cursor = height
        boxes = {}
        for name in order:
            top_edge = cursor
            bottom_edge = cursor - totals[name]
            boxes[name] = (bottom_edge, top_edge)
            cursor = bottom_edge - gap
        return boxes, height

    # Place destinations so a link leaves its source near the matching side.
    source_boxes, total_height = _stack(source_order, source_totals, gap=28)
    dest_names = [name for name in dest_totals if name != "User"] + ["User"]

    def _center(box: tuple[float, float]) -> float:
        return (box[0] + box[1]) / 2

    bary = {}
    for name in dest_names:
        weight = 0.0
        mass = 0.0
        for source in source_order:
            count = flows.get((source, name), 0)
            weight += count * _center(source_boxes[source])
            mass += count
        bary[name] = weight / mass if mass else 0
    dest_order = sorted(dest_names, key=lambda name: (bary[name], dest_totals[name]), reverse=True)
    dest_boxes, _ = _stack(dest_order, dest_totals, gap=28)

    sns.set_theme(style="white", palette="pastel")
    pastel = sns.color_palette("pastel", 6)
    source_color = {"Bot": pastel[0], "User": pastel[1]}
    dest_color = {"User": pastel[1]}
    for name, color in zip([name for name in dest_order if name != "User"], pastel[2:]):
        dest_color[name] = color

    fig, ax = plt.subplots(figsize=(7.16, 5.0))
    x0, x1 = 0.20, 0.68
    node_w = 0.028
    ribbon_x0 = x0 + node_w

    # Ribbons leave each source in destination order, and arrive in source order.
    source_cursor = {name: source_boxes[name][1] for name in source_order}
    dest_cursor = {name: dest_boxes[name][1] for name in dest_order}
    streams = []
    for source in source_order:
        for dest in dest_order:
            count = flows.get((source, dest), 0)
            if count <= 0:
                continue
            y0_hi = source_cursor[source]
            y0_lo = y0_hi - count
            source_cursor[source] = y0_lo
            y1_hi = dest_cursor[dest]
            y1_lo = y1_hi - count
            dest_cursor[dest] = y1_lo
            _sankey_ribbon(ax, ribbon_x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, source_color[source])
            streams.append(
                {
                    "y0_lo": y0_lo,
                    "y0_hi": y0_hi,
                    "y1_lo": y1_lo,
                    "y1_hi": y1_hi,
                    "pct": 100 * count / source_totals[source],
                }
            )

    def _draw_node(box: tuple[float, float], x: float, color) -> None:
        lo, hi = box
        ax.add_patch(
            plt.Rectangle((x, lo), node_w, hi - lo, facecolor=color, edgecolor="#4a4a4a", linewidth=0.6, zorder=3)
        )

    def _stick(x_tip: float, y_tip: float, x_label: float, y_label: float, text: str, ha: str) -> None:
        ax.plot([x_tip, x_label], [y_tip, y_label], color="#6e6e6e", linewidth=0.7, zorder=4, solid_capstyle="round")
        ax.plot(x_tip, y_tip, marker="o", markersize=2.4, color="#4a4a4a", zorder=5)
        ax.text(x_label, y_label, text, ha=ha, va="center", fontsize=8, color="#222222", zorder=6)

    for name in source_order:
        _draw_node(source_boxes[name], x0, source_color[name])
        center = _center(source_boxes[name])
        _stick(x0, center, x0 - 0.015, center, name, "right")

    outside = []
    for name in dest_order:
        _draw_node(dest_boxes[name], x1, dest_color[name])
        lo, hi = dest_boxes[name]
        center = (lo + hi) / 2
        if hi - lo >= total_height * 0.08:
            ax.text(x1 + node_w / 2, center, name, ha="center", va="center", fontsize=8, color="#222222", zorder=4)
        else:
            outside.append((center, name))

    outside.sort()
    min_gap = total_height * 0.075
    placed = [center for center, _name in outside]
    for _pass in range(8):
        for index in range(1, len(placed)):
            if placed[index] - placed[index - 1] < min_gap:
                shift = (min_gap - (placed[index] - placed[index - 1])) / 2
                placed[index - 1] -= shift
                placed[index] += shift
        if placed:
            placed[0] = max(placed[0], min_gap * 0.6)
            placed[-1] = min(placed[-1], total_height - min_gap * 0.4)
    for (center, name), y_label in zip(outside, placed):
        _stick(x1 + node_w, center, x1 + node_w + 0.045, y_label, name, "left")

    def _at(stream: dict, blend: float) -> tuple[float, float]:
        y_hi = stream["y0_hi"] + (stream["y1_hi"] - stream["y0_hi"]) * blend
        y_lo = stream["y0_lo"] + (stream["y1_lo"] - stream["y0_lo"]) * blend
        return (y_lo + y_hi) / 2, (y_hi - y_lo) / 2

    def _buried(stream: dict, blend: float, others: list[dict]) -> bool:
        y_tip, half = _at(stream, blend)
        for other in others:
            other_y, other_half = _at(other, blend)
            if other_half > half * 2 and abs(y_tip - other_y) < other_half + 24:
                return True
        return False

    # Share of that author's links. The stick ends at the middle of the ribbon.
    # A thin ribbon that crosses a wider one is marked where it is still its own band.
    marked = []
    for stream in streams:
        others = [other for other in streams if other is not stream]
        blend = 0.5
        if _buried(stream, blend, others):
            for step in range(1, 22):
                trial = 0.5 - step * 0.02
                if trial < 0.08:
                    break
                if not _buried(stream, trial, others):
                    blend = trial
                    break
        y_tip, half = _at(stream, blend)
        marked.append(
            {
                "x": ribbon_x0 + blend * (x1 - ribbon_x0),
                "y": y_tip,
                "half": half,
                "pct": stream["pct"],
                "wide": half >= total_height * 0.08,
            }
        )

    marked.sort(key=lambda item: item["y"])
    label_gap = total_height * 0.062
    for index, item in enumerate(marked):
        if item["wide"]:
            item["ly"] = item["y"] + item["half"] * 0.38
            item["lx"] = item["x"]
        else:
            item["ly"] = item["y"] + max(item["half"] + label_gap * 0.55, label_gap * 0.7)
            item["lx"] = item["x"] - 0.055 if index % 2 == 0 else item["x"] + 0.055
    for _pass in range(10):
        for index in range(1, len(marked)):
            below, above = marked[index - 1], marked[index]
            if abs(above["lx"] - below["lx"]) > 0.12:
                continue
            if above["ly"] - below["ly"] < label_gap:
                shift = (label_gap - (above["ly"] - below["ly"])) / 2
                below["ly"] -= shift * 0.35
                above["ly"] += shift
    for item in marked:
        x_line = item["lx"] + (0.028 if item["lx"] < item["x"] else -0.028 if item["lx"] > item["x"] else 0)
        y_line = item["ly"]
        if item["wide"]:
            y_line = item["ly"] - label_gap * 0.22
            x_line = item["lx"]
        ax.plot([item["x"], x_line], [item["y"], y_line], color="#6e6e6e", linewidth=0.7, zorder=4, solid_capstyle="round")
        ax.plot(item["x"], item["y"], marker="o", markersize=3.0, color="#4a4a4a", zorder=5)
        ax.text(
            item["lx"],
            item["ly"],
            f"{item['pct']:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="#222222",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-total_height * 0.03, total_height * 1.04)
    ax.set_yticks([])
    ax.set_xticks([x0 + node_w / 2, x1 + node_w / 2], ["Issue author", "Assignee"])
    ax.tick_params(axis="x", length=0, labelsize=9, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
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
    plot_comments(issues, comments, OUT / "comments.pdf")
    plot_comment_keywords(issues, comments, OUT / "comment_keywords.pdf")
    plot_reactions(issues, OUT / "reaction_types.pdf")
    plot_assignee_sankey(issues, assignees, OUT / "assignee_sankey.pdf")

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
