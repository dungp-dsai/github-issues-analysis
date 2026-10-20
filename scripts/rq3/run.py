"""RQ3: How do the topics of bot-authored and user-authored issue bodies differ?

Unsupervised topics use FASTopic (Wu et al., NeurIPS 2024) with the light
encoder sentence-transformers/all-MiniLM-L6-v2. Closed categories use the
taxonomy in scripts/rq3/taxonomy.py, assigned independently by
deepseek-v4-pro and gpt-6-luna.

The sample is drawn separately inside each author_type from issues that have
a non-empty body. Each stratum uses the 95% sample-size formula from
https://www.calculator.net/sample-size-calculator.html with a 5% margin and
a planning proportion of 0.5.

Run from the repository root:

    uv run python scripts/rq3/run.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_mutual_info_score, cohen_kappa_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import (  # noqa: E402
    BOT_COLOR,
    RESULTS,
    SEED,
    USER_COLOR,
    apply_fdr,
    prepare_issues,
    sample_size_for_proportion,
    split_groups,
    test_binary,
    test_categorical,
)
from scripts.rq3.taxonomy import CATEGORIES, DEFINITIONS, SOURCES, prompt_block  # noqa: E402

OUT = RESULTS / "rq3"
CACHE = OUT / "cache"
BODY_CHARS = 2000
BATCH_SIZE = 5
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TOPIC_CANDIDATES = (6, 8, 10, 12)

MODELS = (
    {
        "key": "deepseek-v4-pro",
        "model": "deepseek-v4-pro",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_base_url": "https://api.deepseek.com",
    },
    {
        "key": "gpt-6-luna",
        "model": "gpt-6-luna",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
    },
)


def _style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_sample(issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = issues.loc[~issues["body_empty"]].copy()
    parts = []
    meta = []
    for author in ["Bot", "User"]:
        part = eligible.loc[eligible["author_type"].eq(author)]
        spec = sample_size_for_proportion(len(part))
        drawn = part.sample(n=spec["n"], random_state=SEED)
        parts.append(drawn)
        meta.append({"author_type": author, "n_eligible": int(len(part)), **spec, "n_drawn": int(len(drawn))})
    sample = pd.concat(parts, ignore_index=True)
    return sample, pd.DataFrame(meta)


def clean_body(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 20:
        cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def cache_path(model_key: str, issue_id: str) -> Path:
    digest = hashlib.sha1(issue_id.encode("utf-8")).hexdigest()
    return CACHE / model_key / f"{digest}.json"


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif hasattr(block, "text") and block.text:
                parts.append(block.text)
        if parts:
            return "\n".join(parts)
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def _parse_results(text: str) -> list[dict]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start_obj, start_arr = raw.find("{"), raw.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("response did not contain JSON")
    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        payload = json.loads(raw[start_arr : raw.rfind("]") + 1])
    else:
        payload = json.loads(raw[start_obj : raw.rfind("}") + 1])
    if isinstance(payload, dict):
        for key in ("results", "labels", "issues", "classifications"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if "issue_id" in payload and "category" in payload:
                payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("JSON payload is not a list of labels")
    return payload


def _client(spec: dict):
    from openai import OpenAI

    load_dotenv(ROOT / "scripts" / ".env")
    return OpenAI(
        api_key=os.environ[spec["api_key_env"]],
        base_url=os.environ.get(spec["base_url_env"], spec["default_base_url"]),
    )


_WORKING_KWARGS: dict[str, dict] = {}


def _complete(client, model: str, messages: list[dict]):
    """Call a chat model, adapting token and temperature parameters.

    gpt-6-luna rejects max_tokens and requires max_completion_tokens.
    deepseek-v4-pro accepts max_tokens. Temperature is omitted on a retry
    when a model rejects it. The first working parameter set is reused.
    """
    if model in _WORKING_KWARGS:
        return client.chat.completions.create(model=model, messages=messages, **_WORKING_KWARGS[model])

    attempts = [
        {"max_tokens": 4000, "temperature": 0, "response_format": {"type": "json_object"}},
        {"max_completion_tokens": 4000, "temperature": 0, "response_format": {"type": "json_object"}},
        {"max_completion_tokens": 4000, "response_format": {"type": "json_object"}},
        {"max_tokens": 4000, "temperature": 0},
        {"max_completion_tokens": 4000},
    ]
    last_error = None
    for extra in attempts:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **extra)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            unsupported = any(
                token in message
                for token in (
                    "unsupported",
                    "not supported",
                    "response_format",
                    "temperature",
                    "max_tokens",
                    "max_completion_tokens",
                )
            )
            if unsupported:
                continue
            raise
        else:
            _WORKING_KWARGS[model] = extra
            return response
    raise last_error


def _classify_batch(client, model: str, batch: pd.DataFrame) -> dict[str, str]:
    alias_to_id = {}
    blocks = []
    for index, row in enumerate(batch.itertuples(index=False)):
        alias = f"S{index:03d}"
        alias_to_id[alias] = row.issue_id
        body = row.body_text.strip()
        if len(body) > BODY_CHARS:
            body = body[:BODY_CHARS] + "\n[truncated]"
        blocks.append(f"issue_id: {alias}\ntitle: {row.title_text.strip()}\nbody:\n{body}")
    messages = [
        {
            "role": "system",
            "content": (
                "You classify GitHub issue reports into exactly one taxonomy category. "
                "Return only a JSON object of the form "
                '{"results":[{"issue_id":"S000","category":"BUG"}]}. '
                f"category must be one of: {', '.join(CATEGORIES)}. "
                "Copy each issue_id exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Taxonomy:\n"
                f"{prompt_block()}\n\n"
                "Classify every issue. Use the issue_id value exactly as given.\n\n"
                + "\n\n".join(blocks)
            ),
        },
    ]
    last_error = None
    last_text = ""
    for attempt in range(5):
        try:
            response = _complete(client, model, messages)
            last_text = _message_text(response.choices[0].message)
            parsed = _parse_results(last_text)
            found = {}
            for item in parsed:
                alias = str(item.get("issue_id", "")).strip()
                category = str(item.get("category", "")).strip().upper()
                issue_id = alias_to_id.get(alias)
                if issue_id and category in CATEGORIES:
                    found[issue_id] = category
            missing = [issue_id for issue_id in alias_to_id.values() if issue_id not in found]
            if missing:
                raise ValueError(f"missing or invalid labels for {missing}")
            return found
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"{model} failed on a batch: {last_error}\nLast text: {last_text[:800]}")


def label_with_model(sample: pd.DataFrame, spec: dict) -> pd.Series:
    folder = CACHE / spec["key"]
    folder.mkdir(parents=True, exist_ok=True)
    labels = {}
    pending = []
    for row in sample.itertuples(index=False):
        path = cache_path(spec["key"], row.issue_id)
        if path.exists():
            labels[row.issue_id] = json.loads(path.read_text())["category"]
        else:
            pending.append(row.issue_id)
    print(f"{spec['key']}: {len(labels)} cached, {len(pending)} to classify", flush=True)
    if not pending:
        return sample["issue_id"].map(labels)

    client = _client(spec)
    probe = sample.loc[sample["issue_id"].eq(pending[0])].iloc[0]
    probe_frame = pd.DataFrame([probe])
    probed = _classify_batch(client, spec["model"], probe_frame)
    for issue_id, category in probed.items():
        cache_path(spec["key"], issue_id).write_text(json.dumps({"issue_id": issue_id, "category": category}))
        labels[issue_id] = category
    pending = pending[1:]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    def run_batch(issue_ids: list[str]) -> dict[str, str]:
        local_client = _client(spec)
        batch = sample.loc[sample["issue_id"].isin(issue_ids)]
        # Keep the requested order so a truncated reply is easier to debug.
        batch = batch.set_index("issue_id").loc[issue_ids].reset_index()
        return _classify_batch(local_client, spec["model"], batch)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_batch, batch): batch for batch in batches}
        done = 0
        for future in as_completed(futures):
            found = future.result()
            for issue_id, category in found.items():
                cache_path(spec["key"], issue_id).write_text(
                    json.dumps({"issue_id": issue_id, "category": category})
                )
                labels[issue_id] = category
            done += 1
            print(f"{spec['key']}: finished {done}/{len(batches)} batches", flush=True)
    missing = [issue_id for issue_id in sample["issue_id"] if issue_id not in labels]
    if missing:
        raise RuntimeError(f"{spec['key']} left {len(missing)} issues unlabeled")
    return sample["issue_id"].map(labels)


def _tokenize(texts: list[str]) -> list[list[str]]:
    tokenized = []
    for text in texts:
        words = re.findall(r"[a-z]{3,}", text.lower())
        tokenized.append([word for word in words if word not in ENGLISH_STOP_WORDS])
    return tokenized


def _build_fastopic(num_topics: int):
    import inspect

    from fastopic import FASTopic

    parameters = inspect.signature(FASTopic.__init__).parameters
    kwargs = {"num_topics": num_topics}
    if "device" in parameters:
        kwargs["device"] = "cpu"
    if "doc_embed_model" in parameters:
        # Default short name resolves to the same Hugging Face MiniLM checkpoint.
        kwargs["doc_embed_model"] = ENCODER
    if "verbose" in parameters:
        kwargs["verbose"] = False
    return FASTopic(**kwargs)


def _as_word_lists(topic_words) -> list[list[str]]:
    lists = []
    for topic in topic_words:
        if isinstance(topic, str):
            lists.append(topic.split())
        else:
            lists.append([str(word) for word in list(topic)])
    return lists


def fit_topics(docs: list[str]) -> tuple[int, list[list[str]], np.ndarray, pd.DataFrame]:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    tokenized = _tokenize(docs)
    dictionary = Dictionary(tokenized)
    scores = []
    fitted = {}
    for k in TOPIC_CANDIDATES:
        print(f"Fitting FASTopic with {k} topics", flush=True)
        model = _build_fastopic(k)
        topic_words, doc_topic = model.fit_transform(docs)
        words = _as_word_lists(topic_words)
        coherence = CoherenceModel(
            topics=[topic[:10] for topic in words],
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
        ).get_coherence()
        scores.append({"k": k, "c_npmi": float(coherence)})
        fitted[k] = (words, np.asarray(doc_topic))
        print(f"  c_npmi={coherence:.4f}", flush=True)
    score_frame = pd.DataFrame(scores).sort_values(["c_npmi", "k"], ascending=[False, True])
    best_k = int(score_frame.iloc[0]["k"])
    words, doc_topic = fitted[best_k]
    return best_k, words, doc_topic, score_frame


def _row(family: str, outcome: str, tested: dict) -> dict:
    tested = {k: v for k, v in tested.items() if k != "table"}
    tested["family"] = family
    tested["outcome"] = outcome
    return tested


def _proportion_tests(frame: pd.DataFrame, column: str, family: str) -> list[dict]:
    rows = []
    bot, user = split_groups(frame)
    for category in sorted(frame[column].dropna().unique()):
        rows.append(
            _row(
                family,
                f"{column}::{category}",
                test_binary(
                    int(bot[column].eq(category).sum()),
                    len(bot),
                    int(user[column].eq(category).sum()),
                    len(user),
                ),
            )
        )
    return rows


def kappa_with_ci(left: pd.Series, right: pd.Series, n_boot: int = 1000) -> dict:
    y1 = left.to_numpy()
    y2 = right.to_numpy()
    point = float(cohen_kappa_score(y1, y2))
    rng = np.random.default_rng(SEED)
    draws = []
    n = len(y1)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y1[idx])) < 2 or len(set(y2[idx])) < 2:
            continue
        draws.append(cohen_kappa_score(y1[idx], y2[idx]))
    low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
    return {
        "cohen_kappa": point,
        "kappa_ci_low": float(low),
        "kappa_ci_high": float(high),
        "percent_agreement": float(np.mean(y1 == y2)),
        "n": int(n),
    }


def plot_grouped_prevalence(frame: pd.DataFrame, column: str, order: list[str], title: str, path: Path) -> None:
    rates = pd.crosstab(frame["author_type"], frame[column], normalize="index").reindex(["Bot", "User"])
    rates = rates.reindex(columns=order).fillna(0) * 100
    fig_w = max(6.2, 0.55 * len(order) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.3))
    x = np.arange(len(order))
    ax.bar(x - 0.18, rates.loc["Bot"], width=0.36, color=BOT_COLOR, label="Bot")
    ax.bar(x + 0.18, rates.loc["User"], width=0.36, color=USER_COLOR, label="Human")
    ax.set_xticks(x, order, rotation=30, ha="right")
    ax.set_ylabel("Percent of sampled issues")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, xlabel: str, ylabel: str) -> None:
    fig_w = max(5.5, 0.55 * matrix.shape[1] + 2.5)
    fig_h = max(3.6, 0.38 * matrix.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=max(matrix.to_numpy().max(), 0.01))
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    issues = prepare_issues()
    sample, sample_meta = draw_sample(issues)
    sample_meta.to_csv(OUT / "sample_size.csv", index=False)
    sample = sample.copy()
    sample["body_clean"] = sample["body_text"].map(clean_body)
    sample["body_truncated_for_llm"] = sample["body_text"].str.len().gt(BODY_CHARS)

    print(sample_meta.to_string(index=False), flush=True)
    for spec in MODELS:
        sample[spec["key"]] = label_with_model(sample, spec).to_numpy()

    sample["models_agree"] = sample["deepseek-v4-pro"].eq(sample["gpt-6-luna"])
    sample["consensus"] = np.where(sample["models_agree"], sample["deepseek-v4-pro"], pd.NA)

    docs = sample["body_clean"].tolist()
    best_k, topic_words, doc_topic, coherence = fit_topics(docs)
    coherence.to_csv(OUT / "topic_coherence.csv", index=False)
    sample["topic"] = doc_topic.argmax(axis=1).astype(int)
    sample["topic_probability"] = doc_topic.max(axis=1)

    topic_rows = []
    for topic_id, words in enumerate(topic_words):
        topic_rows.append(
            {
                "topic": topic_id,
                "top_words": " ".join(words[:10]),
                "short_label": " ".join(words[:4]),
                "n_issues": int((sample["topic"] == topic_id).sum()),
            }
        )
    topics = pd.DataFrame(topic_rows)
    sample["topic_label"] = sample["topic"].map(topics.set_index("topic")["short_label"])

    examples = []
    for topic_id in range(best_k):
        part = sample.loc[sample["topic"].eq(topic_id)].nlargest(3, "topic_probability")
        for _, row in part.iterrows():
            examples.append(
                {
                    "topic": topic_id,
                    "top_words": topics.loc[topics["topic"].eq(topic_id), "top_words"].iloc[0],
                    "issue_id": row["issue_id"],
                    "author_type": row["author_type"],
                    "title": row["title_text"],
                    "topic_probability": row["topic_probability"],
                    "deepseek-v4-pro": row["deepseek-v4-pro"],
                    "gpt-6-luna": row["gpt-6-luna"],
                }
            )
    pd.DataFrame(examples).to_csv(OUT / "topic_examples.csv", index=False)
    topics.to_csv(OUT / "topics.csv", index=False)

    rows = []
    for column, family in [
        ("topic_label", "author_contrast"),
        ("deepseek-v4-pro", "author_contrast"),
        ("gpt-6-luna", "author_contrast"),
    ]:
        tested = test_categorical(
            sample.loc[sample["author_type"].eq("Bot"), column],
            sample.loc[sample["author_type"].eq("User"), column],
        )
        rows.append(_row(family, column, tested))
    agreed = sample.loc[sample["models_agree"]]
    tested = test_categorical(
        agreed.loc[agreed["author_type"].eq("Bot"), "consensus"],
        agreed.loc[agreed["author_type"].eq("User"), "consensus"],
    )
    rows.append(_row("author_contrast", "consensus_where_models_agree", tested))
    bot, user = split_groups(sample)
    rows.append(
        _row(
            "author_contrast",
            "llm_disagreement",
            test_binary(
                int((~bot["models_agree"]).sum()),
                len(bot),
                int((~user["models_agree"]).sum()),
                len(user),
            ),
        )
    )
    rows.extend(_proportion_tests(sample, "deepseek-v4-pro", "posthoc_deepseek"))
    rows.extend(_proportion_tests(sample, "gpt-6-luna", "posthoc_gpt"))
    rows.extend(_proportion_tests(agreed, "consensus", "posthoc_consensus"))
    rows.extend(_proportion_tests(sample, "topic_label", "posthoc_topic"))
    tested = apply_fdr(pd.DataFrame(rows))
    tested.to_csv(OUT / "tests.csv", index=False)

    agreement = kappa_with_ci(sample["deepseek-v4-pro"], sample["gpt-6-luna"])
    agreement["ami_topic_deepseek"] = float(
        adjusted_mutual_info_score(sample["topic"], sample["deepseek-v4-pro"])
    )
    agreement["ami_topic_gpt"] = float(adjusted_mutual_info_score(sample["topic"], sample["gpt-6-luna"]))
    agreement["nmi_topic_deepseek"] = float(
        normalized_mutual_info_score(sample["topic"], sample["deepseek-v4-pro"])
    )
    agreement["nmi_topic_gpt"] = float(normalized_mutual_info_score(sample["topic"], sample["gpt-6-luna"]))
    agreed_topics = agreed["topic"]
    agreed_labels = agreed["consensus"].astype(str)
    agreement["ami_topic_consensus"] = (
        float(adjusted_mutual_info_score(agreed_topics, agreed_labels)) if len(agreed) else float("nan")
    )
    agreement["n_topics"] = best_k
    agreement["encoder"] = ENCODER
    pd.Series(agreement).to_csv(OUT / "alignment_summary.csv", header=["value"])

    confusion = pd.crosstab(sample["deepseek-v4-pro"], sample["gpt-6-luna"]).reindex(index=CATEGORIES, columns=CATEGORIES).fillna(0)
    confusion_rate = confusion.div(confusion.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    confusion.to_csv(OUT / "llm_confusion_counts.csv")
    confusion_rate.to_csv(OUT / "llm_confusion_row_rate.csv")

    sample["topic_code"] = sample["topic"].map(lambda topic_id: f"T{int(topic_id)}")
    topic_order = [f"T{topic_id}" for topic_id in range(best_k)]
    consensus_counts = pd.crosstab(sample["topic_code"], sample["consensus"].fillna("DISAGREE"))
    consensus_counts = consensus_counts.reindex(index=topic_order).fillna(0)
    category_cols = [code for code in CATEGORIES if code in consensus_counts.columns]
    if "DISAGREE" in consensus_counts.columns:
        category_cols = category_cols + ["DISAGREE"]
    consensus_counts = consensus_counts.reindex(columns=category_cols).fillna(0)
    topic_rate = consensus_counts.div(consensus_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    topic_rate.to_csv(OUT / "topic_by_consensus_rate.csv")

    taxonomy = pd.DataFrame(
        [{"category": code, "definition": DEFINITIONS[code], "source": SOURCES[code]} for code in CATEGORIES]
    )
    taxonomy.to_csv(OUT / "taxonomy.csv", index=False)

    keep = [
        "issue_id",
        "repo_name",
        "author_type",
        "author_login",
        "agent_type",
        "title_text",
        "body_truncated_for_llm",
        "topic",
        "topic_label",
        "topic_probability",
        "deepseek-v4-pro",
        "gpt-6-luna",
        "models_agree",
        "consensus",
    ]
    sample[keep].to_csv(OUT / "sample_labels.csv", index=False)

    plot_grouped_prevalence(
        sample,
        "topic_code",
        topic_order,
        "FASTopic topics by author account type",
        OUT / "topic_prevalence.pdf",
    )
    plot_grouped_prevalence(
        sample,
        "deepseek-v4-pro",
        CATEGORIES,
        "deepseek-v4-pro categories by author account type",
        OUT / "taxonomy_deepseek.pdf",
    )
    plot_grouped_prevalence(
        sample,
        "gpt-6-luna",
        CATEGORIES,
        "gpt-6-luna categories by author account type",
        OUT / "taxonomy_gpt.pdf",
    )
    plot_heatmap(
        confusion_rate,
        "Row-normalized agreement of the two LLM labels",
        OUT / "llm_agreement.pdf",
        "gpt-6-luna",
        "deepseek-v4-pro",
    )
    plot_heatmap(
        topic_rate,
        "Topic alignment with the two-model label",
        OUT / "topic_taxonomy_alignment.pdf",
        "Consensus label, or DISAGREE",
        "FASTopic topic",
    )

    print(json.dumps(agreement, indent=2))
    print(tested.loc[tested["family"].eq("author_contrast"), ["outcome", "statistic", "p_value", "p_fdr", "effect"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
