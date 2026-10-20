# Bot-Authored versus Human-Authored GitHub Issues: Characteristics, Engagement, and Topics

Replication package for the empirical comparison of bot-authored and human-authored GitHub issues.

## Overview

GitHub issues are opened by both human accounts and automated agents. This study asks whether those issues follow the same patterns. We compare 2,297 bot-authored issues with 2,685 human-authored issues, after removing 18 pull requests from a 5,000-row export. Group membership is GitHub's `author_type` (`Bot` or `User`). `agent_type` is an automatic guess and is not used to assign groups.

- **RQ1:** What characterizes bot-authored issues compared with human-authored issues?
- **RQ2:** How do engagement patterns differ between bot-authored and human-authored issues?
- **RQ3:** How do issue topics differ between bots and humans?

RQ1 compares lifecycle and resolution, issue metadata, and issue content. RQ2 compares comments, discussion keywords, emoji reactions, and assignees. RQ3 classifies a stratified sample of 330 bot-authored and 335 human-authored issues with non-empty bodies. DeepSeek-V4-Pro and GPT-6-Luna each assign one of seven categories (BUG, RFE, IMPR, DOC, REFAC, QUESTION, OTHER) from the title and the first 2,000 characters of the body.

## Project structure

```
.
├── notebooks/
│   └── eda.ipynb                 # Exploratory checks of the four CSVs
├── scripts/
│   ├── load_data.py              # Load the four CSVs from data/
│   ├── common.py                 # Shared preparation and tests
│   ├── dataset_schema.pdf        # Column definitions
│   ├── rq1/run.py                # RQ1 analysis and figures
│   ├── rq2/run.py                # RQ2 analysis and figures
│   └── rq3/
│       ├── run.py                # RQ3 sample, LLM labels, and figures
│       └── taxonomy.py           # Category definitions and the classification prompt
├── results/
│   ├── rq1/                      # RQ1 tables and figures
│   ├── rq2/                      # RQ2 tables and figures
│   └── rq3/                      # RQ3 sample labels, sample size, and agreement figure
├── data/                         # Input CSVs (not included; see Data requirements)
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Dependency lock file
└── README.md
```

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv)
- The four CSV files described below
- API keys for DeepSeek and OpenAI, only if you rerun RQ3

## Setup

### 1. Install uv

On macOS:

```bash
brew install uv
```

On Linux or Windows, see https://github.com/astral-sh/uv.

### 2. Install dependencies

From the repository root:

```bash
uv sync
```

`uv run` uses the project environment, so a separate activation step is optional. To activate it yourself:

```bash
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Configure API keys for RQ3

RQ1 and RQ2 do not call an API. RQ3 does. Create `scripts/.env`:

```
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
```

Optional overrides:

```
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_BASE_URL=https://api.openai.com/v1
```

Do not commit `scripts/.env`. Responses already returned by the models are cached under `results/rq3/cache/`.

## Data requirements

The analysis reads four CSV files from `data/`. They are not included in this repository. Column meanings are in `scripts/dataset_schema.pdf`.

```
data/
├── issues.csv
├── issue_comments.csv
├── issue_assignees.csv
└── issue_labels.csv
```

| File | Role |
| --- | --- |
| `issues.csv` | One row per issue. `author_type` is `Bot` or `User`. |
| `issue_comments.csv` | Exported comment bodies, joined to issues by `issue_id`. |
| `issue_assignees.csv` | Assignee links, joined to issues by `issue_id`. |
| `issue_labels.csv` | Label names, joined to issues by `issue_id`. |

Place the files in `data/` before running a script. Each script drops rows where `is_pull_request` is true (18 rows), leaving 4,982 issues: 2,297 bot-authored and 2,685 human-authored.

## Running the analysis

Run every command from the repository root.

### RQ1: Issue characteristics

Compares closure, resolution time, closure reasons, labels, milestones, archive events, and the structure of titles and bodies. Tests are a two-proportion Z-test, a chi-square test, or a Mann–Whitney U test, at α = 0.05.

```bash
uv run python scripts/rq1/run.py
```

Writes tables and figures to `results/rq1/`, including `lifecycle_resolution.pdf`, `metadata_counts.pdf`, and `metadata_labels.pdf`.

### RQ2: Engagement

Compares whether an issue has a comment, reaction, or assignee; comment length and keywords; reaction composition; and assignee flow. Keyword counts strip HTML and Markdown, code spans, URLs, and email addresses, then drop English stopwords and lemmatize nouns and verbs. The first run downloads the NLTK data it needs into `.cache/nltk`.

```bash
uv run python scripts/rq2/run.py
```

Writes tables and figures to `results/rq2/`, including `comments.pdf`, `comment_keywords.pdf`, `reaction_types.pdf`, and `assignee_sankey.pdf`.

### RQ3: Issue topics

Draws a stratified sample of non-empty bodies with the Calculator.net sample-size settings used in the paper (95% confidence, 5% margin of error, planning proportion 0.5): 330 bot-authored issues and 335 human-authored issues, seed `20261019`. Both models label every sampled issue from `scripts/rq3/taxonomy.py`.

```bash
uv run python scripts/rq3/run.py
```

This calls the two model APIs and also fits a FASTopic model. The paper reports the two LLM labelings and their agreement, not the topic model. The agreement figure used in the paper is `results/rq3/llm_agreement.pdf`. Per-issue labels are in `results/rq3/sample_labels.csv`.

### Exploratory notebook

`notebooks/eda.ipynb` is an exploratory pass over the same CSVs. It is not required to reproduce the paper figures. Open it in an editor that can run notebooks. The project environment includes `ipykernel`.

## Results

Precomputed tables and figures are already in `results/`. Rerunning RQ1 or RQ2 replaces the files in `results/rq1/` and `results/rq2/`. Rerunning RQ3 calls the model APIs unless a cached response exists for that issue.

| Question | Main outputs |
| --- | --- |
| RQ1 | `results/rq1/lifecycle_resolution.pdf`, `metadata_counts.pdf`, `metadata_labels.pdf`, `tests.csv` |
| RQ2 | `results/rq2/comments.pdf`, `comment_keywords.pdf`, `reaction_types.pdf`, `assignee_sankey.pdf` |
| RQ3 | `results/rq3/sample_labels.csv`, `sample_size.csv`, `llm_agreement.pdf` |

In the figures, the `User` account type is labeled Human.

## Dependencies

Installed from `pyproject.toml` and locked in `uv.lock`:

- pandas and numpy for the tables
- scipy for the Z, chi-square, and Mann–Whitney tests
- scikit-learn for agreement measures in RQ3
- matplotlib and seaborn for the figures
- wordcloud and nltk for the RQ2 keyword clouds
- openai and python-dotenv for the two RQ3 classifiers
- fastopic for the topic-model step inside `scripts/rq3/run.py`
