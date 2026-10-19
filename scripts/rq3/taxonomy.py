"""Issue-report taxonomy used for RQ3.

The six maintenance categories are the manual classification scheme of
Herzig, Just, and Zeller (ICSE 2013). QUESTION is added from the GitHub
issue-type scheme used by Ticket Tagger (Kallis, Di Sorbo, Canfora, and
Panichella), because the Herzig scheme has no question class. A pure
question would otherwise be forced into OTHER.
"""

from __future__ import annotations

CATEGORIES = ["BUG", "RFE", "IMPR", "DOC", "REFAC", "QUESTION", "OTHER"]

DEFINITIONS = {
    "BUG": (
        "Corrective maintenance: the report describes defective behavior "
        "(incorrect output, a crash, a regression, or a broken feature) and "
        "the requested work is a semantic code fix."
    ),
    "RFE": (
        "Request for enhancement: the report asks for new functionality that "
        "the project does not already provide."
    ),
    "IMPR": (
        "Perfective improvement: the report asks to improve the performance "
        "or the handling of functionality that already exists, without adding "
        "a new feature and without only restructuring code."
    ),
    "DOC": (
        "Documentation: the requested work is an update to docs, a README, "
        "comments, examples, or other written project documentation."
    ),
    "REFAC": (
        "Refactoring: the report asks to restructure existing code while "
        "keeping the intended behavior."
    ),
    "QUESTION": (
        "Question: the author wants an explanation, usage help, or "
        "clarification, and is not primarily reporting a defect or asking "
        "for a change. If defective behavior is reported, choose BUG even "
        "when the wording is a question."
    ),
    "OTHER": (
        "None of the above. This residual class includes tests, dependency "
        "and CI changes, release or backport tasks, specification changes, "
        "cleanup, and general development tasks."
    ),
}

SOURCES = {
    "BUG": "Herzig et al., ICSE 2013",
    "RFE": "Herzig et al., ICSE 2013",
    "IMPR": "Herzig et al., ICSE 2013",
    "DOC": "Herzig et al., ICSE 2013",
    "REFAC": "Herzig et al., ICSE 2013",
    "QUESTION": "Kallis et al., Ticket Tagger (GitHub issue types)",
    "OTHER": "Herzig et al., ICSE 2013 residual class",
}


def prompt_block() -> str:
    lines = []
    for code in CATEGORIES:
        lines.append(f"{code}: {DEFINITIONS[code]}")
    return "\n".join(lines)
