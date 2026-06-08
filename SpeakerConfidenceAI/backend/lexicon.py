"""
Rule-based confidence lexicon.

Two curated word lists that every piece of business and consumer text tends
to lean on — hedges (low-confidence) and boosters (high-confidence).  The
combined rule-based score is normalised to the 0-1 range and blended with
the ML model score in scorer.py.

All entries are lower-cased.  Multi-word phrases are matched as whole-word
regex with word boundaries on both sides.
"""

LOW_CONFIDENCE_WORDS: list[str] = [
    "maybe", "perhaps", "probably", "possibly", "might", "could",
    "i think", "i guess", "i suppose", "sort of", "kind of",
    "somewhat", "unsure", "not sure", "don't know", "dont know",
    "uncertain", "i believe", "hopefully", "apparently", "seems",
    "seemed", "like i said", "you know", "uhm", "umm", "err",
    "i feel", "it could be", "it seems", "i'm not sure",
    "not entirely", "a bit", "a little", "sort of", "basically",
    "in a way", "i mean", "if i remember", "i forget", "i can't recall",
    "i don't remember", "sort-of", "kind-of", "kinda", "sorta"
]

HIGH_CONFIDENCE_WORDS: list[str] = [
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "surely", "verified", "confirmed", "guaranteed",
    "without doubt", "of course", "no doubt", "100%", "always",
    "never", "proven", "certain", "sure", "for sure",
    "i am confident", "i'm confident", "i know", "i'm certain",
    "without question", "precisely", "unquestionably", "exactly",
    "it is clear", "there is no doubt", "evidently", "assuredly"
]

NEGATIONS = {"not", "no", "never", "none", "nothing", "nobody"}


def _scan_non_overlapping(wordlist, lowered: str):
    """Longest-match-first, non-overlapping scan."""
    import re

    sorted_words = sorted(wordlist, key=lambda w: -len(w))
    used: list[tuple[int, int]] = []
    hits: list[tuple[str, int, int]] = []

    for w in sorted_words:
        pattern = r"\b" + re.escape(w) + r"\b"
        for m in re.finditer(pattern, lowered):
            s, e = m.start(), m.end()
            # skip if overlaps a previously accepted longer match
            if any(not (e <= us or s >= ue) for us, ue in used):
                continue
            hits.append((w, s, e))
            used.append((s, e))
    hits.sort(key=lambda x: x[1])
    return hits, used


def format_markers(text: str):
    """
    Returns a dict with:
        low            – list of (phrase, start, end)
        high           – list of (phrase, start, end)
        lowered        – input lower-cased
        low_count      – number of low-confidence phrases
        high_count     – number of high-confidence phrases
    """
    lowered = text.lower()

    low_hits, low_spans = _scan_non_overlapping(LOW_CONFIDENCE_WORDS, lowered)

    # high confidence phrases should not overlap already-claimed low spans
    high_raw, _ = _scan_non_overlapping(HIGH_CONFIDENCE_WORDS, lowered)
    high_hits = [
        (w, s, e) for (w, s, e) in high_raw
        if not any(not (e <= ls or s >= le) for ls, le in low_spans)
    ]

    return {
        "lowered": lowered,
        "low": low_hits,
        "high": high_hits,
        "low_count": len(low_hits),
        "high_count": len(high_hits),
    }
