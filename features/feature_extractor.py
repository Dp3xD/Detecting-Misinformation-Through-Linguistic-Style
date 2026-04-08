"""
feature_extractor.py
--------------------
Extracts 7 linguistic-style features from text for misinformation detection.

Features:
  1. hedge_word_density       – fraction of words that are hedge words
  2. emotional_arousal_score  – NRC high-arousal emotion frequency
  3. avg_sentence_length      – mean words per sentence (NLTK)
  4. passive_voice_ratio      – fraction of sentences with passive voice (spaCy)
  5. named_entity_density     – fraction of tokens that are named entities (spaCy)
  6. source_citation_presence – fraction of words that are source-citation keywords
  7. capital_letter_rate      – fraction of alphabetic chars that are uppercase

Run independently:
    python features/feature_extractor.py
"""

import subprocess
import numpy as np
import nltk
import spacy

# Download required NLTK tokenizer data (no-op if already present)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Word / phrase lists
# ---------------------------------------------------------------------------

# Hedge words – signal epistemic uncertainty, common in misinformation
HEDGE_WORDS = [
    "allegedly", "reportedly", "claims", "suggests", "appears",
    "seems", "possibly", "perhaps", "maybe", "might", "could",
    "would", "should", "some say", "rumored", "purportedly",
    "supposedly", "apparently", "believed to", "unconfirmed",
    "speculated", "sources say",
]

# Source-citation keywords – credible text tends to name its sources
SOURCE_KEYWORDS = [
    "according to", "cited", "source", "study shows", "research",
    "data shows", "report", "survey", "analysis", "findings",
    "experts say", "scientists", "published", "journal",
]

# Feature names in the same order returned by extract_features()
FEATURE_NAMES = [
    "hedge_word_density",
    "emotional_arousal_score",
    "avg_sentence_length",
    "passive_voice_ratio",
    "named_entity_density",
    "source_citation_presence",
    "capital_letter_rate",
]


# ---------------------------------------------------------------------------
# spaCy loader (downloads model if missing)
# ---------------------------------------------------------------------------

def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load a spaCy model, auto-downloading it if not yet installed."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found – downloading...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def hedge_word_density(text: str) -> float:
    """
    Fraction of tokens that match hedge words/phrases.
    Multi-word phrases (e.g. 'some say') are counted against total word count.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0

    count = 0
    for hedge in HEDGE_WORDS:
        if " " in hedge:            # multi-word: count phrase occurrences
            count += text_lower.count(hedge)
        else:                       # single word: exact token match
            count += words.count(hedge)

    return count / len(words)


def emotional_arousal_score(text: str) -> float:
    """
    Sum of NRC high-arousal emotion frequencies: fear, anger, surprise,
    disgust, anticipation.  Misinformation often exploits emotional language.
    Returns 0.0 if nrclex is unavailable or text is empty.
    """
    if not text.strip():
        return 0.0

    try:
        from nrclex import NRCLex  # lazy import – avoids hard crash at module load

        nrc = NRCLex(text)
        freqs = nrc.affect_frequencies
        high_arousal = ["fear", "anger", "surprise", "disgust", "anticipation"]
        return float(sum(freqs.get(e, 0.0) for e in high_arousal))
    except Exception:
        return 0.0


def average_sentence_length(text: str) -> float:
    """
    Mean number of words per sentence (NLTK sentence tokenizer).
    Extremely short sentences may indicate sensationalist writing.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    return float(np.mean([len(s.split()) for s in sentences]))


def passive_voice_ratio(text: str, nlp) -> float:
    """
    Fraction of sentences containing passive-voice constructions.
    Detected via spaCy dependency labels 'nsubjpass' (passive subject)
    or 'auxpass' (passive auxiliary verb).
    Passive constructions can obscure agency in reporting.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    passive_count = 0
    for sent in sentences:
        for token in sent:
            if token.dep_ in ("nsubjpass", "auxpass"):
                passive_count += 1
                break   # count at most once per sentence

    return passive_count / len(sentences)


def named_entity_density(text: str, nlp) -> float:
    """
    Fraction of tokens that belong to a named entity span.
    Very high NE density may signal fabricated specificity.
    """
    doc = nlp(text)
    if not doc:
        return 0.0
    ne_token_count = sum(1 for token in doc if token.ent_type_)
    return ne_token_count / len(doc)


def source_citation_presence(text: str) -> float:
    """
    Fraction of tokens (or phrase occurrences) matching source-citation keywords.
    Credible journalism tends to attribute claims to named sources.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0

    count = 0
    for keyword in SOURCE_KEYWORDS:
        if " " in keyword:
            count += text_lower.count(keyword)
        else:
            count += words.count(keyword)

    return count / len(words)


def capital_letter_rate(text: str) -> float:
    """
    Fraction of alphabetic characters that are uppercase.
    Excessive capitalisation is a hallmark of sensationalist / clickbait text.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


# ---------------------------------------------------------------------------
# Main extraction interface
# ---------------------------------------------------------------------------

def extract_features(text: str, nlp) -> np.ndarray:
    """
    Compute all 7 features for a single text sample.

    Returns
    -------
    np.ndarray of shape (7,) – matches FEATURE_NAMES ordering.
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(7, dtype=np.float32)

    return np.array([
        hedge_word_density(text),
        emotional_arousal_score(text),
        average_sentence_length(text),
        passive_voice_ratio(text, nlp),
        named_entity_density(text, nlp),
        source_citation_presence(text),
        capital_letter_rate(text),
    ], dtype=np.float32)


def extract_features_batch(
    texts: list,
    nlp=None,
    batch_size: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features for a list of text samples.

    Parameters
    ----------
    texts      : list of str
    nlp        : loaded spaCy model (loaded automatically if None)
    batch_size : progress-reporting granularity (no memory impact here)
    verbose    : print progress

    Returns
    -------
    np.ndarray of shape (n_samples, 7)
    """
    if nlp is None:
        nlp = load_spacy_model()

    n = len(texts)
    feature_matrix = np.zeros((n, 7), dtype=np.float32)

    if verbose:
        print(f"  Extracting features for {n} samples...")

    for i, text in enumerate(texts):
        feature_matrix[i] = extract_features(text, nlp)
        if verbose and batch_size > 0 and (i + 1) % batch_size == 0:
            print(f"    [{i + 1}/{n}]")

    if verbose:
        print(f"  Done. Feature matrix shape: {feature_matrix.shape}")

    return feature_matrix


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nlp = load_spacy_model()

    sample_texts = [
        "Scientists published a study that according to data shows vaccines are safe.",
        "THEY ARE HIDING THE TRUTH!!! Some say the government allegedly controls everything.",
    ]

    print(f"Feature names: {FEATURE_NAMES}\n")
    for t in sample_texts:
        feats = extract_features(t, nlp)
        print(f"Text: {t[:60]}...")
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"  {name:30s}: {val:.4f}")
        print()
