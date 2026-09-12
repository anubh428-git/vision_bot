"""
sentiment_analysis.py
------------------------
Detects sentiment (positive/neutral/negative) in a user's message, plus an
intensity level and a 0-100 "satisfaction" score, so the chatbot can adapt
its tone and the app can track customer satisfaction over a conversation.

Primary backend: VADER (Valence Aware Dictionary and sEntiment Reasoner),
via NLTK. VADER is specifically tuned for short, informal text (chat
messages, social media, reviews) -- it handles negation ("not good"),
intensifiers ("very good"), punctuation emphasis ("great!!!"), and emoji/
emoticons well, which is exactly the kind of text a chatbot sees. It's also
fast and needs no GPU.

Fallback backend: a small hand-curated lexicon scorer, used automatically if
the VADER lexicon can't be downloaded (e.g. no internet access to NLTK's
data server) -- so the app still works, just with a cruder signal. Both
backends implement the same `analyze()` interface.
"""
import sys, os
import re
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class SentimentResult:
    label: str              # "positive" | "neutral" | "negative"
    compound: float          # -1.0 (most negative) to +1.0 (most positive)
    intensity: str            # "mild" | "moderate" | "strong"
    satisfaction_score: float  # 0-100, derived from compound (100 = very satisfied)
    backend: str                # "vader" | "lexicon_fallback"


def _label_from_compound(compound: float) -> str:
    if compound >= config.SENTIMENT_POSITIVE_THRESHOLD:
        return "positive"
    if compound <= config.SENTIMENT_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _intensity_from_compound(compound: float) -> str:
    magnitude = abs(compound)
    if magnitude >= 0.6:
        return "strong"
    if magnitude >= 0.3:
        return "moderate"
    return "mild"


class SentimentAnalyzer:
    def __init__(self):
        self._vader = None
        self.backend = self._init_backend()

    def _init_backend(self) -> str:
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                self._vader = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self._vader = SentimentIntensityAnalyzer()
            return "vader"
        except Exception:
            return "lexicon_fallback"

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult("neutral", 0.0, "mild", 50.0, self.backend)

        if self.backend == "vader":
            compound = self._vader.polarity_scores(text)["compound"]
        else:
            compound = self._lexicon_fallback_score(text)

        label = _label_from_compound(compound)
        intensity = _intensity_from_compound(compound)
        satisfaction_score = round((compound + 1) / 2 * 100, 1)  # map [-1,1] -> [0,100]
        return SentimentResult(label, round(compound, 3), intensity, satisfaction_score, self.backend)

    # ------------------------------------------------------------------ #
    # Fallback: small hand-curated lexicon + basic negation handling.
    # Only used if VADER's lexicon genuinely can't be loaded/downloaded.
    # ------------------------------------------------------------------ #
    _POSITIVE_WORDS = {
        "good", "great", "excellent", "awesome", "amazing", "fantastic", "wonderful",
        "love", "loved", "loving", "happy", "pleased", "satisfied", "thanks", "thank",
        "helpful", "perfect", "best", "nice", "appreciate", "appreciated", "glad",
        "impressed", "smooth", "easy", "resolved", "fixed", "works", "working", "fast",
    }
    _NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "hated", "angry",
        "annoyed", "annoying", "frustrated", "frustrating", "upset", "disappointed",
        "disappointing", "broken", "useless", "slow", "confusing", "confused", "issue",
        "problem", "error", "fail", "failed", "failing", "stuck", "waste", "ridiculous",
        "unacceptable", "poor", "wrong", "not working", "doesn't work", "never worked",
    }
    _NEGATIONS = {"not", "no", "never", "n't", "cannot", "can't", "won't", "don't", "doesn't"}

    def _lexicon_fallback_score(self, text: str) -> float:
        tokens = re.findall(r"[a-z']+", text.lower())
        score, count = 0, 0
        negate_next = False
        for tok in tokens:
            if tok in self._NEGATIONS:
                negate_next = True
                continue
            polarity = 0
            if tok in self._POSITIVE_WORDS:
                polarity = 1
            elif tok in self._NEGATIVE_WORDS:
                polarity = -1
            if polarity != 0:
                if negate_next:
                    polarity *= -1
                score += polarity
                count += 1
            negate_next = False
        if count == 0:
            return 0.0
        raw = score / count
        return max(-1.0, min(1.0, raw))


# ------------------------------------------------------------------ #
# Escalation detection: repeated strongly-negative turns in a row.
# ------------------------------------------------------------------ #
def should_escalate(sentiment_history: list, window: int = config.ESCALATION_WINDOW,
                     threshold: int = config.ESCALATION_NEGATIVE_COUNT) -> bool:
    """
    sentiment_history: list of SentimentResult, most recent last.
    Flags escalation if `threshold` or more of the last `window` user turns
    were negative -- catches sustained frustration, not just one sharp word.
    """
    recent = sentiment_history[-window:]
    negative_count = sum(1 for s in recent if s.label == "negative")
    return negative_count >= threshold
