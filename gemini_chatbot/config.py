"""config.py -- central configuration for the Gemini chatbot with sentiment analysis."""

# ---------------------------------------------------------------------------
# Sentiment thresholds (VADER compound score, -1.0 to +1.0)
# ---------------------------------------------------------------------------
SENTIMENT_POSITIVE_THRESHOLD = 0.05   # compound >= this -> "positive" (VADER's own recommended cutoff)
SENTIMENT_NEGATIVE_THRESHOLD = -0.05   # compound <= this -> "negative"

# ---------------------------------------------------------------------------
# Escalation detection (sustained negative sentiment -> offer a human agent)
# ---------------------------------------------------------------------------
ESCALATION_WINDOW = 3            # look at the last N user turns
ESCALATION_NEGATIVE_COUNT = 2     # escalate if at least this many of them were negative
