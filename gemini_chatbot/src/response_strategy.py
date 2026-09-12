"""
response_strategy.py
-----------------------
Turns a detected sentiment (+ intensity, + escalation state) into a concrete
tone instruction that gets injected into the model's system instruction for
that turn. This is what actually changes the chatbot's *behavior* based on
sentiment, not just its internal label.

Kept as short, direct behavioral instructions (not essays) since that's what
steers instruction-tuned models most reliably turn-to-turn.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_analysis import SentimentResult


_POSITIVE_GUIDANCE = {
    "mild": "The user's tone is positive. Keep your normal helpful, friendly tone.",
    "moderate": "The user seems pleased. Match their positive energy warmly and reinforce what's going well.",
    "strong": "The user is clearly delighted or very thankful. Warmly acknowledge that, "
              "share in their satisfaction briefly, and keep helping without being over the top.",
}

_NEUTRAL_GUIDANCE = {
    "mild": "The user's tone is neutral. Answer clearly and efficiently.",
    "moderate": "The user's tone is neutral. Answer clearly and efficiently.",
    "strong": "The user's tone is neutral. Answer clearly and efficiently.",
}

_NEGATIVE_GUIDANCE = {
    "mild": "The user seems slightly unhappy or unsure. Be extra clear and reassuring, "
            "and make sure their concern is fully addressed.",
    "moderate": "The user is frustrated or disappointed. Start by briefly acknowledging "
                "their frustration in one short sentence (don't over-apologize), then focus "
                "on solving the actual problem concretely and without excuses.",
    "strong": "The user is clearly upset, angry, or has had a bad experience. Lead with a "
              "brief, genuine acknowledgment of their frustration (e.g. 'That's frustrating, "
              "I'm sorry you're dealing with this' -- but don't over-apologize repeatedly). "
              "Stay calm and solution-focused. Do not be defensive, dismissive, or overly "
              "cheerful -- match the seriousness of their tone while remaining constructive.",
}

_ESCALATION_GUIDANCE = (
    " The user has expressed frustration repeatedly across this conversation. "
    "Proactively and politely offer a concrete next step beyond just continuing to explain "
    "(e.g. offer to escalate to a human representative, or summarize what's been tried so far) "
    "rather than repeating the same kind of answer again."
)


def build_tone_instruction(result: SentimentResult, escalate: bool = False) -> str:
    guidance_table = {
        "positive": _POSITIVE_GUIDANCE,
        "neutral": _NEUTRAL_GUIDANCE,
        "negative": _NEGATIVE_GUIDANCE,
    }[result.label]
    instruction = guidance_table[result.intensity]
    if escalate:
        instruction += _ESCALATION_GUIDANCE
    return instruction


def compose_system_instruction(base_instruction: str, result: SentimentResult, escalate: bool = False) -> str:
    """Combines the user-provided base system instruction (if any) with sentiment-based tone guidance."""
    tone = build_tone_instruction(result, escalate=escalate)
    parts = []
    if base_instruction and base_instruction.strip():
        parts.append(base_instruction.strip())
    parts.append(f"[Customer sentiment guidance for this reply: {tone}]")
    return "\n\n".join(parts)
