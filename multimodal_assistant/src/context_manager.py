"""
context_manager.py
-------------------
Maintains conversational state across turns: message history, which images
are "in play" and in what order, and resolution of references like "it",
"the second image", or "that chart" to a concrete tracked image or prior
claim. This is what lets the assistant handle follow-ups instead of treating
every message as a fresh, context-free query.
"""
import sys, os
import re
from typing import List, Optional, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.schemas import Turn, ImageEvidence, new_id


ORDINAL_WORDS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "last": -1, "latest": -1, "most recent": -1,
}


class ConversationContext:
    def __init__(self):
        self.turns: List[Turn] = []
        self.image_evidence: Dict[str, ImageEvidence] = {}   # image_id -> evidence
        self.image_order: List[str] = []                       # chronological order of images seen

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add_turn(self, role: str, text: str, image_ids: Optional[List[str]] = None, trace=None) -> Turn:
        turn = Turn(turn_id=new_id("turn"), role=role, text=text, image_ids=image_ids or [], trace=trace)
        self.turns.append(turn)
        return turn

    def register_image(self, image_id: str, evidence: ImageEvidence):
        self.image_evidence[image_id] = evidence
        if image_id not in self.image_order:
            self.image_order.append(image_id)
        # Bound memory: keep only the most recent N tracked images.
        if len(self.image_order) > config.MAX_TRACKED_IMAGES:
            oldest = self.image_order.pop(0)
            self.image_evidence.pop(oldest, None)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def recent_history(self, max_turns: int = config.MAX_HISTORY_TURNS) -> List[Turn]:
        return self.turns[-max_turns:]

    def history_as_prompt(self, max_turns: int = config.MAX_HISTORY_TURNS) -> str:
        lines = []
        for t in self.recent_history(max_turns):
            role = "User" if t.role == "user" else "Assistant"
            img_note = f" [with image(s): {', '.join(t.image_ids)}]" if t.image_ids else ""
            lines.append(f"{role}{img_note}: {t.text}")
        return "\n".join(lines)

    def last_user_image_ids(self) -> List[str]:
        for t in reversed(self.turns):
            if t.role == "user" and t.image_ids:
                return t.image_ids
        return []

    def most_recent_image_id(self) -> Optional[str]:
        return self.image_order[-1] if self.image_order else None

    # ------------------------------------------------------------------ #
    # Reference resolution
    # ------------------------------------------------------------------ #
    def contains_reference_term(self, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in config.REFERENCE_TERMS)

    def resolve_ordinal_reference(self, text: str) -> Optional[str]:
        """Resolve phrases like 'the second image' / 'the last one' to an image_id."""
        lowered = text.lower()
        for phrase, idx in ORDINAL_WORDS.items():
            if phrase in lowered:
                if not self.image_order:
                    return None
                try:
                    return self.image_order[idx]
                except IndexError:
                    return None
        return None

    def resolve_reference(self, text: str, newly_uploaded_image_ids: List[str]) -> Optional[str]:
        """
        Best-effort resolution of a pronoun/deictic reference in `text` to a
        concrete image_id, using (in priority order):
          1. An image uploaded in THIS turn (most unambiguous case)
          2. An explicit ordinal phrase ("the second image", "the last one")
          3. The single most recently discussed image, if exactly one is tracked
        Returns None if resolution is genuinely ambiguous -- callers should
        treat that as a signal to ask for clarification rather than guess.
        """
        if newly_uploaded_image_ids:
            return newly_uploaded_image_ids[-1]

        ordinal_match = self.resolve_ordinal_reference(text)
        if ordinal_match:
            return ordinal_match

        if len(self.image_order) == 1:
            return self.image_order[0]

        return None  # ambiguous: multiple candidate images, no ordinal/explicit cue
