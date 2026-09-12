"""
ambiguity_handler.py
---------------------
Decides whether the assistant has enough grounding to answer confidently, or
whether it should ask a clarifying question instead of guessing. This is a
deliberate design choice: a naive single-model pipeline will happily answer
"what's wrong with it?" even when "it" could refer to either of two uploaded
images, silently picking one and possibly answering the wrong question. This
module makes that decision explicit and inspectable.
"""
import sys, os
import re
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.schemas import AmbiguityAssessment, AmbiguityType
from src.context_manager import ConversationContext


def _needs_image_but_has_none(text: str, has_any_tracked_image: bool) -> bool:
    if has_any_tracked_image:
        return False
    lowered = text.lower()
    image_dependent_phrases = [
        "in the image", "in this picture", "in the photo", "what is this",
        "what's in", "describe the image", "what do you see", "read the text in",
        "what does it say", "identify", "in the picture",
    ]
    if any(p in lowered for p in image_dependent_phrases):
        return True
    # Generic catch-all: any question mentioning image/picture/photo with no
    # image tracked at all is, by definition, missing the image it needs.
    return bool(re.search(r"\b(image|picture|photo)\b", lowered))


def _has_vague_quantifier(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(q)}\b", lowered) for q in config.VAGUE_QUANTIFIERS)


def assess(
    text: str,
    context: ConversationContext,
    newly_uploaded_image_ids: List[str],
) -> AmbiguityAssessment:
    """
    Runs a sequence of ambiguity checks and returns the first (highest-priority)
    issue found, or a NONE assessment with high confidence if the turn looks
    answerable as-is.
    """
    has_tracked_images = bool(context.image_order) or bool(newly_uploaded_image_ids)

    # 1. Question clearly needs an image, but none is available at all.
    if _needs_image_but_has_none(text, has_tracked_images):
        return AmbiguityAssessment(
            ambiguity_type=AmbiguityType.MISSING_IMAGE,
            confidence=0.1,
            clarifying_question=(
                "I don't have an image to look at yet -- could you upload the "
                "image you'd like me to analyze?"
            ),
            explanation="Query language implies visual analysis, but no image has been shared.",
        )

    # 2. Explicit ordinal reference ("the first image", "the last one") is
    #    unambiguous by construction -- resolve it directly, independent of
    #    the generic pronoun list, so it doesn't fall through to "most
    #    recent image" defaulting below.
    ordinal_resolved = context.resolve_ordinal_reference(text)
    if ordinal_resolved:
        return AmbiguityAssessment(
            ambiguity_type=AmbiguityType.NONE,
            confidence=0.9,
            resolved_image_id=ordinal_resolved,
            explanation=f"Ordinal reference resolved directly to image '{ordinal_resolved}'.",
        )

    # 3. Query uses a reference term ("it", "this", "the second one") --
    #    try to resolve it; if resolution fails because multiple images are
    #    in play, that's a genuine ambiguity worth asking about.
    if context.contains_reference_term(text):
        resolved = context.resolve_reference(text, newly_uploaded_image_ids)
        if resolved is None and len(context.image_order) > 1:
            candidates = ", ".join(context.image_order)
            return AmbiguityAssessment(
                ambiguity_type=AmbiguityType.MULTIPLE_CANDIDATE_IMAGES,
                confidence=0.3,
                clarifying_question=(
                    f"You've shared {len(context.image_order)} images with me so far -- "
                    "which one are you asking about? You can say e.g. 'the first one' "
                    "or 'the most recent one'."
                ),
                explanation=f"Reference term used with {len(context.image_order)} candidate images tracked: {candidates}.",
            )
        elif resolved is None:
            return AmbiguityAssessment(
                ambiguity_type=AmbiguityType.UNRESOLVED_REFERENCE,
                confidence=0.2,
                clarifying_question=(
                    "I want to make sure I answer about the right thing -- could you "
                    "clarify what 'it' / 'this' refers to?"
                ),
                explanation="Reference term found but no image or prior subject to resolve it to.",
            )
        else:
            return AmbiguityAssessment(
                ambiguity_type=AmbiguityType.NONE,
                confidence=0.85,
                resolved_image_id=resolved,
                explanation=f"Reference resolved to image '{resolved}'.",
            )

    # 3. Vague scope ("some of the objects", "a few things") without an image
    #    to ground the vagueness in -- worth a light clarification, but this
    #    is lower-severity than an unresolved reference, so confidence is
    #    borderline rather than very low.
    if _has_vague_quantifier(text) and not has_tracked_images:
        return AmbiguityAssessment(
            ambiguity_type=AmbiguityType.VAGUE_SCOPE,
            confidence=0.5,
            clarifying_question=(
                "Could you be a bit more specific about what you'd like me to focus on?"
            ),
            explanation="Vague quantifier used with no image/context to anchor scope.",
        )

    # 4. Looks answerable.
    resolved_image = newly_uploaded_image_ids[-1] if newly_uploaded_image_ids else context.most_recent_image_id()
    return AmbiguityAssessment(
        ambiguity_type=AmbiguityType.NONE,
        confidence=0.9,
        resolved_image_id=resolved_image,
        explanation="No ambiguity signals detected.",
    )


def should_clarify(assessment: AmbiguityAssessment) -> bool:
    return (
        assessment.ambiguity_type != AmbiguityType.NONE
        and assessment.confidence < config.AMBIGUITY_CONFIDENCE_THRESHOLD
    )
