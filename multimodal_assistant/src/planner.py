"""
planner.py
----------
This is the piece that makes the assistant "intelligent decision-making
rather than simple model inference": instead of always doing
image -> caption -> LLM -> answer, the planner inspects the current turn
(text, images present, ambiguity assessment, conversation state) and decides
which actions are actually needed, in what order. A follow-up question about
an already-analyzed image skips re-running vision models; a question that
needs outside knowledge triggers retrieval; a genuinely ambiguous turn short-
circuits straight to a clarification step instead of generating an answer.
"""
import sys, os
import re
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.schemas import Plan, PlanStep, ActionType, AmbiguityAssessment, AmbiguityType
from src.context_manager import ConversationContext


EXTERNAL_KNOWLEDGE_TRIGGERS = [
    "who is", "who was", "when did", "when was", "what year", "history of",
    "historical background", "historical context",
    "compare to", "compared with", "is it true that", "fact check", "how does this compare",
    "background on", "background of", "context on", "what's the significance of",
]


def _needs_external_knowledge(text: str) -> bool:
    lowered = text.lower()
    return any(trigger in lowered for trigger in EXTERNAL_KNOWLEDGE_TRIGGERS)


def build_plan(
    text: str,
    newly_uploaded_image_ids: List[str],
    ambiguity: AmbiguityAssessment,
    context: ConversationContext,
) -> Plan:
    """
    Produces an ordered list of PlanSteps. The orchestrator executes these
    in order, feeding each step's output into the evidence bundle (or, for
    ASK_CLARIFICATION, short-circuiting the turn entirely).
    """
    steps: List[PlanStep] = []

    # --- Ambiguity gate: if we shouldn't guess, the ENTIRE plan is just "ask". ---
    from src.ambiguity_handler import should_clarify
    if should_clarify(ambiguity):
        return Plan(
            steps=[PlanStep(
                action=ActionType.ASK_CLARIFICATION,
                reason=f"Ambiguity detected ({ambiguity.ambiguity_type.value}, confidence={ambiguity.confidence:.2f}); "
                       f"answering now risks addressing the wrong subject.",
                params={"question": ambiguity.clarifying_question},
            )],
            needs_clarification=True,
        )

    # --- Vision analysis: only for images newly uploaded THIS turn. Images
    #     already analyzed in a prior turn are reused from context (no
    #     redundant model calls) unless the question explicitly asks to
    #     re-look at something. ---
    for image_id in newly_uploaded_image_ids:
        steps.append(PlanStep(
            action=ActionType.ANALYZE_IMAGE,
            reason="New image uploaded this turn -- needs captioning/tagging/OCR before we can reason about it.",
            params={"image_id": image_id},
        ))

    # --- Reference resolution: if the ambiguity assessment already resolved
    #     a reference to a *previously* analyzed image, pull its cached
    #     evidence into this turn's bundle rather than re-analyzing. ---
    if ambiguity.resolved_image_id and ambiguity.resolved_image_id not in newly_uploaded_image_ids:
        steps.append(PlanStep(
            action=ActionType.RESOLVE_REFERENCE,
            reason=f"Question refers back to previously discussed image '{ambiguity.resolved_image_id}'.",
            params={"image_id": ambiguity.resolved_image_id},
        ))

    # --- Conversation history: always useful for continuity, cheap to include. ---
    if context.turns:
        steps.append(PlanStep(
            action=ActionType.USE_CONVERSATION_HISTORY,
            reason="Prior turns exist -- include them so follow-up phrasing resolves correctly.",
        ))

    # --- External knowledge: only if the question needs facts beyond what's
    #     visible in the image / said in the conversation. ---
    if _needs_external_knowledge(text):
        steps.append(PlanStep(
            action=ActionType.RETRIEVE_EXTERNAL_KNOWLEDGE,
            reason="Question asks for outside context/facts not derivable from the image alone.",
            params={"query": text},
        ))

    # --- Always finish with generation. ---
    steps.append(PlanStep(
        action=ActionType.GENERATE_RESPONSE,
        reason="All required evidence gathered; ready to produce a grounded answer.",
    ))

    return Plan(steps=steps, needs_clarification=False)
