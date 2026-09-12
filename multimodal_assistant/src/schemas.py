"""
schemas.py
----------
Typed data structures shared across the pipeline. Keeping these explicit
(rather than passing raw dicts around) is what makes the multi-stage
orchestration in orchestrator.py legible: every stage has a clear contract
for what it consumes and produces.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time
import uuid


# --------------------------------------------------------------------------- #
# Vision evidence
# --------------------------------------------------------------------------- #
@dataclass
class ImageEvidence:
    image_id: str
    caption: str
    tags: List[str]                 # zero-shot detected concepts, e.g. ["chart or graph", "text or document"]
    tag_scores: Dict[str, float]
    ocr_text: str                    # "" if no text detected / OCR unavailable
    embedding: Optional[Any] = None   # CLIP embedding, used for cross-turn similarity
    raw_backend: str = "hf"

    def as_evidence_text(self) -> str:
        parts = [f"Caption: {self.caption}"]
        if self.tags:
            parts.append(f"Detected concepts: {', '.join(self.tags)}")
        if self.ocr_text:
            parts.append(f"Text visible in image (OCR): {self.ocr_text}")
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Retrieved external / textual evidence (non-image)
# --------------------------------------------------------------------------- #
@dataclass
class TextEvidence:
    source: str          # e.g. "conversation_history", "external_knowledge"
    content: str
    relevance: float = 1.0


@dataclass
class EvidenceBundle:
    image_evidence: List[ImageEvidence] = field(default_factory=list)
    text_evidence: List[TextEvidence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.image_evidence and not self.text_evidence

    def as_prompt_block(self) -> str:
        lines = []
        for i, img in enumerate(self.image_evidence):
            lines.append(f"[IMAGE {i+1} | id={img.image_id}] {img.as_evidence_text()}")
        for t in self.text_evidence:
            lines.append(f"[{t.source.upper()}] {t.content}")
        return "\n".join(lines) if lines else "(no evidence gathered)"


# --------------------------------------------------------------------------- #
# Conversation turns / context
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    turn_id: str
    role: str                        # "user" | "assistant"
    text: str
    image_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    # Populated on assistant turns for transparency / debugging:
    trace: Optional[Dict[str, Any]] = None


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Ambiguity assessment
# --------------------------------------------------------------------------- #
class AmbiguityType(str, Enum):
    NONE = "none"
    UNRESOLVED_REFERENCE = "unresolved_reference"   # "it"/"this" with no clear referent
    MULTIPLE_CANDIDATE_IMAGES = "multiple_candidate_images"
    VAGUE_SCOPE = "vague_scope"                       # "some of the objects", underspecified
    MISSING_IMAGE = "missing_image"                    # question clearly needs an image, none provided/tracked


@dataclass
class AmbiguityAssessment:
    ambiguity_type: AmbiguityType
    confidence: float                 # confidence that we CAN answer without clarifying (0-1)
    clarifying_question: Optional[str] = None
    resolved_image_id: Optional[str] = None   # if reference resolution succeeded
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ANALYZE_IMAGE = "analyze_image"
    RESOLVE_REFERENCE = "resolve_reference"
    RETRIEVE_EXTERNAL_KNOWLEDGE = "retrieve_external_knowledge"
    USE_CONVERSATION_HISTORY = "use_conversation_history"
    ASK_CLARIFICATION = "ask_clarification"
    GENERATE_RESPONSE = "generate_response"


@dataclass
class PlanStep:
    action: ActionType
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: List[PlanStep]
    needs_clarification: bool = False


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #
@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    grounding_score: float
    supporting_evidence: Optional[str] = None


@dataclass
class ValidationResult:
    passed: bool
    grounding_score: float                 # overall, 0-1
    claim_checks: List[ClaimCheck]
    unsupported_claims: List[str]
    feedback_for_regeneration: str = ""


# --------------------------------------------------------------------------- #
# Final assistant output (with full transparency trace)
# --------------------------------------------------------------------------- #
@dataclass
class AssistantResponse:
    text: str
    response_type: str        # "answer" | "clarification" | "hedged_answer"
    confidence: float
    evidence: EvidenceBundle
    plan: Optional[Plan] = None
    ambiguity: Optional[AmbiguityAssessment] = None
    validation: Optional[ValidationResult] = None
    regeneration_attempts: int = 0
