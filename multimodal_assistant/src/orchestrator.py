"""
orchestrator.py
-----------------
The MultimodalAssistant class is the single entry point the app talks to.
For every turn it runs a multi-stage decision pipeline instead of a single
model call:

    1. Update conversation context with the new turn (text + any images).
    2. Analyze any newly uploaded images independently of the question
       (structured evidence extraction, not "answer this question about
       this image" in one opaque call).
    3. Assess ambiguity. If the turn can't be answered confidently, stop
       here and ask a clarifying question -- no guessing.
    4. Plan: decide which further actions are needed (reference resolution,
       external knowledge retrieval, etc).
    5. Execute the plan, assembling an EvidenceBundle with full provenance.
    6. Generate a grounded answer from the evidence bundle.
    7. Validate the answer against the evidence. If it fails, regenerate
       with explicit feedback (bounded number of attempts), then fall back
       to an explicitly hedged answer rather than shipping an unsupported one.

Every stage's output is attached to the returned AssistantResponse's trace
fields, so the UI can show *why* the assistant answered the way it did.
"""
import sys, os
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.schemas import (
    EvidenceBundle, TextEvidence, AssistantResponse, ActionType, new_id
)
from src.context_manager import ConversationContext
from src.vision_analysis import VisionAnalyzer
from src import ambiguity_handler, planner, knowledge_retrieval, llm_backend, response_validator


class MultimodalAssistant:
    def __init__(self, vision_backend: str = config.VISION_BACKEND, llm_backend_name: str = config.LLM_BACKEND):
        self.context = ConversationContext()
        self.vision = VisionAnalyzer(backend=vision_backend)
        self.llm_backend_name = llm_backend_name
        self._image_cache = {}  # image_id -> PIL.Image, for re-passing to a vision-capable LLM

    # ------------------------------------------------------------------ #
    def process_turn(self, text: str, images: Optional[List] = None) -> AssistantResponse:
        images = images or []

        # --- 1. Register newly uploaded images & register the user turn ---
        newly_uploaded_ids = []
        for img in images:
            image_id = new_id("img")
            self._image_cache[image_id] = img
            newly_uploaded_ids.append(image_id)

        self.context.add_turn("user", text, image_ids=newly_uploaded_ids)

        # --- 2. Ambiguity assessment (before any expensive vision calls on
        #        images that might not even be the relevant one, though new
        #        uploads are cheap to just analyze -- see planner) ---
        ambiguity = ambiguity_handler.assess(text, self.context, newly_uploaded_ids)

        # --- 3. Plan ---
        plan = planner.build_plan(text, newly_uploaded_ids, ambiguity, self.context)

        if plan.needs_clarification:
            response = AssistantResponse(
                text=plan.steps[0].params["question"],
                response_type="clarification",
                confidence=ambiguity.confidence,
                evidence=EvidenceBundle(),
                plan=plan, ambiguity=ambiguity, validation=None,
            )
            self.context.add_turn("assistant", response.text, trace=self._trace(plan, ambiguity, None, 0))
            return response

        # --- 4. Execute plan -> build evidence bundle ---
        evidence = EvidenceBundle()
        primary_image = None
        for step in plan.steps:
            if step.action == ActionType.ANALYZE_IMAGE:
                image_id = step.params["image_id"]
                img = self._image_cache[image_id]
                ev = self.vision.analyze(img, image_id)
                self.context.register_image(image_id, ev)
                evidence.image_evidence.append(ev)
                primary_image = img

            elif step.action == ActionType.RESOLVE_REFERENCE:
                image_id = step.params["image_id"]
                cached_ev = self.context.image_evidence.get(image_id)
                if cached_ev:
                    evidence.image_evidence.append(cached_ev)
                    primary_image = self._image_cache.get(image_id)

            elif step.action == ActionType.USE_CONVERSATION_HISTORY:
                hist = self.context.history_as_prompt()
                if hist:
                    evidence.text_evidence.append(TextEvidence(source="conversation_history", content=hist))

            elif step.action == ActionType.RETRIEVE_EXTERNAL_KNOWLEDGE:
                results = knowledge_retrieval.retrieve(step.params["query"])
                evidence.text_evidence.extend(results)

            elif step.action == ActionType.GENERATE_RESPONSE:
                pass  # handled below, after the loop, so all evidence is assembled first

        # --- 5/6/7. Generate -> validate -> regenerate-with-feedback loop ---
        history_block = self.context.history_as_prompt()
        evidence_block = evidence.as_prompt_block()

        feedback = ""
        attempts = 0
        answer_text = ""
        validation = None
        while True:
            answer_text = llm_backend.generate(
                question=text, evidence_block=evidence_block, history_block=history_block,
                image=primary_image, feedback=feedback, backend=self.llm_backend_name,
            )
            validation = response_validator.validate(answer_text, evidence)
            attempts += 1

            if validation.passed or attempts > config.MAX_REGENERATION_ATTEMPTS:
                break
            feedback = validation.feedback_for_regeneration

        response_type = "answer"
        if not validation.passed:
            # Exhausted retries and still ungrounded -- ship a hedged version
            # rather than an unvalidated claim, and say so explicitly.
            answer_text = (
                answer_text
                + "\n\n⚠️ Note: I wasn't able to fully verify every part of this answer "
                  "against the available evidence after multiple attempts -- please treat "
                  "the flagged points with extra caution: "
                + "; ".join(validation.unsupported_claims[:3])
            )
            response_type = "hedged_answer"

        response = AssistantResponse(
            text=answer_text, response_type=response_type,
            confidence=validation.grounding_score,
            evidence=evidence, plan=plan, ambiguity=ambiguity,
            validation=validation, regeneration_attempts=attempts - 1,
        )
        self.context.add_turn(
            "assistant", answer_text,
            trace=self._trace(plan, ambiguity, validation, attempts - 1),
        )
        return response

    # ------------------------------------------------------------------ #
    def _trace(self, plan, ambiguity, validation, regen_attempts) -> dict:
        return {
            "plan_steps": [f"{s.action.value}: {s.reason}" for s in plan.steps] if plan else [],
            "ambiguity_type": ambiguity.ambiguity_type.value if ambiguity else None,
            "ambiguity_confidence": ambiguity.confidence if ambiguity else None,
            "grounding_score": validation.grounding_score if validation else None,
            "validation_passed": validation.passed if validation else None,
            "regeneration_attempts": regen_attempts,
        }

    def reset(self):
        self.context = ConversationContext()
        self._image_cache = {}
