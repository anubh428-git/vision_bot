"""
tests/test_pipeline.py
------------------------
Exercises the orchestration logic end-to-end using the mock vision and LLM
backends (no model downloads / GPU / network required), so the *decision
logic* -- ambiguity handling, planning, context tracking, validation -- is
verifiable in any environment.

Run:  python -m pytest tests/ -v      (or just: python tests/test_pipeline.py)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import config
from src.orchestrator import MultimodalAssistant
from src.schemas import AmbiguityType


def make_test_image(w=200, h=150, color=(255, 0, 0)):
    return Image.new("RGB", (w, h), color=color)


def new_assistant():
    return MultimodalAssistant(vision_backend="mock", llm_backend_name="mock")


# --------------------------------------------------------------------------- #
def test_missing_image_triggers_clarification():
    a = new_assistant()
    resp = a.process_turn("What is in this image?", images=[])
    assert resp.response_type == "clarification"
    assert resp.ambiguity.ambiguity_type == AmbiguityType.MISSING_IMAGE
    print("PASS: missing image -> clarification")


def test_single_image_question_answers_directly():
    a = new_assistant()
    img = make_test_image()
    resp = a.process_turn("What do you see in this image?", images=[img])
    assert resp.response_type in ("answer", "hedged_answer")
    assert resp.ambiguity.ambiguity_type == AmbiguityType.NONE
    assert len(resp.evidence.image_evidence) == 1
    print("PASS: single image -> direct answer, no clarification needed")


def test_followup_reference_resolves_to_single_tracked_image():
    a = new_assistant()
    img = make_test_image()
    a.process_turn("What is this?", images=[img])
    resp = a.process_turn("What color is it?")  # follow-up, no new image
    assert resp.response_type in ("answer", "hedged_answer")
    assert resp.ambiguity.ambiguity_type == AmbiguityType.NONE
    assert resp.ambiguity.resolved_image_id is not None
    # Evidence should reuse the cached analysis, not re-run vision on nothing.
    assert len(resp.evidence.image_evidence) == 1
    print("PASS: follow-up reference resolves to the single tracked image")


def test_multiple_images_ambiguous_reference_triggers_clarification():
    a = new_assistant()
    img1, img2 = make_test_image(color=(255, 0, 0)), make_test_image(color=(0, 255, 0))
    a.process_turn("What is this?", images=[img1])
    a.process_turn("And what about this one?", images=[img2])
    resp = a.process_turn("What color is it?")  # ambiguous: two images now tracked
    assert resp.response_type == "clarification"
    assert resp.ambiguity.ambiguity_type == AmbiguityType.MULTIPLE_CANDIDATE_IMAGES
    print("PASS: ambiguous reference across multiple images -> clarification")


def test_ordinal_reference_resolves_correctly():
    a = new_assistant()
    img1, img2 = make_test_image(color=(255, 0, 0)), make_test_image(color=(0, 255, 0))
    a.process_turn("Look at this.", images=[img1])
    a.process_turn("Now this one too.", images=[img2])
    resp = a.process_turn("What was in the first image?")
    assert resp.response_type in ("answer", "hedged_answer")
    assert resp.ambiguity.resolved_image_id == a.context.image_order[0]
    print("PASS: ordinal reference ('the first image') resolves without asking")


def test_conversation_context_persists_across_turns():
    a = new_assistant()
    img = make_test_image()
    a.process_turn("Describe this image.", images=[img])
    a.process_turn("Thanks, what else can you tell me?")
    assert len(a.context.turns) == 4  # 2 user + 2 assistant turns
    hist = a.context.history_as_prompt()
    assert "Describe this image" in hist
    print("PASS: conversation context accumulates across turns")


def test_response_validator_flags_unsupported_claims():
    from src.response_validator import validate
    from src.schemas import EvidenceBundle, ImageEvidence

    evidence = EvidenceBundle(image_evidence=[
        ImageEvidence(image_id="img_1", caption="A red bicycle leaning against a brick wall.",
                       tags=["bicycle", "outdoor scene"], tag_scores={}, ocr_text="")
    ])
    grounded_response = "The image shows a red bicycle leaning against a brick wall."
    ungrounded_response = "The image shows a spaceship landing on the moon."

    v1 = validate(grounded_response, evidence)
    v2 = validate(ungrounded_response, evidence)

    assert v1.passed, f"Expected grounded response to pass, got score={v1.grounding_score}"
    assert not v2.passed, "Expected ungrounded response to fail validation"
    print(f"PASS: validator distinguishes grounded (score={v1.grounding_score:.2f}) "
          f"from ungrounded (score={v2.grounding_score:.2f}) claims")


def test_external_knowledge_trigger_adds_text_evidence():
    a = new_assistant()
    img = make_test_image()
    resp = a.process_turn("What is the historical background of this kind of artwork?", images=[img])
    sources = {t.source for t in resp.evidence.text_evidence}
    assert "external_knowledge" in sources
    print("PASS: knowledge-seeking question triggers external retrieval step in the plan")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
