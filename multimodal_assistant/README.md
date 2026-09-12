# Multi-Modal AI Assistant (Text + Image, with Agentic Reasoning)

A conversational assistant that understands and reasons over both text and
images, built as a **multi-stage decision pipeline** rather than a single
model call — the explicit goal of this task. Every turn goes through
context tracking → ambiguity assessment → planning → evidence gathering →
grounded generation → response validation, and every stage's output is kept
for inspection.

## Why not just "call a vision-language model"?

A single VLM call (image + question → answer) can't:
- tell you *why* it answered the way it did,
- refuse to guess when a follow-up question is genuinely ambiguous ("what's
  wrong with **it**?" when two images are in play),
- check its own answer against concrete evidence before showing it to you,
- decide on its own that a question needs outside knowledge vs. only the
  image, and fetch it only when needed.

This project implements those explicitly as separate, testable stages.

## Architecture

```
User turn (text [+ image(s)])
        │
        ▼
context_manager.py     — records the turn; tracks images across turns
        │
        ▼
ambiguity_handler.py    — can we answer confidently, or must we ask first?
        │                  (unresolved reference, multiple candidate images,
        │                   vague scope, missing image entirely)
        ├── ambiguous ──► return clarifying question (turn ends here)
        ▼
planner.py               — decides WHICH actions are needed:
        │                   analyze new image? resolve a reference to a
        │                   prior image? pull conversation history? fetch
        │                   external knowledge? (skips steps that aren't needed)
        ▼
  ┌─────┴──────────────────────────────┐
  ▼                                    ▼
vision_analysis.py            knowledge_retrieval.py
(caption + zero-shot tags      (optional web search for
 + OCR + embeddings,            facts beyond the image)
 per image, independent
 of the question)
  │                                    │
  └─────────────┬──────────────────────┘
                 ▼
        EvidenceBundle (schemas.py)     — structured, provenance-tagged
                 │
                 ▼
        llm_backend.py                    — grounded answer generation
                 │
                 ▼
        response_validator.py              — checks each claim in the answer
                 │                            against the evidence bundle
        ┌────────┴─────────┐
        ▼                  ▼
     passed             failed → regenerate with feedback
        │                  (bounded retries, then hedge explicitly)
        ▼
   AssistantResponse (+ full trace) ──► app.py (Streamlit)
```

## What each stage demonstrates

| Requirement | Where |
|---|---|
| Understand & reason over image + text | `vision_analysis.py` (caption/tags/OCR/embeddings) feeding `llm_backend.py` |
| Maintain context across turns | `context_manager.py` — tracked images, chronological order, history-in-prompt |
| Contextual reasoning | `ambiguity_handler.resolve_reference` + `planner.build_plan` use conversation state, not just the current message |
| Ambiguity handling | `ambiguity_handler.py` — detects unresolved references, multiple candidate images, vague scope, missing images; asks instead of guessing |
| Evidence-based responses | `schemas.EvidenceBundle` with provenance tags; prompt explicitly restricts the LLM to that evidence |
| Response validation | `response_validator.py` — per-claim grounding check, regeneration loop, explicit hedging on failure |
| Intelligent decision-making (not just inference) | `planner.py` — decides *which* tools/steps to run per turn, e.g. skips re-analyzing an already-seen image, only fetches external knowledge when needed |

## Setup

```bash
pip install -r requirements.txt
```

Also install the Tesseract OCR binary if you want OCR text extraction (optional):
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

### LLM backend
**Option A — Ollama (recommended):**
```bash
ollama pull llava     # a vision-capable model, so the LLM can also see raw pixels
ollama serve
```

**Option B — HF transformers (text-only reasoning over the structured evidence):**
```bash
export MM_ASSISTANT_LLM_BACKEND=hf
```

### Run
```bash
streamlit run app.py
```

## Testing without any model downloads

`config.VISION_BACKEND=mock` and `LLM_BACKEND=mock` swap in deterministic,
dependency-light stand-ins so the **decision logic** (ambiguity detection,
planning, context tracking, validation) can be verified without GPUs,
internet, or multi-GB model downloads:

```bash
python tests/test_pipeline.py
```

This covers: missing-image clarification, unambiguous single-image
questions, follow-up reference resolution, multi-image ambiguity
requiring clarification, ordinal reference resolution ("the first
image"), context persistence, the validator distinguishing grounded
from ungrounded claims, and external-knowledge triggering.

## Project layout

```
config.py                    # all settings/model names/thresholds in one place
app.py                        # Streamlit UI with a per-turn reasoning-trace panel
src/
  schemas.py                  # typed contracts between pipeline stages
  vision_analysis.py           # BLIP captioning + CLIP zero-shot tags/embeddings + OCR (or mock)
  context_manager.py            # multi-turn history + image tracking + reference resolution
  ambiguity_handler.py           # decides: answer now, or ask for clarification?
  planner.py                      # decides which actions/tools a turn actually needs
  knowledge_retrieval.py           # optional external-knowledge tool
  llm_backend.py                    # Ollama / HF / mock generation backends
  response_validator.py              # evidence-grounding check + regeneration feedback
  orchestrator.py                     # MultimodalAssistant: wires all of the above together
tests/
  test_pipeline.py                    # offline tests of the decision logic (mock backends)
```

## Extending

- Swap the TF-IDF-based grounding check in `response_validator.py` for a
  proper NLI/entailment cross-encoder for stricter hallucination detection.
- Add more tools to `planner.py` (e.g. a calculator, a code interpreter) —
  each just needs an `ActionType`, a plan-building rule, and an execution
  branch in `orchestrator.py`.
- `context_manager.resolve_reference` is heuristic; for harder coreference
  cases, route through the LLM itself as an extra planning sub-step.
