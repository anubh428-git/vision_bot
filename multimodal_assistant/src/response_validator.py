"""
response_validator.py
------------------------
Checks a generated answer against the evidence it was supposed to be grounded
in, BEFORE it's shown to the user. This is what distinguishes "evidence-based
responses" from "whatever the model said": each declarative sentence in the
response is scored for lexical/semantic overlap against the evidence bundle;
sentences that make claims unsupported by any evidence are flagged. If
grounding is too weak, the orchestrator can send the answer back for
regeneration with explicit feedback, or fall back to a hedged answer that
says what's uncertain.

Uses TF-IDF cosine similarity as a lightweight, dependency-cheap stand-in for
a full NLI (natural language inference) model -- swap in a proper
entailment model (e.g. a cross-encoder NLI model) for stricter checking if
you have the compute budget; the interface (`validate`) stays the same.
"""
import sys, os
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.schemas import ValidationResult, ClaimCheck, EvidenceBundle


# Sentences matching these patterns are meta-commentary / hedges / questions,
# not factual claims about the evidence -- they don't need grounding.
_EXEMPT_PATTERNS = [
    r"^(i'?m not sure|i don'?t have enough|i can'?t confirm|could you|would you|can you)",
    r"\?$",
    r"^(let me know|feel free|happy to help)",
]


def _split_claims(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _is_exempt(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(re.search(p, lowered) for p in _EXEMPT_PATTERNS)


def _grounding_score(claim: str, evidence_texts: List[str]) -> (float, str):
    if not evidence_texts:
        return 0.0, None
    corpus = evidence_texts + [claim]
    try:
        vec = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    except ValueError:
        return 0.0, None  # claim/evidence had no usable vocabulary (e.g. all stopwords)
    claim_vec = vec[-1]
    evidence_vecs = vec[:-1]
    sims = cosine_similarity(claim_vec, evidence_vecs).flatten()
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), evidence_texts[best_idx]


def validate(response_text: str, evidence: EvidenceBundle) -> ValidationResult:
    evidence_texts = [img.as_evidence_text() for img in evidence.image_evidence]
    evidence_texts += [t.content for t in evidence.text_evidence if t.relevance > 0]

    claims = _split_claims(response_text)
    checks: List[ClaimCheck] = []

    for claim in claims:
        if _is_exempt(claim):
            checks.append(ClaimCheck(claim=claim, supported=True, grounding_score=1.0,
                                      supporting_evidence="(exempt: hedge/question, not a factual claim)"))
            continue
        score, best_evidence = _grounding_score(claim, evidence_texts)
        supported = score >= config.MIN_GROUNDING_SCORE
        checks.append(ClaimCheck(claim=claim, supported=supported, grounding_score=score,
                                  supporting_evidence=best_evidence if supported else None))

    scorable = [c for c in checks if c.supporting_evidence != "(exempt: hedge/question, not a factual claim)"]
    overall_score = float(np.mean([c.grounding_score for c in checks])) if checks else 0.0
    unsupported = [c.claim for c in checks if not c.supported]

    # Pass if there's no unsupported claim among the substantive (non-exempt)
    # ones, OR the response was entirely hedges/questions (e.g. a clarification).
    passed = len(unsupported) == 0

    feedback = ""
    if not passed:
        feedback = (
            "Your previous answer included claims not supported by the gathered evidence: "
            + "; ".join(f'"{c}"' for c in unsupported[:3])
            + ". Revise your answer to only state what the evidence actually supports, "
              "and explicitly say when something can't be confirmed from the image/context."
        )

    return ValidationResult(
        passed=passed,
        grounding_score=overall_score,
        claim_checks=checks,
        unsupported_claims=unsupported,
        feedback_for_regeneration=feedback,
    )
