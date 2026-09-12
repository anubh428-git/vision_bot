"""
tests/test_sentiment.py
--------------------------
Two things get tested here:

1. Sentiment detection ACCURACY against a labeled set of 30 realistic
   customer-service-style messages (positive/neutral/negative), reporting
   a confusion matrix and asserting a minimum accuracy bar.
2. Response strategy CORRECTNESS: that different sentiment/intensity
   combinations actually produce different tone instructions, that negative
   sentiment triggers empathetic language, that escalation guidance appears
   only when warranted, and that satisfaction scoring behaves sensibly.

Run:  python tests/test_sentiment.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analysis import SentimentAnalyzer, should_escalate, SentimentResult
from src.response_strategy import build_tone_instruction, compose_system_instruction


# --------------------------------------------------------------------------- #
# Labeled test set -- realistic customer-service-style messages.
# --------------------------------------------------------------------------- #
LABELED_EXAMPLES = [
    # positive
    ("This is amazing, thank you so much for your help!", "positive"),
    ("Great, that fixed it! You are awesome.", "positive"),
    ("Not bad, actually pretty good.", "positive"),
    ("I really appreciate how quickly you resolved this.", "positive"),
    ("Perfect, exactly what I needed, thanks!", "positive"),
    ("Wow, that was way easier than I expected. Love it.", "positive"),
    ("You guys are the best, this saved me so much time.", "positive"),
    ("Excellent support, very happy with the outcome.", "positive"),
    # neutral
    ("Can you tell me your business hours?", "neutral"),
    ("What's the difference between the basic and pro plan?", "neutral"),
    ("How do I reset my password?", "neutral"),
    ("Where can I find my invoice history?", "neutral"),
    ("I'd like to update my shipping address.", "neutral"),
    ("Does this feature work on mobile too?", "neutral"),
    ("Just checking in on the status of my request.", "neutral"),
    # negative
    ("I am extremely frustrated, this has been broken for three days.", "negative"),
    ("This is the worst customer service I have ever experienced.", "negative"),
    ("Why does this keep failing every single time?! So annoying.", "negative"),
    ("This is not good at all.", "negative"),
    ("I want to cancel my subscription immediately, this is ridiculous.", "negative"),
    ("Nothing you've suggested has worked, I'm losing my patience.", "negative"),
    ("This app is useless, it crashes constantly.", "negative"),
    ("I've been waiting for a response for a week now, unacceptable.", "negative"),
    ("Your support team keeps giving me the wrong information.", "negative"),
    ("This is a complete waste of my time and money.", "negative"),
    ("I'm really disappointed with how this was handled.", "negative"),
    ("Still broken. Third time I've had to contact you about this.", "negative"),
    ("The checkout page keeps erroring out, very frustrating.", "negative"),
    ("I don't understand why this is so complicated, it's ridiculous.", "negative"),
    ("Terrible experience, I want a refund.", "negative"),
]

MIN_ACCEPTABLE_ACCURACY = 0.80


def test_sentiment_detection_accuracy():
    sa = SentimentAnalyzer()
    correct = 0
    confusion = {}  # (expected, predicted) -> count
    for text, expected in LABELED_EXAMPLES:
        predicted = sa.analyze(text).label
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
        if predicted == expected:
            correct += 1

    accuracy = correct / len(LABELED_EXAMPLES)
    print(f"Backend: {sa.backend}")
    print(f"Accuracy: {correct}/{len(LABELED_EXAMPLES)} = {accuracy*100:.1f}%")
    print("Confusion (expected -> predicted): counts")
    for (exp, pred), count in sorted(confusion.items()):
        marker = "" if exp == pred else "  <-- misclassified"
        print(f"  {exp:8s} -> {pred:8s} : {count}{marker}")

    assert accuracy >= MIN_ACCEPTABLE_ACCURACY, (
        f"Accuracy {accuracy*100:.1f}% below minimum acceptable {MIN_ACCEPTABLE_ACCURACY*100:.0f}%"
    )
    print(f"PASS: accuracy {accuracy*100:.1f}% >= minimum {MIN_ACCEPTABLE_ACCURACY*100:.0f}%")


def test_satisfaction_score_monotonic_with_compound():
    sa = SentimentAnalyzer()
    very_positive = sa.analyze("This is absolutely fantastic, thank you so much!!!")
    neutral = sa.analyze("Can you check my order status?")
    very_negative = sa.analyze("This is absolutely terrible, I hate this, worst experience ever.")

    assert very_positive.satisfaction_score > neutral.satisfaction_score > very_negative.satisfaction_score
    assert 0 <= very_negative.satisfaction_score <= 100
    assert 0 <= very_positive.satisfaction_score <= 100
    print(f"PASS: satisfaction scores ordered correctly "
          f"({very_positive.satisfaction_score} > {neutral.satisfaction_score} > {very_negative.satisfaction_score})")


def test_tone_instructions_differ_by_sentiment():
    positive = SentimentResult("positive", 0.8, "strong", 90.0, "test")
    neutral = SentimentResult("neutral", 0.0, "mild", 50.0, "test")
    negative = SentimentResult("negative", -0.8, "strong", 10.0, "test")

    tones = {r.label: build_tone_instruction(r) for r in (positive, neutral, negative)}
    assert tones["positive"] != tones["neutral"] != tones["negative"]
    assert "frustrat" in tones["negative"].lower() or "upset" in tones["negative"].lower() or "angry" in tones["negative"].lower()
    print("PASS: tone instructions differ meaningfully across sentiment labels")


def test_negative_intensity_affects_empathy_language():
    mild = SentimentResult("negative", -0.1, "mild", 45.0, "test")
    strong = SentimentResult("negative", -0.8, "strong", 10.0, "test")
    mild_tone = build_tone_instruction(mild)
    strong_tone = build_tone_instruction(strong)
    assert mild_tone != strong_tone
    assert "acknowledg" in strong_tone.lower() or "sorry" in strong_tone.lower()
    print("PASS: stronger negative sentiment produces more explicit empathy guidance")


def test_escalation_triggers_on_sustained_negativity():
    history = [
        SentimentResult("negative", -0.7, "strong", 15.0, "test"),
        SentimentResult("negative", -0.6, "strong", 20.0, "test"),
    ]
    assert should_escalate(history) is True

    calm_history = [
        SentimentResult("positive", 0.6, "moderate", 80.0, "test"),
        SentimentResult("negative", -0.6, "strong", 20.0, "test"),
    ]
    assert should_escalate(calm_history) is False
    print("PASS: escalation triggers on sustained negativity, not a single negative turn")


def test_escalation_guidance_appended_only_when_flagged():
    negative = SentimentResult("negative", -0.8, "strong", 10.0, "test")
    without = build_tone_instruction(negative, escalate=False)
    with_escalation = build_tone_instruction(negative, escalate=True)
    assert without != with_escalation
    assert "escalat" in with_escalation.lower() or "human representative" in with_escalation.lower()
    assert "escalat" not in without.lower()
    print("PASS: escalation guidance only appears in the instruction when escalate=True")


def test_compose_system_instruction_preserves_base_instruction():
    result = SentimentResult("negative", -0.7, "strong", 15.0, "test")
    combined = compose_system_instruction("You are a support agent for Acme Inc.", result)
    assert "Acme Inc." in combined
    assert "sentiment guidance" in combined.lower()
    print("PASS: base system instruction is preserved alongside sentiment guidance")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            print(f"\n--- {t.__name__} ---")
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
