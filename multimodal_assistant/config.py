"""
config.py -- central configuration for the multi-modal AI assistant.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# ---------------------------------------------------------------------------
# Vision backend
# ---------------------------------------------------------------------------
# "hf"   -> real open-source vision models via transformers (BLIP + CLIP + OCR)
# "mock" -> deterministic fake analyzer, used for offline dev/testing of the
#           orchestration logic without downloading multi-GB model weights.
VISION_BACKEND = os.environ.get("MM_ASSISTANT_VISION_BACKEND", "hf")

CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
CLIP_MODEL = "openai/clip-vit-base-patch32"
# Zero-shot tag vocabulary CLIP scores each image against, to get structured
# "detected concept" evidence without training/running a full object detector.
TAG_VOCABULARY = [
    "person", "group of people", "animal", "dog", "cat", "bird", "vehicle", "car",
    "bicycle", "airplane", "boat", "building", "house", "city street", "nature",
    "mountain", "beach", "forest", "food", "meal", "text or document", "chart or graph",
    "screenshot", "diagram", "artwork or painting", "night scene", "daytime scene",
    "indoor scene", "outdoor scene", "crowd", "sports", "technology or electronics",
]
TAG_SCORE_THRESHOLD = 0.12  # min CLIP softmax score to keep a tag as "detected"

# ---------------------------------------------------------------------------
# LLM backend (reasoning / response generation)
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("MM_ASSISTANT_LLM_BACKEND", "ollama")   # "ollama" | "hf" | "mock"
OLLAMA_MODEL = os.environ.get("MM_ASSISTANT_OLLAMA_MODEL", "llava")   # vision-capable model recommended
OLLAMA_HOST = os.environ.get("MM_ASSISTANT_OLLAMA_HOST", "http://localhost:11434")
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ---------------------------------------------------------------------------
# Conversation / context management
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 10          # turns kept in the LLM prompt context
MAX_TRACKED_IMAGES = 5           # most recent images kept "active" for reference resolution

# ---------------------------------------------------------------------------
# Ambiguity handling
# ---------------------------------------------------------------------------
# Pronouns/deictic phrases that need a resolvable referent to answer confidently.
REFERENCE_TERMS = {
    "it", "this", "that", "these", "those", "the image", "the picture", "the photo",
    "the first one", "the second one", "the last one", "here", "there",
}
VAGUE_QUANTIFIERS = {"some", "several", "a few", "many", "most", "certain"}
AMBIGUITY_CONFIDENCE_THRESHOLD = 0.55  # below this, ask for clarification instead of guessing

# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
MIN_GROUNDING_SCORE = 0.35     # min evidence-overlap score per claim to count as "supported"
MAX_REGENERATION_ATTEMPTS = 2   # retries if validation fails before falling back to a hedged answer
