from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from functools import lru_cache
import logging

from app.config import SIM_WEIGHT, KEYWORD_WEIGHT, THRESHOLD_HIGH, THRESHOLD_LOW

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- MODEL LOAD ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------------- FAST EMBEDDING CACHE ----------------
@lru_cache(maxsize=200)
def get_embedding(text):
    return model.encode(text)


def get_similarity(answer, context):
    emb1 = get_embedding(answer)
    emb2 = get_embedding(context)

    score = cosine_similarity([emb1], [emb2])[0][0]
    return score


# ---------------- KEYWORD EXTRACTION ----------------
def extract_keywords(text):
    words = text.lower().split()
    keywords = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return set(keywords)


# ---------------- MISMATCH DETECTION ----------------
def entity_mismatch(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    mismatch = answer_words - context_words

    # 🔥 ignore common words
    important_mismatch = [w for w in mismatch if len(w) > 3]

    return len(important_mismatch) / max(len(answer_words), 1)


# ---------------- KEYWORD OVERLAP ----------------
def keyword_overlap(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words.intersection(context_words)
    return len(overlap) / max(len(answer_words), 1)


# ---------------- SCORE ----------------
def hallucination_score(answer, context):
    sim = get_similarity(answer, context)
    key = keyword_overlap(answer, context)
    mismatch_ratio = entity_mismatch(answer, context)

    # ✅ softer penalty (key fix)
    penalty = mismatch_ratio * 0.3

    final_score = (SIM_WEIGHT * sim) + (KEYWORD_WEIGHT * key) - penalty

    print(f"SIM: {sim:.2f}, KEY: {key:.2f}, MISMATCH: {mismatch_ratio:.2f}, FINAL: {final_score:.2f}")

    return final_score, mismatch_ratio


# ---------------- CLASSIFICATION ----------------
def classify(score, mismatch_ratio):

    # 🔴 Strong mismatch → Hallucinated
    if mismatch_ratio > 0.6:
        return "Hallucinated"

    # 🟡 Moderate mismatch → ALWAYS Partial
    if mismatch_ratio > 0.2:
        return "Partially Hallucinated"

    # 🟢 No mismatch → use score
    if score > THRESHOLD_HIGH:
        return "Grounded"
    elif score > THRESHOLD_LOW:
        return "Partially Hallucinated"
    else:
        return "Hallucinated"


# ---------------- MAIN FUNCTION ----------------
def detect_hallucination(answer, context):
    try:
        score, mismatch_ratio = hallucination_score(answer, context)
        label = classify(score, mismatch_ratio)

        logging.info(f"Processed | Score: {score:.3f} | Mismatch: {mismatch_ratio:.3f} | Label: {label}")

        return {
                    "score": float(score),
                    "label": label,
                    "mismatch": float(mismatch_ratio)
}
        

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise