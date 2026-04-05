import json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from app.detector import detect_hallucination
from app.llm_detector import llm_detect
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 🔥 Hybrid function (same logic as FastAPI)
def hybrid_detect(answer, context):
    ml_result = detect_hallucination(answer, context)

    score = ml_result["score"]
    mismatch = ml_result.get("mismatch", 0)

    # Rule-based
    if mismatch > 0.5:
        return "Hallucinated"

    # LLM override
    if mismatch > 0.2:
        return llm_detect(answer, context)["label"]

    if score > 0.8:
        return "Grounded"
    elif score < 0.4:
        return "Hallucinated"
    else:
        return llm_detect(answer, context)["label"]


# 🔥 Load dataset
with open("data/dataset.json", "r") as f:
    data = json.load(f)


y_true = []
ml_preds = []
llm_preds = []
hybrid_preds = []

for item in data:
    context = item["context"]
    answer = item["answer"]
    label = item["label"]

    y_true.append(label)

    # ML
    ml_result = detect_hallucination(answer, context)
    ml_preds.append(ml_result["label"])

    # LLM
    llm_result = llm_detect(answer, context)
    llm_preds.append(llm_result["label"])

    # HYBRID
    hybrid_preds.append(hybrid_detect(answer, context))


# 🔥 FUNCTION TO PRINT METRICS
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} RESULTS =====")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)


# 🔥 RUN ALL
evaluate_model("ML MODEL", y_true, ml_preds)
evaluate_model("LLM MODEL", y_true, llm_preds)
evaluate_model("HYBRID MODEL", y_true, hybrid_preds)