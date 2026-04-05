from fastapi import FastAPI
from pydantic import BaseModel

from app.detector import detect_hallucination
from app.llm_detector import llm_detect

app = FastAPI()


class InputData(BaseModel):
    context: str
    answer: str


@app.post("/detect")
def detect(input: InputData, mode: str = "hybrid"):

    ml_result = detect_hallucination(input.answer, input.context)

    if mode == "ml":
        return ml_result

    if mode == "llm":
        return llm_detect(input.answer, input.context)

    # 🔥 HYBRID LOGIC
    score = float(ml_result["score"])
    mismatch = ml_result.get("mismatch", 0)

    print(f"DEBUG → Score: {score}, Mismatch: {mismatch}")

    # 🔴 Strong mismatch
    if mismatch > 0.5:
        return {
            "score": score,
            "label": "Hallucinated (Rule-Based)"
        }

    # 🟡 Medium mismatch → LLM
    if mismatch > 0.2:
        llm_result = llm_detect(input.answer, input.context)

        if "Error" in llm_result["label"]:
            return {
                "score": score,
                "label": "Hallucinated (Fallback ML)"
            }

        return {
            "score": score,
            "label": llm_result["label"] + " (LLM - mismatch)"
        }

    # 🟢 High confidence ML
    if score > 0.8:
        return {
            "score": score,
            "label": "Grounded (ML)"
        }

    # 🔴 Low score ML
    elif score < 0.4:
        return {
            "score": score,
            "label": "Hallucinated (ML)"
        }

    # ⚡ Uncertain → LLM
    else:
        llm_result = llm_detect(input.answer, input.context)
        return {
            "score": score,
            "label": llm_result["label"] + " (LLM)"
        }