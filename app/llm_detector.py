import requests
import logging

OLLAMA_URL =  "http://host.docker.internal:11434/api/generate"

def llm_detect(answer, context):

    prompt = f"""
You are an AI hallucination detector.

Context:
{context}

Answer:
{answer}

Task:
- Check if the answer is fully supported by the context
- If fully correct → Grounded
- If partially correct → Partially Hallucinated
- If incorrect → Hallucinated

Respond ONLY in one word:
Grounded / Partially Hallucinated / Hallucinated
"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        },
        timeout=30
        )

        response.raise_for_status()

        result = response.json().get("response", "").strip()

        # ✅ Normalize output
        if "Grounded" in result:
            label = "Grounded"
        elif "Partially" in result:
            label = "Partially Hallucinated"
        else:
            label = "Hallucinated"

        return {"label": label}

    except Exception as e:
        print("LLM ERROR:", str(e)) 

        return {
            "label": "Error (LLM Failed)"
        }