import streamlit as st
import requests
import time

st.title("🧠 AI Hallucination Detector")

context = st.text_area("Enter Context")
answer = st.text_area("Enter Answer")

mode = st.selectbox("Mode", ["ml", "llm", "hybrid"])


def call_api(url, payload):
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response
        except requests.exceptions.ConnectionError:
            time.sleep(2)

    raise Exception("Backend not available")


if st.button("Check"):

    with st.spinner("Analyzing..."):

        response = call_api(
            f"http://backend:8000/detect?mode={mode}",
            {"context": context, "answer": answer}
        )

        result = response.json()

        st.subheader("Result")
        st.write("Score:", result.get("score", "N/A"))
        st.write("Label:", result.get("label"))