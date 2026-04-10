"""
Bias Detection & Mitigation API
Authors: Namritha Diya Lobo, Nidhi K, Navya G N
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json

try:
    from backend.bias_detector import BiasDetector
    from backend.rag_engine import RAGEngine
    from backend.llm_client import LLMClient
except ModuleNotFoundError:
    from bias_detector import BiasDetector
    from rag_engine import RAGEngine
    from llm_client import LLMClient

load_dotenv()

app = Flask(__name__)
CORS(app)

FRONTEND_INDEX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
)

# Initialize components
llm_client = LLMClient()
bias_detector = BiasDetector()
rag_engine = RAGEngine()


def compare_bias(baseline_result: dict, mitigated_result: dict) -> dict:
    """Compare baseline and mitigated scores for easier evaluation in the UI."""
    baseline_scores = baseline_result.get("detailed_scores", {})
    mitigated_scores = mitigated_result.get("detailed_scores", {})

    common_metrics = set(baseline_scores.keys()) & set(mitigated_scores.keys())
    score_improvements = {
        metric: round(float(baseline_scores[metric]) - float(mitigated_scores[metric]), 3)
        for metric in common_metrics
        if isinstance(baseline_scores.get(metric), (int, float))
        and isinstance(mitigated_scores.get(metric), (int, float))
    }

    composite_reduction = round(
        float(baseline_result.get("composite_bias_score", 0.0))
        - float(mitigated_result.get("composite_bias_score", 0.0)),
        3,
    )

    return {
        "bias_removed": baseline_result.get("bias_detected", False)
        and not mitigated_result.get("bias_detected", False),
        "score_improvements": score_improvements,
        "composite_reduction": composite_reduction,
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Bias Detection API is running"})


@app.route("/", methods=["GET"])
def home():
    if os.path.exists(FRONTEND_INDEX):
        return send_file(FRONTEND_INDEX)
    return jsonify(
        {
            "message": "Bias Detection API is running",
            "routes": ["/health", "/analyze"],
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint: returns baseline LLM response and RAG-mitigated response.
    Also returns bias analysis for both.
    """
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        # Step 1: Get baseline LLM response (no bias mitigation)
        baseline_response = llm_client.generate(prompt)

        # Step 2: Analyze baseline for bias
        baseline_bias = bias_detector.analyze(prompt, baseline_response)

        # Step 3: If bias detected, apply RAG + prompt augmentation
        if baseline_bias["bias_detected"]:
            fairness_context = rag_engine.retrieve(prompt, baseline_bias["bias_types"])
            augmented_prompt = rag_engine.build_augmented_prompt(prompt, fairness_context)
            mitigated_response = llm_client.generate(augmented_prompt)
            mitigated_bias = bias_detector.analyze(prompt, mitigated_response)
            rag_applied = True
            retrieved_docs = fairness_context
        else:
            mitigated_response = baseline_response
            mitigated_bias = baseline_bias
            rag_applied = False
            retrieved_docs = []

        return jsonify({
            "prompt": prompt,
            "baseline": {
                "response": baseline_response,
                "bias_analysis": baseline_bias
            },
            "mitigated": {
                "response": mitigated_response,
                "bias_analysis": mitigated_bias,
                "rag_applied": rag_applied,
                "retrieved_docs": retrieved_docs
            },
            "comparison": compare_bias(baseline_bias, mitigated_bias),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
