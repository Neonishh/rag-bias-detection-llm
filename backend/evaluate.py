# evaluate.py

import json
import pandas as pd
import matplotlib.pyplot as plt  # 👈 ADD THIS
from dotenv import load_dotenv
load_dotenv()

from bias_detector import BiasDetector
from rag_engine import RAGEngine
from llm_client import LLMClient

detector = BiasDetector()
rag = RAGEngine()
llm = LLMClient()

results = []

# ── Load prompts ─────────────────────
with open("eval_prompts.json") as f:
    prompts = json.load(f)

# ── Run evaluation ───────────────────
for item in prompts:
    prompt = item["text"]

    baseline = llm.generate(prompt)
    base_analysis = detector.analyze(prompt, baseline)
    base_score = base_analysis["composite_bias_score"]

    if base_analysis["bias_detected"]:
        docs = rag.retrieve(prompt, base_analysis["bias_types"])
        aug_prompt = rag.build_augmented_prompt(prompt, docs)
        mitigated = llm.generate(aug_prompt)
    else:
        mitigated = baseline

    mitigated_analysis = detector.analyze(prompt, mitigated)
    mitigated_score = mitigated_analysis["composite_bias_score"]

    results.append({
        "prompt": prompt,
        "type": item["type"],
        "baseline_score": base_score,
        "mitigated_score": mitigated_score,
        "reduction": base_score - mitigated_score
    })

# ── Convert to DataFrame ─────────────
df = pd.DataFrame(results)

# Save table (for paper)
df.to_csv("results.csv", index=False)

print(df)

# =========================================================
# 🔥 PUT YOUR GRAPH CODE HERE (Step 3)
# =========================================================

# Bar chart: Before vs After
plt.figure()
df[["baseline_score", "mitigated_score"]].mean().plot(kind="bar")
plt.title("Bias Score Before vs After Mitigation")
plt.ylabel("Bias Score")
plt.tight_layout()
plt.savefig("bias_comparison.png")

# Bar chart: Reduction by type
plt.figure()
df.groupby("type")["reduction"].mean().plot(kind="bar")
plt.title("Bias Reduction by Prompt Type")
plt.tight_layout()
plt.savefig("reduction_by_type.png")

plt.figure()
df["baseline_score"].plot(kind="hist", bins=10, alpha=0.5, label="Before")
df["mitigated_score"].plot(kind="hist", bins=10, alpha=0.5, label="After")
plt.legend()
plt.title("Distribution of Bias Scores")
plt.xlabel("Bias Score")
plt.tight_layout()
plt.savefig("bias_distribution.png")

plt.figure()
plt.plot(df["baseline_score"], label="Before")
plt.plot(df["mitigated_score"], label="After")
plt.legend()
plt.title("Bias Score per Prompt")
plt.tight_layout()
plt.savefig("bias_per_prompt.png")

df["percent_reduction"] = (
    (df["baseline_score"] - df["mitigated_score"]) / df["baseline_score"]
) * 100

plt.figure()
df["percent_reduction"].plot(kind="bar")
plt.title("Percentage Bias Reduction per Prompt")
plt.ylabel("% Reduction")
plt.tight_layout()
plt.savefig("percent_reduction.png")

print("✅ Graphs saved!")