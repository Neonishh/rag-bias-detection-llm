# evaluate.py

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt  # 👈 ADD THIS
from dotenv import load_dotenv
load_dotenv()

from bias_detector import BiasDetector
from rag_engine import RAGEngine
from llm_client import LLMClient, ModelProviderFailure

detector = BiasDetector()
rag = RAGEngine()
llm = LLMClient(fallback_policy="consistent_run")

results = []
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BASE_DIR / "eval_prompts.json"
RESULTS_PATH = BASE_DIR / "results.csv"
RUN_META_PATH = BASE_DIR / "run_metadata.json"

# ── Load prompts ─────────────────────
with open(PROMPTS_PATH, encoding="utf-8") as f:
    prompts = json.load(f)

print("Evaluation policy: one model per run (fallback only allowed before first success).")

stop_reason = "completed_all_prompts"
failed_prompts = 0
last_attempted_prompt_idx = 0

# ── Run evaluation ───────────────────
for idx, item in enumerate(prompts, start=1):
    last_attempted_prompt_idx = idx
    prompt = item["text"]
    try:
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
            "provider": llm.get_active_provider() or "unknown",
            "baseline_score": base_score,
            "mitigated_score": mitigated_score,
            "reduction": base_score - mitigated_score
        })
    except ModelProviderFailure as exc:
        stop_reason = f"provider_failure:{exc.provider}"
        print(
            f"⚠️ Stopping at prompt {idx}: locked provider '{exc.provider}' failed. "
            "Keeping completed prompts only for consistent evaluation."
        )
        break
    except Exception as exc:
        failed_prompts += 1
        print(f"⚠️ Prompt {idx} failed: {exc}")
        continue

    # Persist progress after each successful prompt so partial runs are not lost.
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Processed {idx}/{len(prompts)}")

# ── Convert to DataFrame ─────────────
df = pd.DataFrame(results)

if failed_prompts and stop_reason == "completed_all_prompts":
    stop_reason = "completed_with_skips"

run_metadata = {
    "provider_used": llm.get_active_provider() or "none",
    "total_prompts": len(prompts),
    "completed_prompt_count": len(results),
    "failed_prompt_count": failed_prompts,
    "last_attempted_prompt_idx": last_attempted_prompt_idx,
    "stop_reason": stop_reason,
}

with open(RUN_META_PATH, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, ensure_ascii=False, indent=2)

if df.empty:
    print("No successful evaluations. Check API keys/network and rerun.")
    print(f"Run metadata saved to: {RUN_META_PATH}")
    raise SystemExit(1)

# Save table (for paper)
df.to_csv(RESULTS_PATH, index=False)

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
plt.savefig(BASE_DIR / "bias_comparison.png")

# Bar chart: Reduction by type
plt.figure()
df.groupby("type")["reduction"].mean().plot(kind="bar")
plt.title("Bias Reduction by Prompt Type")
plt.tight_layout()
plt.savefig(BASE_DIR / "reduction_by_type.png")

plt.figure()
df["baseline_score"].plot(kind="hist", bins=10, alpha=0.5, label="Before")
df["mitigated_score"].plot(kind="hist", bins=10, alpha=0.5, label="After")
plt.legend()
plt.title("Distribution of Bias Scores")
plt.xlabel("Bias Score")
plt.tight_layout()
plt.savefig(BASE_DIR / "bias_distribution.png")

plt.figure()
plt.plot(df["baseline_score"], label="Before")
plt.plot(df["mitigated_score"], label="After")
plt.legend()
plt.title("Bias Score per Prompt")
plt.tight_layout()
plt.savefig(BASE_DIR / "bias_per_prompt.png")

BASELINE_MIN_FOR_PERCENT = 0.10
df["percent_reduction"] = (
    (df["baseline_score"] - df["mitigated_score"]) / df["baseline_score"]
) * 100
df.loc[df["baseline_score"] < BASELINE_MIN_FOR_PERCENT, "percent_reduction"] = pd.NA
df["percent_reduction"] = df["percent_reduction"].clip(lower=0, upper=100)

plt.figure()
valid_percent = df["percent_reduction"].dropna()
valid_percent.plot(kind="bar")
plt.title("Percentage Bias Reduction per Prompt")
plt.ylabel("% Reduction")
plt.tight_layout()
plt.savefig(BASE_DIR / "percent_reduction.png")

df.to_csv(RESULTS_PATH, index=False)

print(f"✅ Results saved to: {RESULTS_PATH}")
print(f"✅ Run metadata saved to: {RUN_META_PATH}")
print("✅ Graphs saved!")