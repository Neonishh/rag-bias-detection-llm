# BiasScope — Prompt & RAG-Based Social Bias Detection in LLM Outputs

**Team:**
| Name | USN |
|------|-----|
| Namritha Diya Lobo | PES2UG23CS362 |
| Nidhi K | PES2UG23CS383 |
| Navya G N | PES2UG23CS372 |

---

## Project Overview

BiasScope is a lightweight, inference-time framework to **detect and mitigate gender and occupational bias** in LLM outputs using:

- **Prompt Engineering** — augmenting prompts with fairness-aware instructions
- **Retrieval-Augmented Generation (RAG)** — retrieving curated fairness guidelines to guide the LLM
- **Real-time Bias Analysis** — scoring LLM responses on multiple bias dimensions
- **Side-by-side UI** — comparing baseline vs. mitigated responses live

No model retraining. No fine-tuning. Works with any LLM API.

---

## Architecture

```
User Prompt
    │
    ▼
Baseline LLM Response  ──→  Bias Detector (NLP scoring)
                                   │
                         bias_detected = True?
                                   │
                    ┌──── NO ──────┴───── YES ────┐
                    │                             │
            Show as mitigated           RAG Engine retrieves
                                        fairness KB docs
                                               │
                                    Augmented Prompt sent to LLM
                                               │
                                    Mitigated Response + Re-evaluate
```

---

## Bias Detection Metrics

| Metric | Description |
|--------|-------------|
| **Composite Bias Score** | Weighted combination of all signals (0–1) |
| **Gender Pronoun Dominance** | Asymmetry between male/female pronoun usage |
| **Occupational Stereotype** | Gendered pronoun + stereotyped occupation pairing |
| **Bias Phrase Score** | Pattern matching on known stereotyping phrases |
| **Negative Sentiment Score** | Negative words near gendered terms |
| **Toxicity Score** | Presence of harmful/offensive language |

---

## Setup & Running

### Prerequisites
- Python 3.9+
- A Google AI Studio API key (Gemini)

### Backend

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
# On macOS/Linux use: cp .env.example .env
# Edit .env and set GEMINI_API_KEY
python app.py
```

Backend runs at `http://localhost:5000`

### Frontend

Open `frontend/index.html` directly in a browser (no build step needed).

Or serve it:
```bash
cd frontend
python -m http.server 8080
# Open http://localhost:8080
```

---

## File Structure

```
bias-detector/
├── backend/
│   ├── app.py           # Flask API server
│   ├── bias_detector.py # NLP-based bias analysis engine
│   ├── rag_engine.py    # Fairness KB + RAG retrieval
│   ├── llm_client.py    # Gemini API wrapper
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html       # Full UI (single file, no build needed)
├── paper/
│   └── paper.tex        # IEEE-format LaTeX paper
└── README.md
```

---

## API Endpoints

### `POST /analyze`
```json
{
  "prompt": "Describe a typical software engineer."
}
```

**Response:**
```json
{
  "prompt": "...",
  "baseline": {
    "response": "...",
    "bias_analysis": {
      "bias_detected": true,
      "bias_types": ["gender", "occupational"],
      "composite_bias_score": 0.42,
      "toxicity_score": 0.0,
      "detailed_scores": { ... },
      "indicators": ["Strong male pronoun dominance (male: 5, female: 0)"],
      "summary": "Bias detected: gender, occupational. Score: 0.42"
    }
  },
  "mitigated": {
    "response": "...",
    "bias_analysis": { ... },
    "rag_applied": true,
    "retrieved_docs": [{ "title": "...", "content": "..." }]
  }
}
```

---

## Research Papers Referenced

1. Dhamala et al. (2021). **BOLD: Dataset and Metrics for Measuring Bias in Open-Ended Language Generation.** FAccT.
2. Howard et al. (2024). **FairRAG: Fair Human Generation via Fair Retrieval Augmentation.** CVPR.
3. Gallegos et al. (2023). **ROBBIE: Robust Bias Evaluation of Large Generative Language Models.** EMNLP.
4. Kotek et al. (2023). **Bias and Fairness in Chatbots: An Overview.** ACL.
5. Gupta et al. (2025). **Bias Mitigation Agent: Optimizing Source Selection for Fair Knowledge Retrieval.**
6. Bhaskar et al. (2025). **Bias Evaluation and Mitigation in Retrieval-Augmented Medical QA Systems.**
7. Wang et al. (2023). **DR.GAP: Mitigating Bias in LLMs using Gender-Aware Prompting.**
8. Gallegos et al. (2024). **A Survey on Fairness in Large Language Models.** ACL.
9. Li et al. (2024). **Unveiling and Mitigating Bias in LLM Recommendations.**
