# Git Commit Plan — BiasScope Project
# Equal work split across Namritha, Nidhi, Navya
# Files only — no documentation or presentation tasks
# Run each block in sequence on different days

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — PROJECT SETUP & RAG ENGINE
# Assigned to: Namritha Diya Lobo (PES2UG23CS362)
# ─────────────────────────────────────────────────────────────────────────────

# DAY 1 (Week 1 Monday) — Namritha
git init
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
touch .gitignore
# Add to .gitignore: __pycache__/, *.pyc, .env, node_modules/, .DS_Store
git add .gitignore
git commit -m "Initial project setup"

# DAY 2 (Week 1 Wednesday) — Namritha
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
git add backend/requirements.txt backend/.env.example
git commit -m "Add backend requirements and environment config"

# DAY 3 (Week 1 Friday) — Namritha
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
git add backend/rag_engine.py
git commit -m "Implement RAG engine with fairness knowledge base

- Built 10-document fairness KB grounded in IEEE ethics and ACM fairness guidelines
- Implemented keyword+tag overlap retrieval (TF-IDF style)
- Added build_augmented_prompt() to construct fairness-aware prompts
- Tags: gender, occupational, pronouns, stereotype, fairness"

# DAY 4 (Week 2 Tuesday) — Namritha
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
# (Refinement commit after testing retrieval logic)
git add backend/rag_engine.py
git commit -m "Fix retrieval scoring: increase tag-bias_type overlap weight to 2.0

After testing with sample prompts, tag overlap with detected bias types
was under-weighted relative to keyword overlap. Adjusted weights."

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — BIAS DETECTOR MODULE
# Assigned to: Nidhi K (PES2UG23CS383)
# ─────────────────────────────────────────────────────────────────────────────

# DAY 5 (Week 2 Wednesday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
# First commit: skeleton + lexicons
git add backend/bias_detector.py
git commit -m "Add bias detector skeleton with gender and occupational lexicons

- Defined GENDER_MALE_TERMS and GENDER_FEMALE_TERMS sets
- Defined MALE_STEREOTYPED_OCCUPATIONS and FEMALE_STEREOTYPED_OCCUPATIONS
- Added BIAS_PHRASES list with 8 regex patterns
- Scaffold for BiasDetector.analyze()"

# DAY 6 (Week 2 Friday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
git add backend/bias_detector.py
git commit -m "Implement pronoun dominance and occupational stereotype detection

- Gender pronoun dominance: ratio-based score, threshold 0.7
- Occupational stereotype: checks pronoun-occupation co-occurrence
  within 60-token window using regex
- Returns bias_types list and per-metric score dict"

# DAY 7 (Week 3 Monday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
git add backend/bias_detector.py
git commit -m "Add bias phrase pattern matching and negative sentiment scoring

- 8 regex patterns for common stereotyping constructs
- Negative sentiment word proximity check (50-token window)
- Toxicity proxy score using harmful word lexicon"

# DAY 8 (Week 3 Wednesday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
git add backend/bias_detector.py
git commit -m "Add composite bias score with weighted combination

Weights: pronoun_dominance=0.30, occupational=0.35,
phrase=0.25, negative_sentiment=0.10
Bias flagged when composite > 0.15 OR occupational=1 OR phrase_match>0"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — LLM CLIENT & FLASK API
# Assigned to: Navya G N (PES2UG23CS372)
# ─────────────────────────────────────────────────────────────────────────────

# DAY 9 (Week 3 Thursday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add backend/llm_client.py
git commit -m "Implement LLM client wrapping Anthropic Messages API

- Uses claude-sonnet-4-20250514 model
- Supports custom system prompts for baseline vs augmented generation
- Returns extracted text content from response"

# DAY 10 (Week 3 Friday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add backend/app.py
git commit -m "Add Flask REST API with /analyze and /health endpoints

- POST /analyze: baseline generation -> bias detection -> conditional RAG
- Returns structured JSON with bias scores, indicators, retrieved docs
- CORS enabled for frontend integration"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FRONTEND UI
# Assigned to: Navya G N (first half) + Namritha (polish)
# ─────────────────────────────────────────────────────────────────────────────

# DAY 11 (Week 4 Monday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add frontend/index.html
git commit -m "Add frontend skeleton: layout, prompt input, side-by-side panels

- Dark theme UI with CSS variables
- Textarea prompt input with analyze button
- Two-column results grid: Baseline vs Mitigated panels
- Placeholder text while awaiting results"

# DAY 12 (Week 4 Wednesday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add frontend/index.html
git commit -m "Implement analyze() JS function and API integration

- fetch() call to POST /analyze endpoint
- renderResults() populates both panels with response text
- Loading overlay with spinner during API call
- Error banner for failed requests"

# DAY 13 (Week 4 Thursday) — Namritha
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
git add frontend/index.html
git commit -m "Add metric bar charts, bias tags, RAG doc display

- Animated metric bars for all 6 bias scores
- Color-coded bars: green < 0.15, yellow < 0.35, red >= 0.35
- Bias type tags (gender / occupational)
- Retrieved fairness documents section below mitigated panel"

# DAY 14 (Week 4 Friday) — Namritha
git config user.name "Namritha Diya Lobo"
git config user.email "namritha.lobo@pesu.pes.edu"
git add frontend/index.html
git commit -m "Add summary strip, sample prompt chips, keyboard shortcut

- Summary strip shows composite scores + reduction % + RAG status
- 6 pre-written sample bias-eliciting prompts as clickable chips
- Ctrl+Enter keyboard shortcut to submit prompt"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — PAPER & FINAL INTEGRATION
# Assigned to: Nidhi (paper structure) + Navya (integration test fixes)
# ─────────────────────────────────────────────────────────────────────────────

# DAY 15 (Week 5 Monday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
mkdir -p paper
git add paper/paper.tex
git commit -m "Add IEEE LaTeX paper: introduction, related work, architecture sections"

# DAY 16 (Week 5 Wednesday) — Nidhi
git config user.name "Nidhi K"
git config user.email "nidhi.k@pesu.pes.edu"
git add paper/paper.tex
git commit -m "Add evaluation section with results table and qualitative examples to paper"

# DAY 17 (Week 5 Thursday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add backend/app.py backend/bias_detector.py
git commit -m "Fix edge cases: empty response handling, zero division guard in bias detector

- Guard against zero total_gendered tokens in pronoun dominance
- Return 400 error for empty/whitespace prompts in API
- Handle anthropic API exceptions gracefully"

# DAY 18 (Week 5 Friday) — Navya
git config user.name "Navya G N"
git config user.email "navya.gn@pesu.pes.edu"
git add README.md paper/paper.tex
git commit -m "Finalize paper discussion + conclusion sections, update README

- Added limitations and ethical considerations subsections
- Updated README with full API response schema
- Added references table to README"

# ─────────────────────────────────────────────────────────────────────────────
# WORK SPLIT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
# Namritha Diya Lobo (PES2UG23CS362):
#   - backend/rag_engine.py (full)
#   - backend/.env.example, requirements.txt
#   - frontend/index.html (metric bars, tags, RAG docs, summary strip, chips)
#   - .gitignore
#   Commits: 1, 2, 3, 4, 13, 14
#
# Nidhi K (PES2UG23CS383):
#   - backend/bias_detector.py (full)
#   - paper/paper.tex (intro, related work, architecture, evaluation)
#   Commits: 5, 6, 7, 8, 15, 16
#
# Navya G N (PES2UG23CS372):
#   - backend/llm_client.py (full)
#   - backend/app.py (full)
#   - frontend/index.html (layout, API integration, loading states)
#   - Bug fixes + README + paper finalization
#   Commits: 9, 10, 11, 12, 17, 18
# ─────────────────────────────────────────────────────────────────────────────
