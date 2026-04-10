"""
rag_engine.py — Retrieval-Augmented Generation Engine for Bias Mitigation
Loads a curated fairness knowledge base from a structured text file and
retrieves relevant context to augment prompts for debiased LLM generation.

Improvements over v1:
  - KB loaded from fairness_kb.txt (structured, sourced, version-controlled)
  - IDF-weighted retrieval: rare diagnostic terms score higher than stopwords
  - Tag + IDF hybrid scoring for strong recall on both explicit and implicit bias
  - Semantic deduplication: avoids returning near-duplicate KB entries
  - Lean prompt augmentation: extracts actionable principles, not full documents
  - Separate system prompt builder for Gemini 2.5 Flash API
  - Retrieval evaluation harness: measures precision@k and recall@k

Research basis:
  - FairRAG: Gaglione et al. (2024) CVPR
  - ACM FAccT (2022) empirical evaluation requirements
  - NIST RMF 1.0 (2023) governance requirements
  - Kotek et al. (2023) — system prompt + user prompt augmentation ablation

Authors: Namritha Diya Lobo
"""

import re
import math
import os
from typing import List, Dict, Optional
from collections import Counter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KB_FILE = os.path.join(os.path.dirname(__file__), "fairness_kb.txt")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "in", "that", "it", "for", "on", "with", "this",
    "they", "their", "these", "those", "at", "by", "from", "as", "or",
    "but", "not", "all", "when", "which", "who", "what", "how", "its",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "must", "shall", "about", "into",
    "than", "then", "so", "such", "both", "each", "more", "also", "other",
    "any", "no", "only", "same", "very", "just", "if", "up", "out",
}


# ---------------------------------------------------------------------------
# KB parser
# ---------------------------------------------------------------------------

def _parse_kb_file(filepath: str) -> List[Dict]:
    """
    Parses fairness_kb.txt into a list of entry dicts.

    Each entry is delimited by ---ENTRY--- and contains structured fields:
    id, title, tags, sources, content, mitigation_strategies.

    Returns:
        List of parsed entry dicts. Raises FileNotFoundError if KB missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fairness KB file not found at: {os.path.abspath(filepath)}\n"
            f"Ensure fairness_kb.txt is in the same directory as rag_engine.py"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    raw_entries = re.split(r"^---ENTRY---$", raw, flags=re.MULTILINE)
    entries: List[Dict] = []

    for block in raw_entries:
        block = block.strip()
        if not block or block.startswith("#"):
            continue

        entry: Dict = {
            "id": "", "title": "", "tags": [],
            "sources": "", "content": "", "mitigation_strategies": [],
        }

        # Single-line fields
        for field in ("id", "title", "sources"):
            m = re.search(rf"^{field}:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
            if m:
                entry[field] = m.group(1).strip()

        # Tags (comma-separated)
        m = re.search(r"^tags:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
        if m:
            entry["tags"] = [t.strip() for t in m.group(1).split(",")]

        # Content block: from "content:\n" to "mitigation_strategies:" or EOF
        m = re.search(
            r"^content:\s*\n(.*?)(?=^mitigation_strategies:|$)",
            block, re.MULTILINE | re.DOTALL,
        )
        if m:
            entry["content"] = m.group(1).strip()

        # Mitigation strategies: bullet list under "mitigation_strategies:"
        m = re.search(
            r"^mitigation_strategies:\s*\n(.*?)(?=$)",
            block, re.MULTILINE | re.DOTALL,
        )
        if m:
            entry["mitigation_strategies"] = [
                s.strip().lstrip("-").strip()
                for s in m.group(1).splitlines()
                if s.strip().startswith("-")
            ]

        if entry["id"] and entry["content"]:
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Retrieval-Augmented Generation engine for fairness-aware prompt augmentation.

    Retrieval: IDF-weighted keyword overlap + tag matching with semantic
    deduplication. Designed for transparency and reproducibility — all
    scoring is rule-based with no ML dependencies.

    For production with larger KBs (50+ documents), replace _score_doc()
    with sentence-transformer cosine similarity for better semantic recall.
    """

    def __init__(self, kb_filepath: str = KB_FILE):
        """
        Args:
            kb_filepath: Path to fairness_kb.txt. Defaults to same directory
                         as this file.
        """
        self.kb = _parse_kb_file(kb_filepath)
        self._doc_word_sets: List[set] = []
        self._idf: Dict[str, float] = {}
        self._build_idf()

    # -----------------------------------------------------------------------
    # IDF precomputation
    # -----------------------------------------------------------------------

    def _build_idf(self) -> None:
        """
        Precomputes IDF weights across the KB corpus.

        IDF(term) = log((N+1) / (df+1))

        High IDF = rare, diagnostic term (e.g. "counterfactual", "stereotype",
        "agentic"). Low IDF = common term appearing in most documents.
        """
        N = len(self.kb)
        doc_freq: Counter = Counter()

        for doc in self.kb:
            text = (
                doc["content"] + " " + doc["title"]
                + " " + " ".join(doc["tags"])
            ).lower()
            words = set(re.findall(r"\b\w+\b", text)) - _STOPWORDS
            self._doc_word_sets.append(words)
            doc_freq.update(words)

        self._idf = {
            w: math.log((N + 1) / (df + 1))
            for w, df in doc_freq.items()
        }

    # -----------------------------------------------------------------------
    # Public: retrieve
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        prompt: str,
        bias_types: List[str],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieves the top-k most relevant and diverse KB entries.

        Scoring per document:
            idf_score  = sum IDF(w) for w in (prompt_words INTERSECT doc_words)
            tag_score  = |bias_types INTERSECT doc_tags| * 4.0
            ptag_score = |prompt_words INTERSECT doc_tags| * 2.0
            total      = idf_score + tag_score + ptag_score

        After ranking, semantic deduplication ensures the returned documents
        cover diverse facets rather than returning near-identical entries.

        Args:
            prompt:     Original user prompt (keyword extraction target).
            bias_types: Detected bias types from BiasDetector (tag matching).
            top_k:      Max number of documents to return.

        Returns:
            List of up to top_k KB entry dicts.
        """
        prompt_words  = set(re.findall(r"\b\w+\b", prompt.lower())) - _STOPWORDS
        bias_type_set = set(bias_types)

        scored: List[tuple] = []
        for i, doc in enumerate(self.kb):
            overlap   = prompt_words & self._doc_word_sets[i]
            idf_score = sum(self._idf.get(w, 0.0) for w in overlap)

            tag_set      = set(doc["tags"])
            bias_overlap = len(bias_type_set & tag_set)
            ptag_overlap = len(prompt_words & tag_set)

            score = idf_score + (bias_overlap * 4.0) + (ptag_overlap * 2.0)
            scored.append((score, i, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Semantic deduplication: reject candidates sharing >1 tag with
        # already-selected results to ensure topical diversity
        seen_tags: set        = set()
        results:   List[Dict] = []

        for _, _, doc in scored:
            tag_set = set(doc["tags"])
            if not results or len(tag_set & seen_tags) <= 1:
                results.append(doc)
                seen_tags.update(tag_set)
            if len(results) >= top_k:
                break

        # Fallback: if deduplication pruned too aggressively, fill remainder
        if len(results) < top_k:
            selected_ids = {d["id"] for d in results}
            for _, _, doc in scored:
                if doc["id"] not in selected_ids:
                    results.append(doc)
                    if len(results) >= top_k:
                        break

        return results

    # -----------------------------------------------------------------------
    # Public: build_augmented_prompt
    # -----------------------------------------------------------------------

    def build_augmented_prompt(
        self,
        original_prompt: str,
        retrieved_docs: List[Dict],
        bias_types: Optional[List[str]] = None,
    ) -> str:
        """
        Constructs a fairness-augmented user-turn prompt for Gemini 2.5 Flash.

        Design (grounded in FairRAG ablation, Gaglione et al. 2024):
          - Lean injection: one actionable principle per document (not full text)
            Full document injection causes over-hedging quality degradation.
          - Specific over general: concrete actions outperform vague fairness
            statements (Kotek 2023 system prompt ablation).
          - Positive framing: "use they for unspecified gender" outperforms
            "do not use gendered pronouns".
          - Explicit anti-over-correction instruction to preserve answer quality.

        Args:
            original_prompt: The user's original question.
            retrieved_docs:  Output of retrieve().
            bias_types:      Detected bias types for framing the instruction.

        Returns:
            Augmented prompt string (user turn).
        """
        if not retrieved_docs:
            return original_prompt

        # Extract the single most actionable principle from each retrieved doc
        principles: List[str] = []
        for doc in retrieved_docs:
            p = self._extract_actionable_principle(doc["content"])
            if p:
                principles.append(p)

        # Gather concrete mitigation strategies (max 2 per doc, max 4 total)
        strategies: List[str] = []
        for doc in retrieved_docs:
            for s in doc.get("mitigation_strategies", [])[:2]:
                if s and s not in strategies:
                    strategies.append(s)
                if len(strategies) >= 4:
                    break

        bias_ctx = (
            f" with attention to {', '.join(bias_types)} bias"
            if bias_types else ""
        )

        lines = [f"Answer the following question{bias_ctx}.", ""]

        if principles:
            lines.append("Fairness principles to apply:")
            for i, p in enumerate(principles, 1):
                lines.append(f"  {i}. {p}")
            lines.append("")

        if strategies:
            lines.append("Specific actions:")
            for s in strategies:
                lines.append(f"  - {s}")
            lines.append("")

        lines += [
            "Provide a complete, informative response. Do not over-hedge, refuse "
            "to answer, or sacrifice accuracy in the name of fairness. Treat all "
            "genders as equally capable in all professional roles.",
            "",
            f"Question: {original_prompt}",
        ]

        return "\n".join(lines)

    def build_system_prompt(self, bias_types: Optional[List[str]] = None) -> str:
        """
        Builds a Gemini 2.5 Flash system-level fairness instruction.

        Use as the 'system_instruction' in the Gemini API call.
        Combining system prompt + augmented user turn gives the strongest
        bias reduction in ablation studies (Kotek et al. 2023).

        Args:
            bias_types: Detected bias types for targeted instruction framing.

        Returns:
            System instruction string for Gemini 2.5 Flash.
        """
        bias_ctx = (
            f", particularly around {', '.join(bias_types)} bias"
            if bias_types else ""
        )
        return (
            f"You are a helpful, accurate, and fair assistant. "
            f"You are aware of documented gender and occupational bias in LLM "
            f"outputs{bias_ctx}. "
            f"Follow these principles in every response: "
            f"(1) Use 'they/them' for professionals whose gender is unspecified. "
            f"(2) Describe professionals through skills and qualifications, not "
            f"gendered assumptions. "
            f"(3) Attribute agentic traits (decisive, analytical, strategic) and "
            f"communal traits (collaborative, empathetic, caring) equally across "
            f"all genders — do not reserve one type for a single gender. "
            f"(4) Apply counterfactual fairness: your description of a role should "
            f"not change substantively if the subject's gender were swapped. "
            f"Maintain full answer quality and informativeness."
        )

    # -----------------------------------------------------------------------
    # Public: evaluate_retrieval
    # -----------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        top_k: int = 3,
    ) -> Dict:
        """
        Measures retrieval quality against a labeled evaluation set.

        Required by ACM FAccT (2022) and NIST RMF 1.0 for empirically
        validated debiasing systems. A system with a fairness KB but
        untested retrieval cannot claim effective mitigation.

        Args:
            test_cases: List of dicts:
                {
                    "prompt":       str,
                    "bias_types":   List[str],
                    "expected_ids": List[str]  -- KB entry ids
                }
            top_k: k for precision@k and recall@k.

        Returns:
            {
                "precision_at_k": float,
                "recall_at_k":    float,
                "k":              int,
                "n_cases":        int,
                "per_case":       List[dict]
            }
        """
        precisions: List[float] = []
        recalls:    List[float] = []
        per_case:   List[dict]  = []

        for case in test_cases:
            retrieved     = self.retrieve(case["prompt"], case["bias_types"], top_k)
            retrieved_ids = {doc["id"] for doc in retrieved}
            expected_ids  = set(case.get("expected_ids", []))

            if not expected_ids:
                continue

            hits      = retrieved_ids & expected_ids
            precision = len(hits) / top_k
            recall    = len(hits) / len(expected_ids)

            precisions.append(precision)
            recalls.append(recall)
            per_case.append({
                "prompt":        case["prompt"][:60] + "...",
                "retrieved_ids": list(retrieved_ids),
                "expected_ids":  list(expected_ids),
                "hits":          list(hits),
                "precision":     round(precision, 3),
                "recall":        round(recall, 3),
            })

        if not precisions:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "k": top_k,
                "n_cases": 0,
                "per_case": [],
            }

        return {
            "precision_at_k": round(sum(precisions) / len(precisions), 3),
            "recall_at_k":    round(sum(recalls) / len(recalls), 3),
            "k":              top_k,
            "n_cases":        len(precisions),
            "per_case":       per_case,
        }

    # -----------------------------------------------------------------------
    # Public: KB introspection
    # -----------------------------------------------------------------------

    def list_kb_entries(self) -> List[Dict]:
        """Returns a lightweight summary of all loaded KB entries."""
        return [
            {
                "id":           doc["id"],
                "title":        doc["title"],
                "tags":         doc["tags"],
                "sources":      doc["sources"],
                "n_strategies": len(doc.get("mitigation_strategies", [])),
            }
            for doc in self.kb
        ]

    def get_kb_size(self) -> int:
        """Returns the number of documents currently loaded in the KB."""
        return len(self.kb)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_actionable_principle(self, content: str) -> str:
        """
        Extracts the single most actionable sentence from a KB content block.

        Prioritises sentences containing action verbs. Falls back to the first
        sentence. Truncates to 200 chars to keep the augmented prompt lean.
        """
        action_re = re.compile(
            r"\b(use|avoid|prefer|do not|don't|instead|focus on|apply|"
            r"replace|ensure|check|treat|describe|default to|flag|"
            r"omit|vary|attribute|refer to|consider)\b",
            re.IGNORECASE,
        )
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", content.strip())
            if len(s.strip()) > 20
        ]

        for s in sentences:
            if action_re.search(s):
                return s[:200].rstrip(".") + ("." if len(s) <= 200 else "...")

        return sentences[0][:200] if sentences else ""