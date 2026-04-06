"""
RAG Engine - Retrieval-Augmented Generation for Bias Mitigation
Uses a curated fairness knowledge base to retrieve relevant context
and augment prompts for debiased generation.

Authors: Namritha Diya Lobo
"""

import re
from typing import List, Dict


# ─── Fairness Knowledge Base ──────────────────────────────────────────────────
# Curated from IEEE ethics guidelines, ACM fairness principles,
# and social bias mitigation literature (BOLD, DR.GAP, FairRAG papers)

FAIRNESS_KB = [
    {
        "id": "gender_neutral_language",
        "title": "Gender-Neutral Language in Professional Contexts",
        "content": (
            "Use gender-neutral pronouns and titles when discussing professionals. "
            "Prefer 'they/them' or role titles over gendered pronouns unless the person's "
            "gender is known and relevant. For example, use 'the engineer' instead of "
            "'the male engineer' or 'he'. Avoid defaulting to male pronouns for "
            "leadership roles or female pronouns for caregiving roles."
        ),
        "tags": ["gender", "occupational", "pronouns", "language"],
    },
    {
        "id": "occupational_stereotype_avoidance",
        "title": "Avoiding Occupational Gender Stereotypes",
        "content": (
            "Research shows that occupational stereotyping limits opportunity and reinforces "
            "inequality. Do not assume gender based on profession. Engineering, medicine, "
            "law, and leadership have significant representation from all genders. "
            "Similarly, nursing, teaching, and caregiving are practiced by all genders. "
            "When describing job roles, focus on skills, qualifications, and responsibilities "
            "rather than gendered assumptions."
        ),
        "tags": ["occupational", "gender", "stereotype", "profession"],
    },
    {
        "id": "inclusive_job_descriptions",
        "title": "Writing Inclusive and Unbiased Job Descriptions",
        "content": (
            "Effective, fair job descriptions focus on required skills and competencies, "
            "not gender-coded language. Avoid words like 'aggressive', 'dominant', or "
            "'manpower' which skew male, or 'nurturing' and 'supportive' which skew female. "
            "Instead, use competency-based language: 'collaborative', 'analytical', "
            "'results-driven', 'effective communicator'. Inclusive language attracts "
            "diverse, qualified candidates regardless of gender."
        ),
        "tags": ["occupational", "gender", "language", "inclusive"],
    },
    {
        "id": "gender_parity_stem",
        "title": "Gender Parity in STEM Fields",
        "content": (
            "Gender parity in STEM is an active global effort. Studies show that "
            "implicit bias in language and hiring processes significantly reduce female "
            "representation in technical roles. When discussing STEM professionals, "
            "use inclusive language that does not imply a default gender. Acknowledge "
            "contributions of researchers and engineers of all genders. Avoid "
            "reinforcing narratives that frame technical aptitude as gendered."
        ),
        "tags": ["gender", "occupational", "stem", "engineering", "science"],
    },
    {
        "id": "leadership_bias",
        "title": "Challenging Leadership Gender Bias",
        "content": (
            "Leadership qualities such as decisiveness, vision, and competence are not "
            "gender-specific. Research by McKinsey and the World Economic Forum consistently "
            "shows that diverse leadership teams produce better outcomes. When describing "
            "CEOs, managers, directors, and executives, use gender-neutral language "
            "and avoid adjectives that carry implicit gender coding. Do not describe "
            "leadership styles as 'masculine' or 'feminine'."
        ),
        "tags": ["gender", "occupational", "leadership", "management"],
    },
    {
        "id": "caregiver_stereotype",
        "title": "Challenging Caregiver Gender Stereotypes",
        "content": (
            "Caregiving, teaching, and nurturing are skills and values, not gendered traits. "
            "Men are effective parents, nurses, teachers, and caregivers. Women are effective "
            "engineers, surgeons, and executives. Language that implies caregiving is "
            "inherently feminine or leadership is inherently masculine perpetuates harmful "
            "stereotypes that affect both genders. Use role-based rather than gender-based "
            "framing when discussing any profession."
        ),
        "tags": ["gender", "caregiver", "stereotype", "nurse", "teacher"],
    },
    {
        "id": "fairness_principle_1",
        "title": "IEEE Principle: Equity in AI Outputs",
        "content": (
            "According to IEEE Ethically Aligned Design principles, AI systems must avoid "
            "generating outputs that discriminate based on gender, race, or other protected "
            "characteristics. Outputs should reflect equitable representation and avoid "
            "reinforcing historical inequalities embedded in training data. Fairness requires "
            "both equal treatment across groups and awareness of systemic biases."
        ),
        "tags": ["fairness", "gender", "occupational", "ethics", "ieee"],
    },
    {
        "id": "counterfactual_fairness",
        "title": "Counterfactual Fairness in Language Generation",
        "content": (
            "A response is counterfactually fair if it would remain the same when the "
            "gender of the subject is changed. For example, if a response describes a "
            "'female engineer' differently than a 'male engineer' for the same question, "
            "it is not counterfactually fair. Apply this test: replace gendered terms and "
            "check if the response changes substantively. If it does, revise to treat "
            "all genders equivalently."
        ),
        "tags": ["gender", "fairness", "evaluation", "counterfactual"],
    },
    {
        "id": "pronoun_defaulting",
        "title": "Avoiding Default Pronoun Assumptions",
        "content": (
            "Defaulting to 'he' for doctors, engineers, and scientists, or 'she' for nurses "
            "and teachers is a well-documented source of occupational bias. Use 'they' as a "
            "singular pronoun when gender is unspecified. If multiple examples are given, "
            "vary pronouns across genders or use neutral forms consistently. "
            "This applies to fictional scenarios, examples, and hypothetical professionals."
        ),
        "tags": ["gender", "pronouns", "occupational", "language"],
    },
    {
        "id": "intersectional_fairness",
        "title": "Intersectional Fairness Considerations",
        "content": (
            "Bias in language often intersects multiple dimensions: gender, race, culture, "
            "and profession together shape stereotypes. A fair response considers that "
            "professional capability is not predicted by gender, race, or background. "
            "Avoid generalizations about which groups 'naturally' excel at certain roles. "
            "Describe individuals and roles by their actual skills, achievements, and "
            "responsibilities rather than demographic assumptions."
        ),
        "tags": ["gender", "occupational", "fairness", "intersectional"],
    },
]


class RAGEngine:
    """
    Simple keyword and tag-based retrieval over the fairness knowledge base.
    In production this would use vector embeddings (e.g., sentence-transformers).
    For this project, TF-IDF-style overlap is used for transparency and reproducibility.
    """

    def __init__(self):
        self.kb = FAIRNESS_KB

    def retrieve(self, prompt: str, bias_types: List[str], top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant fairness documents given the prompt and detected bias types.
        Returns top_k documents.
        """
        prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))

        scored_docs = []
        for doc in self.kb:
            score = 0.0

            # Tag overlap with bias types
            tag_overlap = len(set(bias_types) & set(doc["tags"]))
            score += tag_overlap * 2.0

            # Keyword overlap between prompt and document content + title
            doc_words = set(
                re.findall(r"\b\w+\b", (doc["content"] + " " + doc["title"]).lower())
            )
            word_overlap = len(prompt_words & doc_words)
            score += word_overlap * 0.3

            # Tag overlap with prompt words
            tag_word_overlap = len(prompt_words & set(doc["tags"]))
            score += tag_word_overlap * 1.5

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def build_augmented_prompt(self, original_prompt: str, retrieved_docs: List[Dict]) -> str:
        """
        Augments the original prompt with retrieved fairness context.
        The system will instruct the LLM to answer using fairness principles.
        """
        if not retrieved_docs:
            return original_prompt

        context_sections = []
        for doc in retrieved_docs:
            context_sections.append(f"[{doc['title']}]\n{doc['content']}")

        fairness_context = "\n\n".join(context_sections)

        augmented = (
            f"You must answer the following question in a way that is free from gender bias "
            f"and occupational stereotypes. Use gender-neutral language. Do not assume gender "
            f"based on profession. Treat all genders as equally capable in all fields.\n\n"
            f"Fairness guidelines to follow:\n{fairness_context}\n\n"
            f"Now answer this question with these principles in mind:\n{original_prompt}"
        )
        return augmented
