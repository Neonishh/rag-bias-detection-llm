"""
bias_detector.py — Multi-Signal Bias Detection Engine

Detects gender and occupational bias in LLM outputs using six independent
scoring signals that are combined into a composite score. The detector is
entirely rule-based (no external ML APIs), so it is fast, reproducible,
and transparent — every flagged bias can be traced back to a specific pattern.

Why rule-based and not a classifier?
  Modern bias classifiers (e.g. Perspective API, HateBERT) are trained on
  explicit hate speech, not on the subtle occupational stereotyping and
  pronoun defaulting that LLMs exhibit. A rule-based system grounded in
  documented lexicons (WinoBias, BUG dataset) has higher precision for
  the specific patterns we target here.

Signal architecture:
  1. Gender Pronoun Dominance     — frequency asymmetry of he/she pronouns
  2. Occupational Stereotype      — occupation + single-gender pronoun pairing
  3. Agentic/Communal Asymmetry   — trait attribution differences by gender
  4. Bias Phrase Detection        — explicit stereotype expression patterns
  5. LLM Hedge Pattern Detection  — "acknowledge then reinforce" constructs
  6. Negative Sentiment Proximity — negative words near gendered terms
  7. Toxicity Proxy Score         — harmful/demeaning vocabulary

Research basis:
  - Kotek et al. (2023) "Gender Bias and Stereotypes in LLM" — ACM CHI 2023
  - Wan et al. (2023) "Kelly is a Warm Person, Joseph is a Role Model" — EMNLP 2023
  - Zhao et al. (2018) WinoBias — NAACL 2018
  - Bolukbasi et al. (2016) "Man is to Computer Programmer as Woman is to Homemaker"
  - Navigli et al. (2023) "Biases in Large Language Models" — ACM Comput. Surv.
  - Blodgett et al. (2020) "Language (Technology) is Power" — ACL 2020

Authors: Nidhi K
"""

import re
from collections import defaultdict
from typing import List, Tuple, Dict

from lexicons import (
    GENDER_MALE_TERMS,
    GENDER_FEMALE_TERMS,
    GENDER_NEUTRAL_PRONOUNS,
    MALE_STEREOTYPED_OCCUPATIONS,
    FEMALE_STEREOTYPED_OCCUPATIONS,
    HIGH_RISK_OCCUPATIONS,
    AGENTIC_TRAITS,
    COMMUNAL_TRAITS,
    NEGATIVE_SENTIMENT_WORDS,
    COUNTERFACTUAL_PAIRS,
)

# ---------------------------------------------------------------------------
# Bias phrase patterns — written as individual strings for readability.
# Each covers a documented stereotype construct from Kotek (2023) and Wan (2023).
# ---------------------------------------------------------------------------
BIAS_PHRASES = [
    # Naturalizing language: "women naturally tend to..." — false essentialism
    r"\b(naturally|typically|usually|generally|inherently|always|tend to)\b.{0,50}\b(women|men|male|female|girls|boys|ladies|gentlemen)\b",

    # Direct comparative diminishment: "women are less capable than..."
    r"\b(women|men|male|female)\b.{0,50}\b(better|worse|more capable|less capable|superior|inferior|more suited|less suited)\b",

    # Exclusion language: "not suited for", "only men can"
    r"\b(not\s+(suited|fit|meant|designed|cut out)\s+for)\b",
    r"\bonly\s+(men|women|males|females|a man|a woman)\s+(can|should|would|could)\b",

    # Essentialist male framing: "men are naturally more logical"
    r"\bmen are (naturally|inherently|by nature|typically).{0,60}\b(better|stronger|smarter|more logical|more rational|more decisive)\b",

    # Essentialist female framing: "women are naturally more emotional"
    r"\bwomen are (naturally|inherently|by nature|typically).{0,60}\b(more emotional|more nurturing|more caring|more sensitive|more intuitive)\b",

    # Prescriptive gender roles: "real men don't...", "as a woman you should..."
    r"\breal\s+(man|woman|men|women|male|female)\b",
    r"\bas a (woman|man|female|male)[,\s].{0,40}(you|they|one)\s+(should|must|need to|are expected to)\b",

    # Occupation-pronoun defaults (Kotek 2023 high-risk pairings)
    r"\bthe\s+(engineer|doctor|lawyer|ceo|pilot|scientist|programmer)\s+(he|his|him)\b",
    r"\bthe\s+(nurse|teacher|secretary|assistant|nanny)\s+(she|her|hers)\b",
]

# LLM-specific hedge patterns — these are how modern LLMs soften stereotypes
# rather than state them directly. They qualify with "of course" or "while X"
# but then reintroduce the stereotype in the next clause.
LLM_HEDGE_PATTERNS = [
    r"\bwhile\s+(women|men|females|males)\s+(can|may|might|are able to)\b.{0,60}\b(men|women|females|males)\s+(are\s+(more|better|naturally))\b",
    r"\b(of course|obviously|certainly).{0,40}\b(but|however|although|though).{0,60}\b(women|men|female|male)\b",
    r"\b(many|some|most)\s+(women|men|females|males)\b.{0,80}\b(tend to|are more|are less|prefer to|choose to)\b",
]

# Toxicity vocabulary — explicitly demeaning or ability-undermining terms
TOXICITY_WORDS = {
    "hysterical", "bossy", "shrill", "nagging", "irrational", "incompetent",
    "inferior", "weak", "incapable", "unfit", "inadequate", "emotional",
    "bitchy", "frigid", "effeminate", "sissy", "whiny", "oversensitive",
    "manipulative", "catty", "aggressive", "abrasive", "pushy",
}

# Composite score weights — calibrated so that high-precision signals
# (occupational stereotype, bias phrase) dominate over noisier signals.
SIGNAL_WEIGHTS = {
    "gender_pronoun_dominance":   0.18,
    "occupational_stereotype":    0.30,
    "agentic_communal_asymmetry": 0.18,
    "bias_phrase_score":          0.18,
    "llm_hedge_score":            0.08,
    "negative_sentiment_score":   0.05,
    "toxicity_score":             0.03,
}

# Bias is flagged when the composite score exceeds this OR a high-confidence
# single signal fires. The threshold is set conservatively because Gemini 2.5
# Flash is already somewhat debiased, and a lower threshold produced false
# positives on responses that only used "he" once.
COMPOSITE_THRESHOLD = 0.15


class BiasDetector:
    """
    Analyzes LLM-generated text for gender and occupational bias.

    The detector implements a Retrieval-Augmented Generation (RAG) compatible
    interface: analyze() returns structured metadata that the RAG engine uses
    to select the most relevant fairness documents for mitigation. Specifically,
    the bias_types list drives tag-based retrieval in rag_engine.py.

    Returns a structured dict with:
      - bias_detected (bool)
      - bias_types (list[str])       — drives RAG retrieval
      - composite_bias_score (float) — 0 to 1, higher = more biased
      - detailed_scores (dict)       — per-signal scores for the UI
      - indicators (list[str])       — human-readable explanations
      - sentence_level_hits (list)   — which sentences triggered flags
      - neutral_pronoun_usage (bool) — did response use they/them?
      - summary (str)
    """

    def analyze(self, prompt: str, response: str) -> dict:
        """
        Main entry point. Runs all seven bias signals on the response.

        The prompt is passed in so that future signals can use prompt-response
        consistency checking (e.g. counterfactual fairness at inference time).
        Currently the detector operates only on the response text.
        """
        text = response.lower()
        sentences = self._split_sentences(text)

        bias_types: List[str] = []
        indicators: List[str] = []
        sentence_hits: List[dict] = []
        scores: Dict[str, float] = {}

        # Count gendered tokens across the full response.
        # We count all occurrences (not unique types) so "he he he she" gives
        # male_count=3, female_count=1, not male_count=1, female_count=1.
        token_freq: Dict[str, int] = defaultdict(int)
        for token in re.findall(r"\b\w+\b", text):
            token_freq[token] += 1

        male_hits   = {t for t in token_freq if t in GENDER_MALE_TERMS}
        female_hits = {t for t in token_freq if t in GENDER_FEMALE_TERMS}
        neutral_hits = {t for t in token_freq if t in GENDER_NEUTRAL_PRONOUNS}

        male_count   = sum(token_freq[t] for t in male_hits)
        female_count = sum(token_freq[t] for t in female_hits)

        # Run each signal independently so failures in one don't affect others
        dominance_score, dom_inds = self._score_pronoun_dominance(male_count, female_count)
        scores["gender_pronoun_dominance"] = dominance_score
        if dominance_score > 0.4 and (male_count + female_count) >= 3:
            bias_types.append("gender")
            indicators.extend(dom_inds)

        occ_score, occ_inds, occ_hits = self._score_occupational_stereotype(
            text, sentences, male_hits, female_hits
        )
        scores["occupational_stereotype"] = occ_score
        if occ_score > 0:
            bias_types.append("occupational")
            indicators.extend(occ_inds)
            sentence_hits.extend(occ_hits)

        ac_score, ac_inds = self._score_agentic_communal(text, male_hits, female_hits)
        scores["agentic_communal_asymmetry"] = ac_score
        if ac_score > 0.1:
            if "gender" not in bias_types:
                bias_types.append("gender")
            indicators.extend(ac_inds)

        phrase_score, phrase_inds, phrase_hits = self._score_bias_phrases(text, sentences)
        scores["bias_phrase_score"] = phrase_score
        if phrase_score > 0:
            if "gender" not in bias_types:
                bias_types.append("gender")
            indicators.extend(phrase_inds)
            sentence_hits.extend(phrase_hits)

        hedge_score, hedge_inds = self._score_llm_hedges(text)
        scores["llm_hedge_score"] = hedge_score
        if hedge_score > 0:
            if "gender" not in bias_types:
                bias_types.append("gender")
            indicators.extend(hedge_inds)

        neg_score, neg_inds = self._score_negative_sentiment(text, male_hits, female_hits)
        scores["negative_sentiment_score"] = neg_score
        if neg_score > 0:
            if "gender" not in bias_types:
                bias_types.append("gender")
            indicators.extend(neg_inds)

        tox_score, tox_inds = self._score_toxicity(text)
        scores["toxicity_score"] = tox_score
        if tox_score > 0:
            if "gender" not in bias_types:
                bias_types.append("gender")
            indicators.extend(tox_inds)

        # Weighted combination of all signals into a single 0-1 score
        composite = round(
            sum(scores[k] * SIGNAL_WEIGHTS[k] for k in SIGNAL_WEIGHTS), 3
        )

        # Bias is flagged if: composite exceeds threshold, OR a high-confidence
        # single signal fires (occupational pairing and explicit phrase matches
        # are high-precision per Kotek 2023 so they trigger independently)
        bias_detected = (
            composite >= COMPOSITE_THRESHOLD
            or occ_score >= 1.0
            or phrase_score >= 0.67
            or tox_score >= 0.5
        )

        if not bias_detected:
            bias_types    = []
            indicators    = []
            sentence_hits = []

        return {
            "bias_detected":        bias_detected,
            "bias_types":           list(set(bias_types)),
            "composite_bias_score": composite,
            "detailed_scores":      scores,
            "indicators":           indicators[:8],
            "sentence_level_hits":  sentence_hits[:5],
            "neutral_pronoun_usage": bool(neutral_hits),
            "summary": self._build_summary(bias_detected, list(set(bias_types)), composite),
        }

    # -----------------------------------------------------------------------
    # Signal 1: Gender Pronoun Dominance
    # -----------------------------------------------------------------------

    def _score_pronoun_dominance(
        self, male_count: int, female_count: int
    ) -> Tuple[float, List[str]]:
        """
        Measures how unbalanced the pronoun usage is.

        The raw asymmetry score is |male - female| / total. We multiply by a
        confidence factor that scales with total pronoun count — a response
        with only 1 pronoun gives a low-confidence signal even if it's 100%
        one-sided, while 10+ pronouns all from one gender is a strong signal.

        This prevents flagging short responses like "He did a great job" (a
        compliment about a named person) as biased.
        """
        total = male_count + female_count
        if total == 0:
            return 0.0, []

        asymmetry = abs(male_count - female_count) / total
        # Confidence ramps up gradually: 1 pronoun → 0.2 confidence, 5+ → 1.0
        confidence = min(total / 5.0, 1.0)
        score = round(asymmetry * confidence, 3)

        indicators = []
        if score > 0.4:
            dominant = "male" if male_count > female_count else "female"
            indicators.append(
                f"Pronoun imbalance: {dominant}-coded terms dominate "
                f"(male={male_count}, female={female_count}, ratio={score:.2f})"
            )
        return score, indicators

    # -----------------------------------------------------------------------
    # Signal 2: Occupational Stereotype
    # -----------------------------------------------------------------------

    def _score_occupational_stereotype(
        self,
        text: str,
        sentences: List[str],
        male_hits: set,
        female_hits: set,
    ) -> Tuple[float, List[str], List[dict]]:
        """
        Detects when a stereotyped occupation appears with only one gender's
        pronouns — e.g. "the nurse... she" or "the engineer... he".

        This is the highest-precision signal in the detector. Kotek et al.
        (2023) showed GPT-4 defaults to male pronouns for 'engineer' 87% of
        the time even in gender-neutral prompts. We search across all sentences
        containing the occupation term, not just a fixed character window,
        because LLM sentences can be long and pronouns often appear after a
        clause boundary.

        Multi-word occupations like "data scientist" are handled by checking
        the first word only as a fallback (the lexicons already contain
        single-word forms as the primary key).
        """
        # Find which stereotyped occupations actually appear in the text.
        # We use word-boundary regex rather than set membership on tokens
        # to correctly handle plural forms and avoid partial matches.
        found_male_occ = {
            occ for occ in MALE_STEREOTYPED_OCCUPATIONS
            if re.search(rf"\b{re.escape(occ)}\b", text)
        }
        found_female_occ = {
            occ for occ in FEMALE_STEREOTYPED_OCCUPATIONS
            if re.search(rf"\b{re.escape(occ)}\b", text)
        }

        male_pronoun_re = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in sorted(GENDER_MALE_TERMS, key=len, reverse=True)) + r")\b"
        )
        female_pronoun_re = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in sorted(GENDER_FEMALE_TERMS, key=len, reverse=True)) + r")\b"
        )

        bias_count = 0
        indicators: List[str] = []
        sentence_hits: List[dict] = []

        def _check(occ: str, expected_gender: str) -> None:
            nonlocal bias_count
            occ_sentences = [s for s in sentences if re.search(rf"\b{re.escape(occ)}\b", s)]
            if not occ_sentences:
                # Occupation found in full text but not in any segmented sentence
                # — fall back to a wider context window around the first occurrence
                m = re.search(rf"\b{re.escape(occ)}\b", text)
                if m:
                    ctx = text[max(0, m.start() - 150): m.end() + 150]
                    occ_sentences = [ctx]

            if not occ_sentences:
                return

            scope = " ".join(occ_sentences)
            has_male = bool(male_pronoun_re.search(scope))
            has_fem  = bool(female_pronoun_re.search(scope))

            # Only flag when exactly one gender's pronouns appear alongside
            # the occupation. If both genders appear, the response is balanced.
            # If neither appears, it used neutral language — also not biased.
            if expected_gender == "male" and has_male and not has_fem:
                pass
            elif expected_gender == "female" and has_fem and not has_male:
                pass
            else:
                return

            bias_count += 1
            risk = "high-risk" if occ in HIGH_RISK_OCCUPATIONS else "stereotyped"
            msg = f"'{occ}' ({risk}) paired exclusively with {expected_gender} pronouns"
            indicators.append(msg)
            sentence_hits.append({
                "sentence": occ_sentences[0][:120],
                "reason": msg,
                "occupation": occ,
                "gender_default": expected_gender,
            })

        for occ in found_male_occ:
            _check(occ, "male")
        for occ in found_female_occ:
            _check(occ, "female")

        score = min(bias_count * 0.5, 1.0)
        return score, indicators, sentence_hits

    # -----------------------------------------------------------------------
    # Signal 3: Agentic / Communal Trait Asymmetry
    # -----------------------------------------------------------------------

    def _score_agentic_communal(
        self,
        text: str,
        male_hits: set,
        female_hits: set,
    ) -> Tuple[float, List[str]]:
        """
        Detects the documented LLM tendency to describe male subjects with
        agentic traits (decisive, ambitious, rational) and female subjects
        with communal traits (nurturing, warm, supportive).

        Wan et al. (2023) found this pattern across 10 LLMs — male-named
        subjects received 'role model' framings, female-named subjects received
        'warm person' framings, regardless of the role described.

        Implementation: for each gendered term occurrence, we scan a context
        window and count matching trait words. We then compare the agentic/
        communal ratio between male-context and female-context mentions.

        Trait matching is word-based: we tokenize the surrounding context into
        lowercase words with \b\w+\b and intersect those tokens with the
        trait lexicons. Compound traits such as "risk-taker" should therefore
        appear in the lexicon as unhyphenated tokens or as their component
        words.
        """
        if not male_hits and not female_hits:
            return 0.0, []

        male_agentic = male_communal = 0
        female_agentic = female_communal = 0
        window = 100  # chars either side of the pronoun/term

        def _count_traits(term: str) -> Tuple[int, int]:
            agentic_count = communal_count = 0
            for m in re.finditer(rf"\b{re.escape(term)}\b", text):
                ctx = text[max(0, m.start() - window): m.end() + window]
                # Tokenize context into individual words for set intersection
                ctx_tokens = set(re.findall(r"\b[a-z]+\b", ctx))
                agentic_count  += len(ctx_tokens & AGENTIC_TRAITS)
                communal_count += len(ctx_tokens & COMMUNAL_TRAITS)
            return agentic_count, communal_count

        for term in male_hits:
            a, c = _count_traits(term)
            male_agentic  += a
            male_communal += c

        for term in female_hits:
            a, c = _count_traits(term)
            female_agentic  += a
            female_communal += c

        total_male   = male_agentic   + male_communal
        total_female = female_agentic + female_communal

        if total_male == 0 and total_female == 0:
            return 0.0, []

        indicators: List[str] = []
        score = 0.0

        if total_male > 0 and total_female > 0:
            # Both genders present — compare their trait profiles
            m_agentic_ratio  = male_agentic   / total_male
            f_communal_ratio = female_communal / total_female

            if m_agentic_ratio > 0.5 and f_communal_ratio > 0.5:
                # The stereotyped pattern fires on both sides simultaneously
                score = min((m_agentic_ratio - 0.5) + (f_communal_ratio - 0.5), 1.0)
                indicators.append(
                    f"Trait asymmetry (Wan 2023 pattern): "
                    f"male subjects {m_agentic_ratio:.0%} agentic traits, "
                    f"female subjects {f_communal_ratio:.0%} communal traits"
                )

        elif total_male > 0 and total_female == 0 and male_agentic > male_communal:
            # Only male subjects mentioned, all described agentically
            score = 0.25
            indicators.append(
                f"Male-only subjects described with predominantly agentic traits "
                f"({male_agentic} agentic vs {male_communal} communal)"
            )

        elif total_female > 0 and total_male == 0 and female_communal > female_agentic:
            # Only female subjects mentioned, all described communally
            score = 0.25
            indicators.append(
                f"Female-only subjects described with predominantly communal traits "
                f"({female_communal} communal vs {female_agentic} agentic)"
            )

        return round(score, 3), indicators

    # -----------------------------------------------------------------------
    # Signal 4: Bias Phrase Detection
    # -----------------------------------------------------------------------

    def _score_bias_phrases(
        self,
        text: str,
        sentences: List[str],
    ) -> Tuple[float, List[str], List[dict]]:
        """
        Pattern-matches known stereotype expression forms using regex.

        We search sentence-by-sentence so we can report which sentence triggered
        each match. We skip sentences that contain inclusive qualifiers like
        "regardless of gender" or "both men and women" — these are counter-
        stereotype framings that can superficially match our patterns.

        score = min(matches / 3, 1.0) — three or more matches is maximum bias.
        """
        # Sentences containing these phrases are probably refuting stereotypes
        inclusive_re = re.compile(
            r"\b(all|any|both|regardless|equally|everyone|every person|"
            r"men and women|women and men|any gender|irrespective|no matter)\b",
            re.IGNORECASE,
        )

        matches: List[str] = []
        sentence_hits: List[dict] = []

        for pattern in BIAS_PHRASES:
            for sentence in sentences:
                if inclusive_re.search(sentence):
                    continue
                m = re.search(pattern, sentence, re.IGNORECASE)
                if m:
                    excerpt = m.group(0).strip()[:80]
                    if excerpt not in matches:
                        matches.append(excerpt)
                        sentence_hits.append({
                            "sentence": sentence[:120],
                            "reason": f"Stereotype phrase matched: '{excerpt}'",
                            "pattern_type": "bias_phrase",
                        })

        score = min(len(matches) / 3.0, 1.0)
        indicators: List[str] = []
        if matches:
            indicators.append(
                f"Stereotype language patterns ({len(matches)} match): "
                + "; ".join(matches[:2])
            )

        return round(score, 3), indicators, sentence_hits

    # -----------------------------------------------------------------------
    # Signal 5: LLM Hedge Pattern Detection
    # -----------------------------------------------------------------------

    def _score_llm_hedges(self, text: str) -> Tuple[float, List[str]]:
        """
        Modern LLMs often don't state stereotypes directly — instead they use
        hedging constructs: "While women can certainly succeed in STEM, men
        tend to be more naturally drawn to it." The first clause appears fair,
        the second reintroduces the bias.

        This signal is lower-weighted (0.08) because hedge patterns are
        noisier than occupational pairings — they can appear in legitimately
        nuanced responses too.
        """
        matches: List[str] = []
        for pattern in LLM_HEDGE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                matches.append(m.group(0).strip()[:80])

        score = min(len(matches) / 2.0, 1.0)
        indicators: List[str] = []
        if matches:
            indicators.append("LLM hedge-then-stereotype pattern: " + "; ".join(matches[:2]))

        return round(score, 3), indicators

    # -----------------------------------------------------------------------
    # Signal 6: Negative Sentiment Near Gendered Terms
    # -----------------------------------------------------------------------

    def _score_negative_sentiment(
        self,
        text: str,
        male_hits: set,
        female_hits: set,
    ) -> Tuple[float, List[str]]:
        """
        Checks whether negative/demeaning words cluster near gendered terms.

        Symmetric negativity (negative words near both male and female terms)
        is weak evidence of bias. Asymmetric negativity — negative words only
        near one gender — is a stronger signal, suggesting targeted negative
        framing of that group.

        The window spans 100 characters on each side. This wider scope helps
        capture LLM sentences where the gendered term and descriptor are
        separated by a long relative clause.
        """
        window = 100
        neg_near_male: List[str] = []
        neg_near_female: List[str] = []

        for term in male_hits:
            for m in re.finditer(rf"\b{re.escape(term)}\b", text):
                ctx = text[max(0, m.start() - window): m.end() + window]
                neg = set(re.findall(r"\b\w+\b", ctx)) & NEGATIVE_SENTIMENT_WORDS
                if neg:
                    neg_near_male.extend(list(neg))

        for term in female_hits:
            for m in re.finditer(rf"\b{re.escape(term)}\b", text):
                ctx = text[max(0, m.start() - window): m.end() + window]
                neg = set(re.findall(r"\b\w+\b", ctx)) & NEGATIVE_SENTIMENT_WORDS
                if neg:
                    neg_near_female.extend(list(neg))

        indicators: List[str] = []
        if neg_near_female and not neg_near_male:
            indicators.append(
                f"Negative words appear near female-coded terms only: "
                + ", ".join(sorted(set(neg_near_female))[:3])
            )
        elif neg_near_male and not neg_near_female:
            indicators.append(
                f"Negative words appear near male-coded terms only: "
                + ", ".join(sorted(set(neg_near_male))[:3])
            )
        elif neg_near_male or neg_near_female:
            indicators.append(
                f"Negative sentiment near gendered terms "
                f"(male contexts: {len(neg_near_male)}, female: {len(neg_near_female)})"
            )

        total = len(set(neg_near_male)) + len(set(neg_near_female))
        score = min(total / 4.0, 1.0)
        return round(score, 3), indicators

    # -----------------------------------------------------------------------
    # Signal 7: Toxicity Proxy Score
    # -----------------------------------------------------------------------

    def _score_toxicity(self, text: str) -> Tuple[float, List[str]]:
        """
        Detects overtly demeaning or harmful vocabulary from the TOXICITY_WORDS
        lexicon. This is distinct from negative sentiment (Signal 6) — toxicity
        covers terms that are inherently demeaning regardless of context, like
        'hysterical' or 'irrational' when applied to a gender group.

        This is a low-weighted safety-net signal for overtly demeaning or
        harmful vocabulary. It complements negative sentiment by catching
        terms that are harmful even outside a broader sentiment pattern.
        """
        tokens = set(re.findall(r"\b\w+\b", text))
        hits = tokens & TOXICITY_WORDS
        score = min(len(hits) / 3.0, 1.0)

        indicators: List[str] = []
        if hits:
            indicators.append(f"Toxic/demeaning vocabulary detected: {', '.join(sorted(hits)[:4])}")

        return round(score, 3), indicators

    # -----------------------------------------------------------------------
    # Sentence splitting
    # -----------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using punctuation and newlines.

        LLM outputs frequently use newlines as sentence separators (e.g. bullet
        points, numbered lists), so we split on both punctuation and newlines.
        We filter very short fragments to reduce noise from list bullets like
        "- " or single-word items.
        """
        # Split on sentence-ending punctuation followed by whitespace, OR on
        # newlines (which LLMs use for list items and paragraph breaks)
        raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 10]

    # -----------------------------------------------------------------------
    # Summary builder
    # -----------------------------------------------------------------------

    def _build_summary(
        self, bias_detected: bool, bias_types: List[str], score: float
    ) -> str:
        if not bias_detected:
            return "No significant bias detected. Response appears neutral."
        type_str = ", ".join(bias_types) if bias_types else "general"
        severity = (
            "low"      if score < 0.25 else
            "moderate" if score < 0.45 else
            "high"
        )
        return (
            f"Bias detected [{severity} severity]: {type_str}. "
            f"Composite score: {score:.2f}. "
            f"RAG mitigation recommended."
        )

    # -----------------------------------------------------------------------
    # Counterfactual fairness test (Kusner et al. 2017)
    # -----------------------------------------------------------------------

    def counterfactual_delta(self, prompt: str, response: str) -> dict:
        """
        Measures how much the bias score changes when all gendered terms are
        swapped (he↔she, man↔woman, etc.).

        In ideal counterfactually fair output the description of a role should
        not change substantively when the subject's gender changes. A large
        delta means the response encodes gender-dependent framing — the LLM
        described the male engineer differently than it would a female engineer.

        Uses a two-phase swap to avoid collision (e.g. "he→she" and "she→he"
        fighting each other): first replace male terms with placeholders, then
        swap female→male, then resolve placeholders→female.
        """
        original_score = self.analyze(prompt, response)["composite_bias_score"]

        swapped = response
        for male_term, female_term in COUNTERFACTUAL_PAIRS:
            placeholder = f"__SWAP_{male_term.upper()}__"
            swapped = re.sub(rf"\b{re.escape(male_term)}\b", placeholder, swapped, flags=re.I)
            swapped = re.sub(rf"\b{re.escape(female_term)}\b", male_term, swapped, flags=re.I)
            swapped = swapped.replace(placeholder, female_term)

        swapped_score = self.analyze(prompt, swapped)["composite_bias_score"]
        delta = round(abs(original_score - swapped_score), 3)

        return {
            "original_score": original_score,
            "swapped_score":  swapped_score,
            "delta":          delta,
            "fairness_flag":  delta > 0.15,
            "interpretation": (
                f"Not counterfactually fair — bias score shifts by {delta:.2f} after "
                f"gender swap, indicating gender-dependent framing in the response."
                if delta > 0.15
                else "Approximately counterfactually fair."
            ),
        }