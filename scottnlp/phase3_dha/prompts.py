"""DHA 5-strategy prompt templates for DeepSeek classification."""

from scottnlp.config import DHA_STRATEGIES


# ── Strategy-specific instruction blocks ──────────────────────────────

_STRATEGY_INSTRUCTIONS = {
    "nomination": """\
=== TASK: NOMINATION STRATEGY ANALYSIS ===
Analyze how languages (English, Gaelic, Scots) and social actors (institutions,
communities, speakers) are NAMED and REFERRED TO in this text.

Look for:
- Referential devices: proper names, pronouns, metaphors, metonymies, synecdoches
- Naming conventions: "the Gaelic language" vs "Gaelic" vs "Gàidhlig" vs "native tongue"
- Categorization: how language communities are labeled (e.g., "Gaelic speakers",
  "the Gaelic community", "minority language users")
- Institutional naming: how bodies/authorities are designated in relation to language
- Collectivization vs individualization: are speakers treated as groups or individuals?""",

    "predication": """\
=== TASK: PREDICATION STRATEGY ANALYSIS ===
Analyze what QUALITIES, CHARACTERISTICS, and ATTRIBUTES are assigned to languages
and actors in this text.

Look for:
- Evaluative adjectives: "important", "valuable", "vulnerable", "fragile", "rich"
- Stereotypical characterizations: implicit or explicit value judgments
- Predicative nouns/nominalization: "a national asset", "a burden", "a right"
- Prepositional phrases attributing properties: "of significance", "under threat"
- Implicit predication through collocation: what words routinely co-occur with each language?
- How dependency parsing data (verbs, modifiers) reveals attributed qualities""",

    "argumentation": """\
=== TASK: ARGUMENTATION STRATEGY ANALYSIS ===
Analyze what REASONING SCHEMES (topoi) are used to justify or challenge claims
about language in this text.

Common topoi in language policy discourse:
- Topos of ADVANTAGE/USEFULNESS: "If X is useful, it should be promoted"
- Topos of THREAT/DANGER: "If X is endangered, action must be taken"
- Topos of CULTURE/HERITAGE: "Because X is part of heritage, it must be preserved"
- Topos of HISTORY/TRADITION: "Because X has always been..., it should continue"
- Topos of NUMBERS: "Because X many speakers exist / are declining..."
- Topos of LAW/RIGHTS: "Because international law requires..., Scotland must..."
- Topos of BURDEN/COST: "Because X is expensive / impractical..."
- Topos of AUTHORITY: "Because [expert/body] says..., it must be so"
- Topos of DEFINITION: "Because X is defined as..., it follows that..."

Identify which topoi are present, what claims they support, and what language(s) they apply to.""",

    "perspectivization": """\
=== TASK: PERSPECTIVIZATION STRATEGY ANALYSIS ===
Analyze from WHOSE VIEWPOINT or STANDPOINT claims about language are expressed
in this text.

Look for:
- Narrator/speaker positioning: legislative voice, governmental authority, community advocate
- Framing devices: "it is recognized that", "Scotland's view is", "speakers themselves"
- Involvement vs distance: first person plural ("we shall"), impersonal constructions
  ("it is provided that"), passive voice
- Reported speech/thought: "the Minister stated", "communities have expressed"
- Whose interests are foregrounded vs backgrounded?
- Internal vs external perspective on language communities""",

    "intensification_mitigation": """\
=== TASK: INTENSIFICATION / MITIGATION STRATEGY ANALYSIS ===
Analyze whether assertions about language are STRENGTHENED (intensified) or
WEAKENED (mitigated) in this text.

Look for:
INTENSIFICATION devices:
- Deontic modals: "shall", "must", "is required to"
- Amplifiers: "absolutely", "fundamentally", "vitally important"
- Epistemic certainty: "it is clear that", "undoubtedly", "without question"
- Superlatives and maximizers: "the most important", "essential"
- Repetition and accumulation of terms

MITIGATION devices:
- Hedging modals: "may", "might", "could"
- Downtoners: "somewhat", "to some extent", "where practicable"
- Epistemic uncertainty: "it appears that", "arguably"
- Conditional framing: "if resources permit", "subject to..."
- Vague quantifiers: "certain", "some", "appropriate"

Note: Legal texts often use "shall" as mandatory obligation, not mere prediction.
Distinguish legal "shall" (intensification) from hedged permissions "may" (mitigation).""",
}


# ── Output schema ─────────────────────────────────────────────────────

_BASE_SCHEMA = """\
=== REQUIRED OUTPUT FORMAT ===
Respond with a single JSON object matching this exact schema:
{{
  "strategy_name": "{strategy}",
  "present": true or false,
  "confidence": 0.0 to 1.0,
  "evidence_quotes": ["exact quote from the text", "..."],
  "linguistic_devices": [
    {{"device": "name of device", "example": "text example", "function": "what it does"}}
  ],
  "target_languages": ["Gaelic", "Scots", "English"],{topoi_field}
  "notes": "Brief analytical note explaining the classification"
}}

Rules:
- "evidence_quotes" must be EXACT substrings from the provided text
- "confidence" reflects how clearly this strategy is present (0.0 = absent/unsure, 1.0 = prominent)
- "target_languages" lists which languages this strategy instance applies to
- If the strategy is not present, set "present": false, "confidence": 0.0, empty lists, and explain in "notes"
"""

_TOPOI_FIELD = '\n  "topoi": ["topos_of_law", "topos_of_culture", "..."],'


# ── Dependency frame formatting ───────────────────────────────────────

_ROLE_PRIORITY = {"agent": 0, "patient": 1, "modifier": 2, "oblique": 3, "other": 4}


def format_dep_frames_for_prompt(frames: list[dict], max_frames: int = 15) -> str:
    """Format dependency frames as concise linguistic context for the LLM."""
    if not frames:
        return ""

    # Deduplicate by (target_term, sentence, syntactic_role)
    seen = set()
    unique = []
    for f in frames:
        key = (f["target_term"], f.get("sentence", ""), f.get("syntactic_role", ""))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    # Sort by role priority
    unique.sort(key=lambda f: _ROLE_PRIORITY.get(f.get("syntactic_role", "other"), 4))
    unique = unique[:max_frames]

    lines = []
    for f in unique:
        term = f["target_term"]
        role = f.get("syntactic_role", "unknown")
        dep_rel = f.get("dep_relation", "")
        verb = f.get("governing_verb", "")
        mods = f.get("modifiers", [])

        parts = [f'- "{term}" as {role.upper()}']
        if dep_rel:
            parts.append(f"(dep: {dep_rel}")
            if verb:
                parts.append(f", verb: {verb})")
            else:
                parts.append(")")
        if mods:
            mod_strs = [m["text"] if isinstance(m, dict) else str(m) for m in mods[:3]]
            parts.append(f" [modifiers: {', '.join(mod_strs)}]")

        lines.append("".join(parts))

    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────

def build_strategy_prompt(
    strategy: str,
    chunk: dict,
    dep_frames: list[dict] | None = None,
) -> str:
    """Build the full user prompt for a single DHA strategy classification."""
    if strategy not in DHA_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {DHA_STRATEGIES}")

    # 1. Document context
    lang_focus = chunk.get("language_focus", [])
    if isinstance(lang_focus, list):
        lang_focus = ", ".join(lang_focus)

    doc_context = (
        f"=== DOCUMENT CONTEXT ===\n"
        f"Title: {chunk.get('doc_title', 'Unknown')}\n"
        f"Year: {chunk.get('doc_year', 'Unknown')}\n"
        f"Type: {chunk.get('doc_type', 'Unknown')}\n"
        f"Jurisdiction: {chunk.get('jurisdiction', 'Unknown')}\n"
        f"Section: {chunk.get('section_title', 'Unknown')}\n"
        f"Language focus: {lang_focus}\n"
    )

    # 2. Dependency frame context (optional)
    dep_context = ""
    if dep_frames:
        formatted = format_dep_frames_for_prompt(dep_frames)
        if formatted:
            dep_context = (
                "\n=== LINGUISTIC EVIDENCE FROM DEPENDENCY PARSING ===\n"
                "The following syntactic relationships involving language terms "
                "were detected in this text:\n"
                f"{formatted}\n"
            )

    # 3. Strategy instruction
    instruction = _STRATEGY_INSTRUCTIONS[strategy]

    # 4. Output schema
    topoi_field = _TOPOI_FIELD if strategy == "argumentation" else ""
    schema = _BASE_SCHEMA.format(strategy=strategy, topoi_field=topoi_field)

    # 5. Chunk text
    text_block = f"\n=== TEXT TO ANALYZE ===\n{chunk.get('text', '')}\n"

    return f"{doc_context}{dep_context}\n{instruction}\n{text_block}\n{schema}"
