"""Phase 2 Deep-Dive: SVO Power Triangle & Markedness Theory Analysis."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from scottnlp.config import (
    OUTPUT_DIR, ERA_DEFINITIONS, YEAR_TO_ERA, LANG_CANONICAL_MAP,
)

PHASE1_DIR = OUTPUT_DIR / "phase1"
PHASE2_DIR = OUTPUT_DIR / "phase2"
PHASE3_DIR = OUTPUT_DIR / "phase3"
DEEP_OUTPUT_DIR = PHASE2_DIR / "deep"

# Alias for backward compatibility within this module
LANG_MAP = {k: v for k, v in LANG_CANONICAL_MAP.items() if k.islower()}

# Heads that indicate "Scots" refers to a legal system, not the language.
# e.g. "Scots private law", "Scots criminal law"
SCOTS_LAW_HEADS = {"law", "private", "criminal"}

# Agent classification taxonomy for SVO analysis
AGENT_TAXONOMY = {
    "INSTITUTIONAL": [
        "minister", "parliament", "government", "bòrd", "bord", "council",
        "authority", "board", "body", "commission", "executive", "committee",
        "secretary", "department", "office", "officer", "inspector",
        "party", "parties",
    ],
    "LEGAL_PROCESS": [
        "act", "section", "regulation", "provision", "order", "schedule",
        "charter", "treaty", "plan", "policy", "law", "article", "measure",
        "directive", "statute", "subsection", "paragraph",
    ],
    "ABSTRACT": [
        "scotland", "country", "state", "kingdom", "nation", "territory",
        "area", "region", "crown", "majesty",
    ],
    "PERSON_GROUP": [
        "speaker", "community", "person", "people", "pupil", "teacher",
        "child", "parent", "member", "user", "group", "population",
        "learner", "resident", "citizen", "individual",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Markedness Theory — "Invisible Hegemony"
# ═══════════════════════════════════════════════════════════════════════


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_markedness_ratios(dep_frames_path: Path) -> dict:
    """Count language mentions per document/era, compute markedness ratios.

    Normalizes language variants (Gàidhlig→Gaelic, Beurla→English, Albais→Scots).
    Only counts frames where target_lemma maps to a specific language (excludes
    generic terms like "language", "tongue", "Scottish", etc.).
    """
    frames = _load_jsonl(dep_frames_path)

    by_doc = defaultdict(lambda: {
        "mentions": defaultdict(int),
        "year": 0,
        "era": "",
    })

    for f in frames:
        lemma = f["target_lemma"]
        if lemma not in LANG_MAP:
            continue
        lang = LANG_MAP[lemma]
        doc_title = f["doc_title"]
        by_doc[doc_title]["mentions"][lang] += 1
        by_doc[doc_title]["year"] = f["doc_year"]
        by_doc[doc_title]["era"] = YEAR_TO_ERA.get(f["doc_year"], "unknown")

    # Compute per-document markedness ratio
    result_by_doc = {}
    for doc_title, data in by_doc.items():
        mentions = dict(data["mentions"])
        eng = mentions.get("English", 0)
        gae = mentions.get("Gaelic", 0)
        sco = mentions.get("Scots", 0)
        minority_total = gae + sco
        total = eng + gae + sco

        result_by_doc[doc_title] = {
            "year": data["year"],
            "era": data["era"],
            "mentions": mentions,
            "total_language_mentions": total,
            "markedness_ratio": round(minority_total / eng, 1) if eng > 0 else float("inf"),
            "english_invisibility_index": round(1 - eng / total, 3) if total > 0 else None,
        }

    # Aggregate by era
    result_by_era = {}
    for era in ERA_DEFINITIONS:
        era_mentions = defaultdict(int)
        for doc_data in result_by_doc.values():
            if doc_data["era"] == era:
                for lang, count in doc_data["mentions"].items():
                    era_mentions[lang] += count

        eng = era_mentions.get("English", 0)
        gae = era_mentions.get("Gaelic", 0)
        sco = era_mentions.get("Scots", 0)
        total = eng + gae + sco

        result_by_era[era] = {
            "mentions": dict(era_mentions),
            "total": total,
            "markedness_ratio": round((gae + sco) / eng, 1) if eng > 0 else float("inf"),
            "english_invisibility_index": round(1 - eng / total, 3) if total > 0 else None,
        }

    # Overall totals
    all_mentions = defaultdict(int)
    for doc_data in result_by_doc.values():
        for lang, count in doc_data["mentions"].items():
            all_mentions[lang] += count

    eng_total = all_mentions.get("English", 0)
    total_all = sum(all_mentions.values())

    return {
        "by_document": result_by_doc,
        "by_era": result_by_era,
        "totals": {
            "mentions": dict(all_mentions),
            "total": total_all,
            "overall_markedness_ratio": round(
                (all_mentions.get("Gaelic", 0) + all_mentions.get("Scots", 0)) / eng_total, 1
            ) if eng_total > 0 else float("inf"),
            "overall_english_invisibility": round(1 - eng_total / total_all, 3) if total_all > 0 else None,
        },
    }


def analyze_dha_strategy_targeting(classifications_path: Path) -> dict:
    """Compute how often each language is targeted by each DHA strategy.

    Returns: strategy → language → {count, by_year}.
    """
    clss = _load_jsonl(classifications_path)

    targeting = defaultdict(lambda: defaultdict(lambda: {"count": 0, "by_year": defaultdict(int)}))

    for c in clss:
        if not c.get("present"):
            continue
        strategy = c.get("strategy_name", "")
        yr = c.get("doc_year", 0)
        for lang in c.get("target_languages", []):
            targeting[strategy][lang]["count"] += 1
            targeting[strategy][lang]["by_year"][yr] += 1

    # Convert to serializable dict
    result = {}
    for strategy, langs in targeting.items():
        result[strategy] = {}
        for lang, data in langs.items():
            result[strategy][lang] = {
                "count": data["count"],
                "by_year": dict(data["by_year"]),
            }

    return result


def compute_protection_discourse_density(
    dep_frames_path: Path,
    classifications_path: Path,
) -> dict:
    """Compute the 'naming paradox' metric.

    For each document, measure the proportion of chunks that BOTH:
    1. Mention a minority language (from dep_frames), AND
    2. Apply predication to that language (from DHA classifications)

    The hypothesis: frequent naming + predication reinforces 'otherness',
    while English avoids this treatment entirely.
    """
    frames = _load_jsonl(dep_frames_path)
    clss = _load_jsonl(classifications_path)

    # Step 1: Which chunks mention which languages (from dep_frames)?
    chunk_languages = defaultdict(set)
    chunk_doc = {}
    for f in frames:
        lemma = f["target_lemma"]
        if lemma in LANG_MAP:
            lang = LANG_MAP[lemma]
            chunk_languages[f["chunk_id"]].add(lang)
            chunk_doc[f["chunk_id"]] = (f["doc_title"], f["doc_year"])

    # Step 2: Which chunks have predication present for which languages?
    chunk_predication = defaultdict(set)
    chunk_nomination = defaultdict(set)
    for c in clss:
        if not c.get("present"):
            continue
        cid = c.get("chunk_id", "")
        if c["strategy_name"] == "predication":
            for lang in c.get("target_languages", []):
                chunk_predication[cid].add(lang)
        elif c["strategy_name"] == "nomination":
            for lang in c.get("target_languages", []):
                chunk_nomination[cid].add(lang)

    # Step 3: Count chunks per doc that both name AND predicate each language
    # Also gather total chunks per doc from classifications
    doc_total_chunks = Counter()
    seen = set()
    for c in clss:
        cid = c.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            doc_total_chunks[c.get("doc_title", "")] += 1

    docs = defaultdict(lambda: {
        "year": 0,
        "era": "",
        "total_chunks": 0,
        "naming_and_predication": defaultdict(int),
        "naming_only": defaultdict(int),
    })

    for chunk_id, langs_mentioned in chunk_languages.items():
        if chunk_id not in chunk_doc:
            continue
        doc_title, doc_year = chunk_doc[chunk_id]
        docs[doc_title]["year"] = doc_year
        docs[doc_title]["era"] = YEAR_TO_ERA.get(doc_year, "unknown")

        for lang in langs_mentioned:
            if lang in chunk_predication.get(chunk_id, set()):
                docs[doc_title]["naming_and_predication"][lang] += 1
            else:
                docs[doc_title]["naming_only"][lang] += 1

    # Add total chunk counts
    for doc_title in docs:
        docs[doc_title]["total_chunks"] = doc_total_chunks.get(doc_title, 0)

    # Compute density
    result_by_doc = {}
    for doc_title, data in docs.items():
        total = data["total_chunks"]
        density = {}
        for lang in ["English", "Gaelic", "Scots"]:
            n_both = data["naming_and_predication"].get(lang, 0)
            n_name_only = data["naming_only"].get(lang, 0)
            density[lang] = {
                "chunks_named_and_predicated": n_both,
                "chunks_named_only": n_name_only,
                "protection_discourse_density": round(n_both / total, 3) if total > 0 else 0,
            }
        result_by_doc[doc_title] = {
            "year": data["year"],
            "era": data["era"],
            "total_chunks": total,
            "density": density,
        }

    return result_by_doc


def build_markedness_profile(
    markedness_ratios: dict,
    strategy_targeting: dict,
    protection_density: dict,
) -> dict:
    """Combine all markedness metrics into a unified profile.

    Computes:
    - english_predication_absence: how rarely English is predicated
    - naming_paradox: correlation between naming frequency and protection language
    """
    # English predication absence from strategy targeting
    pred_targeting = strategy_targeting.get("predication", {})
    english_pred = pred_targeting.get("English", {}).get("count", 0)
    gaelic_pred = pred_targeting.get("Gaelic", {}).get("count", 0)
    scots_pred = pred_targeting.get("Scots", {}).get("count", 0)

    english_predication_absence = {
        "english_predication_count": english_pred,
        "gaelic_predication_count": gaelic_pred,
        "scots_predication_count": scots_pred,
        "english_share_of_predication": round(
            english_pred / (english_pred + gaelic_pred + scots_pred), 3
        ) if (english_pred + gaelic_pred + scots_pred) > 0 else 0,
    }

    # Strategy targeting asymmetry: how much more Gaelic/Scots are targeted vs English
    targeting_asymmetry = {}
    for strategy, langs in strategy_targeting.items():
        eng_c = langs.get("English", {}).get("count", 0)
        gae_c = langs.get("Gaelic", {}).get("count", 0)
        sco_c = langs.get("Scots", {}).get("count", 0)
        total = eng_c + gae_c + sco_c
        targeting_asymmetry[strategy] = {
            "english_share": round(eng_c / total, 3) if total > 0 else 0,
            "minority_share": round((gae_c + sco_c) / total, 3) if total > 0 else 0,
            "asymmetry_ratio": round((gae_c + sco_c) / eng_c, 1) if eng_c > 0 else float("inf"),
        }

    return {
        "markedness_ratios": markedness_ratios["totals"],
        "english_predication_absence": english_predication_absence,
        "strategy_targeting_asymmetry": targeting_asymmetry,
        "protection_density_by_document": protection_density,
    }


def save_markedness_results(
    markedness_ratios: dict,
    strategy_targeting: dict,
    markedness_profile: dict,
    output_dir: Path,
) -> None:
    """Save all Markedness Theory outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markedness ratios
    path = output_dir / "markedness_ratios.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(markedness_ratios, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved markedness ratios to {path}")

    # Strategy targeting
    path = output_dir / "strategy_targeting.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(strategy_targeting, f, ensure_ascii=False, indent=2)
    print(f"Saved strategy targeting to {path}")

    # Markedness profile
    path = output_dir / "markedness_profile.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(markedness_profile, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved markedness profile to {path}")

    # Flat summary CSV
    rows = []
    for doc_title, data in markedness_ratios["by_document"].items():
        prot = markedness_profile["protection_density_by_document"].get(doc_title, {})
        prot_density = prot.get("density", {}) if prot else {}
        rows.append({
            "doc_title": doc_title,
            "doc_year": data["year"],
            "era": data["era"],
            "english_mentions": data["mentions"].get("English", 0),
            "gaelic_mentions": data["mentions"].get("Gaelic", 0),
            "scots_mentions": data["mentions"].get("Scots", 0),
            "markedness_ratio": data["markedness_ratio"],
            "english_invisibility_index": data["english_invisibility_index"],
            "protection_density_gaelic": prot_density.get("Gaelic", {}).get("protection_discourse_density", 0),
            "protection_density_scots": prot_density.get("Scots", {}).get("protection_discourse_density", 0),
            "protection_density_english": prot_density.get("English", {}).get("protection_discourse_density", 0),
        })
    summary_df = pd.DataFrame(rows).sort_values("doc_year").reset_index(drop=True)
    path = output_dir / "markedness_summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"Saved markedness summary to {path}")


def run_markedness_analysis(
    dep_frames_path: Path | None = None,
    classifications_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run the full Markedness Theory analysis."""
    if dep_frames_path is None:
        dep_frames_path = PHASE2_DIR / "dep_frames.jsonl"
    if classifications_path is None:
        classifications_path = PHASE3_DIR / "dha_classifications.jsonl"
    if output_dir is None:
        output_dir = DEEP_OUTPUT_DIR

    print("=" * 60)
    print("ANALYSIS 3: Markedness Theory — Invisible Hegemony")
    print("=" * 60)

    # Step 1: Markedness ratios
    print("\n[1/4] Computing markedness ratios...")
    ratios = compute_markedness_ratios(dep_frames_path)
    totals = ratios["totals"]
    print(f"  Mentions: {totals['mentions']}")
    print(f"  Overall markedness ratio (minority/English): {totals['overall_markedness_ratio']}")
    print(f"  English invisibility index: {totals['overall_english_invisibility']}")

    # Step 2: DHA strategy targeting
    print("\n[2/4] Analyzing DHA strategy targeting...")
    targeting = analyze_dha_strategy_targeting(classifications_path)
    for strategy in sorted(targeting):
        langs = targeting[strategy]
        lang_str = ", ".join(f"{l}={d['count']}" for l, d in sorted(langs.items()))
        print(f"  {strategy}: {lang_str}")

    # Step 3: Protection discourse density
    print("\n[3/4] Computing protection discourse density...")
    protection = compute_protection_discourse_density(dep_frames_path, classifications_path)
    for doc_title in sorted(protection, key=lambda x: protection[x]["year"]):
        data = protection[doc_title]
        gae = data["density"]["Gaelic"]["protection_discourse_density"]
        sco = data["density"]["Scots"]["protection_discourse_density"]
        eng = data["density"]["English"]["protection_discourse_density"]
        if gae > 0 or sco > 0 or eng > 0:
            print(f"  {doc_title} ({data['year']}): Gaelic={gae}, Scots={sco}, English={eng}")

    # Step 4: Build profile
    print("\n[4/4] Building markedness profile...")
    profile = build_markedness_profile(ratios, targeting, protection)
    asym = profile["strategy_targeting_asymmetry"]
    print("  Strategy targeting asymmetry (minority/English ratio):")
    for s, data in sorted(asym.items()):
        print(f"    {s}: {data['asymmetry_ratio']}x")

    # Save
    print("\nSaving results...")
    save_markedness_results(ratios, targeting, profile, output_dir)

    return {
        "markedness_ratios": ratios,
        "strategy_targeting": targeting,
        "protection_density": protection,
        "markedness_profile": profile,
    }


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: SVO Power Triangle
# ═══════════════════════════════════════════════════════════════════════


def _classify_agent(text: str, lemma: str) -> str:
    """Classify an agent entity into INSTITUTIONAL/LEGAL_PROCESS/ABSTRACT/PERSON_GROUP.

    Uses the head noun (last noun in the NP) for classification.
    Multi-word entities checked as whole phrase first, then by head noun.
    """
    text_lower = text.lower()
    lemma_lower = lemma.lower()

    # Phase 1: Check full-phrase patterns (catches "Creative Scotland", "BBC Scotland")
    full_phrase_institutional = [
        "creative scotland", "bbc scotland", "education scotland",
        "scottish ministers", "scottish government", "scottish parliament",
        "scottish qualifications authority", "bòrd na gàidhlig",
        "contracting state", "lord advocate",
    ]
    for phrase in full_phrase_institutional:
        if phrase in text_lower:
            return "INSTITUTIONAL"

    # Phase 2: Head noun check (last significant word)
    tokens = lemma_lower.split()
    for check_token in reversed(tokens):
        for category, patterns in AGENT_TAXONOMY.items():
            if check_token in patterns:
                return category

    # Phase 3: Fallback substring check
    for category, patterns in AGENT_TAXONOMY.items():
        for pattern in patterns:
            if pattern in text_lower:
                return category

    return "OTHER"


def _resolve_relative_pronoun(token) -> str | None:
    """Resolve a relative pronoun (which, that, who) to its antecedent."""
    if token.lemma_.lower() not in ("which", "that", "who", "whom"):
        return None
    # Walk up to the relcl head, then find the noun it modifies
    current = token
    visited = set()
    while current.head != current and current.i not in visited:
        visited.add(current.i)
        if current.dep_ == "relcl":
            # The head of relcl is the antecedent
            antecedent = current.head
            subtree = " ".join(t.text for t in antecedent.subtree)
            return subtree
        current = current.head
    return None


def _extract_agent_from_verb(verb_tok) -> dict | None:
    """Extract agent information from a single verb token."""
    # Check for passive: has auxpass child or tag is VBN with aux
    has_auxpass = any(c.dep_ == "auxpass" for c in verb_tok.children)
    is_vbn_with_aux = verb_tok.tag_ == "VBN" and any(
        c.dep_ in ("auxpass", "aux") and c.lemma_ == "be" for c in verb_tok.children
    )
    is_passive = has_auxpass or is_vbn_with_aux

    if is_passive:
        # Look for 'agent' dependency (spaCy labels "by"-agent as 'agent')
        for child in verb_tok.children:
            if child.dep_ == "agent":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        subtree = " ".join(t.text for t in grandchild.subtree)
                        return {
                            "agent_text": subtree,
                            "agent_lemma": " ".join(
                                t.lemma_ for t in grandchild.subtree
                                if t.pos_ not in ("DET", "PUNCT")
                            ),
                            "voice": "passive",
                            "by_phrase": True,
                        }
                subtree = " ".join(t.text for t in child.subtree)
                return {
                    "agent_text": subtree,
                    "agent_lemma": child.lemma_,
                    "voice": "passive",
                    "by_phrase": True,
                }

        # No by-agent found — agentless passive
        return {
            "agent_text": None,
            "agent_lemma": None,
            "voice": "passive",
            "by_phrase": False,
        }
    else:
        # Active voice: find nsubj
        for child in verb_tok.children:
            if child.dep_ in ("nsubj", "csubj"):
                raw_text = " ".join(t.text for t in child.subtree)
                lemma_parts = " ".join(
                    t.lemma_ for t in child.subtree
                    if t.pos_ not in ("DET", "PUNCT")
                )
                # Resolve relative pronouns
                if child.lemma_.lower() in ("which", "that", "who", "whom"):
                    resolved = _resolve_relative_pronoun(child)
                    if resolved:
                        return {
                            "agent_text": resolved,
                            "agent_lemma": resolved.lower(),
                            "voice": "active",
                            "by_phrase": None,
                        }
                return {
                    "agent_text": raw_text,
                    "agent_lemma": lemma_parts,
                    "voice": "active",
                    "by_phrase": None,
                }
    return None


def _find_verb_agent_in_doc(parsed_doc, verb_lemma: str, verb_text: str) -> dict | None:
    """Given a parsed spaCy doc and the target verb, find the subject/agent.

    Searches ALL sentences in the doc for the verb match.
    For active verbs: return nsubj child.
    For passive verbs (VBN with auxpass): return 'agent' dep or by-PP agent.
    """
    # Find the verb token(s) matching the lemma across entire doc
    verb_tokens = [
        t for t in parsed_doc
        if t.lemma_.lower() == verb_lemma.lower() and t.pos_ in ("VERB", "AUX")
    ]
    if not verb_tokens:
        # Fallback: match by text
        verb_tokens = [
            t for t in parsed_doc
            if t.text.lower() == verb_text.lower() and t.pos_ in ("VERB", "AUX")
        ]
    if not verb_tokens:
        # Last resort: match any token with same lemma (may be tagged differently)
        verb_tokens = [
            t for t in parsed_doc
            if t.lemma_.lower() == verb_lemma.lower()
        ]
    if not verb_tokens:
        return None

    for verb_tok in verb_tokens:
        result = _extract_agent_from_verb(verb_tok)
        if result is not None:
            return result

    # If we found verb tokens but no agent, check if it's actually passive
    # (verb might be in a participial clause without explicit aux)
    for verb_tok in verb_tokens:
        if verb_tok.tag_ == "VBN":
            return {
                "agent_text": None,
                "agent_lemma": None,
                "voice": "passive",
                "by_phrase": False,
            }

    return None


def extract_svo_triples(
    dep_frames_path: Path,
    nlp=None,
    gpu_id: int = 0,
) -> list[dict]:
    """Re-parse patient-frame sentences to extract full SVO triples.

    For each patient frame (language term as dobj/nsubjpass/pobj):
    1. Parse the sentence with spaCy
    2. Find the governing verb
    3. Extract the nsubj (active) or agent-PP (passive) or mark as AGENTLESS
    4. Classify the agent entity
    """
    frames = _load_jsonl(dep_frames_path)
    patient_frames = [f for f in frames if f["syntactic_role"] == "patient"]

    print(f"  Total patient frames: {len(patient_frames)}")

    # Deduplicate sentences for efficient parsing
    unique_sentences = {}
    for f in patient_frames:
        sent = f["sentence"]
        if sent not in unique_sentences:
            unique_sentences[sent] = sent

    sent_list = list(unique_sentences.keys())
    print(f"  Unique sentences to parse: {len(sent_list)}")

    # Load spaCy if not provided
    if nlp is None:
        import spacy
        print(f"  Loading spaCy model (GPU {gpu_id})...")
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        nlp = spacy.load("en_core_web_trf")
        print("  spaCy loaded.")

    # Parse all unique sentences in batch
    print(f"  Parsing {len(sent_list)} sentences...")
    parsed_sents = {}
    for i, doc in enumerate(nlp.pipe(sent_list, batch_size=32)):
        if (i + 1) % 100 == 0:
            print(f"    Parsed {i + 1}/{len(sent_list)}...")
        parsed_sents[sent_list[i]] = doc

    # Extract SVO triples
    triples = []
    for frame in patient_frames:
        sent_text = frame["sentence"]
        gov_verb = frame.get("governing_verb")
        if not gov_verb:
            # No governing verb — cannot extract agent
            triples.append({
                "chunk_id": frame["chunk_id"],
                "doc_year": frame["doc_year"],
                "doc_title": frame["doc_title"],
                "era": YEAR_TO_ERA.get(frame["doc_year"], "unknown"),
                "target_term": frame["target_term"],
                "target_lemma": frame["target_lemma"],
                "syntactic_role": frame["syntactic_role"],
                "dep_relation": frame["dep_relation"],
                "governing_verb": None,
                "sentence": sent_text,
                "agent_text": None,
                "agent_lemma": None,
                "agent_category": "NO_VERB",
                "voice": "unknown",
                "by_phrase": None,
            })
            continue

        parsed_doc = parsed_sents.get(sent_text)
        if parsed_doc is None:
            continue

        # Find agent in the parsed document
        agent_info = _find_verb_agent_in_doc(
            parsed_doc, gov_verb["lemma"], gov_verb["text"]
        )

        if agent_info is None:
            # Verb found in frame but agent extraction failed
            agent_info = {
                "agent_text": None,
                "agent_lemma": None,
                "voice": "unknown",
                "by_phrase": None,
            }

        # Classify agent
        if agent_info["agent_text"] is None:
            if agent_info["voice"] == "passive":
                category = "AGENTLESS"
            else:
                category = "UNRESOLVED"
        else:
            category = _classify_agent(
                agent_info["agent_text"],
                agent_info["agent_lemma"] or "",
            )

        # Normalize target language
        target_lang = LANG_MAP.get(frame["target_lemma"])

        triples.append({
            "chunk_id": frame["chunk_id"],
            "doc_year": frame["doc_year"],
            "doc_title": frame["doc_title"],
            "era": YEAR_TO_ERA.get(frame["doc_year"], "unknown"),
            "target_term": frame["target_term"],
            "target_lemma": frame["target_lemma"],
            "target_language": target_lang,
            "syntactic_role": frame["syntactic_role"],
            "dep_relation": frame["dep_relation"],
            "governing_verb": gov_verb,
            "sentence": frame["sentence"],
            "agent_text": agent_info["agent_text"],
            "agent_lemma": agent_info["agent_lemma"],
            "agent_category": category,
            "voice": agent_info["voice"],
            "by_phrase": agent_info["by_phrase"],
        })

    print(f"  Extracted {len(triples)} SVO triples")
    return triples


def build_power_triangle(svo_triples: list[dict]) -> dict:
    """Aggregate SVO triples into per-era, per-language power triangles.

    Returns nested dict: era → language → {agent_distribution, top_agents, top_verbs, agentless_ratio}
    """
    # Filter to only language-specific triples
    lang_triples = [t for t in svo_triples if t.get("target_language")]

    result = {}
    for era in ERA_DEFINITIONS:
        era_triples = [t for t in lang_triples if t["era"] == era]
        era_data = {}

        for lang in ["English", "Gaelic", "Scots"]:
            lang_t = [t for t in era_triples if t["target_language"] == lang]
            if not lang_t:
                continue

            # Agent category distribution
            cat_counts = Counter(t["agent_category"] for t in lang_t)

            # Top agent entities (for non-agentless)
            agent_entities = Counter()
            for t in lang_t:
                if t["agent_text"]:
                    agent_entities[t["agent_text"]] += 1

            # Top verbs
            verb_counts = Counter()
            for t in lang_t:
                if t.get("governing_verb"):
                    verb_counts[t["governing_verb"]["lemma"]] += 1

            # Agentless ratio (among passive frames only)
            passive_t = [t for t in lang_t if t["voice"] == "passive"]
            agentless = sum(1 for t in passive_t if t["agent_category"] == "AGENTLESS")

            era_data[lang] = {
                "total_triples": len(lang_t),
                "agent_distribution": dict(cat_counts.most_common()),
                "top_agents": [
                    {"entity": e, "count": n}
                    for e, n in agent_entities.most_common(10)
                ],
                "top_verbs": [
                    {"verb": v, "count": n}
                    for v, n in verb_counts.most_common(10)
                ],
                "passive_count": len(passive_t),
                "agentless_count": agentless,
                "agentless_ratio": round(
                    agentless / len(passive_t), 3
                ) if passive_t else 0,
            }

        result[era] = era_data

    return result


def analyze_agent_backgrounding(
    svo_triples: list[dict],
    dep_frames_path: Path,
) -> dict:
    """Systematic passive voice and agent-suppression analysis.

    Computes per-document and per-era:
    - agentless_passive_ratio: agentless passives / total passives
    - passive_rate: passive frames / (active + passive) frames
    - flagged_sentences: agentless passives for institutional actions
    """
    frames = _load_jsonl(dep_frames_path)

    # Per-document stats from SVO triples
    by_document = defaultdict(lambda: {
        "doc_year": 0,
        "era": "",
        "passive_count": 0,
        "agentless_passive_count": 0,
        "active_count": 0,
        "by_phrase_count": 0,
        "flagged_sentences": [],
    })

    # Institutional terms for backgrounding detection
    institutional_terms = set()
    for patterns in AGENT_TAXONOMY["INSTITUTIONAL"]:
        institutional_terms.add(patterns)

    for t in svo_triples:
        doc = t["doc_title"]
        by_document[doc]["doc_year"] = t["doc_year"]
        by_document[doc]["era"] = t.get("era", "")

        if t["voice"] == "passive":
            by_document[doc]["passive_count"] += 1
            if t["agent_category"] == "AGENTLESS":
                by_document[doc]["agentless_passive_count"] += 1

                # Check if sentence mentions institutional terms
                sent_lower = t["sentence"].lower()
                mentioned_institutions = [
                    term for term in institutional_terms if term in sent_lower
                ]
                if mentioned_institutions:
                    by_document[doc]["flagged_sentences"].append({
                        "sentence": t["sentence"][:200],
                        "target_term": t["target_term"],
                        "governing_verb": t.get("governing_verb", {}).get("lemma", ""),
                        "institutional_terms_in_context": mentioned_institutions,
                    })
            else:
                by_document[doc]["by_phrase_count"] += 1
        elif t["voice"] == "active":
            by_document[doc]["active_count"] += 1

    # Compute ratios
    result_by_doc = {}
    for doc, data in by_document.items():
        total_voiced = data["passive_count"] + data["active_count"]
        result_by_doc[doc] = {
            "doc_year": data["doc_year"],
            "era": data["era"],
            "passive_count": data["passive_count"],
            "agentless_passive_count": data["agentless_passive_count"],
            "by_phrase_count": data["by_phrase_count"],
            "active_count": data["active_count"],
            "agentless_passive_ratio": round(
                data["agentless_passive_count"] / data["passive_count"], 3
            ) if data["passive_count"] > 0 else 0,
            "passive_rate": round(
                data["passive_count"] / total_voiced, 3
            ) if total_voiced > 0 else 0,
            "flagged_sentences": data["flagged_sentences"][:5],  # Top 5
        }

    # By era
    by_era = {}
    for era in ERA_DEFINITIONS:
        era_docs = {d: v for d, v in result_by_doc.items() if v["era"] == era}
        passive_total = sum(d["passive_count"] for d in era_docs.values())
        agentless_total = sum(d["agentless_passive_count"] for d in era_docs.values())
        active_total = sum(d["active_count"] for d in era_docs.values())
        voiced_total = passive_total + active_total

        by_era[era] = {
            "passive_count": passive_total,
            "agentless_passive_count": agentless_total,
            "active_count": active_total,
            "agentless_passive_ratio": round(
                agentless_total / passive_total, 3
            ) if passive_total > 0 else 0,
            "passive_rate": round(
                passive_total / voiced_total, 3
            ) if voiced_total > 0 else 0,
        }

    # Diachronic trend
    diachronic = [
        {
            "era": era,
            "agentless_ratio": by_era[era]["agentless_passive_ratio"],
            "passive_rate": by_era[era]["passive_rate"],
        }
        for era in ERA_DEFINITIONS
    ]

    return {
        "by_document": result_by_doc,
        "by_era": by_era,
        "diachronic_trend": diachronic,
    }


def save_svo_results(
    svo_triples: list[dict],
    power_triangle: dict,
    backgrounding: dict,
    output_dir: Path,
) -> None:
    """Save all SVO Power Triangle outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # SVO triples JSONL
    path = output_dir / "svo_triples.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in svo_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved {len(svo_triples)} SVO triples to {path}")

    # Power triangle JSON
    path = output_dir / "power_triangle.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(power_triangle, f, ensure_ascii=False, indent=2)
    print(f"Saved power triangle to {path}")

    # Power triangle summary CSV
    rows = []
    for era, langs in power_triangle.items():
        for lang, data in langs.items():
            for cat, count in data["agent_distribution"].items():
                rows.append({
                    "era": era,
                    "target_language": lang,
                    "agent_category": cat,
                    "count": count,
                    "proportion": round(count / data["total_triples"], 3) if data["total_triples"] > 0 else 0,
                    "top_agent": data["top_agents"][0]["entity"] if data["top_agents"] else "",
                    "top_verb": data["top_verbs"][0]["verb"] if data["top_verbs"] else "",
                })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "power_triangle_summary.csv", index=False)
        print(f"Saved power triangle summary to {output_dir / 'power_triangle_summary.csv'}")

    # Agent backgrounding JSON
    path = output_dir / "agent_backgrounding.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(backgrounding, f, ensure_ascii=False, indent=2)
    print(f"Saved agent backgrounding to {path}")

    # Agent backgrounding summary CSV
    rows = []
    for doc, data in backgrounding["by_document"].items():
        rows.append({
            "doc_title": doc,
            "doc_year": data["doc_year"],
            "era": data["era"],
            "passive_count": data["passive_count"],
            "agentless_count": data["agentless_passive_count"],
            "by_phrase_count": data["by_phrase_count"],
            "active_count": data["active_count"],
            "agentless_ratio": data["agentless_passive_ratio"],
            "passive_rate": data["passive_rate"],
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("doc_year").reset_index(drop=True)
        df.to_csv(output_dir / "agent_backgrounding_summary.csv", index=False)
        print(f"Saved agent backgrounding summary to {output_dir / 'agent_backgrounding_summary.csv'}")


def run_svo_analysis(
    dep_frames_path: Path | None = None,
    output_dir: Path | None = None,
    nlp=None,
    gpu_id: int = 0,
) -> dict:
    """Run the full SVO Power Triangle analysis."""
    if dep_frames_path is None:
        dep_frames_path = PHASE2_DIR / "dep_frames.jsonl"
    if output_dir is None:
        output_dir = DEEP_OUTPUT_DIR

    print("=" * 60)
    print("ANALYSIS 1: SVO Power Triangle")
    print("=" * 60)

    # Step 1: Extract SVO triples (requires spaCy re-parsing)
    print("\n[1/3] Extracting SVO triples from patient frames...")
    triples = extract_svo_triples(dep_frames_path, nlp=nlp, gpu_id=gpu_id)

    # Quick stats
    cats = Counter(t["agent_category"] for t in triples)
    print(f"  Agent categories: {dict(cats.most_common())}")

    # Step 2: Build power triangle
    print("\n[2/3] Building power triangle...")
    triangle = build_power_triangle(triples)
    for era, langs in triangle.items():
        print(f"\n  {era}:")
        for lang, data in langs.items():
            top_agent = data["top_agents"][0]["entity"] if data["top_agents"] else "N/A"
            print(f"    {lang} (n={data['total_triples']}): top_agent=\"{top_agent}\", agentless_ratio={data['agentless_ratio']}")

    # Step 3: Agent backgrounding
    print("\n[3/3] Analyzing agent backgrounding...")
    backgrounding = analyze_agent_backgrounding(triples, dep_frames_path)
    for era, data in backgrounding["by_era"].items():
        print(f"  {era}: passive_rate={data['passive_rate']}, agentless_ratio={data['agentless_passive_ratio']}")

    # Save
    print("\nSaving results...")
    save_svo_results(triples, triangle, backgrounding, output_dir)

    return {
        "svo_triples": triples,
        "power_triangle": triangle,
        "agent_backgrounding": backgrounding,
    }
