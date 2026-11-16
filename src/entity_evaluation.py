"""
Entity Evaluation and Factual/Numeric Consistency module for Summarization.
Uses spaCy (en_core_web_sm) to extract named entities and compute:
- Entity Precision (groundedness: proportion of summary entities present in source)
- Entity Recall (coverage: proportion of unique source entities preserved in summary)
- Entity Preservation Rate (mention-weighted source entity coverage)
- Entity F1 Score
- Preserved, Missing, and Extra (Hallucinated) Entities
- Numeric Consistency Heuristic (overlap of numbers, dates, currencies, percentages)

NOTE: Numeric consistency is a surface-level overlap heuristic comparing extracted
numerical tokens between summary and source. It does not certify holistic factual truth.
"""

import re
from collections import Counter
from typing import Dict, List, Set, Any, Tuple, Optional
import spacy

# Load spaCy NLP pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model 'en_core_web_sm'. Error: {e}")

# Target Entity Types to Evaluate
TARGET_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "DATE",
    "MONEY", "PERCENT", "CARDINAL", "QUANTITY"
}


def normalize_text(text: str) -> str:
    """
    Case-insensitive and whitespace-normalized string clean-up.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_named_entities(text: str) -> List[Dict[str, str]]:
    """
    Extracts named entities from input text using spaCy.
    Returns list of dicts with normalized text, raw text, and entity label.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in TARGET_ENTITY_TYPES:
            norm_text = normalize_text(ent.text)
            if norm_text:
                entities.append({
                    "text": norm_text,
                    "original": ent.text,
                    "label": ent.label_
                })

    return entities


def evaluate_entity_preservation(source_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Calculates Entity Precision, Recall, and Preservation Rate between source and summary.

    Definitions:
    - Entity Precision: Proportion of summary entities that appeared in source document (groundedness).
      precision = |preserved| / |summary_entities|
    - Entity Recall: Proportion of unique source document entities preserved in summary (coverage).
      recall = |preserved| / |source_entities|
    - Entity Preservation Rate: Mention-weighted proportion of source entity mentions preserved in summary.
      preservation_rate = sum(src_mentions(preserved)) / total_src_mentions
    - Entity F1: Harmonic mean of precision and recall.
    """
    src_entities_list = extract_named_entities(source_text)
    sum_entities_list = extract_named_entities(summary_text)

    # Sets of normalized entity strings
    src_entity_set: Set[str] = {e["text"] for e in src_entities_list}
    sum_entity_set: Set[str] = {e["text"] for e in sum_entities_list}

    preserved_set = src_entity_set.intersection(sum_entity_set)
    missing_set = src_entity_set - sum_entity_set
    extra_set = sum_entity_set - src_entity_set

    num_src = len(src_entity_set)
    num_sum = len(sum_entity_set)
    num_preserved = len(preserved_set)

    # Mention counts for weighted preservation rate
    src_mention_counts = Counter(e["text"] for e in src_entities_list)
    total_src_mentions = len(src_entities_list)
    preserved_mentions = sum(src_mention_counts[e] for e in preserved_set)

    # Precision: proportion of summary entities grounded in source
    if num_sum > 0:
        precision: Optional[float] = num_preserved / num_sum
    else:
        # No summary entities generated: precision is undefined/unavailable
        precision = None

    # Recall: proportion of unique source entities captured
    if num_src > 0:
        recall: Optional[float] = num_preserved / num_src
    else:
        # No source entities to recall
        recall = None

    # Preservation Rate: mention-weighted coverage
    if total_src_mentions > 0:
        preservation_rate: Optional[float] = preserved_mentions / total_src_mentions
    else:
        preservation_rate = None

    # F1 score
    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1: Optional[float] = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
    else:
        f1 = None

    # Explicit status determination
    if num_src == 0 and num_sum == 0:
        status = "no_entities_present"
        status_description = "Neither source nor summary contains recognized named entities."
    elif num_sum == 0:
        status = "no_summary_entities"
        status_description = f"Source contains {num_src} entities, but summary generated no named entities."
    elif num_preserved == 0:
        status = "zero_entities_preserved"
        status_description = f"Summary generated {num_sum} entities, but none matched the {num_src} source entities."
    elif num_preserved == num_sum:
        status = "fully_grounded"
        status_description = f"All {num_sum} summary entities are grounded in the source text."
    else:
        status = "partially_grounded"
        status_description = f"{num_preserved} of {num_sum} summary entities matched source ({len(extra_set)} ungrounded)."

    return {
        "entity_precision": float(precision) if precision is not None else None,
        "entity_recall": float(recall) if recall is not None else None,
        "entity_preservation_rate": float(preservation_rate) if preservation_rate is not None else None,
        "entity_f1": float(f1) if f1 is not None else None,
        "status": status,
        "status_description": status_description,
        "num_source_entities": num_src,
        "num_summary_entities": num_sum,
        "num_preserved_entities": num_preserved,
        "total_source_mentions": total_src_mentions,
        "preserved_source_mentions": preserved_mentions,
        "preserved_entities": sorted(list(preserved_set)),
        "missing_entities": sorted(list(missing_set)),
        "extra_entities": sorted(list(extra_set))
    }


def extract_numeric_values(text: str) -> Set[str]:
    """
    Extracts normalized numbers, percentages, dates, and monetary values using regex.
    Handles comma-separated thousands (100,000 -> 100000), decimals (3.14),
    currencies ($3.4 billion, £100), dates (2026-08-21, 08/21/2026), and percentages (15.5%).
    """
    if not text:
        return set()

    normalized = normalize_text(text)

    # Normalize space-separated punctuation in numeric expressions (e.g. "100 , 000" -> "100000", "$ 100" -> "$100")
    normalized = re.sub(r"(\d)\s*,\s*(\d)", r"\1\2", normalized)
    normalized = re.sub(r"([\$£€])\s+(\d)", r"\1\2", normalized)
    normalized = re.sub(r"(\d)\s+%", r"\1%", normalized)

    patterns = [
        # Currencies with scale: $3.4 billion, £100,000, €50m
        r"[\$£€]\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand|b|m|k))?\b",
        # Percentages: 15.5%, 100%
        r"\b\d+(?:\.\d+)?%\b",
        # ISO Dates: 2026-08-21
        r"\b\d{4}-\d{2}-\d{2}\b",
        # Slash Dates: 08/21/2026 or 8/21/26
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        # Scale numbers: 3.4 billion, 100 million
        r"\b\d+(?:\.\d+)?\s*(?:billion|million|thousand)\b",
        # Floating point numbers / decimals: 3.14, 0.5
        r"\b\d+\.\d+\b",
        # Standalone integers: 100, 2026, 42
        r"\b\d+\b",
    ]

    matches: Set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, normalized):
            val = m.group(0).strip()
            if val:
                matches.add(val)

    return matches


def evaluate_numeric_consistency(source_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Evaluates numeric overlap between generated summary and source text.

    Numeric consistency is a surface-level overlap heuristic:
    - Proportion of summary numbers that are present in the source article.
    - If the summary contains no numbers, the metric is marked 'unavailable' (None)
      rather than claiming 100% factual consistency.
    """
    src_nums = extract_numeric_values(source_text)
    sum_nums = extract_numeric_values(summary_text)

    num_src = len(src_nums)
    num_sum = len(sum_nums)

    if num_sum == 0 and num_src == 0:
        # Case 1: Neither article nor summary contains numbers
        consistency_rate = None
        consistent_nums: Set[str] = set()
        hallucinated_nums: Set[str] = set()
        status = "no_numbers_present"
        status_description = "Neither source document nor summary contains numeric values."
    elif num_sum == 0 and num_src > 0:
        # Case 2: Article contains numbers, but summary generated none
        consistency_rate = None
        consistent_nums = set()
        hallucinated_nums = set()
        status = "no_summary_numbers"
        status_description = f"Source contains {num_src} numbers, but summary generated none (numeric precision undefined)."
    elif num_sum > 0 and num_src == 0:
        # Case 3: Summary contains numbers, but source contained none (all hallucinated)
        consistency_rate = 0.0
        consistent_nums = set()
        hallucinated_nums = set(sum_nums)
        status = "hallucinated_numbers"
        status_description = f"Summary generated {num_sum} numbers, but source has none (all ungrounded)."
    else:
        # Case 4: Both source and summary contain numbers
        consistent_nums = sum_nums.intersection(src_nums)
        hallucinated_nums = sum_nums - src_nums
        consistency_rate = len(consistent_nums) / num_sum

        if len(hallucinated_nums) == 0:
            status = "fully_consistent"
            status_description = f"All {num_sum} summary numbers are grounded in the source text."
        elif len(consistent_nums) > 0:
            status = "partially_consistent"
            status_description = f"{len(consistent_nums)} of {num_sum} summary numbers match source ({len(hallucinated_nums)} ungrounded)."
        else:
            status = "hallucinated_numbers"
            status_description = f"None of the {num_sum} summary numbers match source numbers."

    return {
        "numeric_consistency_rate": float(consistency_rate) if consistency_rate is not None else None,
        "status": status,
        "status_description": status_description,
        "total_source_numbers": num_src,
        "total_summary_numbers": num_sum,
        "consistent_numbers": sorted(list(consistent_nums)),
        "hallucinated_numbers": sorted(list(hallucinated_nums))
    }

