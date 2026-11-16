"""
Entity Evaluation and Factual Consistency module for Summarization.
Uses spaCy (en_core_web_sm) to extract named entities and compute:
- Entity Precision
- Entity Recall
- Entity Preservation Rate
- Preserved, Missing, and Extra (Hallucinated) Entities
- Numeric/Factual Consistency Rate
"""

import re
from typing import Dict, List, Set, Any, Tuple
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
    
    Entity Preservation Rate = (Preserved Entities from Source) / (Total Source Entities)
    Entity Precision = (Preserved Entities from Source) / (Total Summary Entities)
    Entity Recall = (Preserved Entities from Source) / (Total Source Entities)
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

    # Precision: proportion of summary entities that appeared in source document
    if num_sum > 0:
        precision = num_preserved / num_sum
    else:
        precision = 1.0 if num_src == 0 else 0.0

    # Recall & Preservation Rate: proportion of source document entities preserved in summary
    if num_src > 0:
        recall = num_preserved / num_src
        preservation_rate = num_preserved / num_src
    else:
        recall = 1.0
        preservation_rate = 1.0

    return {
        "entity_precision": float(precision),
        "entity_recall": float(recall),
        "entity_preservation_rate": float(preservation_rate),
        "num_source_entities": num_src,
        "num_summary_entities": num_sum,
        "num_preserved_entities": num_preserved,
        "preserved_entities": sorted(list(preserved_set)),
        "missing_entities": sorted(list(missing_set)),
        "extra_entities": sorted(list(extra_set))
    }


def extract_numeric_values(text: str) -> Set[str]:
    """
    Extracts normalized numbers, percentages, dates, and monetary values using regex.
    """
    if not text:
        return set()

    normalized = normalize_text(text)
    
    # Regex patterns for numbers, dates, currencies, percentages
    patterns = [
        r"\b\d+(?:\.\d+)?%?\b",                       # Numbers, decimals, percentages (15.5%)
        r"[\$£€]\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand|b|m|k))?\b",  # Currency ($3.4 billion)
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",               # Dates 08/21/2026
        r"\b\d{4}-\d{2}-\d{2}\b",                     # Dates 2026-08-21
        r"\b\d+(?:\s*(?:billion|million|thousand))\b" # Named scale numbers (3 billion)
    ]

    matches = set()
    for pat in patterns:
        for m in re.finditer(pat, normalized):
            matches.add(m.group(0))

    return matches


def evaluate_numeric_consistency(source_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Evaluates factual consistency for numerical values, dates, percentages, and monetary units.
    Numeric Consistency = (Matching Numbers in Summary present in Source) / (Total Numbers in Summary)
    """
    src_nums = extract_numeric_values(source_text)
    sum_nums = extract_numeric_values(summary_text)

    if not sum_nums:
        # If no numbers were generated, no numerical hallucination occurred
        consistency_rate = 1.0
        consistent_nums = set()
        hallucinated_nums = set()
    else:
        consistent_nums = sum_nums.intersection(src_nums)
        hallucinated_nums = sum_nums - src_nums
        consistency_rate = len(consistent_nums) / len(sum_nums)

    return {
        "numeric_consistency_rate": float(consistency_rate),
        "total_source_numbers": len(src_nums),
        "total_summary_numbers": len(sum_nums),
        "consistent_numbers": sorted(list(consistent_nums)),
        "hallucinated_numbers": sorted(list(hallucinated_nums))
    }
