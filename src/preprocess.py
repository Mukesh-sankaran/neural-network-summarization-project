"""
Text preprocessing and custom tokenization module for summarization.
Handles text cleaning, normalization, and word-level tokenization.
"""

import re

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def clean_text(text: str) -> str:
    """
    Clean and normalize text while preserving numbers, dates, percentages, and monetary values.
    
    Operations:
    - Converts input to lowercase.
    - Normalizes multi-spaces, tabs, and newlines to a single space.
    - Separates punctuation marks with spaces so word-level tokenization can isolate words
      and punctuation, while avoiding splitting decimal numbers, dates, percentages, and currencies.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Normalize all whitespace characters (newlines, tabs, multiple spaces) to a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Separate standard punctuation marks: quotes, parens, brackets, braces, colons, semicolons, commas, exclamations, questions, dashes
    text = re.sub(r"([,\"\'\(\)\[\]\{\}\:\;\!\?\-\~])", r" \1 ", text)

    # Separate periods unless part of a decimal/float number (e.g. 3.14 or 15.5) or abbreviation
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " . ", text)

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenizes text into a list of word/punctuation tokens using the custom clean_text function.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()
