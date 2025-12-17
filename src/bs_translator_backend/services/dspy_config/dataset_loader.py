from __future__ import annotations

import random
from collections.abc import Iterable
from csv import DictWriter
from itertools import permutations
from pathlib import Path
from typing import Any

import dspy
from datasets import load_dataset

EUROPARL_LANGUAGES = ["de", "en", "es", "it", "fr", "pl", "ro", "sl", "nl", "fi"]
LANGUAGE_CODES_TO_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "it": "Italian",
    "fr": "French",
    "pl": "Polish",
    "ro": "Romanian",
    "sl": "Slovenian",
    "nl": "Dutch",
    "fi": "Finnish",
}
EUROPARL_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "europarl_samples.csv"
CUSTOM_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "custom_translation_template.csv"
)
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_DEV_SIZE = 10
DEFAULT_TEST_SIZE = 10


def _ensure_data_dir(path: Path) -> None:
    """Create parent directory for the provided path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_subset(pair: tuple[str, str], sample_size: int):
    """Load a shuffled subset of Europarl for the requested pair."""
    lang_a, lang_b = sorted(pair)
    split = "train[:5000]"
    dataset = load_dataset("Helsinki-NLP/europarl", f"{lang_a}-{lang_b}", split=split)
    return dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))  # pyright: ignore


def _rows_for_direction(
    dataset: Iterable[dict[str, Any]], source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    """Convert dataset rows into directional rows with metadata placeholders."""
    rows: list[dict[str, str]] = []
    for row in dataset:
        translation = row.get("translation", {})
        source_text = translation.get(source_lang)
        target_text = translation.get(target_lang)
        if not source_text or not target_text:
            continue
        rows.append({
            "source_text": str(source_text),
            "target_text": str(target_text),
            "source_language": LANGUAGE_CODES_TO_NAMES[source_lang],
            "target_language": LANGUAGE_CODES_TO_NAMES[target_lang],
            "domain": "",
            "tone": "",
            "glossary": "",
        })
    return rows


def sample_europarl_pairs(
    train_size: int = DEFAULT_SAMPLE_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    languages: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Sample Europarl pairs for the provided languages and return rows ready for CSV.

    Args:
        train_size: Number of training examples per direction.
        dev_size: Number of dev examples per direction.
        test_size: Number of test examples per direction.
        languages: Languages to sample (defaults to EUROPARL_LANGUAGES).
    """
    langs = languages or EUROPARL_LANGUAGES
    all_rows: list[dict[str, str]] = []
    per_split_total = train_size + dev_size + test_size
    for source_lang, target_lang in permutations(langs, 2):
        dataset = _load_subset((source_lang, target_lang), per_split_total)
        directional_rows = _rows_for_direction(dataset, source_lang, target_lang)
        train_rows = directional_rows[:train_size]
        dev_rows = directional_rows[train_size : train_size + dev_size]
        test_rows = directional_rows[train_size + dev_size : train_size + dev_size + test_size]
        for row in train_rows:
            all_rows.append(row | {"split": "train"})
        for row in dev_rows:
            all_rows.append(row | {"split": "dev"})
        for row in test_rows:
            all_rows.append(row | {"split": "test"})
    return all_rows


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Write rows to CSV with a fixed header."""
    _ensure_data_dir(path)
    header = [
        "source_text",
        "target_text",
        "source_language",
        "target_language",
        "domain",
        "tone",
        "glossary",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_or_create_europarl_samples(
    path: Path = EUROPARL_SAMPLE_PATH,
    train_size: int = DEFAULT_SAMPLE_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    languages: list[str] | None = None,
) -> list[dspy.Example]:
    """
    Return DSPy Examples from a cached CSV, creating it from Europarl if missing.
    """
    if path.exists():
        return _read_examples_from_csv(path)

    rows = sample_europarl_pairs(
        train_size=train_size, dev_size=dev_size, test_size=test_size, languages=languages
    )
    _write_csv(path, rows)
    return _read_examples_from_csv(path)


def _read_examples_from_csv(path: Path) -> list[dspy.Example]:
    """Load DSPy Examples from a CSV file."""
    import csv

    examples: list[dspy.Example] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(
                dspy.Example(
                    source_text=row["source_text"],
                    translated_text=row["target_text"],
                    source_language=row["source_language"],
                    target_language=row["target_language"],
                    domain=row.get("domain", ""),
                    tone=row.get("tone", ""),
                    glossary=row.get("glossary", ""),
                    context="",
                    split=row.get("split", ""),
                ).with_inputs(
                    "source_text",
                    "source_language",
                    "target_language",
                    "domain",
                    "tone",
                    "glossary",
                    "context",
                )
            )
    return examples


def load_custom_data_with_split(
    path: Path,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.2,
) -> dict[str, list[dspy.Example]]:
    """
    Load custom translation data (no split column) and assign train/dev/test with 70/20/10 split.
    """
    import csv
    from random import Random

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    RANDOM_SEED = 42
    rng = Random(RANDOM_SEED)  # noqa: S311
    rng.shuffle(rows)

    n = len(rows)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    splits = {
        "train": rows[:train_end],
        "dev": rows[train_end:dev_end],
        "test": rows[dev_end:],
    }

    def to_examples(items: list[dict[str, str]]) -> list[dspy.Example]:
        out: list[dspy.Example] = []
        for row in items:
            out.append(
                dspy.Example(
                    source_text=row.get("source_text", ""),
                    translated_text=row.get("target_text", ""),
                    source_language=row.get("source_language", ""),
                    target_language=row.get("target_language", ""),
                    domain=row.get("domain", ""),
                    tone=row.get("tone", ""),
                    glossary=row.get("glossary", ""),
                    context="",
                    split=row.get("split", ""),
                ).with_inputs(
                    "source_text",
                    "source_language",
                    "target_language",
                    "domain",
                    "tone",
                    "glossary",
                    "context",
                )
            )
        return out

    return {key: to_examples(value) for key, value in splits.items()}


def ensure_custom_template(path: Path = CUSTOM_TEMPLATE_PATH) -> None:
    """
    Ensure a template CSV exists for custom training data.
    """
    if path.exists():
        return
    _ensure_data_dir(path)
    header = [
        "source_text",
        "target_text",
        "source_language",
        "target_language",
        "domain",
        "tone",
        "glossary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=header)
        writer.writeheader()


def combined_splits(
    custom_path: Path | None = None,
    train_size: int = DEFAULT_SAMPLE_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    languages: list[str] | None = None,
) -> dict[str, list[dspy.Example]]:
    """
    Return merged train/dev/test splits from Europarl samples and optional custom data.
    """
    splits: dict[str, list[dspy.Example]] = {"train": [], "dev": [], "test": []}

    euro_examples = get_or_create_europarl_samples(
        train_size=train_size, dev_size=dev_size, test_size=test_size, languages=languages
    )
    for ex in euro_examples:
        split = getattr(ex, "split", "") or "train"
        if split not in splits:
            split = "train"
        splits[split].append(ex)

    if custom_path is not None and custom_path.exists():
        custom = load_custom_data_with_split(custom_path)
        for key, items in custom.items():
            if key not in splits:
                continue
            splits[key].extend(items)
    # shuffle splits
    for split in splits.values():
        random.seed(42)
        random.shuffle(split)
    return splits
