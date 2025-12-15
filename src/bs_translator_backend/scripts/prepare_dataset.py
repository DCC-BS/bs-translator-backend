from __future__ import annotations

from collections import Counter

from bs_translator_backend.services.dspy_config.dataset_loader import (
    CUSTOM_TEMPLATE_PATH,
    EUROPARL_SAMPLE_PATH,
    ensure_custom_template,
    get_or_create_europarl_samples,
)
from bs_translator_backend.utils.logger import get_logger, init_logger


def main() -> None:
    init_logger()
    logger = get_logger("prepare_dataset")

    ensure_custom_template()
    logger.info("Custom template ensured", path=str(CUSTOM_TEMPLATE_PATH))

    examples = get_or_create_europarl_samples()
    logger.info("Europarl samples ready", path=str(EUROPARL_SAMPLE_PATH))

    split_counts = Counter(ex.split for ex in examples)
    direction_counts = Counter(
        (ex.source_language, ex.target_language, ex.split) for ex in examples
    )

    logger.info(
        "Dataset summary",
        total=len(examples),
        splits=dict(split_counts),
    )

    for (src, tgt, split), count in sorted(direction_counts.items()):
        logger.info("Direction stats", source=src, target=tgt, split=split, count=count)


if __name__ == "__main__":
    main()
