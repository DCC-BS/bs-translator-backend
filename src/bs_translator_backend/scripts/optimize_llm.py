from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import dspy
from backend_common.dspy_common import edit_distance_metric
from backend_common.logger import get_logger, init_logger

from bs_translator_backend.container import Container
from bs_translator_backend.services.dspy_config.dataset_loader import (
    EUROPARL_SAMPLE_PATH,
    combined_splits,
)
from bs_translator_backend.services.dspy_config.translation_program import (
    TranslationModule,
    optimize_translation_module,
)


def evaluate_program(program: TranslationModule, valset: Iterable[dspy.Example]) -> None:
    run_name = f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    evaluate = dspy.Evaluate(
        devset=list(valset),
        metric=edit_distance_metric,
        num_threads=2,
        display_table=True,
        display_progress=True,
        save_as_csv=run_name,
        provide_traceback=True,
    )
    result = evaluate(program)
    logger = get_logger(__name__)
    logger.info("Evaluation result", result=result)


def main() -> None:
    init_logger()
    logger = get_logger("optimize_llm")
    container = Container()

    config = container.app_config()

    inference_lm = dspy.LM(
        model=config.llm_model,
        api_key=config.openai_api_key,
        api_base=config.openai_api_base_url,
    )

    optimizer_lm = dspy.LM(
        model=config.optimizer_model,
        api_key=config.optimizer_api_key,
        api_base=config.optimizer_api_base_url,
    )

    dspy.configure(lm=inference_lm)

    logger.info("Loading Europarl samples", path=str(EUROPARL_SAMPLE_PATH))
    splits = combined_splits(
        custom_path=Path("src/bs_translator_backend/data/custom_translation.csv")
    )
    logger.info("Starting GEPA optimization", train_size=len(splits["train"]))
    base_program = TranslationModule(app_config=config)

    logger.info("Evaluate unoptimized program")
    evaluate_program(base_program, splits["dev"])

    optimized = optimize_translation_module(
        base_program=base_program,
        trainset=splits["train"],
        valset=splits["dev"],
        reflection_lm=optimizer_lm,
        task_lm=inference_lm,
    )

    optimized.save(config.translation_module_path)
    logger.info("Optimization complete")

    logger.info("Load optimized program")
    optimized_module = TranslationModule(app_config=config)
    optimized_module.load(config.translation_module_path)
    logger.info("Evaluate optimized program")
    evaluate_program(optimized_module, splits["dev"])


if __name__ == "__main__":
    main()
