# BaseAgent Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the translator's two LLM agents onto `dcc_backend_common.llm_agent.BaseAgent` so the repo stops constructing OpenAI/pydantic-ai clients by hand and inherits common's retry, timeout, postprocessing, and usage-tracking behaviour.

**Architecture:** `BaseAgent` already owns everything `agents/translation_agent.py` hand-rolls, plus hardening the translator lacks (tenacity retries honouring `Retry-After`, deliberate 4xx/5xx classification, configurable timeout, client cleanup, `replace_eszett`). One gap blocks adoption — `BaseAgent.run_stream_events` loses usage accounting when an SSE client disconnects mid-stream — so backend-common is fixed and released first, then the translator migrates on top of the published version. `BaseAgent.create_agent` is abstract, so pydantic-ai imports remain in `agents/*.py` by design; every other layer becomes pydantic-ai-free and the direct `openai` import disappears entirely.

**Tech Stack:** Python 3.12+, uv, pydantic-ai 2.30, dcc-backend-common 0.1.22 → 0.1.23, FastAPI, dependency-injector, pytest/pytest-asyncio, ruff, ty.

**Spec:** This document. Decisions were settled in an architecture interview on 2026-08-21; each is recorded inline at the task it governs.

## Global Constraints

- **Two repos.** `~/code/backend-common` (Tasks 1–2) and `/home/yanick/code/translator/bs-translator-backend` (Tasks 3–8). Task 3 must not start until 0.1.23 is published.
- **No direct `openai` import anywhere in the translator.** It is deliberately *not* declared in `pyproject.toml` and must not be added.
- **pydantic-ai imports are confined to `src/bs_translator_backend/agents/*.py`.** `services/`, `routers/`, `models/`, `utils/` must import zero pydantic-ai when the migration is done.
- **Dependency floor:** translator moves to `dcc-backend-common[fastapi,pydantic-ai]>=0.1.23`. The `pydantic-ai` extra is required — `BaseAgent` is unimportable without it. `pydantic-ai` itself stays a direct dependency, because `agents/translation_agent.py` imports it directly.
- **`[tool.uv] exclude-newer = "P7D"` stays.** `exclude-newer-package = { dcc-backend-common = "P0D" }` already exempts common, so a fresh release resolves immediately.
- **Test baseline:** the translator suite must stay green at every task boundary. Baseline at plan time is **95 passed** plus whatever the in-flight e2e task adds. Never finish a task with a failing suite.
- **One pre-existing lint error is expected and must NOT be fixed:** `S106` at `tests/integration/conftest.py:16`. Any *other* ruff finding is yours.
- **`uv run ty check ./src/bs_translator_backend` must stay "All checks passed!"** at every task boundary.
- **Commit per task.** Never commit to `main` in either repo; both already have working branches.

---

## Reference Implementation

`/home/yanick/code/textmate/text-mate-backend` is the established `BaseAgent` consumer — 20+ agents in production. This plan was cross-checked against it, and every convention below is copied from there rather than invented. **Read these files before starting Task 4:**

| File | What it establishes |
|---|---|
| `src/text_mate_backend/agents/agent_types/word_synonym_agent.py` | The canonical single agent: module-level `INSTRUCTION` constant, `@override` on `create_agent`, `@agent.instructions` decorator *inside* `create_agent`, `name=` / `description=` on the `Agent`. |
| `src/text_mate_backend/agents/agent_types/quick_actions/quick_action_base_agent.py` | How an abstract intermediate agent class layers on top of `BaseAgent`. |
| `src/text_mate_backend/utils/configuration.py:9` | `class Configuration(LlmConfig)` — confirms Task 3's inheritance decision; `:68-69` shows `llm_timeout` / `llm_max_retries` set as literals, not env vars. |
| `src/text_mate_backend/services/fix_service.py:34` | Streaming consumption: `async for chunk in self.agent.run_stream_text(user_prompt=text, deps=request)`. Agent is built in the service's `__init__`. |
| `src/text_mate_backend/app.py:69,92` | Lifespan registration pattern. |

**Version context, already checked — do not re-investigate:**
- `text-mate-backend` pins `pydantic-ai<2.22.1`. This is **specific to that repo**, not a constraint on `BaseAgent`. `backend-common` itself develops and tests against **pydantic-ai 2.31.0**, newer than the translator's 2.30.0, so `BaseAgent` on 2.30 is well within its tested range.
- `text-mate-backend` declares `pydantic-ai` as a **direct** dependency alongside `dcc-backend-common[fastapi]`, which is exactly the arrangement Task 3 adopts.

**Where this plan deliberately diverges from the reference**, each with its reason stated at the relevant task: `trim_text` suppression (Task 4 Step 5), no Logfire `metadata=` (Task 4 Step 3), parameterised `BaseAgent[None, str]` (Task 4 Step 3), and agent cleanup on shutdown (Task 6). Nothing else should differ. If you find yourself inventing a pattern that is not in the reference and not listed here, stop and ask.

---

## File Structure

**backend-common (`~/code/backend-common`)**

| File | Responsibility after this plan |
|---|---|
| `src/dcc_backend_common/llm_agent/base_agent.py` | Gains guaranteed usage logging when a stream consumer exits early. |
| `tests/unit/test_base_agent.py` | Gains coverage for the aborted-stream case. |
| `pyproject.toml` | Version 0.1.22 → 0.1.23. |

**translator (`bs-translator-backend`)**

| File | Responsibility after this plan |
|---|---|
| `src/bs_translator_backend/utils/app_config.py` | `AppConfig` inherits `LlmConfig`; sets `llm_timeout` / `llm_max_retries` as literals. |
| `src/bs_translator_backend/agents/translation_agent.py` | Two `BaseAgent` subclasses + their instruction strings. The only file importing pydantic-ai. No `openai` import, no model/provider/client construction. |
| `src/bs_translator_backend/services/translation_service.py` | Consumes the agents via `run_stream_text`; owns no logging `finally`; gains `aclose()`. |
| `src/bs_translator_backend/app.py` | Lifespan closes the translation service on shutdown. |
| `tests/unit/test_translation_agent.py` | Retargeted at the new classes. |

---

## Task 1: Guarantee usage logging on aborted streams (backend-common)

**Decision:** fix in backend-common, not by overriding in the translator, so every service using `BaseAgent` gets correct accounting.

**Why this is needed:** the translator currently logs usage in a `finally` block, with this comment at `services/translation_service.py`:

```python
# finally: a client disconnect closes this generator mid-stream,
# and the tokens consumed so far must still be logged.
```

`BaseAgent.run_stream_events` logs only when the final `AgentRunResultEvent` arrives, inside an `async for` with no `try`/`finally`. When an SSE consumer stops early the generator is closed and that event never arrives, so the tokens already spent go unrecorded. Migrating without this fix is a silent regression in billing data.

**Files:**
- Modify: `~/code/backend-common/src/dcc_backend_common/llm_agent/base_agent.py` (`run_stream_events`)
- Test: `~/code/backend-common/tests/unit/test_base_agent.py`

**Interfaces:**
- Consumes: `pydantic_ai.usage.RunUsage` (verified importable), `Agent.run_stream_events(..., usage=...)` (verified to accept a `usage` accumulator), `dcc_backend_common.usage_tracking.log_llm_call`.
- Produces: no signature change. `run_stream_events` keeps yielding `AgentStreamEvent | AgentRunResultEvent[OutputType]`.

**Key facts already verified — do not re-derive:**
- `Agent.run_stream_events` accepts a keyword `usage: RunUsage | None`. Pass one in and it accumulates as the run proceeds, so partial usage is readable after an early exit.
- `log_llm_call(result)` is duck-typed on `result.usage` and `result.response` only. Its own docstring says: *"Never raises: it is also called from `finally` blocks on aborted streams, where the result may be incomplete."*
- `log_llm_call` reads `response.parts` (iterated for `part_kind == "tool-call"`) and `response.finish_reason`, and tolerates `finish_reason is None`. A stand-in response with `parts=[]` and `finish_reason=None` is therefore safe.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_base_agent.py`. Reuse the existing module-level helpers `fake_stream_events`, `make_text_delta`, and `make_run_result_event` — read them first; they are already defined at the top of that file.

```python
@pytest.mark.asyncio
async def test_usage_is_logged_when_consumer_exits_before_result_event():
    """An SSE client that disconnects mid-stream must still have its tokens recorded."""
    agent = make_test_agent()  # follow the existing pattern in this file

    with patch("dcc_backend_common.llm_agent.base_agent.log_llm_call") as mock_log:
        stream = agent.run_stream_events("prompt")
        async for _event in stream:
            break  # consumer gives up before AgentRunResultEvent arrives
        await stream.aclose()

        assert mock_log.call_count == 1, "aborted stream must log usage exactly once"
        logged = mock_log.call_args[0][0]
        assert logged.usage is not None
        assert logged.response.finish_reason is None
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd ~/code/backend-common
uv run python -m pytest tests/unit/test_base_agent.py -k aborted -v
```

Expected: FAIL — `mock_log.call_count == 0`, because nothing logs without the result event.

- [ ] **Step 3: Implement the fix**

In `base_agent.py`, add near the other module-level helpers:

```python
@dataclass(frozen=True)
class _AbortedResponse:
    """Minimal response stand-in for a run that never produced a final response."""

    parts: tuple[()] = ()
    finish_reason: None = None


@dataclass(frozen=True)
class _AbortedRun:
    """Duck-typed ``log_llm_call`` input carrying only the usage accrued so far."""

    usage: Any
    response: _AbortedResponse = _AbortedResponse()
```

Then rewrite `run_stream_events` so the accumulator is passed in and a `finally` covers the early-exit path:

```python
    async def run_stream_events(
        self,
        user_prompt: UserPrompt = None,
        deps: DepsType | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[OutputType]]:
        """Stream raw pydantic-ai events. No postprocessing; use run() for a postprocessed final result.

        Usage is logged exactly once: from the final AgentRunResultEvent when the run
        completes, or — if the consumer stops early, as an SSE client that disconnects
        does — from the usage accrued up to that point, so abandoned runs are still
        accounted for.
        """
        ms = self._extract_model_settings(kwargs)
        usage = RunUsage()
        logged = False

        try:
            async with self._agent.run_stream_events(  # ty: ignore[no-matching-overload]
                user_prompt=self.process_prompt(user_prompt, deps),
                deps=deps,
                model_settings=ms,
                usage=usage,
                **kwargs,
            ) as stream:
                async for event in stream:
                    if isinstance(event, AgentRunResultEvent):
                        self._log_result(event.result)
                        logged = True
                    yield event
        finally:
            if not logged:
                log_llm_call(_AbortedRun(usage=usage))
```

Add the imports: `from dataclasses import dataclass` and `from pydantic_ai.usage import RunUsage`.

- [ ] **Step 4: Run the new test and the whole unit suite**

```bash
cd ~/code/backend-common
uv run python -m pytest tests/unit/test_base_agent.py -k aborted -v
uv run python -m pytest tests/unit --doctest-modules
```

Expected: the new test PASSES and every pre-existing test still passes. If a pre-existing test now sees an extra `log_llm_call`, that is a real behaviour change — read it and decide whether the test's expectation was encoding the bug. Explain either way; do not silently retrofit assertions.

- [ ] **Step 5: Confirm the normal path still logs exactly once**

The risk in this change is double-logging: once from `AgentRunResultEvent`, once from the `finally`. The `logged` flag prevents it. Prove it:

```bash
uv run python -m pytest tests/unit/test_base_agent.py -v -k "stream"
```

Expected: PASS. If any existing test asserts `log_llm_call` call counts, it must still see exactly 1 for a fully-consumed stream.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
cd ~/code/backend-common
uv run ruff check ./src ./tests && uv run ruff format --check ./src ./tests
uv run ty check ./src/dcc_backend_common
git checkout -b fix/log-usage-on-aborted-stream
git add src/dcc_backend_common/llm_agent/base_agent.py tests/unit/test_base_agent.py
git commit -m "fix(llm_agent): log usage when a stream consumer exits early"
```

---

## Task 2: Release backend-common 0.1.23

**Files:**
- Modify: `~/code/backend-common/pyproject.toml` (line 3, `version`)

- [ ] **Step 1: Bump the version**

```bash
cd ~/code/backend-common
uv version 0.1.23
uv lock
```

- [ ] **Step 2: Verify the full check target passes**

```bash
make check && make test
```

Expected: both green.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: release 0.1.23"
```

- [ ] **Step 4: Hand off to the human for merge + publish**

**STOP HERE — this step is not automatable.** Publishing is a manual `workflow_dispatch` on `.github/workflows/publish.yml`, which reads `uv version --short` and pushes a `v$VERSION` tag. Report to the user that `0.1.23` is ready and that they need to merge the branch and trigger the **Publish to PyPI** workflow. Do not proceed to Task 3 until they confirm it is published.

---

## Task 3: `AppConfig` inherits `LlmConfig` (translator)

**Decision:** inherit rather than compose. `AppConfig` already duplicates `llm_url` / `llm_api_key` / `llm_model` by name; inheriting removes the duplication and picks up `llm_timeout` and `llm_max_retries`, which the hand-rolled agent has no way to set today.

**Files:**
- Modify: `src/bs_translator_backend/utils/app_config.py`
- Modify: `pyproject.toml` (dependency line)
- Test: `tests/unit/test_app_config.py` (create if absent)

**Interfaces:**
- Consumes: `dcc_backend_common.config.app_config.LlmConfig` — fields `llm_model: str`, `llm_url: str`, `llm_api_key: str`, `llm_timeout: int = 300` (`ge=0`), `llm_max_retries: int = 2` (`ge=0`).
- Produces: `AppConfig` is now a valid `LlmConfig`, so it can be handed straight to a `BaseAgent` constructor. Tasks 4–5 depend on this.

- [ ] **Step 1: Move the dependency to the right extras**

In `pyproject.toml`, change the dependency line:

```toml
    "dcc-backend-common[fastapi,pydantic-ai]>=0.1.23",
```

Two extras, both currently missing:
- `pydantic-ai` is **mandatory** — `dcc_backend_common.llm_agent` imports pydantic-ai and tenacity, which only that extra declares.
- `fastapi` is a **latent bug fix**. The repo already imports `dcc_backend_common.fastapi_health_probes` and `fastapi_logging_middleware` without declaring the extra; it only works because the translator happens to declare `fastapi` itself. The reference consumer uses `dcc-backend-common[fastapi]`. Fixing it here costs nothing.

**Keep `"pydantic-ai>=2.7.0"` as a direct dependency.** `agents/translation_agent.py` imports `Agent` and `Model` directly, so it must be declared directly — the extra covers common's needs, not yours. The reference consumer does the same. (This is the opposite of `openai`, which must *not* be declared because after Task 4 nothing imports it.)

Then:

```bash
uv lock && uv sync
uv run python -c "from dcc_backend_common.llm_agent import BaseAgent; print('BaseAgent import OK')"
```

Expected: `BaseAgent import OK`.

- [ ] **Step 2: Write the failing test**

Create or extend `tests/unit/test_app_config.py`:

```python
from dcc_backend_common.config.app_config import LlmConfig

from bs_translator_backend.utils.app_config import AppConfig


class TestAppConfigIsLlmConfig:
    def test_app_config_is_an_llm_config(self) -> None:
        """AppConfig must be usable anywhere BaseAgent expects an LlmConfig."""
        assert issubclass(AppConfig, LlmConfig)

    def test_llm_timeout_and_retries_have_defaults(self, app_config: AppConfig) -> None:
        assert app_config.llm_timeout == 300
        assert app_config.llm_max_retries == 2
```

The `app_config` fixture lives in `tests/integration/conftest.py`. If it is not visible from `tests/unit/`, construct an `AppConfig` inline with the same field values rather than moving the fixture.

- [ ] **Step 3: Run it and confirm it fails**

```bash
uv run python -m pytest tests/unit/test_app_config.py -v
```

Expected: FAIL on `issubclass` — `AppConfig` currently extends `AbstractAppConfig`.

- [ ] **Step 4: Change the base class and delete the duplicated fields**

In `app_config.py`, change the import and the class declaration, and remove the three now-inherited field declarations:

```python
from dcc_backend_common.config import get_env_or_throw, log_secret
from dcc_backend_common.config.app_config import LlmConfig


class AppConfig(LlmConfig):
    # llm_url, llm_api_key, llm_model are inherited from LlmConfig.
    # llm_timeout and llm_max_retries come with it and are wired in from_env below.
    reasoning: bool = Field(
        default=False,
        description="Enable LLM reasoning; when false, disable with /no_think hint",
    )
    ...
```

Keep every other field, both `field_validator`s, and `__str__` exactly as they are. `LlmConfig` extends `AbstractAppConfig` itself, so nothing else in the hierarchy changes.

- [ ] **Step 5: Set the LLM tuning values explicitly in `from_env`**

**Do not add new env vars.** The reference consumer sets these as literals rather than exposing them to configuration, and matching that keeps the two services consistent. Add to the `cls(...)` call in `from_env`:

```python
            llm_timeout=60 * 5,
            llm_max_retries=2,
```

These equal `LlmConfig`'s own defaults, so passing them changes nothing today — they are written out so the intended values are visible at the call site, exactly as `text-mate-backend/src/text_mate_backend/utils/configuration.py` does. If tuning is ever needed, promoting them to env vars is a one-line change then.

- [ ] **Step 6: Run the tests**

```bash
uv run python -m pytest tests/unit/test_app_config.py -v
uv run python -m pytest --doctest-modules
```

Expected: new tests PASS, full suite still green.

- [ ] **Step 7: Lint, typecheck, commit**

```bash
uv run ruff check ./src ./tests && uv run ruff format --check ./src ./tests
uv run ty check ./src/bs_translator_backend
git add pyproject.toml uv.lock src/bs_translator_backend/utils/app_config.py tests/unit/test_app_config.py
git commit -m "refactor(config): inherit LlmConfig for timeout and retry settings"
```

---

## Task 4: Rewrite the agents as `BaseAgent` subclasses

**Decision:** accept a thin, contained pydantic-ai surface. `BaseAgent.create_agent` is abstract, so subclasses must build a `pydantic_ai.Agent`. That import stays in this one file; nothing else in the repo imports pydantic-ai.

**Files:**
- Modify: `src/bs_translator_backend/agents/translation_agent.py` (full rewrite)
- Test: `tests/unit/test_translation_agent.py`

**Interfaces:**
- Consumes: `AppConfig` as an `LlmConfig` (Task 3).
- Produces — Task 5 depends on these exact names:
  - `class TranslationAgent(BaseAgent[None, str])`, constructed as `TranslationAgent(app_config)`
  - `class ShortTextTranslationAgent(BaseAgent[None, str])`, constructed as `ShortTextTranslationAgent(app_config)`
  - both expose the inherited `async run_stream_text(user_prompt, *, delta=True) -> AsyncGenerator[str, None]` and `async close() -> None`

**Three deletions this task makes, each with its reason:**

1. **`transform_to_swissgerman_style` → delete.** `BaseAgent`'s default postprocessor `replace_eszett` does exactly `str.replace("ß", "ss")` and is applied automatically to both streamed chunks and final output. The local copy is redundant. Its unit tests in `tests/unit/test_translation_agent.py` should be deleted too — the behaviour is covered by common's `tests/unit/test_postprocessing.py`.
2. **`TextOutput(...)` → delete.** It existed only to attach the transform above.
3. **`keep_recent_message` + `ProcessHistory` → verify, then almost certainly delete.** `keep_recent_message` returns `messages[-1:] if len(messages) > 1 else messages`. The service never passes `message_history`, and pydantic-ai carries instructions on the `ModelRequest` rather than as a separate message — so the list it sees should always have length 1, making the capability a no-op. **Verify before deleting:** add a temporary `print(len(messages))` inside `keep_recent_message`, run one deterministic e2e test that exercises a real agent run, and confirm it only ever prints `1`. If it ever prints more, keep the capability and pass it through `create_agent` via `capabilities=[ProcessHistory(keep_recent_message)]`. Report what you observed.

- [ ] **Step 1: Write the failing test**

Replace the contents of `tests/unit/test_translation_agent.py`:

```python
import pytest
from dcc_backend_common.llm_agent import BaseAgent

from bs_translator_backend.agents.translation_agent import (
    ShortTextTranslationAgent,
    TranslationAgent,
)


class TestAgentConstruction:
    def test_translation_agent_is_a_base_agent(self, app_config) -> None:
        assert issubclass(TranslationAgent, BaseAgent)

    def test_short_text_agent_is_a_base_agent(self, app_config) -> None:
        assert issubclass(ShortTextTranslationAgent, BaseAgent)

    def test_agents_construct_from_app_config(self, app_config) -> None:
        assert TranslationAgent(app_config) is not None
        assert ShortTextTranslationAgent(app_config) is not None

    def test_agents_carry_distinct_instructions(self) -> None:
        """The whole point of the short agent is a different prompt."""
        from bs_translator_backend.agents.translation_agent import (
            SHORT_TEXT_TRANSLATION_INSTRUCTION,
            TRANSLATION_INSTRUCTION,
        )

        assert TRANSLATION_INSTRUCTION != SHORT_TEXT_TRANSLATION_INSTRUCTION
        assert "lexical" in SHORT_TEXT_TRANSLATION_INSTRUCTION.lower()

    @pytest.mark.asyncio
    async def test_agents_expose_close(self, app_config) -> None:
        agent = TranslationAgent(app_config)
        await agent.close()


class TestNoDirectSdkUsage:
    def test_module_does_not_import_openai(self) -> None:
        """The openai SDK is not a declared dependency; it must not be imported."""
        from pathlib import Path

        import bs_translator_backend.agents.translation_agent as mod

        source = Path(mod.__file__).read_text()
        assert "import openai" not in source
        assert "from openai" not in source
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run python -m pytest tests/unit/test_translation_agent.py -v
```

Expected: FAIL with `ImportError: cannot import name 'TranslationAgent'`.

- [ ] **Step 3: Rewrite `translation_agent.py`**

**Follow the house idiom from `text-mate-backend`** (see the Reference Implementation section): module-level instruction constants, `@override` on `create_agent`, the `@agent.instructions` decorator *inside* `create_agent`, and `name=` / `description=` on the `Agent`.

```python
"""Translation agents built on the shared dcc-backend-common BaseAgent.

This is the only module in the service that imports pydantic-ai: BaseAgent.create_agent
is abstract and must return a pydantic_ai.Agent. Everything else — services, routers,
models — stays free of it. The openai SDK is never imported here; BaseAgent owns client
construction, retries, and timeouts.
"""

from typing import override

from dcc_backend_common.llm_agent import BaseAgent, Preprocessor
from dcc_backend_common.llm_agent.postprocessing import replace_eszett
from pydantic_ai import Agent
from pydantic_ai.models import Model

from bs_translator_backend.utils.app_config import AppConfig

TRANSLATION_INSTRUCTION = """..."""  # see Step 4

SHORT_TEXT_TRANSLATION_INSTRUCTION = """..."""  # see Step 4


class TranslationAgent(BaseAgent[None, str]):
    """Full-text translation for the Cantonal Administration of Basel-Stadt."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config, output_type=str, enable_thinking=config.reasoning)

    @override
    def create_agent(self, model: Model) -> Agent[None, str]:
        agent = Agent[None, str](
            model=model,
            output_type=str,
            name="Translation Agent",
            description="Translates full texts into the target language, preserving markdown",
        )

        @agent.instructions
        def get_instruction() -> str:
            return TRANSLATION_INSTRUCTION

        return agent


class ShortTextTranslationAgent(BaseAgent[None, str]):
    """Dictionary-style translation for inputs of 1-3 words."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config, output_type=str, enable_thinking=config.reasoning)

    @override
    def create_agent(self, model: Model) -> Agent[None, str]:
        agent = Agent[None, str](
            model=model,
            output_type=str,
            name="Short Text Translation Agent",
            description="Translates 1-3 word inputs as a lexical lookup rather than copying them verbatim",
        )

        @agent.instructions
        def get_instruction() -> str:
            return SHORT_TEXT_TRANSLATION_INSTRUCTION

        return agent
```

Two deliberate deviations from the reference, both justified:
- The reference declares `class WordSynonymAgent(BaseAgent)` unparameterised. Parameterise as `BaseAgent[None, str]` here — this repo runs `ty`, and the precise types cost nothing.
- The reference passes `metadata=lambda ctx: build_agent_metadata(...)` to feed Logfire spans. **The translator has no Logfire** (`grep -rn logfire src/ pyproject.toml` returns nothing), so adding a metadata helper with no consumer would be speculative. Skip it. If Logfire is adopted later, mirror the reference then.

- [ ] **Step 4: Move both instruction strings across verbatim**

Copy the existing long prompt (currently returned by the `get_instructions` closure in `create_translation_agent`) into `TRANSLATION_INSTRUCTION`, and the short prompt from `create_short_text_translation_agent` into `SHORT_TEXT_TRANSLATION_INSTRUCTION`. **Character-for-character, including the `/no_think` rule.** Removing that rule is Task 8's job and must not be smuggled in here — this task changes plumbing only, so any behaviour change later is attributable to exactly one commit.

- [ ] **Step 5: Suppress the `trim_text` postprocessor**

`BaseAgent._get_postprocessors` appends `trim_text` (which does `text.lstrip()`) whenever `output_type is str`. Both agents' prompts demand *"Maintain line breaks and whitespace exactly (including non-breaking spaces)"*, and document translation is chunked, so stripping leading whitespace from a chunk would corrupt markdown indentation. Override it on **both** classes:

```python
    @override
    def _get_postprocessors(self) -> list[Preprocessor]:
        """Keep replace_eszett; drop trim_text — the prompts require exact whitespace."""
        return [replace_eszett]
```

The imports are already in the module header shown in Step 3. Note `_get_stream_postprocessors` already excludes `trim_text`, so this only affects the non-streaming path — do it anyway, so the two paths cannot drift.

This override is the one place the translator genuinely needs to differ from the reference: text-mate's agents emit prose where `lstrip()` is harmless, whereas chunked markdown translation depends on leading whitespace surviving.

- [ ] **Step 6: Do the `ProcessHistory` verification described above**

Report the observed message-list lengths, and whether you deleted or kept the capability.

- [ ] **Step 7: Run the tests**

```bash
uv run python -m pytest tests/unit/test_translation_agent.py -v
uv run python -m pytest --doctest-modules
```

The service still imports `create_translation_agent` / `create_short_text_translation_agent` at this point, so **the suite will fail** until Task 5. That is expected and is the one sanctioned red state in this plan. If you prefer a green boundary, do Tasks 4 and 5 as a single commit — say which you chose.

- [ ] **Step 8: Commit**

```bash
git add src/bs_translator_backend/agents/translation_agent.py tests/unit/test_translation_agent.py
git commit -m "refactor(agents): rebuild translation agents on BaseAgent"
```

---

## Task 5: Switch `TranslationService` onto the new agents

**Files:**
- Modify: `src/bs_translator_backend/services/translation_service.py`
- Test: `tests/unit/test_translation_service.py`, `tests/integration/translation_service_test.py`

**Interfaces:**
- Consumes: `TranslationAgent(app_config)`, `ShortTextTranslationAgent(app_config)` from Task 4.
- Produces: `TranslationService.aclose()` — Task 6 wires it into the app lifespan.

**The streaming call changes shape.** Today:

```python
async with translation_agent.run_stream(user_message) as stream:
    try:
        async for text_part in stream.stream_text(delta=True):
            chunk_translation += text_part
            yield text_part
    finally:
        log_llm_call(stream)
```

After migration, `BaseAgent.run_stream_text` is a plain async generator that applies `replace_eszett` per chunk and — after Task 1 — logs usage itself, including on early exit:

```python
async for text_part in translation_agent.run_stream_text(user_prompt=user_message, delta=True):
    chunk_translation += text_part
    yield text_part
```

Use the keyword form `user_prompt=`, matching the reference consumer (`text-mate-backend/src/text_mate_backend/services/fix_service.py:34`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_translation_service.py`:

```python
class TestServiceUsesBaseAgents:
    def test_service_builds_base_agent_instances(self, app_config) -> None:
        from dcc_backend_common.llm_agent import BaseAgent

        service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
        assert isinstance(service.translation_agent, BaseAgent)
        assert isinstance(service.short_text_translation_agent, BaseAgent)

    @pytest.mark.asyncio
    async def test_aclose_closes_both_agents(self, app_config) -> None:
        service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
        service.translation_agent.close = AsyncMock()
        service.short_text_translation_agent.close = AsyncMock()

        await service.aclose()

        service.translation_agent.close.assert_awaited_once()
        service.short_text_translation_agent.close.assert_awaited_once()

    def test_module_does_not_import_pydantic_ai(self) -> None:
        from pathlib import Path

        import bs_translator_backend.services.translation_service as mod

        assert "pydantic_ai" not in Path(mod.__file__).read_text()
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run python -m pytest tests/unit/test_translation_service.py -k "BaseAgent or aclose or pydantic_ai" -v
```

Expected: FAIL — `aclose` does not exist and the agents are bare `Agent` objects.

- [ ] **Step 3: Update the imports and constructor**

```python
from bs_translator_backend.agents.translation_agent import (
    ShortTextTranslationAgent,
    TranslationAgent,
)
```

Delete `from dcc_backend_common.usage_tracking import log_llm_call` — the service no longer logs; `BaseAgent` does. In `__init__`:

```python
        self.translation_agent = TranslationAgent(app_config)
        self.short_text_translation_agent = ShortTextTranslationAgent(app_config)
```

- [ ] **Step 4: Replace the streaming block**

Inside the chunk loop in `translate_text`, replace the whole `async with ... finally: log_llm_call(stream)` block with:

```python
            chunk_translation = ""
            async for text_part in translation_agent.run_stream_text(
                user_prompt=user_message, delta=True
            ):
                chunk_translation += text_part
                yield text_part
```

Leave the surrounding logic untouched: the short-vs-long agent selection, `SHORT_TEXT_WORD_THRESHOLD`, `SHORT_TEXT_SOURCE_LANGUAGE_CONFIDENCE_THRESHOLD`, the `source_language_trustworthy` gate and its `functools.partial`, the `accumulated_context` tail, and both early returns all stay exactly as they are.

- [ ] **Step 5: Add `aclose`**

```python
    async def aclose(self) -> None:
        """Close both agents' HTTP clients. Called from the FastAPI lifespan."""
        await self.translation_agent.close()
        await self.short_text_translation_agent.close()
```

- [ ] **Step 6: Update the agent mocks in the existing tests**

`tests/integration/translation_service_test.py` and `tests/unit/test_translation_service.py` currently mock `run_stream` with a `MagicMock` exposing `stream_text` plus `__aenter__`/`__aexit__`. That contract is gone. Replace it with a plain async generator:

```python
def fake_agent(captured: dict, name: str):
    agent = MagicMock()

    async def run_stream_text(user_prompt, delta=True):
        captured[name] = user_prompt
        yield "[translated]"

    agent.run_stream_text = run_stream_text
    return agent
```

Every assertion about *which* agent ran and *what prompt* it received must be preserved — those cover the short-text feature and the source-language gate. Only the mocking mechanism changes.

- [ ] **Step 7: Run the full suite**

```bash
uv run python -m pytest --doctest-modules
```

Expected: fully green again. Confirm the short-vs-long routing tests and the three source-language-assertion tests still pass — they are the regression net for the feature this migration must not break.

- [ ] **Step 8: Prove the pydantic-ai boundary holds**

```bash
grep -rn "pydantic_ai\|from openai\|import openai" src/bs_translator_backend/ | grep -v "agents/translation_agent.py"
```

Expected: **no output.** Any hit is a leak to fix before committing.

- [ ] **Step 9: Lint, typecheck, commit**

```bash
uv run ruff check ./src ./tests && uv run ruff format --check ./src ./tests
uv run ty check ./src/bs_translator_backend
git add src/bs_translator_backend/services/translation_service.py tests/
git commit -m "refactor(translation): consume BaseAgent-based translation agents"
```

---

## Task 6: Close the agents on shutdown

**Decision:** two agents, cleanup wired into the FastAPI lifespan. This fixes a connection-pool leak that exists today — nothing currently closes any client.

**Deliberate divergence from the reference.** `text-mate-backend` has a lifespan (`app.py:69`) but never calls `close()` or `cleanup()` on any agent either — the leak is common to both services, not unique to the translator. This task fixes it here rather than propagating it. `BaseAgent.close()` exists and is unused across both consumers, which suggests it is worth raising for `text-mate-backend` as well; note that in your report but do not change that repo.

**Files:**
- Modify: `src/bs_translator_backend/app.py`
- Test: `tests/unit/test_app_lifespan.py` (create)

**Verified facts about `app.py` — do not re-derive:**
- `create_app()` at line 121 is the factory.
- `_build_fastapi_app()` at line 17 constructs the `FastAPI(...)` object. **Its docstring claims "and lifespan", but no lifespan is actually registered** — you are adding the first one.
- `_configure_container` at line 81 builds the `Container`, wires it, and stores it at `app.state.container` (line 90).
- Ordering matters: `_build_fastapi_app()` runs *before* `_configure_container`, so the lifespan cannot close over the container directly. Reach it through `app.state.container` at shutdown time instead.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_app_lifespan.py`:

```python
from unittest.mock import AsyncMock

import pytest

from bs_translator_backend.app import create_app


@pytest.mark.asyncio
async def test_lifespan_closes_translation_service(monkeypatch) -> None:
    """Shutdown must release the agents' HTTP connection pools."""
    app = create_app()
    service = app.state.container.translation_service()
    service.aclose = AsyncMock()

    async with app.router.lifespan_context(app):
        pass

    service.aclose.assert_awaited_once()
```

`create_app()` reads config from the environment via `AppConfig.from_env()`. If that fails in the test environment, monkeypatch the required env vars (`LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`, `CLIENT_URL`, `DOCLING_URL`, `DOCLING_API_KEY`, `HMAC_SECRET`, `WHISPER_URL`) using the same values as the `app_config` fixture in `tests/integration/conftest.py`.

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run python -m pytest tests/unit/test_app_lifespan.py -v
```

Expected: FAIL — `aclose` is never awaited, because no lifespan is registered at all.

- [ ] **Step 3: Implement**

Add to `app.py`, with `from contextlib import asynccontextmanager` and `from collections.abc import AsyncIterator`:

```python
@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release the LLM clients' connection pools on shutdown."""
    yield
    container: Container | None = getattr(app.state, "container", None)
    if container is not None:
        await container.translation_service().aclose()
```

and register it in `_build_fastapi_app`:

```python
    app = FastAPI(
        title="BS Translator API",
        ...
        lifespan=_lifespan,
    )
```

The `getattr` guard matters: `_build_fastapi_app()` runs before `_configure_container`, so a partially-built app (or a test that never configures the container) must not raise on shutdown.

- [ ] **Step 4: Run tests, lint, typecheck, commit**

```bash
uv run python -m pytest --doctest-modules
uv run ruff check ./src ./tests && uv run ty check ./src/bs_translator_backend
git add src/bs_translator_backend/app.py tests/unit/test_app_lifespan.py
git commit -m "fix(app): close LLM clients on shutdown"
```

---

## Task 7: Verify the migration against a live model

**This task is a gate, not a code change.** Tasks 4–6 changed how the model is called — retries, timeout, postprocessing, and streaming all moved to `BaseAgent`. Nothing so far has proven a real model still behaves.

**Prerequisite:** the live e2e tier from the e2e task (`tests/e2e/test_live_llm.py`, marked `@pytest.mark.live`, gated on `E2E_LIVE_LLM=1`).

- [ ] **Step 1: Run the live suite against the real endpoint**

```bash
E2E_LIVE_LLM=1 uv run python -m pytest tests/e2e -m live -v
```

- [ ] **Step 2: Confirm each of these explicitly**

- `Hirsch`, German → French, explicit source: returns a real French translation, not `Hirsch`.
- `Der Hirsch`, German → French: translates.
- A longer markdown passage: translates, structure preserved, whitespace intact — this is the `trim_text` suppression from Task 4 Step 5 proving itself.
- Output contains no `ß` — `replace_eszett` is doing what `transform_to_swissgerman_style` used to.
- Output contains no visible reasoning/thinking text — `enable_thinking` plumbing is correct.

- [ ] **Step 3: Confirm usage logging still fires**

Run one live translation and confirm an `llm_call` line appears in the logs with non-zero `input_tokens`/`output_tokens`. Then abort one mid-stream (disconnect the client) and confirm an `llm_call` line **still** appears — this is Task 1 proving itself end to end, the whole reason common was released first.

- [ ] **Step 4: Report**

Write up what passed and what did not. **If anything fails, stop and report rather than patching prompts to make a live assertion go green** — a prompt edit at this point would hide a migration defect.

---

## Task 8: Remove the `/no_think` marker

**Decision:** drop the marker and rely on `enable_thinking`. Sequenced last, deliberately, because it is the only behavioural change in this plan and it needs a live model to verify.

**Read this before starting — the current state is not what it looks like:**

`_create_translation_model` hardcodes `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` **regardless of `app_config.reasoning`**, while `/no_think` is appended only when `reasoning` is `False`. So today thinking is suppressed by the chat template in *both* modes, and the `reasoning` flag effectively only toggles a redundant marker. After Task 4, `enable_thinking=app_config.reasoning` makes the flag real — **setting `LLM_REASONING=true` will genuinely enable thinking for the first time.** Confirm with the user that this is intended before shipping, and call it out in the commit message.

**Files:**
- Modify: `src/bs_translator_backend/agents/translation_agent.py` (both instruction constants)
- Modify: `src/bs_translator_backend/services/translation_service.py`
- Test: `tests/unit/test_translation_service.py`

**No live pre-check is needed.** The user has confirmed that `chat_template_kwargs.enable_thinking` alone suppresses thinking on this model — treat that as settled and do not spend a live run re-verifying it.

- [ ] **Step 1: Write the failing test**

```python
def test_user_message_has_no_no_think_marker(self, app_config) -> None:
    service = TranslationService(app_config, TextChunkService(), lambda: MagicMock())
    config = TranslationConfig(
        target_language=Language.FR, source_language=Language.DE,
        domain="", tone="", glossary="", context="",
    )
    for build in (service._create_user_message, service._create_short_text_user_message):
        assert "/no_think" not in build(text="Hirsch", translation_config=config)
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run python -m pytest tests/unit/test_translation_service.py -k no_think -v
```

Expected: FAIL — both builders still append the marker.

- [ ] **Step 3: Remove the marker**

In `translation_service.py`, delete both `if not reasoning: prompt += "/no_think"` blocks and the now-unused `reasoning` parameter from `_create_user_message` and `_create_short_text_user_message`, plus the `reasoning=self.app_config.reasoning` argument at the call site. `app_config.reasoning` is still used — it now feeds `enable_thinking` in Task 4's constructors.

- [ ] **Step 4: Remove the prompt rule from both instruction strings**

Delete this block from `TRANSLATION_INSTRUCTION`:

```
Special instruction:
- If source_text ends with the postfix "/no_think", ignore this marker for the purposes of translation. Remove "/no_think" from the end before translating, and do not include it in the translated_text or in your output.
```

and the equivalent `/no_think` rule from `SHORT_TEXT_TRANSLATION_INSTRUCTION`.

- [ ] **Step 5: Run everything, including live**

```bash
uv run python -m pytest --doctest-modules
E2E_LIVE_LLM=1 uv run python -m pytest tests/e2e -m live -v
```

Expected: all green. The live run here is a regression check on translation quality after the prompt edit, not a re-verification of thinking suppression.

- [ ] **Step 6: Commit**

```bash
git add src/bs_translator_backend/agents/translation_agent.py src/bs_translator_backend/services/translation_service.py tests/
git commit -m "refactor(prompts): drop /no_think marker in favour of enable_thinking

LLM_REASONING now genuinely controls thinking; previously it was suppressed
unconditionally via chat_template_kwargs and the flag only toggled the marker."
```

---

## Final verification

- [ ] Full suite green: `uv run python -m pytest --doctest-modules`
- [ ] `make check` passes
- [ ] `uv run ty check ./src/bs_translator_backend` → "All checks passed!"
- [ ] `uv run ruff check ./src ./tests` → only the pre-existing `S106` at `tests/integration/conftest.py:16`
- [ ] Boundary holds: `grep -rn "pydantic_ai\|openai" src/bs_translator_backend/ | grep -v "agents/translation_agent.py"` → no output
- [ ] `openai` still absent from `pyproject.toml`
- [ ] Live e2e green against the real endpoint
