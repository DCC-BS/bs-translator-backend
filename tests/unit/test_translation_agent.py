import pytest

from bs_translator_backend.agents.translation_agent import (
    keep_recent_message,
    transform_to_swissgerman_style,
)


class TestTransformToSwissgermanStyle:
    def test_replaces_eszett_with_double_s(self):
        assert transform_to_swissgerman_style("große") == "grosse"

    def test_replaces_multiple_eszett(self):
        assert transform_to_swissgerman_style("große Straße") == "grosse Strasse"

    def test_preserves_text_without_eszett(self):
        assert transform_to_swissgerman_style("hello world") == "hello world"

    def test_handles_empty_string(self):
        assert transform_to_swissgerman_style("") == ""

    def test_handles_only_eszett(self):
        assert transform_to_swissgerman_style("ß") == "ss"


class TestKeepRecentMessage:
    @pytest.mark.asyncio
    async def test_keeps_single_message(self):
        messages = ["msg1"]
        result = await keep_recent_message(messages)
        assert result == ["msg1"]

    @pytest.mark.asyncio
    async def test_keeps_only_last_message_when_multiple(self):
        messages = ["msg1", "msg2", "msg3"]
        result = await keep_recent_message(messages)
        assert result == ["msg3"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_input(self):
        messages = []
        result = await keep_recent_message(messages)
        assert result == []
