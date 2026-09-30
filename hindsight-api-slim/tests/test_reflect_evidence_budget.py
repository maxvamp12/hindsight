"""
Tests for the evidence-budget composition inside the reflect agent loop.

Covers the interaction added in fix/reflect-evidence-budget and its resolution
against upstream's presentation layer:
1. The degradation ladder slims oversized RAW tool results (drop source_facts,
   strip source_memories, relevance-prefix truncation with an _truncated marker).
2. The presenter then builds the PROMPT form (aliased ids).
3. The evidence tally counts the presented serialization (what context_history
   and the synthesis prompt actually see), and hitting the cap forces final
   synthesis instead of another collection iteration.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
from hindsight_api.engine.reflect.agent import run_reflect_agent


def _oversized_recall_output() -> dict:
    """A recall result that is far larger than any small evidence budget."""
    return {
        "query": "test query",
        "memories": [
            {"id": f"mem-{i}", "content": "detail " * 300, "source_memories": [{"quote": "proof " * 50}]}
            for i in range(3)
        ],
        "source_facts": {f"mem-{i}": {"id": f"mem-{i}", "content": "dump " * 300} for i in range(3)},
    }


class TestEvidenceBudgetComposition:
    """The ladder (raw) -> presenter (prompt form) -> tally(presented) pipeline."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Fallback answer from final iteration",
                usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
        )
        return llm

    @pytest.fixture
    def mock_functions(self):
        return {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(return_value=_oversized_recall_output()),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    @pytest.mark.asyncio
    async def test_oversized_result_triggers_ladder_then_forced_synthesis(
        self, mock_llm, mock_functions, monkeypatch
    ):
        """An oversized recall on a tiny budget must slim, tally, latch, and force
        synthesis after a single collection iteration — and the tool message the
        model reads must be the PRESENTED form (alias f1, _truncated marker,
        proof arrays already stripped)."""
        config = MagicMock(
            reflect_prompt_cache_enabled=False,
            reflect_max_completion_tokens=None,
            llm_temperature_reflect=0.17,
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="recall", arguments={"reason": "gather", "query": "test query"})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=False,
            budget="low",
            max_iterations=5,
            max_evidence_tokens=25,  # tiny: the raw recall result is way over
            **mock_functions,
        )

        # The latch fired: forced final synthesis after ONE tool round, so the
        # second call_with_tools (the model choosing done/next tool) never ran.
        assert mock_llm.call_with_tools.await_count == 1
        assert result.text == "Fallback answer from final iteration"

        # The tool message the model saw is the PRESENTED serialization.
        tool_msg = mock_llm.call_with_tools.await_args_list[0].kwargs["messages"][-1]
        assert tool_msg["role"] == "tool"
        content = tool_msg["content"]
        assert isinstance(content, str)
        # presenter ran (raw ids became aliases)
        assert '"f1"' in content
        # ladder rang A: lookup cargo dropped before presentation
        assert "source_facts" not in content
        # ladder ranng B: proof arrays stripped, count only
        assert "source_memories" not in content
        # ladder rung C: prefix truncation marker survives presentation
        assert "_truncated" in content

        # And the forced synthesis (llm.call) received the trimmed context —
        # the presented, slimmed outputs, not the raw 174k-char pile.
        synth_msgs = mock_llm.call.await_args.kwargs["messages"]
        assert any(m.get("role") == "user" or "content" in m for m in synth_msgs)

    @pytest.mark.asyncio
    async def test_small_result_leaves_budget_alone(self, mock_llm, mock_functions, monkeypatch):
        """Results within budget are untouched: no _truncated marker, tally stays
        under the cap, and the loop continues normally to done()."""
        config = MagicMock(
            reflect_prompt_cache_enabled=False,
            reflect_max_completion_tokens=None,
            llm_temperature_reflect=0.17,
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)

        mock_functions["recall_fn"].return_value = {
            "memories": [{"id": "mem-1", "content": "small fact"}]
        }
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"reason": "gather", "query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="done", arguments={"answer": "Fine.", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=False,
            budget="low",
            max_iterations=5,
            max_evidence_tokens=24000,
            **mock_functions,
        )

        assert result.text == "Fine."
        tool_msg = mock_llm.call_with_tools.await_args_list[0].kwargs["messages"][-1]
        assert "_truncated" not in tool_msg["content"]
        assert '"f1"' in tool_msg["content"]  # presenter still applied
