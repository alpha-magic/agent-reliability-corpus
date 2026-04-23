"""Initial framework shortlist — scraped weekly.

Additions/removals happen via the curator agent (Agent B) proposing changes
and a human confirming. Order here is alphabetical by slug for diff-friendly
history.
"""

from __future__ import annotations

from agentfail.schema import Framework

FRAMEWORKS: tuple[Framework, ...] = (
    Framework(
        slug="agno",
        repo="agno-agi/agno",
        display_name="Agno",
        homepage="https://docs.agno.com",
    ),
    Framework(
        slug="autogen",
        repo="microsoft/autogen",
        display_name="AutoGen",
        homepage="https://microsoft.github.io/autogen/",
    ),
    Framework(
        slug="autogpt",
        repo="Significant-Gravitas/AutoGPT",
        display_name="AutoGPT",
        homepage="https://agpt.co/",
    ),
    Framework(
        slug="crewai",
        repo="crewAIInc/crewAI",
        display_name="CrewAI",
        homepage="https://www.crewai.com/",
    ),
    Framework(
        slug="langchain",
        repo="langchain-ai/langchain",
        display_name="LangChain",
        homepage="https://www.langchain.com/",
    ),
    Framework(
        slug="langgraph",
        repo="langchain-ai/langgraph",
        display_name="LangGraph",
        homepage="https://langchain-ai.github.io/langgraph/",
    ),
    Framework(
        slug="letta",
        repo="letta-ai/letta",
        display_name="Letta",
        homepage="https://www.letta.com/",
    ),
    Framework(
        slug="llamaindex",
        repo="run-llama/llama_index",
        display_name="LlamaIndex",
        homepage="https://www.llamaindex.ai/",
    ),
    Framework(
        slug="mastra",
        repo="mastra-ai/mastra",
        display_name="Mastra",
        homepage="https://mastra.ai/",
    ),
    Framework(
        slug="semantic_kernel",
        repo="microsoft/semantic-kernel",
        display_name="Semantic Kernel",
        homepage="https://learn.microsoft.com/semantic-kernel/",
    ),
    Framework(
        slug="smolagents",
        repo="huggingface/smolagents",
        display_name="smolagents",
        homepage="https://huggingface.co/docs/smolagents",
    ),
    Framework(
        slug="swarm",
        repo="openai/swarm",
        display_name="OpenAI Swarm",
        homepage="https://github.com/openai/swarm",
    ),
)


FRAMEWORK_BY_SLUG: dict[str, Framework] = {f.slug: f for f in FRAMEWORKS}
