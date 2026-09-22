import asyncio
import sys
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uuid
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    on_topic: bool
    context: str


class QuestionClassification(BaseModel):
    is_chemical_related: bool = Field(
        description="True if related to chemical/process engineering or castor oil."
    )


classifier_llm = llm.with_structured_output(
    QuestionClassification, method="function_calling"
)


def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    result = classifier_llm.invoke(
        f"""Determine whether this question is related to a chemical process
engineering assistant covering: castor oil extraction, physicochemical
properties, density, viscosity, moisture, flash/fire/pour point,
saponification value, esterification, transesterification, biodiesel
characterization, mass balance, oil yield, unit conversion.

Question: {question}
"""
    )
    return {"on_topic": result.is_chemical_related}


def topic_router(state: AgentState):
    return "retriever" if state["on_topic"] else "off_topic"


def off_topic_response(state: AgentState):
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry! I can only help with chemical engineering, "
                    "process engineering, castor oil production, and castor oil extraction."
                )
            )
        ]
    }


_mcp_session: ClientSession | None = None


def _expand_query(question: str) -> str:
    """Rewrite the question into a retrieval-friendly query."""
    try:
        resp = llm.invoke(
            f"""Rewrite the user's question into a search query that will match
the wording of a chemical engineering lab report about castor oil.
Include likely keywords such as 'determination', 'formula', 'method',
'procedure', and any unit names. Return ONLY the rewritten query.

Question: {question}"""
        )
        text = resp.content
        if isinstance(text, list):
            text = "".join(
                item.get("text", "") for item in text if isinstance(item, dict)
            )
        return text.strip() or question
    except Exception:
        return question


def make_retriever_node():

    async def retriever_node(state: AgentState):

        if _mcp_session is None:
            return {
                "context": "",
                "messages": [
                    AIMessage(
                        content="MCP session is not connected."
                    )
                ],
            }

        question = state["messages"][-1].content

        result = await _mcp_session.call_tool(
            "search_castor_oil",
            {"query": question}
        )

        text = "\n".join(
            block.text
            for block in result.content
            if hasattr(block, "text")
        )

        return {
            "context": text
        }

    return retriever_node


answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following context and chat history.
Take the latest question into consideration.

IMPORTANT:
- Do not use LaTeX.
- Do not use \\boxed.
- Do not use \\begin{{aligned}}.
- Write formulas using normal text.
- Use simple Markdown when useful.
- Show calculations clearly.
- Keep the answer concise.

Chat history:
{history}

Context:
{context}

Question:
{question}
"""
)


def answer_node(state: AgentState):
    question = state["messages"][-1].content
    history = "\n".join(
        f"{type(m).__name__}: {m.content}" for m in state["messages"][:-1]
    )
    prompt = answer_prompt.format(
        history=history or "(none)",
        context=state.get("context", "") or "(no context)",
        question=question,
    )
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return {"messages": [AIMessage(content=content)]}


builder = StateGraph(AgentState)
builder.add_node("classifier", question_classifier)
builder.add_node("off_topic", off_topic_response)
builder.add_node("retriever", make_retriever_node())
builder.add_node("answer", answer_node)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier",
    topic_router,
    {"retriever": "retriever", "off_topic": "off_topic"},
)
builder.add_edge("retriever", "answer")
builder.add_edge("answer", END)
builder.add_edge("off_topic", END)

graph = builder.compile(checkpointer=MemorySaver())


_mcp_context = None
_mcp_session_context = None


async def start_mcp():
    global _mcp_session
    global _mcp_context
    global _mcp_session_context

    server_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "McpServer.py"
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
    )

    # Start MCP server
    _mcp_context = stdio_client(server_params)

    read, write = await _mcp_context.__aenter__()

    # Create MCP session
    _mcp_session_context = ClientSession(read, write)

    _mcp_session = await _mcp_session_context.__aenter__()

    # Initialize MCP
    await _mcp_session.initialize()

    # Check available tools
    tools = await _mcp_session.list_tools()

    print("MCP connected!", file=sys.stderr)

    for tool in tools.tools:
        print(f"  - {tool.name}", file=sys.stderr)


async def stop_mcp():
    global _mcp_session
    global _mcp_context
    global _mcp_session_context

    _mcp_session = None

    if _mcp_session_context:
        await _mcp_session_context.__aexit__(None, None, None)

    if _mcp_context:
        await _mcp_context.__aexit__(None, None, None)

    print("MCP connection closed.", file=sys.stderr)