"""
Web Research Agent using LangGraph + Tavily Search.

Searches the web for a given topic, synthesizes findings,
and returns a structured research report.

Usage:
    python agent.py
    python agent.py --query "latest advances in quantum computing"
"""

import argparse
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY not found in .env")

# --------------------------------------------------
# Initialize LLM only once
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# --------------------------------------------------
# State Definition
# --------------------------------------------------

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_results: list[dict]
    report: str


# --------------------------------------------------
# Search Node
# --------------------------------------------------

def search_web(state: ResearchState) -> ResearchState:
    tool = TavilySearch(max_results=5)

    raw_results = tool.invoke(state["query"])

    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results = raw_results
    else:
        results = []

    return {"search_results": results}


# --------------------------------------------------
# Report Generation Node
# --------------------------------------------------

def synthesize_report(state: ResearchState) -> ResearchState:

    results_text = "\n\n".join(
        f"Source: {r.get('url', 'N/A')}\n"
        f"Title: {r.get('title', 'N/A')}\n"
        f"Content: {r.get('content', '')[:500]}"
        for r in state["search_results"]
    )

    messages = [
        SystemMessage(
            content=(
                "You are an expert research analyst.\n"
                "Create a professional report with:\n\n"
                "1. Executive Summary\n"
                "2. Key Findings (bullet points)\n"
                "3. Important Insights\n"
                "4. Conclusion\n"
                "5. Sources"
            )
        ),
        HumanMessage(
            content=f"""
Research Query:
{state['query']}

Search Results:
{results_text}
"""
        ),
    ]

    try:
        response = llm.invoke(messages)

        return {
            "report": response.content,
            "messages": [response],
        }

    except Exception as e:
        return {
            "report": f"❌ Failed to generate report.\n\n{str(e)}",
            "messages": [],
        }


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

def build_graph() -> StateGraph:

    graph = StateGraph(ResearchState)

    graph.add_node("search", search_web)
    graph.add_node("synthesize", synthesize_report)

    graph.set_entry_point("search")

    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Web Research Agent"
    )

    parser.add_argument(
        "--query",
        default="latest advances in AI agents 2024",
        help="Research topic",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🔍 Researching: {args.query}")
    print("=" * 70 + "\n")

    agent = build_graph()

    result = agent.invoke(
        {
            "query": args.query,
            "messages": [],
            "search_results": [],
            "report": "",
        }
    )

    print("=" * 70)
    print("📄 RESEARCH REPORT")
    print("=" * 70)
    print(result["report"])
    print("=" * 70)


if __name__ == "__main__":
    main()