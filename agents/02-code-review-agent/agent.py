"""
Code Review Agent using LangChain + Groq.

Reviews Python code for bugs, security issues, style violations,
performance issues and suggests improvements.

Usage:
    python agent.py --file test.py
    python agent.py --code "print('Hello World')"
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in .env")

SYSTEM_PROMPT = """
You are an expert software engineer and code reviewer.

Analyze the provided code and generate a professional review covering:

1. Bugs & Correctness
2. Security Issues
3. Performance
4. Code Style
5. Best Practices
6. Refactoring Suggestions

Finally include:

- Overall Rating (1-10)
- Summary
- Improved Code (if applicable)

Return everything in Markdown.
"""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


def review_code(code: str, language: str = "python") -> str:

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Review this {language} code:\n\n```{language}\n{code}\n```"
        ),
    ]

    response = llm.invoke(messages)

    return response.content


def main():

    parser = argparse.ArgumentParser(description="AI Code Review Agent")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--file",
        help="Path to the source code file",
    )

    group.add_argument(
        "--code",
        help="Inline source code",
    )

    parser.add_argument(
        "--language",
        default="python",
        help="Programming language",
    )

    args = parser.parse_args()

    if args.file:

        if not os.path.exists(args.file):
            print(f"\n❌ File not found: {args.file}")
            return

        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()

        print(f"\n🔍 Reviewing file: {args.file}\n")

    else:

        code = args.code

        print("\n🔍 Reviewing inline code\n")

    review = review_code(code, args.language)

    print("=" * 70)
    print("📋 AI CODE REVIEW")
    print("=" * 70)
    print(review)


if __name__ == "__main__":
    main()