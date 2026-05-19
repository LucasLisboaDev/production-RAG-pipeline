"""
LLM Client (Week 3)
====================
Thin wrapper around OpenAI and Anthropic APIs.
Reads the provider from the .env file.

Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in your .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


def call_llm(messages: list[dict], max_tokens: int = 1024) -> str:
    if LLM_PROVIDER == "openai":
        return _call_openai(messages, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(messages, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_openai(messages: list[dict], max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content


def _call_anthropic(messages: list[dict], max_tokens: int) -> str:
    import anthropic
    # Extract system prompt if present
    system = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_messages.append(m)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=user_messages,
    )
    return response.content[0].text
