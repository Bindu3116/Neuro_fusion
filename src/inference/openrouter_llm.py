"""
OpenRouter LLM integration for stress feedback generation.
Uses Grok (x-ai/grok-4.1-fast) or any OpenRouter model.
API key should be set in environment: OPENROUTER_API_KEY
"""

import os
import json
import requests
from typing import Dict, Optional


OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "x-ai/grok-4.1-fast"


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Set it in your environment or in a .env file (use python-dotenv)."
        )
    return key


def generate_feedback(
    predicted_label: str,
    confidence: float,
    probabilities: Dict[str, float],
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 250,
    temperature: float = 0.7,
) -> str:
    """
    Call OpenRouter (Grok) to generate wellness feedback from stress prediction.

    Args:
        predicted_label: e.g. "Moderate Stress"
        confidence: 0–1
        probabilities: dict like {"Calm": 0.1, "Mild Stress": 0.2, ...}
        model: OpenRouter model id (default: x-ai/grok-4.1-fast)
        api_key: Override env OPENROUTER_API_KEY if provided
        max_tokens: Max response length
        temperature: Sampling temperature

    Returns:
        Generated feedback text.
    """
    key = api_key or get_api_key()

    prompt = f"""You are a brief, empathetic wellness assistant. Based on this stress assessment from brain (EEG) and heart (ECG) signals, give short advice.

Stress level: {predicted_label}
Confidence: {confidence:.1%}

Probability distribution:
{json.dumps(probabilities, indent=2)}

Respond in 2–4 short sentences: (1) one line interpreting the result, (2) one or two actionable tips. Be supportive and concise. Do not repeat the numbers."""

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENROUTER_BASE, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "").strip()

    if not content:
        return "Please take a moment to rest and reassess how you feel."

    return content


def generate_feedback_stream(
    predicted_label: str,
    confidence: float,
    probabilities: Dict[str, float],
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
):
    """
    Stream OpenRouter response (e.g. for UI). Yields text chunks.
    """
    key = api_key or get_api_key()

    prompt = f"""You are a brief, empathetic wellness assistant. Based on this stress assessment from brain (EEG) and heart (ECG) signals, give short advice.

Stress level: {predicted_label}
Confidence: {confidence:.1%}

Probability distribution:
{json.dumps(probabilities, indent=2)}

Respond in 2–4 short sentences: (1) one line interpreting the result, (2) one or two actionable tips. Be supportive and concise. Do not repeat the numbers."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    with requests.post(
        OPENROUTER_BASE, json=payload, headers=headers, timeout=60, stream=True
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {resp.status_code}: {resp.text}"
            )

        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass
