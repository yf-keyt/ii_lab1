from dataclasses import dataclass


class InputTooLong(ValueError):
    pass


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


def estimate_tokens(text: str) -> int:
    # Conservative byte estimate, independent of the provider's tokenizer.
    return len(text.encode("utf-8")) + 16


def build_context(
    prompt: str, history: list[Turn], text: str, max_messages: int, max_tokens: int
) -> list[dict[str, str]]:
    budget = max_tokens - estimate_tokens(prompt) - estimate_tokens(text)
    if budget < 0:
        raise InputTooLong
    selected = []
    for turn in reversed(history):
        cost = estimate_tokens(turn.user) + estimate_tokens(turn.assistant)
        if len(selected) * 2 + 2 > max_messages or cost > budget:
            break
        selected.append(turn)
        budget -= cost
    messages = [{"role": "system", "content": prompt}]
    for turn in reversed(selected):
        messages.extend(
            [
                {"role": "user", "content": turn.user},
                {"role": "assistant", "content": turn.assistant},
            ]
        )
    messages.append({"role": "user", "content": text})
    return messages
