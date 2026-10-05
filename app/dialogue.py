from app.config import Settings
from app.context import build_context
from app.llm import Completion, LLMClient
from app.prompts import PROMPTS
from app.storage import UserSession


class Dialogue:
    def __init__(self, client: LLMClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def respond(self, session: UserSession, text: str) -> Completion:
        preferences = await session.preferences()
        messages = build_context(
            PROMPTS[preferences.mode],
            await session.history(),
            text,
            self.settings.history_max_messages,
            self.settings.history_max_tokens,
        )
        answer = await self.client.complete(messages, preferences.temperature)
        await session.save_turn(text, answer.text)
        return answer
