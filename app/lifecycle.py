import asyncio

from aiogram import BaseMiddleware


class TaskTracker(BaseMiddleware):
    """Cancel handlers before closing their HTTP sessions and database pool."""

    def __init__(self):
        self.tasks = set()
        self.stopping = False

    async def __call__(self, handler, event, data):
        if self.stopping:
            return None
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            return await handler(event, data)
        finally:
            self.tasks.discard(task)

    async def close(self):
        self.stopping = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
