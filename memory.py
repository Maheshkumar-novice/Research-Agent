import logging

from llama_index.core.memory import FactExtractionMemoryBlock, Memory

logging.info("Initializing memory")


def get_memory() -> Memory:
    return Memory.from_defaults(
        session_id="user_session",
        token_limit=4000,
        chat_history_token_ratio=0.3,
        memory_blocks=[FactExtractionMemoryBlock(priority=1)],
        async_database_uri="sqlite+aiosqlite:///./memory.db",
    )
