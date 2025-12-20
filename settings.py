import logging

import anthropic
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from ratelimit import limits, sleep_and_retry
from rich.logging import RichHandler
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
load_dotenv()

logging.info("Initializing embedding model")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
Settings.embed_model = embed_model

logging.info("Initializing LLM with rate limiting and retries")


class SafeRateLimitedLLM(LLM):
    """LLM wrapper with rate limiting and automatic retries."""

    def __init__(self, llm):
        self._llm = llm

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logging.warning(f"API error, retrying in {retry_state.next_action.sleep:.1f}s..."),
    )
    def complete(self, *args, **kwargs):
        return self._llm.complete(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    def chat(self, *args, **kwargs):
        return self._llm.chat(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    def structured_predict(self, *args, **kwargs):
        return self._llm.structured_predict(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    def stream_complete(self, *args, **kwargs):
        return self._llm.stream_complete(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    def stream_chat(self, *args, **kwargs):
        return self._llm.stream_chat(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    async def acomplete(self, *args, **kwargs):
        return await self._llm.acomplete(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    async def achat(self, *args, **kwargs):
        return await self._llm.achat(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    async def astream_complete(self, *args, **kwargs):
        return await self._llm.astream_complete(*args, **kwargs)

    @sleep_and_retry
    @limits(calls=50, period=60)
    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
    )
    async def astream_chat(self, *args, **kwargs):
        return await self._llm.astream_chat(*args, **kwargs)

    @property
    def metadata(self):
        return self._llm.metadata

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped LLM."""
        return getattr(self._llm, name)


# Create rate-limited LLM
base_llm = Anthropic(model="claude-sonnet-4-20250514", max_tokens=1000, timeout=60.0)
llm = SafeRateLimitedLLM(base_llm)
Settings.llm = llm

PERSIST_DIR = "./storage"
