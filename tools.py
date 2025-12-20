import logging
from enum import Enum

from llama_index.core import PromptTemplate, Settings
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    KEYWORD_EXTRACTION = "keyword_extraction"
    SUMMARIZATION = "summarization"
    CONTENT_ANALYSIS = "content_analysis"
    QUESTION_ANSWERING = "question_answering"


class QueryAnalysis(BaseModel):
    intent: QueryIntent = Field(description="Primary intent of the query")
    requires_tools: list[str] = Field(description="List of tools needed")
    complexity: str = Field(description="simple, medium, or complex")


class Analysis(BaseModel):
    themes: list[str]
    sentiment: str
    key_entities: list[str]


class Keywords(BaseModel):
    keywords: list[str]


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    response = Settings.llm.structured_predict(
        Keywords, PromptTemplate(f"Extract the {top_k} most important keywords:\n\n{text}"), top_k=top_k, text=text
    )
    logging.info(response)
    return response.dict()


def summarize_text(text: str, max_length: int = 500) -> str:
    response = Settings.llm.complete(f"Summarize in under {max_length} words:\n\n{text}")
    logging.info(response)
    return str(response)


def analyze_content(text: str) -> dict:
    response = Settings.llm.structured_predict(Analysis, PromptTemplate(f"Analyze this content:\n\n{text}"), text=text)
    logging.info(response)
    return response.dict()


TOOLS = {
    "keyword_extraction": FunctionTool.from_defaults(fn=extract_keywords, name="keyword_extraction"),
    "summarization": FunctionTool.from_defaults(fn=summarize_text, name="summarization"),
    "content_analysis": FunctionTool.from_defaults(fn=analyze_content, name="content_analysis"),
}
