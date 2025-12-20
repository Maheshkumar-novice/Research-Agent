import logging

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field

from index import get_query_engine, get_ranker, get_retriever
from memory import get_memory
from prompts import INTENT, QUESTION_ANSWER, RESPONSE, SUB_QUESTIONS
from tools import TOOLS, QueryAnalysis, QueryIntent
from workflow_events import SubQuestionsAnsweredEvent, SubQuestionsEvent

logging.info("Setting up the workflow")


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(description="List of sub-questions to answer the main query")


class ResearchAgent(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranker = get_ranker()
        self.retriever = get_retriever()
        self.query_engine = get_query_engine()
        self.memory = get_memory()
        self.tools = TOOLS

    @step
    async def initialize_research(self, ctx: Context, ev: StartEvent) -> SubQuestionsEvent:
        logging.info(self.memory.get())

        await ctx.store.set("original_query", ev.query)
        user_query = ev.query
        analysis = Settings.llm.structured_predict(QueryAnalysis, INTENT, query=user_query)

        await ctx.store.set("query_analysis", analysis)
        logging.info(f"Query intent: {analysis.intent}, Tools needed: {analysis.requires_tools}")

        if analysis.intent == QueryIntent.QUESTION_ANSWERING:
            sub_q_response = Settings.llm.structured_predict(SubQuestions, SUB_QUESTIONS, query=user_query)
            logging.info(sub_q_response)
            return SubQuestionsEvent(queries=sub_q_response.sub_questions)
        else:
            return SubQuestionsEvent(queries=[user_query])

    @step
    async def answer_sub_question(self, ctx: Context, ev: SubQuestionsEvent) -> SubQuestionsAnsweredEvent:
        query_analysis = await ctx.store.get("query_analysis")
        question_answers = {}

        for query in ev.queries:
            nodes = self.retriever.retrieve(query)
            nodes = self.ranker.postprocess_nodes(nodes, query_str=query)
            context = "\n\n".join([node.text for node in nodes])

            if query_analysis.intent == QueryIntent.KEYWORD_EXTRACTION:
                result = self.tools["keyword_extraction"](context).raw_output["keywords"]
                answer = f"Key terms extracted: {', '.join(result)}"

            elif query_analysis.intent == QueryIntent.SUMMARIZATION:
                answer = self.tools["summarization"](context).raw_output

            elif query_analysis.intent == QueryIntent.CONTENT_ANALYSIS:
                result = self.tools["content_analysis"](context).raw_output
                answer = f"Analysis:\nThemes: {result['themes']}\nSentiment: {result['sentiment']}\nKey Entities: {result['key_entities']}"

            elif query_analysis.intent == QueryIntent.QUESTION_ANSWERING:
                if "keyword_extraction" in query_analysis.requires_tools:
                    keywords = self.tools["keyword_extraction"](context).raw_output["keywords"]
                    context = f"Key Terms: {', '.join(keywords)}\n\n{context}"

                if len(context) > 5000 and "summarization" in query_analysis.requires_tools:
                    context = self.tools["summarization"](context)

                response = Settings.llm.complete(QUESTION_ANSWER.format(context=context, query=query))
                answer = str(response)

            else:
                answer = Settings.llm.complete(QUESTION_ANSWER.format(context=context, query=query))

            question_answers[query] = str(answer)
            logging.info(f"Executed {query_analysis.intent} for: {query}")

        logging.info(question_answers)
        return SubQuestionsAnsweredEvent(question_answers=question_answers)

    @step
    async def collect_results(self, ctx: Context, ev: SubQuestionsAnsweredEvent) -> StopEvent:
        query_analysis = await ctx.store.get("query_analysis")
        original_query = await ctx.store.get("original_query")

        if len(ev.question_answers) == 1:
            result = list(ev.question_answers.values())[0]
        else:
            context = "\n\n".join([f"Q: {k}\nA: {v}" for k, v in ev.question_answers.items()])

            history = self.memory.get()
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history])

            result = Settings.llm.complete(RESPONSE.format(context=context, query=original_query, history=history_text))

        self.memory.put_messages([ChatMessage(role="user", content=original_query), ChatMessage(role="assistant", content=str(result))])

        logging.info(f"Final result for {query_analysis.intent}: {result}")
        return StopEvent(result=str(result))
