from llama_index.core.prompts import PromptTemplate

SUB_QUESTIONS = PromptTemplate(
    """
We are doing a RAG application for adobe_annual_reports call PDF. We have our own index.

Given a user question, output a list of relevant sub-questions, such that the answers to all the
sub-questions put together will answer the question. Respond
in pure JSON without any markdown, like this:
{{
    "sub_questions": [
        "What is the population of San Francisco?",
        "What is the budget of San Francisco?",
        "What is the GDP of San Francisco?"
    ]
}}

We will be passing these sub-questions to an LLM with relevant context from the Vector Store (RAG) to get the answers.

If the question feels like not relevant to our RAG requirement then just output empty JSON object.

Here is the user question: {query}
    """
)

QUESTION_ANSWER = PromptTemplate(
    """
We are answering questions based on the retrieved RAG context. Please use it to arrive at your answer.

Be concise in your answer.

Context:
{context}

Query:
Here is the user question: {query}
    """
)

RESPONSE = PromptTemplate(
    """
For the initial user query, we have done a sub-questions split and got answers from an LLM using RAG.
I want you to use that information as context and answer the original user query.
I have also attached our conversation history so far that you can refer to.

Be very concise in your answer. If the context is empty, then use your own knowledge to answer.

Context:
{context}

History:
{history}

Query:
Here is the user question: {query}
    """
)

INTENT = PromptTemplate("""
Analyze this user query and determine:
1. What is the primary intent? (keyword_extraction, summarization, content_analysis, question_answering)
2. Which tools are needed?
3. Query complexity?

User query: {query}

Examples:
- "Extract key terms from the report" → keyword_extraction
- "Summarize the document" → summarization
- "What themes are in this?" → content_analysis
- "What is the revenue?" → question_answering
""")
