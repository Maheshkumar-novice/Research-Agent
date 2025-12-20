# Design Decisions

A quick overview of why things are the way they are.

---

## Architecture

### Workflow-Based Design (`workflow.py`)
Using LlamaIndex's Workflow framework instead of simple function chains. Why? It gives us:
- Clear step-by-step flow
- Context sharing between steps via `ctx.store`
- Async support out of the box

The workflow has 3 steps:
1. `initialize_research` → Analyze query intent, break into sub-questions if needed
2. `answer_sub_question` → Get answers for each sub-question
3. `collect_results` → Combine everything into final response

### Event-Driven Communication (`workflow_events.py`)
Custom events to pass data between steps. Kept it simple with just two:
- `SubQuestionsEvent` → carries the sub-questions list
- `SubQuestionsAnsweredEvent` → carries the Q&A dict

---

## Query Handling

### Intent Classification (`tools.py`)
Before doing anything, we figure out what the user actually wants:
- **keyword_extraction** → Extract key terms
- **summarization** → Summarize content
- **content_analysis** → Analyze themes/sentiment
- **question_answering** → Answer questions (default)

This way we don't waste tokens doing unnecessary work.

### Sub-Question Splitting
Complex questions get broken into smaller ones. Each sub-question hits the retriever separately. Better retrieval results this way.

---

## RAG Pipeline

### Two-Stage Retrieval (`index.py`)
1. **Retriever** → Get top 10 similar chunks
2. **Reranker** → Re-rank and pick top 5

The reranker (SentenceTransformerRerank) improves relevance. Worth the extra compute.

### Persistent Index
Index gets saved to `./storage`. No need to rebuild every time. Checks if folder exists on startup.

---

## Memory (`memory.py`)

Using LlamaIndex's Memory with:
- **FactExtractionMemoryBlock** → Pulls out important facts automatically
- **SQLite backend** → Persists across sessions
- **Token limit of 4000** → Keeps context manageable
- **30% for chat history** → Rest for facts

---

## LLM & Rate Limiting (`settings.py`)

### SafeRateLimitedLLM Wrapper
Wraps the Anthropic LLM with:
- **Rate limiting** → 50 calls/minute max
- **Exponential backoff** → Waits 4s, then 8s, 16s, etc. on failures
- **5 retry attempts** → Handles transient API errors

Catches `RateLimitError` and `APIError` specifically.

### Embeddings
Using `bge-small-en-v1.5` from HuggingFace. Runs on CPU. Small but good enough.

---

## Prompts (`prompts.py`)

Four main prompts:
1. **SUB_QUESTIONS** → Break query into sub-parts
2. **QUESTION_ANSWER** → Answer based on context
3. **RESPONSE** → Final answer with history
4. **INTENT** → Classify what user wants

All use PromptTemplate for variable injection.

---

## Entry Point (`app.py`)

Simple REPL loop. Nothing fancy:
- Creates workflow instance
- Takes input
- Runs query
- Repeats until 'quit'

---

## File Organization

```
app.py           → Entry point
workflow.py      → Main logic
workflow_events.py → Event definitions
index.py         → Vector store & retrieval
memory.py        → Memory config
tools.py         → Tool definitions & intent enum
prompts.py       → All prompts
settings.py      → LLM, embeddings, rate limiting
```

Each file does one thing. Makes it easy to find stuff.
