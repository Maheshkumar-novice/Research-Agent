import asyncio
import logging

from workflow import ResearchAgent


async def run_agent_loop():
    """Simple agent loop with memory"""
    print("Agent ready. Type 'quit' to exit.\n")
    workflow = ResearchAgent(timeout=60, verbose=True)

    while True:
        question = input("You: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            logging.info("Getting out of loop")
            break

        if not question:
            continue

        await workflow.run(query=question)


if __name__ == "__main__":
    asyncio.run(run_agent_loop())
