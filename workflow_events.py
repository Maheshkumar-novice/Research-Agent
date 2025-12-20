from llama_index.core.workflow import Event


class SubQuestionsEvent(Event):
    queries: list[str]


class SubQuestionsAnsweredEvent(Event):
    question_answers: dict[str, str]
