# Functions to perform (raw text) retrieval-augmented queries against LLMs.

def build_rag_prompts(
    context: str,
    question: str,
    topic: str | None = None,
) -> tuple[str, str]:
    sys_inst = (
        """
        You are an assistant and expert in answering questions from chunks of
        content.
        """
    )
    if topic:
        sys_inst += (
            """
            Only answer questions related to {}, else say that you cannot
            answer this question.
            """
        ).format(topic)

    user_prompt = (
        """
        Read the following information that might contain the context you
        require to answer the question. You can use the information starting
        from the <START_CONTEXT> tag up to the <END_CONTEXT> tag. Here is the
        context: <START_CONTEXT>\n{context}\n<END_CONTEXT>\n{question}
        """
    )

    user_prompt = user_prompt.format(context=context, question=question)
    return sys_inst, user_prompt

def perform_rag_query(
    llm: LLMBackend,
    context: str,
    question: str,
    topic: str | None = None,
) -> str:
    sys_inst, user_prompt = build_rag_prompts(
        context=context,
        question=question,
        topic=topic,
    )
    return llm.generate(sys_inst, user_prompt)
