
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
    build_job_vector_store,
    load_job_vector_store,
)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def create_rag_chain(vector_store, job_id=None):

    retriever = get_retriever(
        vector_store,
        k=4,
        job_id=job_id
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert AI Video Assistant.

Answer the user's question based ONLY on the video
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video transcript."

Always be concise and precise.

Context from video transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(
    transcript: str,
    job_id: Optional[str] = None
):

    if job_id is not None:
        vector_store = build_job_vector_store(
            transcript,
            job_id
        )
    else:
        vector_store = build_vector_store(transcript)

    return create_rag_chain(
        vector_store,
        job_id=job_id
    )


def load_rag_chain(job_id: Optional[str] = None):

    if job_id is not None:
        vector_store = load_job_vector_store(job_id)
    else:
        vector_store = load_vector_store()

    return create_rag_chain(
        vector_store,
        job_id=job_id
    )


def ask_question(rag_chain, question: str) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer
