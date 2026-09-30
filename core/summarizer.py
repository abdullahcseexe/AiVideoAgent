from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.rate_limiters import InMemoryRateLimiter
import httpx
import os

# Shared by summarizer.py and extractor.py (extractor imports it from here)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_llm():
    llm = ChatMistralAI(
        model="open-mistral-nemo",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        rate_limiter=rate_limiter,
    )
    return llm.with_retry(
        retry_if_exception_type=(httpx.HTTPStatusError,),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of the transcript concisely. "
                "Return only the summary.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert summarizer. Combine these partial summaries "
                "into one clear, polished summary in bullet points. "
                "Return only the bullet points, with no introduction or closing line.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the transcript, generate a short, clear title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])