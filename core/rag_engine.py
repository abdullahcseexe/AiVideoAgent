from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core.summarizer import get_llm   # same model, rate limiter and retry as the rest


RAG_SYSTEM_PROMPT = """You are an expert assistant. Answer the user's question
based ONLY on the transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from transcript:
{context}"""


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _make_chain(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    return _make_chain(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)   # was missing vector_store
    return _make_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer