from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .llm import chat_llm
from .vectorstore import create_or_load_vectorstore

load_dotenv()


def build_retrieval_chain():
    vectorstore = create_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = chat_llm(streaming=True)

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    # Build RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
