import textwrap
from typing import Dict, List
from langchain.schema.document import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.messages import (
    AIMessage,
)

from chains.base import BaseChain, keep_chain_params, log_chain_params
from chains.prompts import PromptFormatter
from lib.helpers import print_params

question = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context and conversation history to answer the question.
        If you don't know the answer, say that you don't know. Limit your response to three sentences maximum
        and keep the answer concise. Don't reveal that the context is empty, just say you don't know.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Question: {question}
        """
    ),
)

def rerank_documents(list_of_documents: list[Document], query: str, amount=5):
    global compressor

    if len(list_of_documents) > 5:
        if compressor is None:
            compressor = FlashrankRerank(top_n=amount)

        ranked_documents = compressor.compress_documents(
            documents=list_of_documents, query=query
        )
        return ranked_documents

    return list_of_documents


def rerank_rag(params: Dict):
    query = params["question"]
    new_content = []
    new_content_strs = []

    def sort_by_references(document: Document):
        return len(document.metadata)

    documents: list[Document] = params["documents"]

    documents.sort(reverse=True, key=sort_by_references)
    for document in documents:
        if document.page_content not in new_content_strs:
            new_content.append(document)
            new_content_strs.append(document.page_content)

    new_content = rerank_documents(new_content, query, 10)

    return new_content


def set_metadata(params):
    print_params("metadata", params)
    return params["documents"] if "documents" in params.keys() else params["context"]["documents"] if "documents" in params["context"].keys() else []


def store_documents(params):
    if isinstance(params, List) and len(params) > 0 and isinstance(params[0], Document):
        return {"documents": params}

    return params


class RagChain(BaseChain):
    def __init__(self, retriever: BaseRetriever, prompt:PromptFormatter=question, **kwargs):
        super().__init__(prompt=prompt, **kwargs)
        self.retriever = retriever

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if self.chain is not None and (
            custom_prompt is None or self.custom_prompt is custom_prompt
        ):
            return self.chain

        self._setup_prompt(custom_prompt)

        self.chain = super().__call__(custom_prompt)

        self.chain = (
            keep_chain_params
            | RunnableParallel(
                {
                    "context": (
                        RunnableParallel(
                            {
                                "documents": self.retriever,
                                "orig_params": RunnablePassthrough(),
                            }
                        )
                        | keep_chain_params
                        | rerank_rag
                        | store_documents
                    ),
                    "question": RunnableLambda(lambda x: x["question"]),
                    "chat_history": RunnableLambda(
                        lambda x: (
                            x["chat_history"] if "chat_history" in x.keys() else []
                        )
                    ),
                }
            )
            | RunnableParallel({
                "answer": self.chain,
                "documents": RunnableLambda(lambda x: x["documents"] if "documents" in x.keys() else x["context"]["documents"] if "documents" in x["context"].keys() else []),
                "question": RunnableLambda(lambda x: x["question"]),
                "metadata": RunnableLambda(set_metadata)
            })
            | log_chain_params

        )

        return self.chain
