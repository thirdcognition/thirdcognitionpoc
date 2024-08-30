import textwrap
from typing import Any, Dict, List
from langchain.schema.document import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)
from lib.db_tools import get_vectorstore_as_retriever
# from langchain_core.messages import (
#     AIMessage,
# )

from chains.base import BaseChain, keep_chain_params, log_chain_params
from chains.init import get_chain, get_llm
from chains.prompts import PromptFormatter, question
from lib.helpers import get_chain_with_history, print_params

compressor = None
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

def remove_duplicates(documents: Dict[Any, Document]) -> List[str]:
    if isinstance(documents, Dict):
        if all(isinstance(document, Document) for document in documents.values()):
            doc_dict = {val.page_content: val for val in documents.values()}
            # Return the documents as a list
            return list(doc_dict.values())
        return list(set(documents.values()))

    return documents

class RagChain(BaseChain):
    def __init__(self, retrievers: List[BaseRetriever], prompt:PromptFormatter=question, **kwargs):
        self.retrievers = retrievers
        super().__init__(prompt=prompt, **kwargs)

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if self.chain is not None and (
            custom_prompt is None or self.custom_prompt is custom_prompt
        ):
            return self.chain

        self._setup_prompt(custom_prompt)

        self.chain = super().__call__(custom_prompt)

        retriever = self.retrievers[0]
        if len(self.retrievers) > 1:
            retrievers = {i: retriever for i, retriever in enumerate(self.retrievers)}
            retriever = (
                RunnableParallel(retrievers) |
                remove_duplicates
            )

        self.chain = (
            keep_chain_params
            | RunnableParallel(
                {
                    "context": (
                        RunnableParallel(
                            {
                                "documents": retriever,
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

class RagChatChain(RagChain):
    # def __init__(self, retriever: BaseRetriever, prompt:PromptFormatter=question, **kwargs):
    #     super().__init__(prompt=prompt, **kwargs)
    #     self.retriever = retriever

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if self.chain is not None and (
            custom_prompt is None or self.custom_prompt is custom_prompt
        ):
            return self.chain

        self._setup_prompt(custom_prompt)
        self.chain = super().__call__(custom_prompt)

        chat_chain = get_chain("chat")

        self.chain = (
            RunnableParallel({
                "is_question": get_chain("question_classification") | RunnableLambda(lambda x: x[0] if isinstance(x, tuple) else x),
                "orig_params": RunnablePassthrough()
            }) |
            RunnableBranch(
                (lambda x: x["is_question"], self.chain),
                keep_chain_params | chat_chain | RunnableLambda(lambda x: {"answer": x})
            )
        )

        return self.chain


rag_chains = {}

def get_rag_chain(store_ids:List[str], embedding_id="hyde", reset=False, amount_of_documents=5, chat=False) -> RunnableSequence:
    global rag_chains

    chain_type = "chat" if chat else "search"
    chain_id = f"{"+".join(store_ids)}-{embedding_id}-{chain_type}-#{amount_of_documents}"

    if chain_id in rag_chains and not reset:
        return rag_chains[chain_id]

    print(f"Initializing RAG {chain_type} chain: {chain_id}")

    retrievers = [get_vectorstore_as_retriever(store_id, embedding_id, amount_of_documents) for store_id in store_ids]

    if chat:
        rag_chain = RagChatChain(retrievers, llm=get_llm("chat"))()
    else:
        rag_chain = RagChain(retrievers, llm=get_llm("chat"))()

    rag_chains[chain_id] = rag_chain

    return rag_chain
