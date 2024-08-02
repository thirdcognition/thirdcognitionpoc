from functools import cache
import os
import random
import textwrap
import chromadb
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.openai_functions import create_extraction_chain
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_compressors import FlashrankRerank
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma

from langchain.globals import set_debug, set_verbose
from dotenv import load_dotenv

# Load .env file from the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


set_debug(os.getenv("DEBUG", False))
set_verbose(os.getenv("VERBOSE", True))

CHROMA_PATH = os.getenv("CHROMA_PATH", "db/chroma_db")
SQLITE_DB = os.getenv("SQLITE_DB", "db/files.db")

chat_llm = os.getenv("CHAT_LLM", "phi3:mini")
instruct_llm = os.getenv("INSTRUCT_LLM", "phi3:instruct")
ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

CTX_MULTP = int(os.getenv("CTX_MULTP", 16))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", 1024 )) * CTX_MULTP * 2
CHAR_LIMIT = int(os.getenv("CONTEXT_SIZE", 1024 )) * (CTX_MULTP // 3 * 2) * 2

llms = {}
embeddings = {}
prompts = {}
chains = {}
templates = {
    "text_formatter": {
        "system": """Act as a document formatter. Rewrite the text specified by the user in full detail. Use only information from the text. Don't add or remove anything, just return the result. Return only the unformatted text.""",
        "user": """{context}""",
    },
    "text_formatter_compress": {
        "system": """Act as a document formatter. The text is a part of a longer document. Rewrite the text specified by the user in full detail. Use only information from the available in the text. Don't add or remove anything, just return the result. Return only the unformatted text.""",
        "user": """{context}""",
    },
    "text_formatter_guided": {
        "system": """Act as a document formatter. Use only information from the context. Don't add or remove anything, just return the result. Return only the unformatted text.""",
        "user": """Instructions: {question}
        Context: {context}
        Answer:""",
    },
    "md_formatter": {
        "system": """Act as a document formatter. Rewrite the text in context using markdown syntax. Use only information from the context. Don't add or remove anything, just return the result.""",
        "user": """Context: {context}
        Answer:""",
    },
    "md_formatter_guided": {
        "system": """Act as a document formatter. Rewrite the text in context using markdown syntax. Use only information from the context. Don't add or remove anything, just return the result.""",
        "user": """Instructions: {question}
        Context: {context}
        Answer:""",
    },
    "action": {
        "system": """Act as a task completing machine. Use the following pieces of retrieved context to complete the task. If you don't know the answer, just say that you don't know how.""",
        "user": """Task: {action}
        Context: {context}
        Answer:""",
    },  # Use ten sentences maximum and
    "helper": {
        "system": """Act as a startup coach and answer questions thoroughly and exactly. Use the context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.
            Do not mention that you are using the context. Avoid sentences like: Based on the provided context.
            Terms:
            PRE-IC Pre-IC refers to Pre-Investment Committee
            IC refers to Investment Committee
            POC refers to Proof of Concept
            MVP refers to Minimum viable product""",
        "user": """Context: {context}
        Question: {question}""",
    },
    "question": {
        "system": """You are a helpful startup coach from antler trying to answer questions thoroughly and exactly. Use the following pieces of retrieved context to answer the question. Use history if you don't undestand the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise unless question requests for more.""",
        "user": """Question: {question}
        Context: {context}
        """,
    },
    "hyde": {
        "system": """Act as a startup coach and answer questions thoroughly and exactly. If you don't know the answer, just say that you don't know. Answer the question. Use five sentences maximum, keep the answer concise and discover keywords from the question.
        Terms:
        PRE-IC Pre-IC refers to Pre-Investment Committee
        IC refers to Investment Committee
        POC refers to Proof of Concept
        MVP refers to Minimum viable product""",
        "user": """{question}""",
    },  # <|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>
    "summary": {
        "system": """Summarize this content: """,
        "user": """Content: {context} """,
    },
    "journey_text": {
        "system": """Act as a task orginizer and scheduler. Create a schedule with {amount} tasks using the specified format using context.""",
        "user": """Format: {format_example}
        Context: {context}""",
    },
}

journey_json_template = {
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the task",
            "title": "Name",
        },
        "description": {
            "type": "string",
            "description": "Description for the task",
            "title": "Description",
        },
        "actions": {
            "description": "List actions within the task.",
            "items": {"type": "string"},
            "title": "Actions",
            "type": "array",
        },
        "priority": {
            "type": "int",
            "description": "How important the task is",
            "title": "Priority",
        },
    },
    "required": ["name", "description"],
}


def validate_json_format(data, template=journey_json_template):
    keys = data.keys()
    template_keys = template["properties"].keys()

    for key in template_keys:
        if key not in keys:
            data[key] = None

    return data


def get_journey_format_example(amount=10):
    template_text = """
Task [number]: Title
* Description: Task description
* Actions: Actions in the task
* Priority: [priority]
"""

    result_text = ""

    priorities = ["LOW", "MEDIUM", "HIGH"]

    for i in range(0, amount):
        result_text += template_text.replace("[number]", str(i + 1)).replace(
            "[priority]", random.choice(priorities)
        )

    return result_text


def get_chain(chain, size="") -> LLMChain:
    if len(chains.keys()) == 0:
        init_llms()

    if size:
        return chains[chain + "_" + size]
    else:
        return chains[chain]


def get_llm_prompt(llm_id="default", prompt_id="helper", size=""):
    if len(chains.keys()) == 0:
        init_llms()

    if llm_id in llms.keys():
        llm = llms[llm_id]
    else:
        llm = llms["default"]

    return {"llm": llm, "prompt": prompts[prompt_id]}


def init_chain(
    id, llm="default", input_variables=["context"], templates=templates, init_llm=True
):  # templates_mistral
    if f"{id}" not in prompts:
        messages = [
            ("system", templates[f"{id}"]["system"])
        ]
        if id == "helper" or id == "question" or id == "hyde":
            messages.append(MessagesPlaceholder("chat_history"))
        messages.append(("user", templates[f"{id}"]["user"]))
        prompts[f"{id}"] = ChatPromptTemplate.from_messages(messages)
        # prompts[f"{id}"] = PromptTemplate(
        #     template=templates[f"{id}"], input_variables=input_variables
        # )

    # if f"{id}_large" not in chains:
    #     chains[f"{id}_large"] = LLMChain(
    #         llm=llms[f"{llm}_large"], prompt=prompts[f"{id}"]
    #     )
    if f"{id}" not in chains and init_llm:
        chains[f"{id}"] = LLMChain(llm=llms[llm], prompt=prompts[f"{id}"])
    # if f"{id}_small" not in chains:
    #     chains[f"{id}_small"] = LLMChain(
    #         llm=llms[f"{llm}_small"], prompt=prompts[f"{id}"]
    #     )


def init_llm(
    id,
    model=chat_llm,
    temperature=0,
    verbose=True,
    Type=Ollama,
):
    if f"{id}" not in llms:
        llms[f"{id}"] = Type(
            base_url=ollama_url,
            model=model,
            verbose=verbose,
            temperature=temperature,
            num_ctx=CONTEXT_SIZE,
            num_predict=CONTEXT_SIZE,
            # num_ctx=CONTEXT_SIZE,
            # num_predict=CONTEXT_SIZE,
            repeat_penalty=1.5,
            timeout=20 * 1000,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

    # if f"{id}_large" not in llms:
    #     llms[f"{id}_large"] = Type(
    #         model=model_small,
    #         verbose=verbose,
    #         temperature=temperature,
    #         num_ctx=1024 * 32,
    #         num_predict=1024 * 32,
    #         repeat_penalty=1.5,
    #         timeout=300*1000,
    #         # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    #     )

    # if f"{id}_small" not in llms:
    #     llms[f"{id}_small"] = Type(
    #         model=model_large,
    #         verbose=verbose,
    #         temperature=temperature,
    #         num_ctx=1024 * 8,
    #         num_predict=1024 * 8,
    #         repeat_penalty=1.5,
    #         timeout=100*1000,
    #         # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    #     )


initialized = False


def init_llms():
    print("Initialize llms...")
    initialized = True

    init_llm("default")
    init_llm("instruct", model=instruct_llm)

    # init_llm("json", Type=OllamaFunctions)

    if "json" not in llms:
        llms["json"] = OllamaFunctions(
            # model=llama3_llm,
            base_url=ollama_url,
            model=instruct_llm,
            temperature=0,
            # num_ctx=1024 * 8,
            # num_predict=1024 * 8,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

    init_llm("chat", temperature=0.1, Type=ChatOllama)

    init_llm("warm", temperature=0.3, model=instruct_llm)

    init_chain("summary", "instruct", templates=templates)
    init_chain(
        "action",
        input_variables=["context", "action"],
        llm="instruct",
        templates=templates,
    )
    init_chain("text_formatter", llm="instruct", templates=templates)
    init_chain("text_formatter_compress", llm="instruct", templates=templates)
    init_chain(
        "text_formatter_guided",
        "warm",
        input_variables=["context", "question"],
        templates=templates,
    )
    init_chain("md_formatter", llm="instruct", templates=templates)
    init_chain(
        "md_formatter_guided",
        "warm",
        input_variables=["context", "question"],
        templates=templates,
    )
    init_chain(
        "journey_text",
        input_variables=["context", "amount"],
        llm="instruct",
        templates=templates,
    )
    init_chain(
        "question", llm="chat", input_variables=["context", "question", "history"]
    )

    # template = """<s> <<SYS>> Act as a a helpful startup coach from antler trying to answer questions thoroughly. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> </s>

    # if "question" not in prompts:
    #     prompts["question"] = ChatPromptTemplate.from_template(templates["question"])

    if "helper" not in prompts:
        # prompts["helper"] = ChatPromptTemplate.from_template(templates["helper"])
        # prompts["helper"] = ChatPromptTemplate.from_template(templates["helper"])
        init_chain("helper", llm="chat", templates=templates, init_llm=False)

    if "hyde" not in prompts:
        init_chain("hyde", llm="chat", templates=templates, init_llm=False)
        # prompts["hyde"] = PromptTemplate(
        #     template=templates["hyde"],
        #     input_variables=["question"],
        #     # template=templates["hyde"], input_variables=["question"]
        # )

    global embeddings

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if "base" not in embeddings:
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {"allow_download": "True"}
        embeddings["base"] = GPT4AllEmbeddings(
            model_name=model_name, gpt4all_kwargs=gpt4all_kwargs
        )

    if "hyde" not in embeddings:
        embeddings["hyde"] = HypotheticalDocumentEmbedder.from_llm(
            llms["default"], embeddings["base"], custom_prompt=prompts["hyde"]
        )

    # journey_text_prompt = PromptTemplate(
    #     template=templates["journey_text"], input_variables=["context"]
    # )

    if "summary_documents" not in chains:
        chain = StuffDocumentsChain(
            llm_chain=chains["summary"],
            document_prompt=PromptTemplate(
                input_variables=["page_content"], template="{page_content}"
            ),
            document_variable_name="context",
            verbose=True,
        )
        chains["summary_documents"] = ReduceDocumentsChain(
            combine_documents_chain=chain, token_max=CONTEXT_SIZE
        )

    if "reduce_journey_documents" not in chains:
        chains["reduce_journey_documents"] = StuffDocumentsChain(
            llm_chain=chains["text_formatter_compress"],
            document_prompt=PromptTemplate(
                input_variables=["page_content"], template="{page_content}"
            ),
            document_variable_name="context",
            verbose=True,
        )
    chains["reduce_journey_documents"].verbose = True

    if "journey_json" not in chains:
        chains["journey_json"] = create_extraction_chain(
            journey_json_template,
            llms["json"],
        )


chroma_client = None


def create_document_lists(list_of_strings, source="local"):
    doc_list = []

    for item in list_of_strings:
        doc_list.append(Document(page_content=item, metadata={"source": source}))

    return doc_list


def get_chroma_collection(name, update=False, path=CHROMA_PATH) -> chromadb.Collection:
    global chroma_client
    chroma_client = chroma_client or chromadb.PersistentClient(path=path)

    if update:
        chroma_client.delete_collection(name=name)

    print(f"Get collection {name}")

    return chroma_client.get_or_create_collection(name)


def get_vectorstore(id, embedding_id="base") -> Chroma:
    init_llms()

    global embeddings

    return Chroma(
        client=chroma_client,
        collection_name=id,
        embedding_function=embeddings[embedding_id],
    )


@cache
def get_text_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


compressor = None


def rerank_documents(list_of_documents: list[Document], query: str, amount=5):
    global compressor

    if len(list_of_documents) > 0:
        if compressor is None:
            compressor = FlashrankRerank(top_n=amount)

        ranked_documents = compressor.compress_documents(
            documents=list_of_documents, query=query
        )
        return ranked_documents

    return list_of_documents


def split_text(text, split=CHAR_LIMIT, overlap=0):
    splitter = get_text_splitter(chunk_size=split, chunk_overlap=overlap)
    return splitter.split_text(text)


def join_documents(texts, split=CHAR_LIMIT):
    joins = []
    text_join = ""

    total_len = 0
    for text in texts:
        _text = ""
        if isinstance(text, str):
            _text = text
        else:
            _text = text.page_content

        total_len += len(_text)

    chunks = total_len // split + 1
    chunk_length = total_len // chunks

    # print(f"{len(texts) = }, {chunks = }, { chunk_length = }")

    for text in texts:
        _text = ""
        if isinstance(text, str):
            _text = text
        else:
            _text = text.page_content

        if len(text_join) > 100 and (len(text_join) + len(_text)) > chunk_length:
            joins.append(text_join)
            text_join = _text
        else:
            text_join += _text + "\n\n"

    joins.append(text_join)

    return joins


def semantic_splitter(text, split=CHAR_LIMIT):
    init_llms()

    if len(text) > 1000:
        print("Split")
        less_text = split_text(text, 1000)
    else:
        less_text = [text]

    semantic_splitter = SemanticChunker(
        embeddings["base"], breakpoint_threshold_type="percentile"
    )

    texts = []
    for txt in less_text:
        texts = texts + semantic_splitter.split_text(txt)

    return join_documents(texts, split)


def split_markdown(text, split=CHAR_LIMIT):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    texts = markdown_splitter.split_text(text)

    return join_documents(texts, split)


# chroma_collection = get_chroma_collection("rag-all")
