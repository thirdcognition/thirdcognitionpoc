from functools import cache
import os
import random
import textwrap
from typing import List
import chromadb
from langchain_core.language_models.llms import BaseLLM
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
from langchain_groq import ChatGroq
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma

from langchain.globals import set_debug, set_verbose
from dotenv import load_dotenv

from prompts import (
    PromptFormatter,
    text_formatter,
    text_formatter_compress,
    text_formatter_guided,
    md_formatter,
    md_formatter_guided,
    action,
    check,
    helper,
    chat,
    question,
    hyde,
    summary,
    summary_guided,
    journey_steps,
    journey_step_details,
    journey_step_intro
)

# Load .env file from the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
print("Loading env: ", os.path.join(os.path.dirname(__file__), "..", ".env"))
# for key, value in os.environ.items():
#     print(f'\t{key}= {value}')

set_debug(os.getenv("DEBUG", False))
set_verbose(os.getenv("VERBOSE", True))

CHROMA_PATH = os.getenv("CHROMA_PATH", "db/chroma_db")
SQLITE_DB = os.getenv("SQLITE_DB", "db/files.db")

use_ollama = os.getenv("USE_OLLAMA", "True") == "True" or False
use_groq = os.getenv("USE_GROQ", "True") == "True" or False
DEFAULT_LLM_MODEL:BaseLLM = None

if use_ollama:
    DEFAULT_LLM_MODEL = ChatOllama
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    chat_llm = os.getenv("OLLAMA_CHAT_LLM", "phi3:mini")
    CHAT_CONTEXT_SIZE = int(os.getenv("OLLAMA_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("OLLAMA_CHAT_CHAR_LIMIT", 1024))

    instruct_llm = os.getenv("OLLAMA_INSTRUCT_LLM", "phi3:instruct")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("OLLAMA_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("OLLAMA_INSTRUCT_CHAR_LIMIT", 1024))

    tool_llm = os.getenv("OLLAMA_TOOL_LLM", "phi3:instruct")
    TOOL_CONTEXT_SIZE = int(os.getenv("OLLAMA_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("OLLAMA_TOOL_CHAR_LIMIT", 1024))

    print(f"Ollama: {chat_llm=} {CHAT_CONTEXT_SIZE=} {CHAT_CHAR_LIMIT}")
    print(f"Ollama: {instruct_llm=} {INSTRUCT_CONTEXT_SIZE=} {INSTRUCT_CHAR_LIMIT}")
    print(f"Ollama: {tool_llm=} {TOOL_CONTEXT_SIZE=} {TOOL_CHAR_LIMIT}")

if use_groq:
    DEFAULT_LLM_MODEL = ChatGroq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    chat_llm = os.getenv("GROQ_CHAT_LLM", "llama-3.1-8b-instant")
    CHAT_CONTEXT_SIZE = int(os.getenv("GROQ_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("GROQ_CHAT_CHAR_LIMIT", 1024))

    instruct_llm = os.getenv("GROQ_INSTRUCT_LLM", "llama-3.1-8b-instant")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("GROQ_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("GROQ_INSTRUCT_CHAR_LIMIT", 1024))

    tool_llm = os.getenv("GROQ_TOOL_LLM", "llama-3.1-8b-instant")
    TOOL_CONTEXT_SIZE = int(os.getenv("GROQ_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("GROQ_TOOL_CHAR_LIMIT", 1024))

    print(f"Groq: {chat_llm=} {CHAT_CONTEXT_SIZE=} {CHAT_CHAR_LIMIT}")
    print(f"Groq: {instruct_llm=} {INSTRUCT_CONTEXT_SIZE=} {INSTRUCT_CHAR_LIMIT}")
    print(f"Groq: {tool_llm=} {TOOL_CONTEXT_SIZE=} {TOOL_CHAR_LIMIT}")

llms = {}
embeddings = {}
prompts = {}
chains = {}

journey_json_template = {
    "properties": {
        "title": {
            "type": "string",
            "description": "Title of the class",
            "title": "Title",
        },
        "intro": {
            "type": "string",
            "description": "Introduction to the class",
            "title": "Intro",
        },
        "content": {
            "type": "string",
            "description": "Detailed content of the class",
            "title": "Content",
        },
        "actions": {
            "description": "List actions within the class.",
            "items": {"type": "string"},
            "title": "Actions",
            "type": "array",
        },
        "priority": {
            "type": "int",
            "description": "How important the class is",
            "title": "Priority",
        },
    },
    "required": ["name", "intro", "content"],
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


import re
from fuzzywuzzy import fuzz


def handle_thinking(text_provider, min_len=100, retry=True) -> tuple[str, str]:

    # Rest of the function remains the same
    # Define the tags to match
    start_tag = "thinking_start"
    end_tag = "thinking_end"

    # Initialize the lists to store the thinking contents and remaining contents
    thinking_contents = []
    text_contents = []

    text_contents_joined = ""
    thinking_contents_joined = ""

    if isinstance(text_provider, str):
        text = text_provider
    else:
        text = text_provider()

#     print(f"""Provider response:
# === RESPONSE START ===

# Length: {len(text)}
# {text}

# === RESPONSE END ===
# """)

    # Split the text into matches using regex
    matches = re.split(r"([\[{\(][/ ]*[^\[\]]+[\]}\)])", text, 0, re.IGNORECASE)

    # Initialize a flag to indicate whether we're inside a thinking block
    in_thinking = False

    # print(f"Start matching fors {len(matches)} matches")

    # Iterate over the matches
    for match in matches:
        match=match.strip()
        # print(f"Match: {match}")
        # If the match is a start tag, set the flag to True
        if fuzz.ratio(match, start_tag) > 80:
            in_thinking = True
        # If the match is an end tag, set the flag to False
        elif fuzz.ratio(match, end_tag) > 80:
            in_thinking = False
        # If we're inside a thinking block, add the match to the thinking contents
        elif in_thinking:
            thinking_contents.append(match)
        # Otherwise, add the match to the text contents
        else:
            text_contents.append(match)

    text_contents_joined = "\n".join(text_contents)
    thinking_contents_joined = "\n".join(thinking_contents)

    if retry and not isinstance(text_provider, str) and (
        text_contents_joined == "" or len(text_contents_joined) < min_len
    ):
        max_retries = 5
        try_nmb = 0
        while text_contents_joined == "" or (len(text_contents_joined) < min_len and try_nmb < max_retries):
            try_nmb += 1
            text_contents_joined, thinking_contents_joined = handle_thinking(
                text_provider,
                min_len=min_len,
                retry=False
            )

    return text_contents_joined, thinking_contents_joined

def verify_step_result(get_result, amount: int, format:str = None) -> str:
    if format is None:
        format = """
Format for 5 items:
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
"""

    max_retries = 3

    success = False
    steps: str = None
    retries = 0
    while not success and retries < max_retries:
        retries += 1

        steps = get_result()
        steps = re.sub(r':\s*\n', ': ', steps)
        steps = re.sub(r'\n\s*:', ':', steps)
        steps = "\n".join([step.strip() for step in steps.split("\n") if step.strip()])
        correct_response = False
        resp_retr = 0
        while(not correct_response and resp_retr < max_retries):
            resp_retr += 1

            check_response = (get_chain("check").invoke({
                "context": steps,
                "options": "if matches the format respond: yes, if matches the format but not right amount of items respond: maybe, if does not match respond: no",
                "expected_count": f"Expected approximately {amount} items.",
                "count": len(steps.split("\n")),
                "format": format

            })["text"])
            resp = check_response.lower().split('\n')[0].strip()
            # print(f"{resp = }")
            correct_response = resp in ["yes", "y", "no", "n", "maybe", "m"]
            success = resp in ["yes", "y", "maybe", "m"]

    return steps

def init_chain(
    id, llm="default", prompt: PromptFormatter = text_formatter, init_llm=True
):  # templates_mistral
    if f"{id}" not in prompts:
        prompts[f"{id}"] = (
            prompt.get_chat_prompt_template()
        )  # ChatPromptTemplate.from_messages(messages)

    if f"{id}" not in chains and init_llm:
        chains[f"{id}"] = LLMChain(llm=llms[llm], prompt=prompts[f"{id}"])


def init_llm(
    id,
    model=chat_llm,
    temperature=0.2,
    verbose=True,
    Type=DEFAULT_LLM_MODEL,
    ctx_size=CHAT_CONTEXT_SIZE,
):
    if f"{id}" not in llms:
        print(f"Initialize llm {id}: {model=} with {ctx_size=} and {temperature=}...")
        if use_ollama:
            llms[f"{id}"] = Type(
                base_url=ollama_url,
                model=model,
                verbose=verbose,
                temperature=temperature,
                num_ctx=ctx_size,
                num_predict=ctx_size,
                repeat_penalty=1.5,
                timeout=20 * 1000,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        else:
            llms[f"{id}"] = Type(
                streaming=True,
                api_key=GROQ_API_KEY,
                model=model,
                verbose=verbose,
                temperature=temperature,
                timeout=30000,
                max_retries=5,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )


initialized = False


def init_llms():
    print("Initialize llms...")
    initialized = True

    init_llm("default")
    init_llm(
        "instruct", temperature=0.2, model=instruct_llm, ctx_size=INSTRUCT_CONTEXT_SIZE
    )
    init_llm(
        "instruct_0", temperature=0, model=instruct_llm, ctx_size=INSTRUCT_CONTEXT_SIZE
    )
    init_llm(
        "instruct_warm", temperature=0.5, model=instruct_llm, ctx_size=INSTRUCT_CONTEXT_SIZE
    )

    # init_llm("json", Type=OllamaFunctions)

    if "json" not in llms:
        if use_ollama:
            llms["json"] = DEFAULT_LLM_MODEL(
                format="json",
                # model=llama3_llm,
                base_url=ollama_url,
                model=tool_llm,
                temperature=0,
                num_ctx=TOOL_CONTEXT_SIZE,
                num_predict=TOOL_CONTEXT_SIZE,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        if use_groq:
            llms["json"] = DEFAULT_LLM_MODEL(
                api_key=GROQ_API_KEY,
                format="json",
                model=tool_llm,
                temperature=0,
                timeout=30000,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )

    init_llm("tool", temperature=0, model=tool_llm, ctx_size=TOOL_CONTEXT_SIZE)

    init_llm("chat", temperature=0.5, Type=DEFAULT_LLM_MODEL)

    init_llm("warm", temperature=0.4)

    init_chain("summary", "instruct", summary)
    init_chain("summary_guided", "instruct", summary_guided)
    init_chain(
        "action",
        "instruct_0",
        action,
    )
    init_chain(
        "check",
        "instruct_0",
        check,
    )
    init_chain("text_formatter", "instruct", text_formatter)
    init_chain("text_formatter_compress", "instruct", text_formatter_compress)
    init_chain(
        "text_formatter_guided_0",
        "instruct",
        text_formatter_guided,
    )
    init_chain("md_formatter", "instruct", md_formatter)
    init_chain(
        "md_formatter_guided",
        "instruct_0",
        md_formatter_guided,
    )
    init_chain(
        "journey_steps",
        "instruct",
        journey_steps,
    )
    init_chain(
        "journey_step_details",
        "instruct_warm",
        journey_step_details,
    )
    init_chain(
        "journey_step_intro",
        "instruct_warm",
        journey_step_intro
    )
    init_chain("question", "chat", chat)

    # template = """<s> <<SYS>> Act as a a helpful startup coach from antler trying to answer questions thoroughly. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> </s>

    # if "question" not in prompts:
    #     prompts["question"] = ChatPromptTemplate.from_template(templates["question"])

    if "helper" not in prompts:
        # prompts["helper"] = ChatPromptTemplate.from_template(templates["helper"])
        # prompts["helper"] = ChatPromptTemplate.from_template(templates["helper"])
        init_chain("helper", "chat", helper, init_llm=False)

    if "hyde" not in prompts:
        init_chain("hyde", "chat", hyde, init_llm=False)
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
            llm_chain=chains["text_formatter_compress"],
            document_prompt=PromptTemplate(
                input_variables=["page_content"], template="{page_content}"
            ),
            document_variable_name="context",
            verbose=True,
        )
        chains["summary_documents"] = ReduceDocumentsChain(
            combine_documents_chain=chain
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


def create_document_lists(
    list_of_strings: List[str], list_of_thoughts: List[str] = None, source="local"
):
    doc_list = []

    for index, item in enumerate(list_of_strings):
        thinking = list_of_thoughts[index] if list_of_thoughts else None
        if len(item) > 3000:
            split_texts = split_text(item, split=3000, overlap=100)
            for split_item in split_texts:
                doc = Document(
                    page_content=split_item,
                    metadata={"source": source, "thoughts": thinking, "index": index},
                )
                doc_list.append(doc)
        else:
            doc = Document(
                page_content=item,
                metadata={"source": source, "thoughts": thinking, "index": index},
            )
            doc_list.append(doc)

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


def split_text(text, split=INSTRUCT_CHAR_LIMIT, overlap=100):
    text_len = len(text)
    split = text_len // (text_len / split)
    if (text_len - split) > overlap:
        splitter = get_text_splitter(chunk_size=split, chunk_overlap=overlap)
        return splitter.split_text(text)
    else:
        return [text]


def join_documents(texts, split=INSTRUCT_CHAR_LIMIT):
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


def semantic_splitter(text, split=INSTRUCT_CHAR_LIMIT):
    init_llms()

    if len(text) > 1000:
        # print("Split")
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


def split_markdown(text, split=INSTRUCT_CHAR_LIMIT):
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
