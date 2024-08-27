import re
import textwrap
import time
from typing import Dict, List

from groq import RateLimitError
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_huggingface import (
    # HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.language_models.llms import BaseLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import (
    RunnableLambda,
    # Runnable,
)


from chains.base import BaseChain
from chains.rag_chain import question

from chains.chain import Chain
from lib.helpers import print_params
from lib.load_env import (
    CHAT_CONTEXT_SIZE,
    CHAT_LLM,
    DEBUGMODE,
    DEFAULT_LLM_MODEL,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    HF_API_KEY,
    INSTRUCT_CONTEXT_SIZE,
    INSTRUCT_DETAILED_CONTEXT_SIZE,
    INSTRUCT_DETAILED_LLM,
    INSTRUCT_LLM,
    OLLAMA_URL,
    RATE_LIMIT_INTEVAL,
    RATE_LIMIT_PER_SECOND,
    STRUCTURED_CONTEXT_SIZE,
    STRUCTURED_DETAILED_LLM,
    STRUCTURED_LLM,
    TESTER_APIKEY,
    TESTER_LLM,
    TESTER_LLM_MODEL,
    TOOL_CONTEXT_SIZE,
    TOOL_LLM,
    USE_AZURE,
    USE_BEDROCK,
    USE_GROQ,
    USE_HF_EMBEDDINGS,
    USE_LOCAL_EMBEDDINGS,
    USE_OLLAMA,
    USE_OLLAMA_EMBEDDINGS,
)
from chains.prompts import (
    PromptFormatter,
    text_formatter,
    text_formatter_compress,
    text_formatter_guided,
    md_formatter,
    md_formatter_guided,
    action,
    check,
    question_classifier,
    helper,
    chat,
    grader,
    hyde,
    hyde_document,
    summary,
    summary_guided,
    journey_steps,
    journey_step_details,
    journey_step_intro,
    journey_step_actions,
    journey_step_action_details,
    journey_structured,
)

from langchain_core.rate_limiters import InMemoryRateLimiter

RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=RATE_LIMIT_PER_SECOND,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=RATE_LIMIT_INTEVAL,  # 0.1 <-- Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=1,  # Controls the maximum burst size.
)
RATE_LIMITER_DETAILED = InMemoryRateLimiter(
    requests_per_second=RATE_LIMIT_PER_SECOND,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=RATE_LIMIT_INTEVAL,  # 0.1 <-- Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=1,  # Controls the maximum burst size.
)
CHAT_RATE_LIMITER = None

llms:Dict[str, BaseLLM] = {}
embeddings = {}

from typing import Dict, Union, Callable

chains: Dict[str, Union[BaseChain, Callable]] = {}

def get_chain(chain) -> BaseChain:
    if len(chains.keys()) == 0:
        init_llms()

    return chains[chain]


def get_llm(llm_id="default"):
    if len(llms.keys()) == 0:
        init_llms()

    if llm_id in llms.keys():
        llm = llms[llm_id]
    else:
        llm = llms["default"]

    return llm


def get_embeddings(embedding_id="base"):
    if len(embeddings.keys()) == 0:
        init_llms()

    if embedding_id in embeddings.keys():
        return embeddings[embedding_id]

def init_chain(id, prompt:PromptFormatter, check_for_hallucinations=False) -> Chain:
    return Chain(llm=llms[id], llm_id=id, prompt=prompt, validation_llm=llms["instruct_detailed"] if check_for_hallucinations else None)


def init_llm(
    id,
    model=CHAT_LLM,
    temperature=0.2,
    verbose=DEBUGMODE,
    Type=DEFAULT_LLM_MODEL,
    ctx_size=CHAT_CONTEXT_SIZE,
    rate_limiter=RATE_LIMITER,
    structured=False
):
    if f"{id}" not in llms:
        print(f"Initialize llm {id}: {model=} with {ctx_size=} and {temperature=}...")
        if USE_BEDROCK:
            from lib.load_env import BEDROCK_REGION
            llms[f"{id}"] = Type(
                model_id=model,
                region_name=BEDROCK_REGION,
                verbose=verbose,
                model_kwargs={"temperature": temperature},
                # num_ctx=ctx_size,
                # num_predict=ctx_size,
                # repeat_penalty=2,
                # timeout=10000,
                rate_limiter=rate_limiter,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if verbose
                    else None
                ),
            )
        if USE_AZURE:
            from lib.load_env import AZURE_API_VERSION
            llms[f"{id}"] = Type(
                azure_deployment=model,
                api_version=AZURE_API_VERSION,
                verbose=verbose,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}} if structured else {},
                # num_ctx=ctx_size,
                # num_predict=ctx_size,
                # repeat_penalty=2,
                # timeout=10000,
                rate_limiter=rate_limiter,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if verbose
                    else None
                ),
            )

        if USE_OLLAMA:
            llms[f"{id}"] = Type(
                base_url=OLLAMA_URL,
                model=model,
                verbose=verbose,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}} if structured else {},
                num_ctx=ctx_size,
                num_predict=ctx_size,
                repeat_penalty=2,
                timeout=10000,
                rate_limiter=rate_limiter,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if verbose
                    else None
                ),
            )
        if USE_GROQ:
            llms[f"{id}"] = Type(
                streaming=verbose,
                api_key=GROQ_API_KEY,
                model=model,
                model_kwargs={"response_format": {"type": "json_object"}} if structured else {},
                verbose=verbose,
                temperature=temperature,
                timeout=10000,
                max_retries=5,
                rate_limiter=rate_limiter,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if verbose
                    else None
                ),
            )


initialized = False


def init_llms():
    print("Initialize llms...")
    global initialized
    initialized = True

    init_llm("default")
    init_llm(
        "instruct", temperature=0.2, model=INSTRUCT_LLM, ctx_size=INSTRUCT_CONTEXT_SIZE
    )
    init_llm(
        "instruct_0", temperature=0, model=INSTRUCT_LLM, ctx_size=INSTRUCT_CONTEXT_SIZE
    )
    init_llm(
        "instruct_warm",
        temperature=0.5,
        model=INSTRUCT_LLM,
        ctx_size=INSTRUCT_CONTEXT_SIZE,
    )
    init_llm(
        "instruct_detailed",
        temperature=0.2,
        model=INSTRUCT_DETAILED_LLM,
        ctx_size=INSTRUCT_DETAILED_CONTEXT_SIZE,
        rate_limiter=RATE_LIMITER_DETAILED
    )
    init_llm(
        "instruct_detailed_0",
        temperature=0,
        model=INSTRUCT_DETAILED_LLM,
        ctx_size=INSTRUCT_DETAILED_CONTEXT_SIZE,
        rate_limiter=RATE_LIMITER_DETAILED
    )
    init_llm(
        "instruct_detailed_warm",
        temperature=0.5,
        model=INSTRUCT_DETAILED_LLM,
        ctx_size=INSTRUCT_DETAILED_CONTEXT_SIZE,
        rate_limiter=RATE_LIMITER_DETAILED
    )
    init_llm(
        "structured", temperature=0.2, model=STRUCTURED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE
    )
    init_llm(
        "structured_0", temperature=0, model=STRUCTURED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE
    )

    init_llm(
        "structured_detailed", temperature=0.2, model=STRUCTURED_DETAILED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE,
            rate_limiter=RATE_LIMITER_DETAILED
    )
    init_llm(
        "structured_detailed_0", temperature=0, model=STRUCTURED_DETAILED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE,
            rate_limiter=RATE_LIMITER_DETAILED
    )

    init_llm(
        "json", temperature=0, model=STRUCTURED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE, structured=True
    )

    init_llm(
        "json_detailed", temperature=0, model=STRUCTURED_DETAILED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE, structured=True,
            rate_limiter=RATE_LIMITER_DETAILED
    )

    if "tester" not in llms:
        if USE_AZURE:
            from langchain_community.chat_models.azureml_endpoint import (
                AzureMLEndpointApiType,
                CustomOpenAIChatContentFormatter,
            )
            tester = TESTER_LLM_MODEL(
                endpoint_url=TESTER_LLM,
                endpoint_api_type=AzureMLEndpointApiType.serverless,
                endpoint_api_key=TESTER_APIKEY,
                content_formatter=CustomOpenAIChatContentFormatter(),
                model_kwargs={"temperature": 0.2},
            )
            llms["tester"] = tester
        else:
            llms["tester"] = llms["structured_0"]

    init_llm("tool", temperature=0, model=TOOL_LLM, ctx_size=TOOL_CONTEXT_SIZE)
    init_llm("chat", temperature=0.5, model=CHAT_LLM, ctx_size=CHAT_CONTEXT_SIZE, rate_limiter=CHAT_RATE_LIMITER)
    init_llm("warm", temperature=0.7, model=CHAT_LLM, ctx_size=CHAT_CONTEXT_SIZE, rate_limiter=CHAT_RATE_LIMITER)

    chains["summary"] = init_chain(
        "instruct_detailed", summary, check_for_hallucinations=True
    )  # init_chain("summary", "instruct_detailed", summary)
    chains["summary_guided"] = init_chain("instruct_detailed", summary_guided, check_for_hallucinations=True)
    chains["action"] = init_chain("instruct_0", action)
    chains["grader"] = init_chain("json", grader)
    chains["check"] = init_chain("instruct_0", check)
    chains["text_formatter"] = init_chain("instruct_detailed", text_formatter, check_for_hallucinations=True)
    chains["text_formatter_compress"] = init_chain(
        "instruct_detailed", text_formatter_compress, check_for_hallucinations=True
    )
    chains["text_formatter_guided_0"] = init_chain(
        "instruct_detailed", text_formatter_guided, check_for_hallucinations=True
    )
    chains["md_formatter"] = init_chain("instruct_detailed", md_formatter, check_for_hallucinations=True)
    chains["md_formatter_guided"] = init_chain("instruct_detailed_0", md_formatter_guided, check_for_hallucinations=True)
    chains["journey_structured"] = init_chain("json", journey_structured)
    chains["journey_steps"] = init_chain("json_detailed", journey_steps, check_for_hallucinations=True)
    chains["journey_step_details"] = init_chain("instruct_detailed_warm", journey_step_details, check_for_hallucinations=True)
    chains["journey_step_intro"] = init_chain("instruct_warm", journey_step_intro, check_for_hallucinations=True)
    chains["journey_step_actions"] = init_chain(
        "instruct_detailed", journey_step_actions, check_for_hallucinations=True
    )
    chains["journey_step_action_details"] = init_chain(
        "instruct_warm", journey_step_action_details, check_for_hallucinations=True
    )
    chains["question"] = init_chain("chat", question, check_for_hallucinations=True)
    chains["helper"] = init_chain("chat", helper)
    chains["chat"] = init_chain("chat", chat)

    global embeddings

    if "base" not in embeddings:
        if USE_LOCAL_EMBEDDINGS:
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings["base"] = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif USE_OLLAMA_EMBEDDINGS:
            embeddings["base"] = OllamaEmbeddings(
                model=EMBEDDING_MODEL, base_url=OLLAMA_URL
            )
        elif USE_HF_EMBEDDINGS:
            embeddings["base"] = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_API_KEY, model_name=EMBEDDING_MODEL
            )

    if "hyde" not in embeddings:
        embeddings["hyde"] = HypotheticalDocumentEmbedder.from_llm(
            llms["tester"],
            embeddings["base"],
            custom_prompt=hyde.get_chat_prompt_template(),  # prompts["hyde"]
        )

    if "hyde_document" not in embeddings:
        embeddings["hyde_document"] = HypotheticalDocumentEmbedder.from_llm(
            llms["tester"],
            embeddings["base"],
            custom_prompt=hyde_document.get_chat_prompt_template(),  # prompts["hyde"]
        )

    def compress_doc(params):
        resp = chains["text_formatter_compress"]().invoke(params)
        if isinstance(resp, tuple):
            return resp[0]
        return resp

    if "summary_documents" not in chains:

        chain = create_stuff_documents_chain(
            RunnableLambda(compress_doc),
            chains[
                "text_formatter_compress"
            ].prompt.get_chat_prompt_template(),  # prompts["text_formatter_compress"],
            output_parser=text_formatter_compress.parser,
        )

        chains["summary_documents"] = lambda: (
            RunnableLambda(lambda params: chain.invoke(params)[0]) | chains["summary"]()
        )

    if "reduce_journey_documents" not in chains:
        chains["reduce_journey_documents"] = lambda: create_stuff_documents_chain(
            RunnableLambda(compress_doc),
            chains[
                "text_formatter_compress"
            ].prompt.get_chat_prompt_template(),  # prompts["text_formatter_compress"],
            output_parser=text_formatter_compress.parser,
        )
