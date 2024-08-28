from typing import Dict, Union
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder

from langchain_core.embeddings import Embeddings
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.language_models.llms import BaseLLM
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from chains.base import BaseChain
from chains.base import drop_thoughts
from chains.chain import Chain
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
    question,
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

llms: Dict[str, BaseLLM] = {}

def init_llm(llm_model:str=CHAT_LLM, llm_ctx_size:int=CHAT_CONTEXT_SIZE, llm_temperature:float=0.5, llm_structured:bool = False, llm_type:BaseLLM=DEFAULT_LLM_MODEL, llm_ratelimiter:BaseRateLimiter = RATE_LIMITER, debug_mode:bool=DEBUGMODE) -> BaseLLM:
    print(f"Initializing llm: {llm_model=} with {llm_ctx_size=} and {llm_temperature=}...")

    common_kwargs = {
        "verbose": debug_mode,
        "temperature": llm_temperature,
        "rate_limiter": llm_ratelimiter,
        "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]) if debug_mode else None
    }

    if USE_BEDROCK:
        from lib.load_env import BEDROCK_REGION
        llm = llm_type(
            model_id=llm_model,
            region_name=BEDROCK_REGION,
            model_kwargs={"temperature": llm_temperature},
            **common_kwargs
        )

    if USE_AZURE:
        from lib.load_env import AZURE_API_VERSION
        llm = llm_type(
            azure_deployment=llm_model,
            api_version=AZURE_API_VERSION,
            model_kwargs={"response_format": {"type": "json_object"}} if llm_structured else {},
            **common_kwargs
        )

    if USE_OLLAMA:
        llm = llm_type(
            base_url=OLLAMA_URL,
            model=llm_model,
            model_kwargs={"response_format": {"type": "json_object"}} if llm_structured else {},
            num_ctx=llm_ctx_size,
            num_predict=llm_ctx_size,
            repeat_penalty=2,
            timeout=10000,
            **common_kwargs
        )

    if USE_GROQ:
        llm = llm_type(
            streaming=debug_mode,
            api_key=GROQ_API_KEY,
            model=llm_model,
            model_kwargs={"response_format": {"type": "json_object"}} if llm_structured else {},
            verbose=debug_mode,
            timeout=10000,
            max_retries=5,
            **common_kwargs
        )

    return llm

LLM_CONFIGS = {
    "instruct": {
        "temperature": 0.2,
        "model": INSTRUCT_LLM,
        "ctx_size": INSTRUCT_CONTEXT_SIZE,
    },
    "instruct_detailed_0": {
        "temperature": 0,
        "model": INSTRUCT_DETAILED_LLM,
        "ctx_size": INSTRUCT_DETAILED_CONTEXT_SIZE,
        "rate_limiter": RATE_LIMITER_DETAILED,
    },
    "instruct_detailed_warm": {
        "temperature": 0.5,
        "model": INSTRUCT_DETAILED_LLM,
        "ctx_size": INSTRUCT_DETAILED_CONTEXT_SIZE,
        "rate_limiter": RATE_LIMITER_DETAILED,
    },
    "structured": {
        "temperature": 0.2,
        "model": STRUCTURED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
    },
    "structured_0": {
        "temperature": 0,
        "model": STRUCTURED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
    },
    "structured_detailed": {
        "temperature": 0.2,
        "model": STRUCTURED_DETAILED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
        "rate_limiter": RATE_LIMITER_DETAILED,
    },
    "structured_detailed_0": {
        "temperature": 0,
        "model": STRUCTURED_DETAILED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
        "rate_limiter": RATE_LIMITER_DETAILED,
    },
    "json": {
        "temperature": 0,
        "model": STRUCTURED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
        "structured": True,
    },
    "json_detailed": {
        "temperature": 0,
        "model": STRUCTURED_DETAILED_LLM,
        "ctx_size": STRUCTURED_CONTEXT_SIZE,
        "structured": True,
        "rate_limiter": RATE_LIMITER_DETAILED,
    },
}

def get_llm(id) -> BaseLLM:
    global llms
    global LLM_CONFIGS
    if id in llms:
        return llms[id]

    if id in LLM_CONFIGS:
        llms[id] = init_llm(**LLM_CONFIGS[id])
    if "tester" == id:
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
            llms[id] = tester
        else:
            llms[id] = get_llm("structured_0")
    else:
        llms[id] = init_llm()

    return llms[id]

def init_chain(
    id,
    prompt: PromptFormatter,
    retry_id: str = None,
    validate_id: str = 'instruct_detailed',
    check_for_hallucinations=False,
    ChainType: BaseChain = Chain,
) -> BaseChain:
    return ChainType(
        llm=get_llm(id),
        retry_llm=get_llm(retry_id if retry_id is not None else id if id == "json" else "instruct_detailed"),
        prompt=prompt,
        validation_llm=get_llm(validate_id) if check_for_hallucinations else None,
    )

CHAIN_CONFIG: Dict[str, tuple[str, PromptFormatter, bool]] = {
    "summary": ("instruct_detailed", summary, True),
    "summary_guided": ("instruct_detailed", summary_guided, True),
    "action": ("instruct_0", action, False),
    "grader": ("json", grader, False),
    "check": ("instruct_0", check, False),
    "text_formatter": ("instruct_detailed", text_formatter, True),
    "text_formatter_compress": ("instruct_detailed", text_formatter_compress, True),
    "text_formatter_guided_0": ("instruct_detailed", text_formatter_guided, True),
    "md_formatter": ("instruct_detailed", md_formatter, True),
    "md_formatter_guided": ("instruct_detailed_0", md_formatter_guided, True),
    "journey_structured": ("json", journey_structured, False),
    "journey_steps": ("json_detailed", journey_steps, True),
    "journey_step_details": ("instruct_detailed_warm", journey_step_details, True),
    "journey_step_intro": ("instruct_warm", journey_step_intro, True),
    "journey_step_actions": ("instruct_detailed", journey_step_actions, True),
    "journey_step_action_details": ("instruct_warm", journey_step_action_details, True),
    "question": ("chat", question, True),
    "helper": ("chat", helper, False),
    "chat": ("chat", chat, False),
    "question_classification": ("tester", question_classifier, False),
}

chains: Dict[str, Union[BaseChain, RunnableSequence]] = {}

def get_base_chain(chain) -> Union[BaseChain, RunnableSequence]:
    global chains
    if chain in chains:
        return chains[chain]

    if chain in CHAIN_CONFIG:
        llm_id, prompt, check_for_hallucinations = CHAIN_CONFIG[chain]
        chains[chain] = init_chain(llm_id, prompt, check_for_hallucinations)
        return chains[chain]

    if "stuff_documents" == chain:
        base_chain = get_base_chain(
            "text_formatter_compress"
        )

        chains[chain] = create_stuff_documents_chain(
            base_chain() | drop_thoughts,
            base_chain.prompt.get_chat_prompt_template(),
            output_parser=base_chain.prompt.parser,
        )

    if "summary_documents" == chain:
        chains[chain] = (
            get_chain("stuff_documents") | drop_thoughts | get_chain("summary")
        )

    raise ValueError(f"Unknown chain: {chain}")

def get_chain(chain, custom_prompt: tuple[str, str] | None = None) -> RunnableSequence:
    return get_base_chain(chain)(custom_prompt) if isinstance(get_base_chain(chain), BaseChain) else get_base_chain(chain)

def init_embeddings(model:str=EMBEDDING_MODEL) -> Embeddings:
    if USE_LOCAL_EMBEDDINGS:
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif USE_OLLAMA_EMBEDDINGS:
        return OllamaEmbeddings(
            model=model, base_url=OLLAMA_URL
        )
    elif USE_HF_EMBEDDINGS:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_API_KEY, model_name=model
        )

embeddings:Dict[str, Embeddings] = {}

def get_embeddings(embedding_id):
    global embeddings
    if embedding_id in embeddings:
        return embeddings[embedding_id]

    if "base" == embedding_id:
        embeddings[embedding_id] = init_embeddings()
    if "hyde" == embedding_id:
        embeddings[embedding_id] = HypotheticalDocumentEmbedder.from_llm(
            get_llm("tester"),
            get_embeddings("base"),
            custom_prompt=hyde.get_chat_prompt_template(),
        )

    if "hyde_document" == embedding_id:
        embeddings["hyde_document"] = HypotheticalDocumentEmbedder.from_llm(
            get_llm("tester"),
            get_embeddings("base"),
            custom_prompt=hyde_document.get_chat_prompt_template(),
        )

    return embeddings[embedding_id]