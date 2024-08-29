from typing import Dict, Literal, Union
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.language_models.llms import BaseLLM
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_core.embeddings import Embeddings
from chains.base import BaseChain
from chains.base import drop_thoughts
from chains.chain import Chain
from lib.load_env import (
    DEBUGMODE,
    SETTINGS,
    EmbeddingModelSettings,
    EmbeddingProviderSettings,
    ProviderModelSettings,
    ProviderSettings,
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


# RATE_LIMITER = InMemoryRateLimiter(
#     requests_per_second=RATE_LIMIT_PER_SECOND,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
#     check_every_n_seconds=RATE_LIMIT_INTEVAL,  # 0.1 <-- Wake up every 100 ms to check whether allowed to make a request,
#     max_bucket_size=1,  # Controls the maximum burst size.
# )
# RATE_LIMITER_DETAILED = InMemoryRateLimiter(
#     requests_per_second=RATE_LIMIT_PER_SECOND,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
#     check_every_n_seconds=RATE_LIMIT_INTEVAL,  # 0.1 <-- Wake up every 100 ms to check whether allowed to make a request,
#     max_bucket_size=1,  # Controls the maximum burst size.
# )
CHAT_RATE_LIMITER = None

limiters: Dict[str, BaseRateLimiter] = {}
llms: Dict[str, BaseLLM] = {}


def get_limiter(model_config: ProviderModelSettings) -> Union[BaseRateLimiter, None]:
    id = f"{model_config.provider}_{model_config.type}"
    if id in limiters.keys():
        return limiters[id]

    # llm_config: ProviderSettings = next(
    #     (config for config in SETTINGS.llms if config.type == llm), None
    # )
    # model_config: ProviderModelSettings = (
    #     next((config for config in llm_config.models if config.type == llm_model), None)
    #     if llm_config
    #     else None
    # )

    limiter = (
        InMemoryRateLimiter(
            requests_per_second=model_config.ratelimit_per_sec,
            check_every_n_seconds=model_config.ratelimit_interval,
            max_bucket_size=model_config.ratelimit_bucket,
        )
        if model_config
        else None
    )

    limiters[id] = limiter

    return limiter


def init_llm(
    # llm_model: str = SETTINGS.default_llms.chat.llm,
    # llm_ctx_size: int = SETTINGS.default_llms.chat.context_size,
    # temperature: float = 0.5,
    # llm_structured: bool = False,
    # llm_type: BaseLLM = SETTINGS.default_provider.class_model,
    # llm_ratelimiter: BaseRateLimiter = get_limiter(SETTINGS.default_provider.type, SETTINGS.default_llms.chat.type),
    provider: ProviderSettings = SETTINGS.default_provider,
    model: ProviderModelSettings = SETTINGS.default_llms.default,
    temperature=0.5,
    debug_mode: bool = DEBUGMODE,
) -> BaseLLM:
    print(
        f"Initializing llm: {model.model=} with {model.context_size=} and {temperature=}..."
    )

    common_kwargs = {
        "verbose": debug_mode,
        "temperature": temperature,
        "rate_limiter": get_limiter(model),
        "callback_manager": (
            CallbackManager([StreamingStdOutCallbackHandler()]) if debug_mode else None
        ),
    }

    if model.provider == "BEDROCK":
        llm = model.class_model(
            model_id=model.model,
            region_name=provider.region,
            model_kwargs={"temperature": temperature},
            **common_kwargs,
        )

    if model.provider == "AZURE":
        llm = model.class_model(
            azure_deployment=model.model,
            api_version=provider.api_version,
            model_kwargs=(
                {"response_format": {"type": "json_object"}}
                if "structured" in model.type
                else {}
            ),
            **common_kwargs,
        )

    if model.provider == "AZURE_ML":
        llm = model.class_model(
            endpoint_url=model.endpoint,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=model.api_key,
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs={"temperature": temperature},
            **common_kwargs,
        )

    if model.provider == "OLLAMA":
        llm = model.class_model(
            base_url=model.url,
            model=model.model,
            model_kwargs=(
                {"response_format": {"type": "json_object"}}
                if "structured" in model.type
                else {}
            ),
            num_ctx=model.context_size,
            num_predict=model.context_size,
            repeat_penalty=2,
            timeout=10000,
            **common_kwargs,
        )

    if model.provider == "GROQ":
        llm = model.class_model(
            streaming=debug_mode,
            api_key=model.api_key,
            model=model.model,
            model_kwargs=(
                {"response_format": {"type": "json_object"}}
                if "structured" in model.type
                else {}
            ),
            timeout=10000,
            max_retries=5,
            **common_kwargs,
        )

    return llm


# LLM_CONFIGS = {
#     "instruct": {
#         "temperature": 0.2,
#         "model": INSTRUCT_LLM,
#         "ctx_size": INSTRUCT_CONTEXT_SIZE,
#     },
#     "instruct_detailed_0": {
#         "temperature": 0,
#         "model": INSTRUCT_DETAILED_LLM,
#         "ctx_size": INSTRUCT_DETAILED_CONTEXT_SIZE,
#         "rate_limiter": RATE_LIMITER_DETAILED,
#     },
#     "instruct_detailed_warm": {
#         "temperature": 0.5,
#         "model": INSTRUCT_DETAILED_LLM,
#         "ctx_size": INSTRUCT_DETAILED_CONTEXT_SIZE,
#         "rate_limiter": RATE_LIMITER_DETAILED,
#     },
#     "structured": {
#         "temperature": 0.2,
#         "model": STRUCTURED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#     },
#     "structured_0": {
#         "temperature": 0,
#         "model": STRUCTURED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#     },
#     "structured_detailed": {
#         "temperature": 0.2,
#         "model": STRUCTURED_DETAILED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#         "rate_limiter": RATE_LIMITER_DETAILED,
#     },
#     "structured_detailed_0": {
#         "temperature": 0,
#         "model": STRUCTURED_DETAILED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#         "rate_limiter": RATE_LIMITER_DETAILED,
#     },
#     "json": {
#         "temperature": 0,
#         "model": STRUCTURED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#         "structured": True,
#     },
#     "json_detailed": {
#         "temperature": 0,
#         "model": STRUCTURED_DETAILED_LLM,
#         "ctx_size": STRUCTURED_CONTEXT_SIZE,
#         "structured": True,
#         "rate_limiter": RATE_LIMITER_DETAILED,
#     },
# }

temperature_map = {"default": 0.2, "zero": 0, "warm": 0.5}


def get_llm(
    id: str = "default",
    provider: Literal[
        "OLLAMA", "GROQ", "BEDROCK", "OPENAI", "ANTHROPIC", "AZURE", "AZURE_ML", None
    ] = None,
    temperature: Literal["default", "zero", "warm", None] = None,
) -> BaseLLM:
    global llms
    global SETTINGS

    if id in llms:
        return llms[id]

    llm_type = id

    if temperature is None:
        if "_0" in id:
            temperature = "zero"
            llm_type.replace("_0", "")
        elif "_warm" in id:
            temperature = "warm"
            llm_type.replace("_warm", "")
        else:
            temperature = "default"

    temperature_value = temperature_map[temperature]

    llm_config = None
    provider_config = SETTINGS.default_provider
    if provider is not None:
        provider_config = (
            next((config for config in SETTINGS.llms if config.type == provider), None)
            or provider_config
        )

    if provider_config is not None:
        llm_config = (
            next(
                (
                    config
                    for config in provider_config.models
                    if config.type == llm_type
                ),
                None,
            )
            if provider_config
            else None
        )

    if llm_config is None:
        llm_config = (
            SETTINGS.default_llms.__getattribute__(llm_type)
            if hasattr(SETTINGS.default_llms, llm_type)
            else None
        )
        if llm_config is None:
            llm_config = SETTINGS.default_llms.default

    llms[id] = init_llm(
        provider=provider_config, model=llm_config, temperature=temperature_value
    )

    return llms[id]


def init_chain(
    id,
    prompt: PromptFormatter,
    retry_id: str = None,
    validate_id: str = "instruct_detailed",
    check_for_hallucinations=False,
    ChainType: BaseChain = Chain,
) -> BaseChain:
    if retry_id is None:
        retry_id = "json" if id == "json" else "instruct_detailed"

    return ChainType(
        llm=get_llm(id),
        retry_llm=get_llm(retry_id),
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
        chains[chain] = init_chain(llm_id, prompt, check_for_hallucinations=check_for_hallucinations)
        return chains[chain]

    if "stuff_documents" == chain:
        base_chain = get_base_chain("text_formatter_compress")

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
    return (
        get_base_chain(chain)(custom_prompt)
        if isinstance(get_base_chain(chain), BaseChain)
        else get_base_chain(chain)
    )


def init_embeddings(
    embedding_provider: EmbeddingProviderSettings = SETTINGS.default_embedding_provider,
    embedding_model: EmbeddingModelSettings = SETTINGS.default_embeddings.default,
) -> Embeddings:
    if embedding_provider.type == "LOCAL":
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return embedding_provider.class_model(
            model_name=embedding_model.model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif embedding_provider.type == "OLLAMA":
        return embedding_provider.class_model(
            model=embedding_model.model, base_url=embedding_provider.url
        )
    elif embedding_provider.type == "HUGGINGFACE":
        return embedding_provider.class_model(
            api_key=embedding_provider.api_key, model_name=embedding_model.model
        )


embeddings: Dict[str, Embeddings] = {}


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
