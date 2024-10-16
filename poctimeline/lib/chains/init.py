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
from lib.chains.base import BaseChain
from lib.chains.chain import Chain
from lib.load_env import (
    DEBUGMODE,
    DEVMODE,
    SETTINGS,
    EmbeddingModelSettings,
    EmbeddingProviderSettings,
    ProviderModelSettings,
    ProviderSettings,
)
from lib.prompts.base import PromptFormatter
from lib.prompts.journey import (
    plan,
    module_intro,
    action_details,
    module_actions,
    module_content,
    journey_template_selector,
)
from lib.prompts.journey_structured import module_structured
from lib.prompts.actions import (
    action,
    summary,
    summary_guided,
    summary_with_title,
    question_classifier,
    check,
    grader,
    combine_description,
)
from lib.prompts.formatters import (
    text_formatter,
    text_formatter_simple,
    text_formatter_compress,
    text_formatter_guided,
    text_formatter_compress_guided,
    md_formatter,
    md_formatter_guided,
)
from lib.prompts.topics import (
    page_formatter,
    page_formatter_guided,
    topic_formatter,
    topic_formatter_guided,
    topic_hierarchy,
    topic_combiner
)
from lib.prompts.taxonomy import (
    taxonomy,
    taxonomy_hierarchy,
    taxonomy_combiner,
)
from lib.prompts.concepts import (
    concept_structured,
    concept_more,
    concept_hierarchy,
    concept_combiner,
)
from lib.prompts.chat import chat, question, helper
from lib.prompts.hyde import hyde, hyde_document
from lib.chains.prompt_generator import journey_prompts

CHAT_RATE_LIMITER = None

limiters: Dict[str, BaseRateLimiter] = {}
llms: Dict[str, BaseLLM] = {}


def get_limiter(model_config: ProviderModelSettings) -> Union[BaseRateLimiter, None]:
    id = f"{model_config.provider}_{model_config.type}"
    if id in limiters.keys():
        return limiters[id]

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

use_structured_mode = False

def init_llm(
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
            timeout=30000,
            max_tokens = model.max_tokens,
            max_retries = 2,
            **common_kwargs,
        )
        use_structured_mode = True

    if model.provider == "AZURE":
        llm = model.class_model(
            azure_deployment=model.model,
            api_version=provider.api_version,
            model_kwargs=(
                {"response_format": {"type": "json_object"}}
                if "structured" in model.type
                else {}
            ),
            timeout=60000,
            request_timeout=120,
            max_tokens = model.max_tokens,
            max_retries = 2,
            **common_kwargs,
        )

    if model.provider == "AZURE_ML":
        llm = model.class_model(
            endpoint_url=model.endpoint,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=model.api_key,
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs={"temperature": temperature},
            timeout=1000,
            max_tokens = model.max_tokens,
            max_retries = 2,
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
            timeout=30000,
            max_retries = 2,
            max_tokens = model.max_tokens,
            **common_kwargs,
        )

    if model.provider == "GROQ":
        llm = model.class_model(
            # streaming=debug_mode,
            api_key=model.api_key,
            model=model.model,
            model_kwargs=(
                {"response_format": {"type": "json_object"}}
                if "structured" in model.type
                else {}
            ),
            timeout=30000,
            max_retries=2,
            max_tokens = model.max_tokens,
            **common_kwargs,
        )

    return llm


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
        if hasattr(SETTINGS.default_llms, llm_type):
            llm_config = SETTINGS.default_llms.__getattribute__(llm_type)
        else:
            llm_config = SETTINGS.default_llms.default

    llms[id] = init_llm(
        provider=provider_config, model=llm_config, temperature=temperature_value
    )

    return llms[id]


def init_chain(
    id,
    prompt: PromptFormatter,
    retry_id: str = None,
    validate_id: str = "instruct" if DEVMODE else "instruct_detailed",
    check_for_hallucinations=False,
    ChainType: BaseChain = Chain,
) -> BaseChain:
    if retry_id is None:
        retry_id = "structured_detailed" if "structured" in id else "instruct_detailed"

    return ChainType(
        llm=get_llm(id),
        retry_llm=get_llm(retry_id),
        prompt=prompt,
        validation_llm=(
            get_llm(validate_id) if (check_for_hallucinations and not DEVMODE) else None
        ),
        async_mode=True
    )


CHAIN_CONFIG: Dict[str, tuple[str, PromptFormatter, bool]] = {
    "combine_bullets": ("instruct", combine_description, False),
    "summary": ("instruct_detailed" if not DEVMODE else "instruct", summary, True),
    "summary_guided": (
        "instruct_detailed" if not DEVMODE else "instruct",
        summary_guided,
        True,
    ),
    "summary_with_title": (
        "structured_detailed" if not DEVMODE else "structured",
        summary_with_title,
        True,
    ),
    "action": ("instruct_0", action, False),
    "grader": ("structured", grader, False),
    "check": ("instruct_0", check, False),
    "text_formatter_simple": ("instruct", text_formatter_simple, False),
    "text_formatter": ("instruct", text_formatter, False),
    "text_formatter_compress": ("instruct", text_formatter_compress, False),
    "text_formatter_guided": (
        "instruct_detailed_0" if not DEVMODE else "instruct_0",
        text_formatter_guided,
        True,
    ),
    "text_formatter_compress_guided": (
        "instruct_detailed_0" if not DEVMODE else "instruct_0",
        text_formatter_compress_guided,
        True,
    ),
    "md_formatter": ("instruct", md_formatter, False),
    "md_formatter_guided": (
        "instruct_detailed_0" if not DEVMODE else "instruct_0",
        md_formatter_guided,
        True,
    ),
    "page_formatter": ("instruct", page_formatter, True),
    "page_formatter_guided": (
        "instruct_detailed_0" if not DEVMODE else "instruct_0",
        page_formatter_guided,
        True,
    ),
    "topic_formatter": ("instruct", topic_formatter, True),
    "topic_formatter_guided": (
        "instruct_detailed_0" if not DEVMODE else "instruct_0",
        topic_formatter_guided,
        True,
    ),
    "topic_hierarchy": (
        "structured_detailed" if not DEVMODE else "structured",
        topic_hierarchy,
        False,
    ),
    "topic_combiner": (
        "structured_detailed" if not DEVMODE else "structured",
        topic_combiner,
        False,
    ),
    "taxonomy": (
        "instruct",
        taxonomy,
        False,
    ),
    "taxonomy_hierarchy": (
        "structured_detailed" if not DEVMODE else "structured",
        taxonomy_hierarchy,
        True,
    ),
    "taxonomy_combiner": (
        "structured_detailed" if not DEVMODE else "structured",
        taxonomy_combiner,
        False,
    ),
    "concept_structured": ("structured", concept_structured, False),
    "concept_more": ("structured", concept_more, False),
    # "concept_unique": (
    #     "structured_detailed" if not DEVMODE else "structured",
    #     concept_unique,
    #     True,
    # ),
    "concept_hierarchy": (
        "structured_detailed" if not DEVMODE else "structured",
        concept_hierarchy,
        False,
    ),
    "concept_combiner": (
        "structured_detailed" if not DEVMODE else "structured",
        concept_combiner,
        False,
    ),
    "journey_prompt_generator": (
        "structured_detailed" if not DEVMODE else "structured",
        journey_prompts,
        True,
    ),
    "module_structured": ("structured", module_structured, False),
    "plan": (
        "structured_detailed" if not DEVMODE else "structured",
        plan,
        True,
    ),
    "module_content": ("instruct_detailed_warm", module_content, True),
    "module_intro": ("instruct_warm", module_intro, True),
    "module_actions": (
        "instruct_detailed" if not DEVMODE else "instruct",
        module_actions,
        True,
    ),
    "journey_template_selector": ("instruct", journey_template_selector, True),
    "action_details": ("instruct_warm", action_details, True),
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
        chains[chain] = init_chain(
            llm_id, prompt, check_for_hallucinations=check_for_hallucinations
        )
        return chains[chain]

    if "stuff_documents" == chain:
        base_chain = get_base_chain("text_formatter_compress")

        chains[chain] = create_stuff_documents_chain(
            base_chain(),
            base_chain.prompt.get_chat_prompt_template(),
            output_parser=base_chain.prompt.parser,
        )
        return chains[chain]

    if "summary_documents" == chain:
        chains[chain] = get_chain("stuff_documents") | get_chain("summary")
        return chains[chain]

    if "summary_documents_with_title" == chain:
        chains[chain] = get_chain("stuff_documents") | get_chain("summary_with_title")
        return chains[chain]

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
