# Load .env file from the parent directory
import os

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_core.language_models.llms import BaseLLM

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../", ".env"))
print("Loading env: ", os.path.join(os.path.dirname(__file__), "../../../", ".env"))
# for key, value in os.environ.items():
#     print(f'\t{key}= {value}')

DEBUGMODE = os.getenv("LLM_DEBUG", "True") == "True" or False

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "False") == "True" or False
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

set_debug(DEBUGMODE)
set_verbose(DEBUGMODE)

from langchain_community.chat_models.ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint

LLM_PROVIDERS = os.getenv("LLM_PROVIDERS", "OLLAMA").upper().split(",")

LLM_MODEL_MAP:dict[str, BaseLLM] = {
    "OLLAMA": ChatOllama,
    "GROQ": ChatGroq,
    "BEDROCK": ChatBedrock,
    "OPENAI": ChatOpenAI,
    "ANTHROPIC": ChatAnthropic,
    "AZURE": AzureChatOpenAI,
    "AZURE_ML": AzureMLChatOnlineEndpoint,
}

# from langchain_core.embeddings import Embeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
EMBEDDING_MODEL_MAP:dict[str, Embeddings] = {
    "LOCAL": HuggingFaceEmbeddings,
    "OLLAMA": OllamaEmbeddings,
    "HUGGINGFACE": HuggingFaceInferenceAPIEmbeddings,
}

# DEFAULT_LLM_MODEL = LLM_MODEL_MAP.get(LLM_PROVIDERS[0])
LLM_MODELS = [
    "chat",
    "instruct",
    "instruct_detailed",
    "structured",
    "structured_detailed",
    "tool",
    "tester",
]

from pydantic import BaseModel
from typing import Any, Literal, Optional, List, Union

class ProviderModelSettings(BaseModel):
    type: Literal[
        "chat",
        "instruct",
        "instruct_detailed",
        "structured",
        "structured_detailed",
        "tool",
        "tester",
    ]
    class_model: Optional[Any] = None
    provider: Optional[str] = None
    url: Optional[str] = None
    model: Optional[str] = None
    context_size: Optional[int] = None
    char_limit: Optional[int] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    ratelimit_per_sec: Optional[float] = None
    ratelimit_interval: Optional[float] = None
    ratelimit_bucket: Optional[float] = None

class ModelDefaults(BaseModel):
    default: Optional[ProviderModelSettings] = None
    chat: Optional[ProviderModelSettings] = None
    instruct:Optional[ProviderModelSettings] = None
    instruct_detailed:Optional[ProviderModelSettings] = None
    structured:Optional[ProviderModelSettings] = None
    structured_detailed:Optional[ProviderModelSettings] = None
    tool:Optional[ProviderModelSettings] = None
    tester:Optional[ProviderModelSettings] = None

class ProviderSettings(BaseModel):
    type: Literal[
        "OLLAMA", "GROQ", "BEDROCK", "OPENAI", "ANTHROPIC", "AZURE", "AZURE_ML"
    ]
    class_model: Optional[Any] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    models: List[ProviderModelSettings] = []


class EmbeddingModelSettings(BaseModel):
    type: Literal["large", "medium", "small"]
    model: Optional[str] = None
    char_limit: Optional[int] = None
    overlap: Optional[int] = None

class EmbeddingDefaults(BaseModel):
    default: Optional[EmbeddingModelSettings] = None
    large: Optional[EmbeddingModelSettings] = None
    medium: Optional[EmbeddingModelSettings] = None
    small: Optional[EmbeddingModelSettings] = None

class EmbeddingProviderSettings(BaseModel):
    type: Literal["LOCAL", "HUGGINGFACE", "OPENAI", "OLLAMA", "BEDROCK", "AZURE"]
    class_model: Any = None #Union[HuggingFaceEmbeddings, OllamaEmbeddings, HuggingFaceInferenceAPIEmbeddings, None] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    models: List[EmbeddingModelSettings] = []


class Settings(BaseModel):
    llms: List[ProviderSettings] = []
    default_provider: Optional[ProviderSettings] = None
    default_llms: Optional[ModelDefaults] = None
    embeddings: List[EmbeddingProviderSettings] = []
    default_embedding_provider: Optional[EmbeddingProviderSettings] = None
    default_embeddings: Optional[EmbeddingDefaults] = None
    client_host: str
    admin_host: str
    chroma_path: str
    sqlite_db: str
    file_tablename: str
    journey_tablename: str


SETTINGS = Settings(
    client_host=os.getenv("CLIENT_HOST", "http://localhost:3500"),
    admin_host=os.getenv("ADMIN_HOST", "http://localhost:4000"),
    chroma_path=os.getenv("CHROMA_PATH", "db/chroma_db"),
    sqlite_db=os.getenv("SQLITE_DB", "db/files.db"),
    file_tablename="files",
    journey_tablename="journey",
)

SETTINGS.default_llms = ModelDefaults()
for provider in LLM_PROVIDERS:
    print(f"Loading {provider} settings...")
    provider_settings = ProviderSettings(
        type=provider, class_model=LLM_MODEL_MAP[provider]
    )

    if provider == "OLLAMA":
        provider_settings.url = os.getenv(f"{provider}_URL", "http://127.0.0.1:11434")
    elif provider == "GROQ":
        provider_settings.api_key = os.getenv(f"{provider}_API_KEY", None)
    elif provider == "BEDROCK":
        provider_settings.region = os.getenv("AWS_BEDROCK_REGION", "us-west-2")
        provider_settings.access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        provider_settings.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    elif provider == "OPENAI":
        provider_settings.api_key = os.getenv(f"{provider}_API_KEY", "")
    elif provider == "ANTHROPIC":
        provider_settings.api_key = os.getenv(f"{provider}_API_KEY", "")
    elif provider == "AZURE":
        provider_settings.api_type = os.getenv("AZURE_API_TYPE", "")
        provider_settings.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        provider_settings.api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        provider_settings.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
    elif provider == "AZURE_ML":
        provider_settings.api_key = os.getenv("AZURE_ML_APIKEY", "")

    for type in LLM_MODELS:
        type_model = os.getenv(f"{provider}_{type.upper()}_MODEL", "")
        if type_model != "":
            type_settings = ProviderModelSettings(
                type=type,
                provider=provider,
                url=os.getenv(f"{provider}_{type}_URL", provider_settings.url),
                model=type_model,
                class_model=LLM_MODEL_MAP.get(f"{provider}_{type}", provider_settings.class_model),
                context_size=int(os.getenv(f"{provider}_{type.upper()}_CTX_SIZE", 8192)),
                char_limit=int(os.getenv(f"{provider}_{type.upper()}_CHAR_LIMIT", 12000)),
                api_key=os.getenv(
                    f"{provider}_{type}_API_KEY", provider_settings.api_key
                ),
                endpoint=os.getenv(f"{provider}_{type}_ENDPOINT", None),
                ratelimit_per_sec=float(os.getenv(f"{provider}_{type.upper()}_PER_SEC", 2)),
                ratelimit_interval=float(os.getenv(f"{provider}_{type.upper()}_INTERVAL", 0.5)),
                ratelimit_bucket=float(os.getenv(f"{provider}_{type.upper()}_BUCKET", 1)),
            )
            if SETTINGS.default_llms.__getattribute__("default") is None:
                SETTINGS.default_llms.default = type_settings
            if SETTINGS.default_llms.__getattribute__(type) is None:
                SETTINGS.default_llms.__setattr__(type, type_settings)
            provider_settings.models.append(type_settings)

    SETTINGS.llms.append(provider_settings)
    # Set default provider if not already set
    if SETTINGS.default_provider is None:
        SETTINGS.default_provider = provider_settings

EMBEDDING_PROVIDERS = os.getenv("EMBEDDING_PROVIDERS", "LOCAL").upper().split(",")
EMBEDDING_MODEL_TYPES = ["large", "medium", "small"]

SETTINGS.default_embeddings = EmbeddingDefaults()
for provider in EMBEDDING_PROVIDERS:
    provider_settings = EmbeddingProviderSettings(type=provider, class_model=EMBEDDING_MODEL_MAP.get(provider))

    if provider == "OLLAMA":
        provider_settings.url = os.getenv(f"{provider}_EMBEDDING_URL", "")
    if provider == "HUGGINGFACE":
        provider_settings.api_key = os.getenv(f"{provider}_EMBEDDDING_API_KEY", "")
    # if provider == "OPENAI":
    #     provider_settings.api_key = os.getenv(f"{provider}_API_KEY", "")
    # elif provider == "BEDROCK":
    #     provider_settings.region = os.getenv("AWS_BEDROCK_REGION", "us-west-2")
    #     provider_settings.access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    #     provider_settings.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    # elif provider == "AZURE":
    #     provider_settings.api_type = os.getenv("AZURE_API_TYPE", "")
    #     provider_settings.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    #     provider_settings.api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    #     provider_settings.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
    for model_type in EMBEDDING_MODEL_TYPES:
        model = os.getenv(f"{provider}_EMBEDDING_{model_type.upper()}_MODEL", "")
        if model != "":
            model_settings = EmbeddingModelSettings(
                type=model_type,
                model=model,
                char_limit=int(
                    os.getenv(f"{provider}_EMBEDDING_{model_type.upper()}_CHAR_LIMIT", 1000)
                ),
                overlap=int(os.getenv(f"{provider}_EMBEDDING_{model_type.upper()}_OVERLAP", 100)),
            )
            if SETTINGS.default_embeddings.__getattribute__("default") is None:
                SETTINGS.default_embeddings.default = model_settings
            if SETTINGS.default_embeddings.__getattribute__(model_type) is None:
                SETTINGS.default_embeddings.__setattr__(model_type, model_settings)
            provider_settings.models.append(model_settings)

    SETTINGS.embeddings.append(provider_settings)
    if SETTINGS.default_embedding_provider is None:
        SETTINGS.default_embedding_provider = provider_settings


for provider_settings in SETTINGS.llms:
    print(f"+++ {provider_settings.type} +++")
    for model_settings in provider_settings.models:
        print(
            f"\t{model_settings.type.capitalize()}: {model_settings.model=} {model_settings.context_size=} {model_settings.char_limit=}"
        )

for embedding_provider_settings in SETTINGS.embeddings:
    print(f"+++ {embedding_provider_settings.type} EMBEDDINGS +++")
    for model_settings in embedding_provider_settings.models:
        print(
            f"\t{model_settings.type.capitalize()}: {model_settings.model=} {model_settings.char_limit=} {model_settings.overlap=}"
        )
