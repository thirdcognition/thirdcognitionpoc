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

FILE_TABLENAME = "files"
JOURNEY_TABLENAME = "journey"
CHROMA_PATH = os.getenv("CHROMA_PATH", "db/chroma_db")
SQLITE_DB = os.getenv("SQLITE_DB", "db/files.db")

set_debug(DEBUGMODE)
set_verbose(DEBUGMODE)

USE_OLLAMA = os.getenv("USE_OLLAMA", "True") == "True" or False
USE_GROQ = os.getenv("USE_GROQ", "True") == "True" or False
USE_BEDROCK = os.getenv("USE_AWS_BEDROCK", "False") == "True" or False
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "False") == "True" or False
USE_AZURE = os.getenv("USE_AZURE", "False") == "True" or False
USE_OPENAI = os.getenv("USE_OPENAI", "False") == "True" or False

DEFAULT_LLM_MODEL: BaseLLM = None
CHAT_LLM: str = None
CHAT_CHAR_LIMIT: int = None
CHAT_CONTEXT_SIZE: int = None
INSTRUCT_LLM: str = None
INSTRUCT_CHAR_LIMIT: int = None
INSTRUCT_CONTEXT_SIZE: int = None
STRUCTURED_LLM: str = None
STRUCTURED_CHAR_LIMIT: int = None
STRUCTURED_CONTEXT_SIZE: int = None
STRUCTURED_DETAILED_LLM: str = None
STRUCTURED_DETAILED_CHAR_LIMIT: int = None
STRUCTURED_DETAILED_CONTEXT_SIZE: int = None
TOOL_LLM: str = None
TOOL_CHAR_LIMIT: int = None
TOOL_CONTEXT_SIZE: int = None
TESTER_LLM_MODEL: BaseLLM = None
TESTER_LLM: str = None
TESTER_CHAR_LIMIT: int = None
TESTER_CONTEXT_SIZE: int = None
TESTER_APIKEY: str = None
EMBEDDING_MODEL: str = None
EMBEDDING_CHAR_LIMIT: int = 1000
EMBEDDING_OVERLAP: int = 100

CLIENT_HOST = os.getenv("CLIENT_HOST", "http://localhost:3500")
ADMIN_HOST = os.getenv("ADMIN_HOST", "http://localhost:4000")

RATE_LIMIT_PER_SECOND = float(os.getenv("RATE_LIMIT_PER_SECOND", 2))
RATE_LIMIT_INTEVAL = float(os.getenv("RATE_LIMIT_INTEVAL", 0.1))

OLLAMA_URL = None
if USE_OLLAMA:
    from langchain_community.chat_models.ollama import ChatOllama
    DEFAULT_LLM_MODEL = ChatOllama
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    CHAT_LLM = os.getenv("OLLAMA_CHAT_LLM", "phi3:mini")
    CHAT_CONTEXT_SIZE = int(os.getenv("OLLAMA_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("OLLAMA_CHAT_CHAR_LIMIT", 1024))

    INSTRUCT_LLM = os.getenv("OLLAMA_INSTRUCT_LLM", "phi3:instruct")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("OLLAMA_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("OLLAMA_INSTRUCT_CHAR_LIMIT", 1024))

    INSTRUCT_DETAILED_LLM = os.getenv("OLLAMA_INSTRUCT_DETAILED_LLM", "phi3:instruct")
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(
        os.getenv("OLLAMA_INSTRUCT_DETAILED_CTX_SIZE", 8192)
    )
    INSTRUCT_DETAILED_CHAR_LIMIT = int(
        os.getenv("OLLAMA_INSTRUCT_DETAILED_CHAR_LIMIT", 1024)
    )

    STRUCTURED_LLM = os.getenv("OLLAMA_STRUCTURED_LLM", "phi3:instruct")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("OLLAMA_STRUCTURED_CTX_SIZE", 8192))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("OLLAMA_STRUCTURED_CHAR_LIMIT", 1024))

    STRUCTURED_DETAILED_LLM = os.getenv("OLLAMA_STRUCTURED_DETAILED_LLM", "phi3:instruct")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("OLLAMA_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("OLLAMA_STRUCTURED_DETAILED_CHAR_LIMIT", 1024))

    TOOL_LLM = os.getenv("OLLAMA_TOOL_LLM", "phi3:instruct")
    TOOL_CONTEXT_SIZE = int(os.getenv("OLLAMA_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("OLLAMA_TOOL_CHAR_LIMIT", 1024))

    print("+++ OLLAMA +++")

GROQ_API_KEY = None
if USE_GROQ:
    from langchain_groq import ChatGroq
    DEFAULT_LLM_MODEL = ChatGroq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    CHAT_LLM = os.getenv("GROQ_CHAT_LLM", "llama-3.1-8b-instant")
    CHAT_CONTEXT_SIZE = int(os.getenv("GROQ_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("GROQ_CHAT_CHAR_LIMIT", 1024))

    INSTRUCT_LLM = os.getenv("GROQ_INSTRUCT_LLM", "llama-3.1-8b-instant")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("GROQ_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("GROQ_INSTRUCT_CHAR_LIMIT", 1024))

    INSTRUCT_DETAILED_LLM = os.getenv(
        "GROQ_INSTRUCT_DETAILED_LLM", "llama-3.1-8b-instant"
    )
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(
        os.getenv("GROQ_INSTRUCT_DETAILED_CTX_SIZE", 8192)
    )
    INSTRUCT_DETAILED_CHAR_LIMIT = int(os.getenv("GROQ_INSTRUCT_CHAR_LIMIT", 1024))

    STRUCTURED_LLM = os.getenv("GROQ_STRUCTURED_LLM", "llama-3.1-8b-instant")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("GROQ_STRUCTURED_CTX_SIZE", 8192))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("GROQ_STRUCTURED_CHAR_LIMIT", 1024))

    STRUCTURED_DETAILED_LLM = os.getenv("GROQ_STRUCTURED_DETAILED_LLM", "llama-3.1-8b-instant")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("GROQ_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("GROQ_STRUCTURED_DETAILED_CHAR_LIMIT", 1024))

    TOOL_LLM = os.getenv("GROQ_TOOL_LLM", "llama-3.1-8b-instant")
    TOOL_CONTEXT_SIZE = int(os.getenv("GROQ_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("GROQ_TOOL_CHAR_LIMIT", 1024))

    print("+++ GROQ +++")

if USE_BEDROCK:
    from langchain_aws import ChatBedrock
    BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-west-2")
    DEFAULT_LLM_MODEL = ChatBedrock
    BEDROCK_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
    BEDROCK_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    CHAT_LLM = os.getenv("AWS_BEDROCK_CHAT_LLM", "meta.llama3-1-8b-instruct-v1:0")
    CHAT_CONTEXT_SIZE = int(os.getenv("AWS_BEDROCK_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("AWS_BEDROCK_CHAT_CHAR_LIMIT", 1024))

    INSTRUCT_LLM = os.getenv("AWS_BEDROCK_INSTRUCT_LLM", "meta.llama3-1-8b-instruct-v1:0")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("AWS_BEDROCK_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("AWS_BEDROCK_INSTRUCT_CHAR_LIMIT", 1024))

    INSTRUCT_DETAILED_LLM = os.getenv(
        "AWS_BEDROCK_INSTRUCT_DETAILED_LLM", "meta.llama3-1-70b-instruct-v1:0"
    )
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(
        os.getenv("AWS_BEDROCK_INSTRUCT_DETAILED_CTX_SIZE", 8192)
    )
    INSTRUCT_DETAILED_CHAR_LIMIT = int(
        os.getenv("AWS_BEDROCK_INSTRUCT_DETAILED_CHAR_LIMIT", 1024)
    )

    STRUCTURED_LLM = os.getenv("AWS_BEDROCK_STRUCTURED_LLM", "meta.llama3-1-8b-instruct-v1:0")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("AWS_BEDROCK_STRUCTURED_CTX_SIZE", 8192))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("AWS_BEDROCK_STRUCTURED_CHAR_LIMIT", 1024))

    STRUCTURED_DETAILED_LLM = os.getenv("AWS_BEDROCK_STRUCTURED_DETAILED_LLM", "meta.llama3-1-70b-instruct-v1:0")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("AWS_BEDROCK_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("AWS_BEDROCK_STRUCTURED_DETAILED_CHAR_LIMIT", 1024))

    TOOL_LLM = os.getenv("AWS_BEDROCK_TOOL_LLM", "meta.llama3-1-8b-instruct-v1:0")
    TOOL_CONTEXT_SIZE = int(os.getenv("AWS_BEDROCK_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("AWS_BEDROCK_TOOL_CHAR_LIMIT", 1024))

    print("+++ BEDROCK +++")

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    DEFAULT_LLM_MODEL = ChatOpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    CHAT_LLM = os.getenv("OPENAI_CHAT_LLM", "gpt-3.5-turbo")
    CHAT_CONTEXT_SIZE = int(os.getenv("OPENAI_CHAT_CTX_SIZE", 4096))
    CHAT_CHAR_LIMIT = int(os.getenv("OPENAI_CHAT_CHAR_LIMIT", 2048))

    INSTRUCT_LLM = os.getenv("OPENAI_INSTRUCT_LLM", "gpt-3.5-turbo-instruct")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("OPENAI_INSTRUCT_CTX_SIZE", 4096))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("OPENAI_INSTRUCT_CHAR_LIMIT", 2048))

    INSTRUCT_DETAILED_LLM = os.getenv("OPENAI_INSTRUCT_DETAILED_LLM", "gpt-4")
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(os.getenv("OPENAI_INSTRUCT_DETAILED_CTX_SIZE", 8192))
    INSTRUCT_DETAILED_CHAR_LIMIT = int(os.getenv("OPENAI_INSTRUCT_DETAILED_CHAR_LIMIT", 4096))

    STRUCTURED_LLM = os.getenv("OPENAI_STRUCTURED_LLM", "gpt-3.5-turbo-instruct")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("OPENAI_STRUCTURED_CTX_SIZE", 4096))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("OPENAI_STRUCTURED_CHAR_LIMIT", 2048))

    STRUCTURED_DETAILED_LLM = os.getenv("OPENAI_STRUCTURED_DETAILED_LLM", "gpt-4")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("OPENAI_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("OPENAI_STRUCTURED_DETAILED_CHAR_LIMIT", 4096))

    TOOL_LLM = os.getenv("OPENAI_TOOL_LLM", "gpt-3.5-turbo-instruct")
    TOOL_CONTEXT_SIZE = int(os.getenv("OPENAI_TOOL_CTX_SIZE", 4096))
    TOOL_CHAR_LIMIT = int(os.getenv("OPENAI_TOOL_CHAR_LIMIT", 2048))

    print("+++ OPENAI +++")

if USE_ANTHROPIC:
    from langchain_anthropic import ChatAnthropic
    DEFAULT_LLM_MODEL = ChatAnthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    CHAT_LLM = os.getenv("ANTHROPIC_CHAT_LLM", "claude-2")
    CHAT_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_CHAT_CTX_SIZE", 8192))
    CHAT_CHAR_LIMIT = int(os.getenv("ANTHROPIC_CHAT_CHAR_LIMIT", 1024))

    INSTRUCT_LLM = os.getenv("ANTHROPIC_INSTRUCT_LLM", "claude-instant-1")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_INSTRUCT_CTX_SIZE", 8192))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("ANTHROPIC_INSTRUCT_CHAR_LIMIT", 1024))

    INSTRUCT_DETAILED_LLM = os.getenv("ANTHROPIC_INSTRUCT_DETAILED_LLM", "claude-2")
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_INSTRUCT_DETAILED_CTX_SIZE", 8192))
    INSTRUCT_DETAILED_CHAR_LIMIT = int(os.getenv("ANTHROPIC_INSTRUCT_DETAILED_CHAR_LIMIT", 1024))

    STRUCTURED_LLM = os.getenv("ANTHROPIC_STRUCTURED_LLM", "claude-instant-1")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_STRUCTURED_CTX_SIZE", 8192))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("ANTHROPIC_STRUCTURED_CHAR_LIMIT", 1024))

    STRUCTURED_DETAILED_LLM = os.getenv("ANTHROPIC_STRUCTURED_DETAILED_LLM", "claude-2")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("ANTHROPIC_STRUCTURED_DETAILED_CHAR_LIMIT", 1024))

    TOOL_LLM = os.getenv("ANTHROPIC_TOOL_LLM", "claude-instant-1")
    TOOL_CONTEXT_SIZE = int(os.getenv("ANTHROPIC_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("ANTHROPIC_TOOL_CHAR_LIMIT", 1024))

    print("+++ ANTHROPIC +++")

if USE_AZURE:
    from langchain_openai import AzureChatOpenAI
    from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
    DEFAULT_LLM_MODEL = AzureChatOpenAI
    AZURE_API_TYPE = os.getenv("AZURE_API_TYPE", "")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

    CHAT_LLM = os.getenv("AZURE_CHAT_LLM", "gpt-3.5-turbo")
    CHAT_CONTEXT_SIZE = int(os.getenv("AZURE_CHAT_CTX_SIZE", 4096))
    CHAT_CHAR_LIMIT = int(os.getenv("AZURE_CHAT_CHAR_LIMIT", 2048))

    INSTRUCT_LLM = os.getenv("AZURE_INSTRUCT_LLM", "gpt-3.5-turbo-instruct")
    INSTRUCT_CONTEXT_SIZE = int(os.getenv("AZURE_INSTRUCT_CTX_SIZE", 4096))
    INSTRUCT_CHAR_LIMIT = int(os.getenv("AZURE_INSTRUCT_CHAR_LIMIT", 2048))

    INSTRUCT_DETAILED_LLM = os.getenv("AZURE_INSTRUCT_DETAILED_LLM", "gpt-4")
    INSTRUCT_DETAILED_CONTEXT_SIZE = int(os.getenv("AZURE_INSTRUCT_DETAILED_CTX_SIZE", 8192))
    INSTRUCT_DETAILED_CHAR_LIMIT = int(os.getenv("AZURE_INSTRUCT_DETAILED_CHAR_LIMIT", 4096))

    STRUCTURED_LLM = os.getenv("AZURE_STRUCTURED_LLM", "gpt-3.5-turbo-instruct")
    STRUCTURED_CONTEXT_SIZE = int(os.getenv("AZURE_STRUCTURED_CTX_SIZE", 4096))
    STRUCTURED_CHAR_LIMIT = int(os.getenv("AZURE_STRUCTURED_CHAR_LIMIT", 2048))

    STRUCTURED_DETAILED_LLM = os.getenv("AZURE_STRUCTURED_DETAILED_LLM", "gpt-4")
    STRUCTURED_DETAILED_CONTEXT_SIZE = int(os.getenv("AZURE_STRUCTURED_DETAILED_CTX_SIZE", 8192))
    STRUCTURED_DETAILED_CHAR_LIMIT = int(os.getenv("AZURE_STRUCTURED_DETAILED_CHAR_LIMIT", 4096))

    TOOL_LLM = os.getenv("AZURE_TOOL_LLM", "gpt-3.5-turbo-instruct")
    TOOL_CONTEXT_SIZE = int(os.getenv("AZURE_TOOL_CTX_SIZE", 4096))
    TOOL_CHAR_LIMIT = int(os.getenv("AZURE_TOOL_CHAR_LIMIT", 2048))

    TESTER_LLM_MODEL = AzureMLChatOnlineEndpoint
    TESTER_LLM = os.getenv("AZURE_ML_TESTER_ENDPOINT", "")
    TESTER_APIKEY = os.getenv("AZURE_ML_TESTER_APIKEY", "")
    TESTER_CONTEXT_SIZE = int(os.getenv("AZURE_ML_TESTER_CTX_SIZE", 8192))
    TESTER_CHAR_LIMIT = int(os.getenv("AZURE_ML_TESTER_CHAR_LIMIT", 4096))

    print("+++ AZURE +++")


print(f"\tLimits: {RATE_LIMIT_INTEVAL=} {RATE_LIMIT_PER_SECOND=}")
print(f"\tChat: {CHAT_LLM=} {CHAT_CONTEXT_SIZE=} {CHAT_CHAR_LIMIT}")
print(f"\tInstruct: {INSTRUCT_LLM=} {INSTRUCT_CONTEXT_SIZE=} {INSTRUCT_CHAR_LIMIT}")
print(
    f"\tInstruct detailed: {INSTRUCT_DETAILED_LLM=} {INSTRUCT_DETAILED_CONTEXT_SIZE=} {INSTRUCT_DETAILED_CHAR_LIMIT}"
)
print(
    f"\tStructured: {STRUCTURED_LLM=} {STRUCTURED_CONTEXT_SIZE=} {STRUCTURED_CHAR_LIMIT}"
)
print(
    f"\tStructured: {STRUCTURED_DETAILED_LLM=} {STRUCTURED_DETAILED_CONTEXT_SIZE=} {STRUCTURED_DETAILED_CHAR_LIMIT}"
)
print(f"\tTool: {TOOL_LLM=} {TOOL_CONTEXT_SIZE=} {TOOL_CHAR_LIMIT}")
if TESTER_LLM_MODEL is not None:
    print(f"\tTester: {TESTER_LLM=} {TESTER_CONTEXT_SIZE=} {TESTER_CHAR_LIMIT}")

USE_OLLAMA_EMBEDDINGS = os.getenv("USE_OLLAMA_EMBEDDING", "False") == "True" or False
USE_HF_EMBEDDINGS = os.getenv("USE_HF_EMBEDDING", "False") == "True" or False
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDING", "True") == "True" or False

if USE_LOCAL_EMBEDDINGS:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    EMBEDDING_CHAR_LIMIT = int(os.getenv("EMBEDDING_CHAR_LIMIT", 1000))
    EMBEDDING_OVERLAP = int(os.getenv("EMBEDDING_OVERLAP", 100))

    print("+++ LOCAL EMBEDDING +++")

if USE_OLLAMA_EMBEDDINGS:
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-small-en")
    EMBEDDING_CHAR_LIMIT = int(os.getenv("OLLAMA_EMBEDDING_CHAR_LIMIT", 1000))
    EMBEDDING_OVERLAP = int(os.getenv("OLLAMA_EMBEDDING_OVERLAP", 100))

    print("+++ OLLAMA EMBEDDING +++")

HF_API_KEY = None
if USE_HF_EMBEDDINGS:
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en")
    EMBEDDING_CHAR_LIMIT = int(os.getenv("HF_EMBEDDING_CHAR_LIMIT", 1000))
    EMBEDDING_OVERLAP = int(os.getenv("HF_EMBEDDING_OVERLAP", 100))

    print("+++ HUGGINGFACE EMBEDDING +++")

print(f"\tEmbedding: {EMBEDDING_MODEL=}, {EMBEDDING_CHAR_LIMIT=}, {EMBEDDING_OVERLAP=}")
