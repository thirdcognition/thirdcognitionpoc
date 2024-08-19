# Load .env file from the parent directory
import os

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_core.language_models.llms import BaseLLM
from langchain_community.chat_models.ollama import ChatOllama
from langchain_groq import ChatGroq

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
TOOL_LLM: str = None
TOOL_CHAR_LIMIT: int = None
TOOL_CONTEXT_SIZE: int = None
EMBEDDING_MODEL: str = None
EMBEDDING_CHAR_LIMIT: int = 1000
EMBEDDING_OVERLAP: int = 100

CLIENT_HOST = os.getenv("CLIENT_HOST", "http://localhost:3100")
ADMIN_HOST = os.getenv("ADMIN_HOST", "http://localhost:4000")

OLLAMA_URL = None
if USE_OLLAMA:
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

    TOOL_LLM = os.getenv("OLLAMA_TOOL_LLM", "phi3:instruct")
    TOOL_CONTEXT_SIZE = int(os.getenv("OLLAMA_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("OLLAMA_TOOL_CHAR_LIMIT", 1024))

    print("+++ OLLAMA +++")

GROQ_API_KEY = None
if USE_GROQ:
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

    TOOL_LLM = os.getenv("GROQ_TOOL_LLM", "llama-3.1-8b-instant")
    TOOL_CONTEXT_SIZE = int(os.getenv("GROQ_TOOL_CTX_SIZE", 8192))
    TOOL_CHAR_LIMIT = int(os.getenv("GROQ_TOOL_CHAR_LIMIT", 1024))

    print("+++ GROQ +++")

print(f"\tChat: {CHAT_LLM=} {CHAT_CONTEXT_SIZE=} {CHAT_CHAR_LIMIT}")
print(f"\tInstruct: {INSTRUCT_LLM=} {INSTRUCT_CONTEXT_SIZE=} {INSTRUCT_CHAR_LIMIT}")
print(
    f"\tInstruct detailed: {INSTRUCT_DETAILED_LLM=} {INSTRUCT_DETAILED_CONTEXT_SIZE=} {INSTRUCT_DETAILED_CHAR_LIMIT}"
)
print(
    f"\tStructured: {STRUCTURED_LLM=} {STRUCTURED_CONTEXT_SIZE=} {STRUCTURED_CHAR_LIMIT}"
)
print(f"\tTool: {TOOL_LLM=} {TOOL_CONTEXT_SIZE=} {TOOL_CHAR_LIMIT}")

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
