from typing import Dict, List

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_huggingface import (
    # HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    # RunnablePassthrough,
    RunnableLambda,
    # Runnable,
)
# from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryOutputParser

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
    STRUCTURED_CONTEXT_SIZE,
    STRUCTURED_LLM,
    TOOL_CONTEXT_SIZE,
    TOOL_LLM,
    USE_GROQ,
    USE_HF_EMBEDDINGS,
    USE_LOCAL_EMBEDDINGS,
    USE_OLLAMA,
    USE_OLLAMA_EMBEDDINGS,
)
from lib.prompts import (
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
    grader,
    hyde,
    summary,
    summary_guided,
    journey_steps,
    journey_step_details,
    journey_step_intro,
    journey_step_actions,
    journey_structured,
)

llms = {}
embeddings = {}
# prompts: Dict[str, PromptTemplate] = {}


class Chain:
    def __init__(
        self,
        llm_id="default",
        prompt: PromptFormatter = text_formatter,
        custom_prompt: tuple[str, str] | None = None,
    ):
        self.llm_id = llm_id
        self.prompt = prompt
        self.custom_prompt = custom_prompt
        self.chain = None
        self.prompt_template = None

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if len(llms.keys()) == 0:
            init_llms()

        if self.prompt_template is None or custom_prompt is not None:
            if custom_prompt is not None:
                self.custom_prompt = custom_prompt

            if self.custom_prompt is None:
                self.prompt_template = (
                    self.prompt.get_chat_prompt_template()
                )  # ChatPromptTemplate.from_messages(messages)
            else:
                self.prompt_template = self.prompt.get_chat_prompt_template(
                    custom_system=self.custom_prompt[0],
                    custom_user=self.custom_prompt[1],
                )

        if self.chain is None or custom_prompt is not None:
            self.chain = self.prompt_template | llms[self.llm_id]

            if self.prompt.parser is not None:
                retry_parser = RetryOutputParser.from_llm(
                    parser=self.prompt.parser, llm=llms[self.llm_id], max_retries=5
                )

                def add_format_instructions(params: Dict):
                    if "format_instructions" not in params.keys():
                        params["format_instructions"] = (
                            self.prompt.parser.get_format_instructions()
                        )
                    return params

                # print(f"Prompt {id = }")

                def rerun_parser(x):
                    x["completion"] = x["completion"].content.strip()
                    return retry_parser.parse_with_prompt(**x)

                self.chain = RunnableParallel(
                    completion=self.chain,
                    prompt_value=self.prompt_template,
                ) | RunnableLambda(rerun_parser)

                if (
                    isinstance(self.prompt.parser, PydanticOutputParser)
                    and "format_instructions" in self.prompt_template.input_variables
                ):
                    # print("Add format instructions")
                    self.chain = add_format_instructions | self.chain
                # else:
                #     self.chain = self.chain | self.prompt.parser

            else:
                self.chain = self.chain | StrOutputParser()

        return self.chain


from typing import Dict, Union, Callable

chains: Dict[str, Union[Chain, Callable]] = {}


def get_chain(chain) -> Chain:
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


def init_llm(
    id,
    model=CHAT_LLM,
    temperature=0.2,
    verbose=DEBUGMODE,
    Type=DEFAULT_LLM_MODEL,
    ctx_size=CHAT_CONTEXT_SIZE,
):
    if f"{id}" not in llms:
        print(f"Initialize llm {id}: {model=} with {ctx_size=} and {temperature=}...")
        if USE_OLLAMA:
            llms[f"{id}"] = Type(
                base_url=OLLAMA_URL,
                model=model,
                verbose=verbose,
                temperature=temperature,
                num_ctx=ctx_size,
                num_predict=ctx_size,
                repeat_penalty=2,
                timeout=10000,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if verbose
                    else None
                ),
            )
        else:
            llms[f"{id}"] = Type(
                streaming=verbose,
                api_key=GROQ_API_KEY,
                model=model,
                verbose=verbose,
                temperature=temperature,
                timeout=10000,
                max_retries=5,
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
    )
    init_llm(
        "instruct_detailed_0",
        temperature=0,
        model=INSTRUCT_DETAILED_LLM,
        ctx_size=INSTRUCT_DETAILED_CONTEXT_SIZE,
    )
    init_llm(
        "instruct_detailed_warm",
        temperature=0.5,
        model=INSTRUCT_DETAILED_LLM,
        ctx_size=INSTRUCT_DETAILED_CONTEXT_SIZE,
    )

    if "json" not in llms:
        if USE_OLLAMA:
            llms["json"] = DEFAULT_LLM_MODEL(
                # model=llama3_llm,
                base_url=OLLAMA_URL,
                model=STRUCTURED_LLM,
                model_kwargs={"response_format": {"type": "json_object"}},
                temperature=0,
                num_ctx=STRUCTURED_CONTEXT_SIZE,
                num_predict=STRUCTURED_CONTEXT_SIZE,
                verbose=DEBUGMODE,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if DEBUGMODE
                    else None
                ),
            )
        if USE_GROQ:
            llms["json"] = DEFAULT_LLM_MODEL(
                api_key=GROQ_API_KEY,
                model=STRUCTURED_LLM,
                model_kwargs={"response_format": {"type": "json_object"}},
                temperature=0,
                timeout=30000,
                verbose=DEBUGMODE,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if DEBUGMODE
                    else None
                ),
            )

    init_llm("tool", temperature=0, model=TOOL_LLM, ctx_size=TOOL_CONTEXT_SIZE)
    init_llm("chat", temperature=0.5, Type=DEFAULT_LLM_MODEL)
    init_llm("warm", temperature=0.4)

    chains["summary"] = Chain(
        "instruct_detailed", summary
    )  # init_chain("summary", "instruct_detailed", summary)
    chains["summary_guided"] = Chain("instruct", summary_guided)
    chains["action"] = Chain("instruct_detailed_0", action)
    chains["grader"] = Chain("json", grader)
    chains["check"] = Chain("instruct_0", check)
    chains["text_formatter"] = Chain("instruct_detailed", text_formatter)
    chains["text_formatter_compress"] = Chain(
        "instruct_detailed", text_formatter_compress
    )
    chains["text_formatter_guided_0"] = Chain(
        "instruct_detailed", text_formatter_guided
    )
    chains["md_formatter"] = Chain("instruct_detailed", md_formatter)
    chains["md_formatter_guided"] = Chain("instruct_detailed_0", md_formatter_guided)
    chains["journey_structured"] = Chain("json", journey_structured)
    chains["journey_steps"] = Chain("json", journey_steps)
    chains["journey_step_details"] = Chain("instruct_warm", journey_step_details)
    chains["journey_step_intro"] = Chain("instruct_detailed_warm", journey_step_intro)
    chains["journey_step_actions"] = Chain(
        "instruct_detailed_warm", journey_step_actions
    )
    chains["question"] = Chain("chat", question)
    chains["helper"] = Chain("chat", helper)

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
            llms["default"],
            embeddings["base"],
            custom_prompt=hyde.get_chat_prompt_template(),  # prompts["hyde"]
        )

    if "summary_documents" not in chains:
        chain = create_stuff_documents_chain(
            RunnableLambda(
                lambda params: chains["text_formatter_compress"]().invoke(params)[0]
            ),
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
            RunnableLambda(
                lambda params: chains["text_formatter_compress"]().invoke(params)[0]
            ),
            chains[
                "text_formatter_compress"
            ].prompt.get_chat_prompt_template(),  # prompts["text_formatter_compress"],
            output_parser=text_formatter_compress.parser,
        )
