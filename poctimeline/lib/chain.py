from typing import Dict, List

from langchain_chroma import Chroma

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
    RunnablePassthrough,
    RunnableLambda,
    Runnable,
)
from langchain_core.prompts.prompt import PromptTemplate
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
    USE_OLLAMA_embeddings,
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
prompts: Dict[str, PromptTemplate] = {}
chains: Dict[str, RunnableSequence] = {}

def get_chain(chain) -> RunnableSequence:
    if len(chains.keys()) == 0:
        init_llms()

    return chains[chain]


def get_llm(llm_id="default"):
    if len(chains.keys()) == 0:
        init_llms()

    if llm_id in llms.keys():
        llm = llms[llm_id]
    else:
        llm = llms["default"]

    return llm


def get_prompt(prompt_id="helper") -> PromptTemplate:
    if len(chains.keys()) == 0:
        init_llms()

    return prompts[prompt_id]


def get_embeddings(embedding_id="base"):
    if len(embeddings.keys()) == 0:
        init_llms()

    if embedding_id in embeddings.keys():
        return embeddings[embedding_id]

def init_chain(
    id,
    llm="default",
    prompt: PromptFormatter = text_formatter,
    init_llm=True,
    structured=False,
):  # templates_mistral
    if f"{id}" not in prompts:
        prompts[f"{id}"] = (
            prompt.get_chat_prompt_template()
        )  # ChatPromptTemplate.from_messages(messages)

    if f"{id}" not in chains and init_llm:
        chains[f"{id}"] = prompts[f"{id}"] | llms[llm]

        if prompt.parser is not None:
            retry_parser = RetryOutputParser.from_llm(
                parser=prompt.parser, llm=llms[llm], max_retries=3
            )

            def add_format_instructions(params: Dict):
                if "format_instructions" not in params.keys():
                    params["format_instructions"] = (
                        prompt.parser.get_format_instructions()
                    )
                return params

            # print(f"Prompt {id = }")

            def rerun_parser(x):
                x["completion"] = x["completion"].content.strip()
                return retry_parser.parse_with_prompt(**x)

            chains[f"{id}"] = RunnableParallel(
                completion=chains[f"{id}"],
                prompt_value=prompts[f"{id}"],
            ) | RunnableLambda(rerun_parser)

            if (
                isinstance(prompt.parser, PydanticOutputParser)
                and "format_instructions" in prompts[f"{id}"].input_variables
            ):
                # print("Add format instructions")
                chains[f"{id}"] = add_format_instructions | chains[f"{id}"]
            # else:
            #     chains[f"{id}"] = chains[f"{id}"] | prompt.parser

        else:
            chains[f"{id}"] = chains[f"{id}"] | StrOutputParser()


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
                repeat_penalty=1.5,
                timeout=20 * 1000,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        else:
            llms[f"{id}"] = Type(
                streaming=DEBUGMODE,
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
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        if USE_GROQ:
            llms["json"] = DEFAULT_LLM_MODEL(
                api_key=GROQ_API_KEY,
                model=STRUCTURED_LLM,
                model_kwargs={"response_format": {"type": "json_object"}},
                temperature=0,
                timeout=30000,
                verbose=DEBUGMODE,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )

    init_llm("tool", temperature=0, model=TOOL_LLM, ctx_size=TOOL_CONTEXT_SIZE)

    init_llm("chat", temperature=0.5, Type=DEFAULT_LLM_MODEL)

    init_llm("warm", temperature=0.4)

    init_chain("summary", "instruct_detailed", summary)
    init_chain("summary_guided", "instruct", summary_guided)
    init_chain(
        "action",
        "instruct_detailed_0",
        action,
    )
    init_chain(
        "grader",
        "json",
        grader,
    )
    init_chain(
        "check",
        "instruct_0",
        check,
    )
    init_chain("text_formatter", "instruct_detailed", text_formatter)
    init_chain("text_formatter_compress", "instruct_detailed", text_formatter_compress)
    init_chain(
        "text_formatter_guided_0",
        "instruct_detailed",
        text_formatter_guided,
    )
    init_chain("md_formatter", "instruct_detailed", md_formatter)
    init_chain(
        "md_formatter_guided",
        "instruct_detailed_0",
        md_formatter_guided,
    )
    init_chain("journey_structured", "json", journey_structured)
    init_chain(
        "journey_steps",
        "json",
        journey_steps,
    )
    init_chain(
        "journey_step_details",
        "instruct_warm",
        journey_step_details,
    )
    init_chain("journey_step_intro", "instruct_detailed_warm", journey_step_intro)
    init_chain("journey_step_actions", "instruct_detailed_warm", journey_step_actions)
    init_chain("question", "chat", question)

    if "helper" not in prompts:
        init_chain("helper", "chat", helper, init_llm=False)

    if "hyde" not in prompts:
        init_chain("hyde", "chat", hyde, init_llm=False)

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
        elif USE_OLLAMA_embeddings:
            embeddings["base"] = OllamaEmbeddings(
                model=EMBEDDING_MODEL, base_url=OLLAMA_URL
            )
        elif USE_HF_EMBEDDINGS:
            embeddings["base"] = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_API_KEY, model_name=EMBEDDING_MODEL
            )

    if "hyde" not in embeddings:
        embeddings["hyde"] = HypotheticalDocumentEmbedder.from_llm(
            llms["default"], embeddings["base"], custom_prompt=prompts["hyde"]
        )

    if "summary_documents" not in chains:
        chain = create_stuff_documents_chain(
            RunnableLambda(
                lambda params: chains["text_formatter_compress"].invoke(params)[0]
            ),
            prompts["text_formatter_compress"],
            output_parser=text_formatter_compress.parser,
        )

        chains["summary_documents"] = (
            RunnableLambda(lambda params: chain.invoke(params)[0]) | chains["summary"]
        )

    if "reduce_journey_documents" not in chains:
        chains["reduce_journey_documents"] = create_stuff_documents_chain(
            RunnableLambda(
                lambda params: chains["text_formatter_compress"].invoke(params)[0]
            ),
            prompts["text_formatter_compress"],
            output_parser=text_formatter_compress.parser,
        )