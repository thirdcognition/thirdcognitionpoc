import pprint as pp
import re
import textwrap
import time
from typing import Dict, List

from groq import RateLimitError
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.schema.document import Document

from langchain_huggingface import (
    # HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda,
    # Runnable,
)
# from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser, NAIVE_COMPLETION_RETRY_WITH_ERROR

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
    STRUCTURED_LLM,
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
from lib.prompts import (
    PromptFormatter,
    text_formatter,
    text_formatter_compress,
    text_formatter_guided,
    md_formatter,
    md_formatter_guided,
    action,
    check,
    error_retry,
    hallucination,
    question_classifier,
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
    journey_step_action_details,
    journey_structured,
)

from langchain_core.rate_limiters import InMemoryRateLimiter

RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=RATE_LIMIT_PER_SECOND,  # 0.1 <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=RATE_LIMIT_INTEVAL,  # 0.1 <-- Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=1,  # Controls the maximum burst size.
)
CHAT_RATE_LIMITER = None

llms:Dict[str, BaseLLM] = {}
embeddings = {}
# prompts: Dict[str, PromptTemplate] = {}

def print_params(msg, params):
    if DEBUGMODE:
        print(f"\n\n\n{msg}")
        print(f"'\n\n{pp.pformat(params).replace("\\n", "\n")}\n\n")

def format_chain_params(params):
    print_params("Format chain", params)
    if "__params" in params.keys() and isinstance(params["__params"], Dict):
        set_params = params["__params"]
        for key in set_params.keys():
            params[key] = set_params[key]
        params.pop("__params", None)
    return params

def log_chain(params):
    print_params("Log chain", params)
    return params

class Chain:
    def __init__(
        self,
        llm_id="default",
        prompt: PromptFormatter = text_formatter,
        custom_prompt: tuple[str, str] | None = None,
        check_for_hallucinations: bool = False,
    ):
        self.llm_id = llm_id
        self.prompt = prompt
        self.custom_prompt = custom_prompt
        self.chain = None
        self.prompt_template = None
        self.hallucination_prompt = hallucination
        self.check_for_hallucinations = check_for_hallucinations
        self.error_prompt = error_retry

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
            response = None
            check_params = None
            store_params:Dict = None
            prompt_value = None
            def retry_setup(params):
                print_params("Retry setup", params)
                nonlocal check_params
                return {
                    "completion": check_params["output"],
                    "prompt": params["prompt"],  #hallucination_check_params["input"],
                    "error": params["completion"]
                }

            def store_params(params):
                print_params("Store params", params)
                nonlocal store_params
                store_params = params
                return params

            def param_restore(result):
                print_params("Param restore", {
                    "result": result, "response": response
                })
                if result[0]:
                    return response
                else:
                    return result[1]

            def param_reset(params):
                print_params("Param reset", params)
                nonlocal check_params
                nonlocal response
                nonlocal store_params
                nonlocal prompt_value

                check_params = None
                response = None
                store_params = None
                prompt_value = None
                return params


            self.chain = self.prompt_template | llms[self.llm_id]
            def chain_sleep(params):
                time.sleep(0.2)
                return params
            # if USE_GROQ:
            #     chain:RunnableSequence = self.chain
            #     def rate_limit(params):
            #         nonlocal chain
            #         retries = 0
            #         while retries < 5:
            #             retries += 1
            #             try:
            #                 return chain.invoke(params)
            #             except RateLimitError as e:
            #                 print(f"Running into rate limits: {e}")
            #                 floats = [float(n) for n in re.findall(r"[-+]?(?:\d*\.*\d+)", e.message)]
            #                 time.sleep(max(floats) + 1)
            #     self.chain = RunnableLambda(rate_limit)

            if self.check_for_hallucinations:
                def param_check(params):
                    nonlocal store_params
                    nonlocal check_params
                    nonlocal response
                    nonlocal prompt_value
                    print_params("Param check", params)
                    response = params["completion"]
                    system_message:SystemMessage = params["prompt_value"].messages[0]
                    human_message:HumanMessage = params["prompt_value"].messages[1]
                    history:List[BaseMessage] = params["params"]["chat_history"] if "chat_history" in params["params"].keys() else []
                    context = params["params"].get("context", [])
                    if isinstance(context, str):
                        context = [context]
                    new_params = {
                        "input": f"Output instruction: {system_message.content.strip()}\n" +
                            (f"Context:\n {"\n".join((doc.page_content if isinstance(doc, Document) else doc).strip() for doc in context)}\n" if len(context) > 0 else "") +
                            f"Human message: {human_message.content.strip()}\n" +
                            (f"Chat history:\n {"\n".join([(f"{item.__class__.__name__}: {item.content}" if isinstance(item, BaseMessage) else item) for item in history]) if len(history) > 0 else ""}\n") if len(history) > 0 else "",
                        "output": params["completion"].content.strip(),
                    }
                    prompt_value = params["prompt_value"]
                    check_params = new_params
                    print_params(f"New params", new_params)
                    return new_params
                # Store params and prepare verification chain values

                # Initial chain
                hallucination_chain = (RunnableParallel(
                        # Run the original chain
                        completion=self.chain,
                        prompt_value=self.prompt_template,
                        params=RunnablePassthrough()
                    )
                    | RunnableLambda(param_check)
                    # Run against param_check params
                    | self.hallucination_prompt.get_chat_prompt_template()
                    | get_llm("structured_0")
                )

                def set_retry_prompt(params):
                    print_params("Set retry prompt", params)
                    nonlocal prompt_value
                    return prompt_value

                # Retry chain if 1st chain fails
                hallucination_chain_retry = (
                    RunnableParallel(
                        # Run the fix chain instead of original chain
                        completion=(
                            RunnableLambda(retry_setup) |
                            self.error_prompt.get_agent_prompt_template() |
                            get_llm("structured_0")
                        ),
                        prompt_value=RunnableLambda(set_retry_prompt),
                        params=RunnablePassthrough()
                    )
                    # Reset the 1st chain params with the new output from the retry chain
                    | RunnableLambda(param_check)
                    | self.hallucination_prompt.get_chat_prompt_template()
                    | get_llm("structured_0")
                )

                hallucination_parser = RetryWithErrorOutputParser(
                    parser=self.hallucination_prompt.parser,
                    retry_chain=hallucination_chain_retry,
                    max_retries=5
                )

                def rerun_parser(x):
                    print_params("Rerun hallucination parser", x)
                    x["completion"] = x["completion"].content.strip()
                    return hallucination_parser.parse_with_prompt(x["completion"], x["prompt_value"])

                hallucination_chain = (
                    RunnableParallel(
                        completion=hallucination_chain,
                        prompt_value=self.prompt_template,
                    ) |
                    RunnableLambda(rerun_parser) |
                    RunnableLambda(param_restore)
                )

                self.chain = (
                    RunnableLambda(store_params) |
                    RunnableBranch(
                        (lambda x: (
                            ("context" in store_params.keys() and len(str(store_params["context"]))>100)) or
                            ("chat_history" in store_params.keys() and len(store_params["chat_history"])>0),
                            hallucination_chain
                        ),
                        self.chain
                    )
                    | RunnableLambda(param_reset)
                )
                # self.chain = RunnableLambda(store_params) | hallucination_chain

            if self.prompt.parser is not None:
                def param_check(params):
                    nonlocal store_params
                    nonlocal check_params
                    nonlocal response
                    print_params("Param check", params)
                    response = params["completion"]
                    new_params = {
                        "output": params["completion"].content.strip(),
                    }
                    check_params = new_params
                    return params
                parser_chain = (RunnableParallel(
                        completion=self.chain,
                        prompt_value=self.prompt_template,
                        params=RunnablePassthrough()
                    )
                    | RunnableLambda(param_check)
                )

                parser_retry_chain = (
                    RunnableLambda(retry_setup) |
                    self.error_prompt.get_agent_prompt_template() |
                    llms[self.llm_id] #get_llm("structured_0")
                )
                retry_parser = RetryWithErrorOutputParser(
                    parser=self.prompt.parser,
                    retry_chain=parser_retry_chain,
                    max_retries=5
                )


                def rerun_parser(x):
                    print_params("Rerun format parser", x)
                    x["completion"] = x["completion"].content.strip()
                    return retry_parser.parse_with_prompt(x["completion"], x["prompt_value"])

                def add_format_instructions(params: Dict):
                    print_params("Add format instructions", params)
                    if "format_instructions" not in params.keys():
                        params["format_instructions"] = (
                            self.prompt.parser.get_format_instructions()
                        )
                    return params


                if (
                    isinstance(self.prompt.parser, PydanticOutputParser)
                    and "format_instructions" in self.prompt_template.input_variables
                ):
                    parser_chain = add_format_instructions | parser_chain

                self.chain = parser_chain | RunnableLambda(rerun_parser)
            else:
                self.chain = self.chain | StrOutputParser()


        fallback_chain = RunnableLambda(lambda x: AIMessage(content=f"I seem to be having some trouble answering, please try again a bit later."))
        self.chain = self.chain.with_fallbacks([fallback_chain])

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
    rate_limiter=RATE_LIMITER
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
    init_llm(
        "structured", temperature=0.2, model=STRUCTURED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE
    )
    init_llm(
        "structured_0", temperature=0, model=STRUCTURED_LLM, ctx_size=STRUCTURED_CONTEXT_SIZE
    )

    if "json" not in llms:
        if USE_BEDROCK:
            from lib.load_env import BEDROCK_REGION
            llms["json"] = DEFAULT_LLM_MODEL(
                model_id=STRUCTURED_LLM,
                region_name=BEDROCK_REGION,
                model_kwargs={"temperature": 0.1},
                # num_ctx=STRUCTURED_CONTEXT_SIZE,
                # num_predict=STRUCTURED_CONTEXT_SIZE,
                verbose=DEBUGMODE,
                rate_limiter=RATE_LIMITER,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if DEBUGMODE
                    else None
                ),
            )
        if USE_AZURE:
            from lib.load_env import AZURE_API_VERSION
            llms["json"] = DEFAULT_LLM_MODEL(
                azure_deployment=STRUCTURED_LLM,
                api_version=AZURE_API_VERSION,
                verbose=DEBUGMODE,
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}},
                # num_ctx=ctx_size,
                # num_predict=ctx_size,
                # repeat_penalty=2,
                # timeout=10000,
                rate_limiter=RATE_LIMITER,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if DEBUGMODE
                    else None
                ),
            )
        if USE_OLLAMA:
            llms["json"] = DEFAULT_LLM_MODEL(
                base_url=OLLAMA_URL,
                model=STRUCTURED_LLM,
                model_kwargs={"response_format": {"type": "json_object"}},
                temperature=0.1,
                num_ctx=STRUCTURED_CONTEXT_SIZE,
                num_predict=STRUCTURED_CONTEXT_SIZE,
                verbose=DEBUGMODE,
                rate_limiter=RATE_LIMITER,
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
                temperature=0.1,
                timeout=30000,
                verbose=DEBUGMODE,
                rate_limiter=RATE_LIMITER,
                callback_manager=(
                    CallbackManager([StreamingStdOutCallbackHandler()])
                    if DEBUGMODE
                    else None
                ),
            )

    init_llm("tool", temperature=0, model=TOOL_LLM, ctx_size=TOOL_CONTEXT_SIZE)
    init_llm("chat", temperature=0.5, model=CHAT_LLM, ctx_size=CHAT_CONTEXT_SIZE, rate_limiter=CHAT_RATE_LIMITER)
    init_llm("warm", temperature=0.7, model=CHAT_LLM, ctx_size=CHAT_CONTEXT_SIZE, rate_limiter=CHAT_RATE_LIMITER)

    chains["summary"] = Chain(
        "instruct_detailed", summary, check_for_hallucinations=True
    )  # init_chain("summary", "instruct_detailed", summary)
    chains["summary_guided"] = Chain("instruct_detailed", summary_guided, check_for_hallucinations=True)
    chains["action"] = Chain("instruct_0", action)
    chains["grader"] = Chain("json", grader)
    chains["check"] = Chain("instruct_0", check)
    chains["text_formatter"] = Chain("instruct_detailed", text_formatter, check_for_hallucinations=True)
    chains["text_formatter_compress"] = Chain(
        "instruct_detailed", text_formatter_compress, check_for_hallucinations=True
    )
    chains["text_formatter_guided_0"] = Chain(
        "instruct_detailed", text_formatter_guided, check_for_hallucinations=True
    )
    chains["md_formatter"] = Chain("instruct_detailed", md_formatter, check_for_hallucinations=True)
    chains["md_formatter_guided"] = Chain("instruct_detailed_0", md_formatter_guided, check_for_hallucinations=True)
    chains["journey_structured"] = Chain("json", journey_structured)
    chains["journey_steps"] = Chain("json", journey_steps, check_for_hallucinations=True)
    chains["journey_step_details"] = Chain("instruct_detailed", journey_step_details, check_for_hallucinations=True)
    chains["journey_step_intro"] = Chain("instruct_warm", journey_step_intro, check_for_hallucinations=True)
    chains["journey_step_actions"] = Chain(
        "instruct", journey_step_actions, check_for_hallucinations=True
    )
    chains["journey_step_action_details"] = Chain(
        "instruct_detailed", journey_step_action_details, check_for_hallucinations=True
    )
    chains["question"] = Chain("chat", question, check_for_hallucinations=True)
    chains["helper"] = Chain("chat", helper)
    chains["chat"] = Chain("chat", chat)

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
