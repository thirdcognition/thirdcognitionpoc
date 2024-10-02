import asyncio
from typing import Callable, List
from langchain_core.exceptions import OutputParserException
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain.schema.document import Document

from lib.chains.base import BaseChain
from lib.chains.base_parser import (
    add_format_instructions,
    error_retry,
    retry_setup,
)
from lib.helpers.shared import get_text_from_completion, print_params
from lib.prompts.base import PromptFormatter
from lib.prompts.hallucination import hallucination


def validation_param_set(params):
    print_params("Param check", params)
    system_message: SystemMessage = params["prompt_value"].messages[0]
    human_message: HumanMessage = params["prompt_value"].messages[1]
    history: List[BaseMessage] = (
        params["params"]["chat_history"]
        if not isinstance(params["params"], PromptValue)
        and "chat_history" in params["params"].keys()
        else []
    )
    context = (
        params["params"].get("context", [])
        if not isinstance(params["params"], PromptValue)
        else str(params["params"])
    )
    if isinstance(context, str):
        context = [context]
    # Extract and format the system message
    system_message_content = system_message.content.strip()

    # Extract and format the context
    context_content = "\n".join(
        (doc.page_content if isinstance(doc, Document) else doc).strip()
        for doc in context
    )
    context_content = f"Context:\n{context_content}\n" if context_content else ""

    # Extract and format the human message
    human_message_content = human_message.content.strip()

    # Extract and format the chat history
    history_content = "\n".join(
        (
            f"{item.__class__.__name__}: {item.content}"
            if isinstance(item, BaseMessage)
            else item
        )
        for item in history
    )
    history_content = f"Chat history:\n{history_content}\n" if history_content else ""

    # Extract and format the completion
    completion_content = get_text_from_completion(params["completion"])

    # Construct the new parameters
    new_params = {
        "input": f"Output instruction: {system_message_content}\n"
        f"{context_content}"
        f"Human message: {human_message_content}\n"
        f"{history_content}",
        "output": completion_content,
    }

    print_params(f"New params", new_params)
    return new_params


class BaseValidationChain(BaseChain):
    def __init__(
        self,
        retry_llm: RunnableSequence | None = None,
        error_prompt: PromptFormatter = error_retry,
        output_parser: BaseOutputParser | None = None,
        validation_llm: RunnableSequence | None = None,
        validation_prompt: PromptFormatter = hallucination,
        validation_setup: Callable[[dict], dict] = None,
        max_retries: int = 5,
        **kwargs,
    ):
        global validation_param_set
        super().__init__(**kwargs)
        self.retry_llm = retry_llm or self.llm
        self.error_prompt = error_prompt
        self.output_parser = output_parser
        self.validation_llm = validation_llm or self.llm
        self.validation_prompt = validation_prompt
        self.validation_setup = validation_setup or validation_param_set
        self.max_retries = max_retries
        self.validation_chain: RunnableSequence = None
        self.retry_chain: RunnableSequence = None
        self.verify_chain: RunnableSequence = None

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if self.chain is not None and (
            custom_prompt is None or repr(self.custom_prompt) == repr(custom_prompt)
        ):
            return self.chain

        self._setup_prompt(custom_prompt)

        self.chain = super().__call__(custom_prompt)

        self.validation_chain: RunnableSequence = (
            self.validation_setup
            | self.validation_prompt.get_chat_prompt_template()
            | self.validation_llm
        )
        self.validation_chain.name = f"{self.name}-validation-initial"

        self.retry_chain: RunnableSequence = (
            self.error_prompt.get_chat_prompt_template() | self.retry_llm
        )
        self.retry_chain.name = f"{self.name}-validation-retry"

        parser = self.output_parser or self.prompt.parser
        self.retry_parser = parser
        if parser is not None:
            self.retry_parser = RetryWithErrorOutputParser(
                parser=parser,
                retry_chain=RunnableLambda(retry_setup) | self.retry_chain,
                max_retries=self.max_retries,
            )

        instance = self
        validation = lambda x: instance._validate(x)
        avalidation = lambda x: asyncio.create_task(instance._avalidate(x))

        self.verify_chain = RunnableParallel(
            completion=self.chain,
            prompt_value=self.prompt_template,
            params=RunnablePassthrough(),
        ) | RunnableLambda((avalidation if self.async_mode else validation))
        if (
            isinstance(parser, PydanticOutputParser)
            and "format_instructions" in self.prompt_template.input_variables
        ):
            self.verify_chain = add_format_instructions(parser) | self.verify_chain

        self.verify_chain.name = f"{self.name}-validation-verify"

        self.chain = RunnableLambda(
            lambda x: {"context": x} if isinstance(x, str) else x
        ) | RunnableBranch(
            (
                lambda x: (
                    (
                        (("context" in x.keys() and len(str(x["context"])) > 100))
                        or ("chat_history" in x.keys() and len(x["chat_history"]) > 0)
                    )
                    if not isinstance(x, PromptValue)
                    else True
                ),
                self.verify_chain,
            ),
            self.chain,
        )

        fallback_chain = RunnableLambda(
            lambda x: AIMessage(
                content=f"I seem to be having some trouble answering, please try again a bit later."
            )
        )
        self.chain = self.chain.with_fallbacks([fallback_chain])
        self.chain.name = f"{self.name}-validation"

        return self.chain

    def _validate(self, params):
        retries_left = self.max_retries
        completion = params["completion"]

        prompt_value: PromptValue = params["prompt_value"]
        while retries_left > 0:
            retries_left -= 1
            validation = None
            try:
                validation = self.validation_chain.invoke(
                    dict(
                        completion=repr(completion),
                        prompt_value=prompt_value,
                        params=params["params"],
                    )
                )
                validation = self.validation_prompt.parser.parse(validation)
                if validation[0]:
                    break
            except OutputParserException as e:
                if retries_left == 0:
                    raise e
                completion = self.retry_chain.invoke(
                    dict(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=f"Error: {e.args[0]}\nObservations:\n{e.observation}",
                    )
                )
                if isinstance(self.retry_parser, RetryWithErrorOutputParser):
                    completion = (
                        completion.content.strip()
                        if isinstance(completion, BaseMessage)
                        else completion.strip()
                    )
                    completion = self.retry_parser.parse_with_prompt(
                        completion=completion, prompt_value=prompt_value
                    )
                else:
                    completion = self.retry_parser.parse(completion)

        return completion

    async def _avalidate(self, params):
        retries_left = self.max_retries
        completion = params["completion"]

        prompt_value: PromptValue = params["prompt_value"]
        while retries_left > 0:
            retries_left -= 1
            validation = None
            try:
                validation = await self.validation_chain.ainvoke(
                    dict(
                        completion=repr(completion),
                        prompt_value=prompt_value,
                        params=params["params"],
                    )
                )
                validation = await self.validation_prompt.parser.aparse(validation)
            except OutputParserException as e:
                if retries_left == 0:
                    raise e

                completion = await self.retry_chain.ainvoke(
                    dict(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=f"Error: {e.args[0]}\nObservations:\n{e.observation}",
                    )
                )
                if isinstance(self.retry_parser, RetryWithErrorOutputParser):
                    completion = (
                        completion.content.strip()
                        if isinstance(completion, BaseMessage)
                        else completion.strip()
                    )
                    completion = self.retry_parser.aparse_with_prompt(
                        completion=completion, prompt_value=prompt_value
                    )
                else:
                    completion = self.retry_parser.aparse(completion)
                    # completion = get_text_from_completion(completion)

        return completion
