import textwrap
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import (
    PydanticOutputParser,
    BaseOutputParser,
    StrOutputParser,
)
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from pydantic import BaseModel, ValidationError
from lib.chains.base import BaseChain
from lib.helpers import get_text_from_completion, pretty_print, print_params
from lib.prompts.base import PromptFormatter
from lib.prompts.actions import error_retry


def add_format_instructions(parser: BaseOutputParser):
    def _add_format_instructions(params: Dict):
        print_params("Add format instructions", params)
        if "format_instructions" not in params.keys():
            params["format_instructions"] = parser.get_format_instructions()
        return params

    return _add_format_instructions


def retry_setup(params):
    print_params("Retry setup", params)
    return {
        "completion": (
            (
                params["completion"].content
                if isinstance(params["completion"], BaseMessage)
                else params["completion"]
            )
            or ""
        ).strip(),
        "prompt": params["prompt"],
        "error": (
            (
                params["error"].content
                if isinstance(params["error"], BaseMessage)
                else params["error"]
            )
            or ""
        ).strip(),
    }


class BaseParserChain(BaseChain):
    def __init__(
        self,
        retry_llm: RunnableSequence | None = None,
        error_prompt: PromptFormatter = error_retry,
        output_parser: BaseOutputParser | None = None,
        max_retries: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retry_llm = retry_llm or self.llm
        self.error_prompt = error_prompt
        self.max_retries = max_retries
        self.output_parser = output_parser

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None
    ) -> RunnableSequence:
        if self.chain is not None and (
            custom_prompt is None or repr(self.custom_prompt) == repr(custom_prompt)
        ):
            return self.chain

        self._setup_prompt(custom_prompt)

        self.chain = super().__call__(custom_prompt)

        parser = self.output_parser or self.prompt.parser

        if parser is None:
            self.chain = self.chain | StrOutputParser()
            return self.chain

        parser_chain = RunnableParallel(
            completion=self.chain,
            prompt_value=self.prompt_template,
            params=RunnablePassthrough(),
        )
        parser_chain.name = f"{self.name}-parser-initial"

        parser_retry_chain = (
            RunnableLambda(retry_setup)
            | self.error_prompt.get_agent_prompt_template()
            | self.retry_llm  # (llms[self.llm_id] if self.llm_id == "json" else get_llm("instruct_detailed"))
        )
        parser_retry_chain.name = f"{self.name}-parser-retry"

        retry_parser = RetryWithErrorOutputParser(
            parser=parser,
            retry_chain=parser_retry_chain,
            max_retries=self.max_retries,
        )

        def rerun_parser(x):
            completion = x["completion"]

            completion = get_text_from_completion(completion)
            try:
                return retry_parser.parse_with_prompt(
                    completion, x["prompt_value"]
                )
            except ValidationError as e:
                completion = parser_retry_chain.invoke(
                    dict(
                        completion=x["completion"],
                        prompt=x["prompt_value"],
                        error=repr(e),
                    )
                )
                return retry_parser.parse_with_prompt(completion, x["prompt_value"])

        async def arerun_parser(x):
            print_params("Rerun format parser", x)

            completion = x["completion"]

            completion = get_text_from_completion(completion)
            try:
                return await retry_parser.aparse_with_prompt(
                    completion, x["prompt_value"]
                )
            except ValidationError as e:
                completion = parser_retry_chain.ainvoke(
                    dict(
                        completion=x["completion"],
                        prompt=x["prompt_value"],
                        error=repr(e),
                    )
                )
                return retry_parser.aparse_with_prompt(completion, x["prompt_value"])

        if (
            isinstance(parser, PydanticOutputParser)
            and "format_instructions" in self.prompt_template.input_variables
        ):
            parser_chain = add_format_instructions(parser) | parser_chain

        self.chain = parser_chain | RunnableLambda(
            arerun_parser if self.async_mode else rerun_parser
        )

        fallback_chain = RunnableLambda(
            lambda x: AIMessage(
                content=f"I seem to be having some trouble answering, please try again a bit later."
            )
        )
        self.chain = self.chain.with_fallbacks([fallback_chain])
        self.chain.name = f"{self.name}-parser"

        return self.chain
