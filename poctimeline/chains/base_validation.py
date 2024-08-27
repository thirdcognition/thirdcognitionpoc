import re
import textwrap
from typing import Callable, List, Union
from langchain_core.exceptions import OutputParserException
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import (
    AIMessage, SystemMessage, HumanMessage, BaseMessage
)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough
)
from langchain.schema.document import Document

from chains.base import BaseChain
from chains.base_parser import error_retry
from chains.prompts import PromptFormatter
from lib.helpers import print_params

class HallucinationParser(BaseOutputParser[tuple[bool, BaseMessage]]):
    """Custom parser to clean specified tag from results."""

    def parse(self, text: Union[str, BaseMessage]) -> tuple[bool, BaseMessage]:
        # print(f"Parsing tags: {text}")
        if isinstance(text, BaseMessage):
            text = text.content

        # Extract all floats from the text using regular expressions
        floats = [float(n) for n in re.findall(r"[-+]?(?:\d*\.*\d+)", text)]

        if len(floats) == 0:
            excpect_msg = f"""Unable to verify if the content is based on context."""
            raise OutputParserException(excpect_msg)
        if floats[0] == 0.0:
            excpect_msg = f"""The generated text is not based on the context. Please rewrite the text so that it matches provided context."""
            raise OutputParserException(excpect_msg)

        # raise OutputParserException(f"Unexpected error: Testing errors.")

        return True, text

    @property
    def _type(self) -> str:
        return "hallucination_parser"

hallucination = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

        Give a binary score 1 or 0, where 1 means that the answer is grounded in / supported by the set of facts or history and answers the question.

        After the score write a short explanation if the answer does not pass test. Always return the score of 0 or 1 regardless.

        Example 1:
        1

        Generated content is grounded in facts

        Example 2:
        0

        Generated content is built on information which is not in scope

        Example 3:
        0

        Generated content is guessing
        """
    ),
    user=textwrap.dedent(
        """
        {input}

        LLM generation: {output}

        Return 1 or 0 score and explanation regardless.
        """
    ),
)
hallucination.parser = HallucinationParser()

def validation_param_set(params):
    print_params("Param check", params)
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
        "output": params["completion"].content.strip() if isinstance(params["completion"], BaseMessage) else params["completion"][1].strip() if isinstance(params["completion"], tuple) else params["completion"].strip()
    }

    print_params(f"New params", new_params)
    return new_params

class BaseValidationChain(BaseChain):
    def __init__(
        self,
        retry_llm: RunnableSequence | None = None,
        error_prompt: PromptFormatter = error_retry,
        validation_llm: RunnableSequence | None = None,
        validation_prompt: PromptFormatter = hallucination,
        validation_setup: Callable[[dict], dict] = None,
        max_retries: int = 5,
        **kwargs
        ):
        global validation_param_set
        super().__init__( **kwargs)
        self.retry_llm = retry_llm or self.llm
        self.error_prompt = error_prompt
        self.validation_llm = validation_llm or self.llm
        self.validation_prompt = validation_prompt
        self.validation_setup = validation_setup or validation_param_set
        self.max_retries = max_retries

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
        ) -> RunnableSequence:
        if self.chain is not None and (custom_prompt is None or self.custom_prompt is custom_prompt):
            return self.chain

        self._setup_prompt(custom_prompt)

        self.chain = super().__call__(custom_prompt)

        validation_chain:RunnableSequence = (
            self.validation_setup
            | self.validation_prompt.get_chat_prompt_template()
            | self.validation_llm
        )

        retry_chain:RunnableSequence = (
            self.error_prompt.get_chat_prompt_template()
            | self.retry_llm
        )

        validation_parser = self.validation_prompt.parser
        max_retries = self.max_retries
        def validation(params):
            nonlocal validation_parser
            nonlocal max_retries
            retries_left = max_retries
            completion = params["completion"].content.strip() if isinstance(params["completion"], BaseMessage) else params["completion"].strip()
            prompt_value:PromptValue = params["prompt_value"]
            while retries_left > 0:
                retries_left -= 1
                try:
                    validation = validation_chain.invoke(dict(
                        completion=completion,
                        prompt_value=prompt_value,
                        params=params["params"],
                    ))
                    validation = validation_parser.parse(validation)
                    if validation[0]:
                        break
                except OutputParserException as e:
                    print(f"{e=}")
                    if retries_left == 0:
                        raise e
                    completion = retry_chain.invoke(
                        dict(
                            prompt=prompt_value.to_string(),
                            completion=completion,
                            error=repr(validation)
                        )
                    )

            return completion[1] if isinstance(completion, tuple) else completion,

        async def avalidation(params):
            nonlocal validation_parser
            nonlocal max_retries
            retries_left = max_retries
            completion = params["completion"].content.strip() if isinstance(params["completion"], BaseMessage) else params["completion"].strip()
            prompt_value:PromptValue = params["prompt_value"]
            while retries_left > 0:
                retries_left -= 1
                try:
                    validation = await validation_chain.ainvoke(dict(
                        completion=completion,
                        prompt_value=prompt_value,
                        params=params["params"],
                    ))
                    validation = await validation_parser.aparse(validation)
                except OutputParserException as e:
                    if retries_left == 0:
                        raise e
                    completion = await retry_chain.ainvoke(
                        dict(
                            prompt=prompt_value.to_string(),
                            completion=completion,
                            error=repr(validation)
                        )
                    )

            return  completion[1] if isinstance(completion, tuple) else completion,

        verify_chain = (
            RunnableParallel(
                completion=self.chain,
                prompt_value=self.prompt_template,
                params=RunnablePassthrough()
            )
            | RunnableLambda(avalidation if self.async_mode else validation)
        )

        self.chain = (
            RunnableBranch(
                (lambda x: (
                    ("context" in x.keys() and len(str(x["context"]))>100)) or
                    ("chat_history" in x.keys() and len(x["chat_history"])>0),
                    verify_chain
                ),
                self.chain
            )
        )

        fallback_chain = RunnableLambda(lambda x: AIMessage(content=f"I seem to be having some trouble answering, please try again a bit later."))
        self.chain = self.chain.with_fallbacks([fallback_chain])

        return self.chain

