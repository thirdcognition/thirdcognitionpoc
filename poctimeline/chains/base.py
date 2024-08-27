from langchain_core.runnables import (
    RunnableSequence,
)
from langchain_core.output_parsers import StrOutputParser

from chains.prompts import PromptFormatter

class BaseChain:
    def __init__(
        self,
        parent_chain: RunnableSequence | None = None,
        prompt: PromptFormatter|None = None,
        llm: RunnableSequence|None = None,
        custom_prompt: tuple[str, str] | None = None,
        async_mode: bool = False,
    ):
        self.parent_chain = parent_chain
        self.llm = llm
        self.prompt = prompt
        self.custom_prompt = custom_prompt
        self.chain = None
        self.prompt_template = None
        self.async_mode = async_mode

    def _setup_prompt(self, custom_prompt: tuple[str, str] | None = None):
        if self.prompt is not None and (self.prompt_template is None or custom_prompt is not None):
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
    def __call__(
        self, custom_prompt: tuple[str, str] | None = None, **kwargs
    ) -> RunnableSequence:
        if self.chain is not None and (custom_prompt is None or self.custom_prompt is custom_prompt):
            return self.chain

        self._setup_prompt(custom_prompt)

        if self.parent_chain is not None and self.llm is not None:
            self.chain = self.parent_chain | self.prompt_template | self.llm
        elif self.llm is not None:
            self.chain = self.prompt_template | self.llm | StrOutputParser()
        elif self.parent_chain is not None:
            self.chain = self.parent_chain
        else:
            raise ValueError(
                "Either parent_chain or prompt_template must be provided."
            )

        return self.chain

# def parser_param_set(params):
#     print_params("Param check", params)
#     return params






