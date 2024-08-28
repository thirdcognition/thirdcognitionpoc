from typing import Dict
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from chains.prompts import PromptFormatter

from lib.helpers import print_params

def keep_chain_params(params:Dict):
    print_params(params)
    if "orig_params" in params.keys() and isinstance(params["orig_params"], Dict):
        set_params = params["orig_params"]
        for key in set_params.keys():
            params[key] = set_params[key]
        # params.pop("orig_params", None)
    params["orig_params"] = params.copy()
    return params

def log_chain_params(params):
    print_params("Log params", params)
    return params

drop_thoughts = RunnableLambda(lambda x: x[0] if isinstance(x, tuple) else x)

class BaseChain:
    def __init__(
        self,
        parent_chain: RunnableSequence | None = None,
        prompt: PromptFormatter|None = None,
        llm: RunnableSequence|None = None,
        custom_prompt: tuple[str, str] | None = None,
        async_mode: bool = False,

    ):
        if not hasattr(self, 'parent_chain') or self.parent_chain is None:
            self.parent_chain = parent_chain
        if not hasattr(self, 'llm') or self.llm is None:
            self.llm = llm
        if not hasattr(self, 'prompt') or self.prompt is None:
            self.prompt = prompt
        if not hasattr(self, 'custom_prompt') or self.custom_prompt is None:
            self.custom_prompt = custom_prompt
        if not hasattr(self, 'chain'):
            self.chain = None
        if not hasattr(self, 'prompt_template'):
            self.prompt_template = None

        self.async_mode = async_mode

        self.id = f"{self.__class__.__name__}-{id(self)}"
        self.name = self.id

    def _setup_prompt(self, custom_prompt: tuple[str, str] | None = None):
        if self.prompt is not None and (self.prompt_template is None or custom_prompt is not None):
            if custom_prompt is not None:
                self.custom_prompt = custom_prompt

            if self.custom_prompt is None:
                self.prompt_template = (
                    self.prompt.get_chat_prompt_template()
                )
            else:
                self.prompt_template = self.prompt.get_chat_prompt_template(
                    custom_system=self.custom_prompt[0],
                    custom_user=self.custom_prompt[1],
                )
    def __call__(
        self, custom_prompt: tuple[str, str] | None = None
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

        self.chain.name = f"{self.name}-base"

        return self.chain
