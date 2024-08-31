from chains.base_parser import BaseParserChain
from langchain_core.runnables import RunnableSequence

from chains.base_validation import BaseValidationChain

class Chain(BaseParserChain):
    def __init__(self, validation_llm: RunnableSequence = None, **kwargs):
        super().__init__(**kwargs)
        self.validation_llm = validation_llm

    def __call__(
        self, custom_prompt: tuple[str, str] | None = None
    ) -> RunnableSequence:
        if self.chain is not None and (custom_prompt is None or self.custom_prompt is custom_prompt):
            return self.chain

        self.chain = super().__call__(custom_prompt)

        if self.validation_llm is not None:
            self.hallunication_chain = BaseValidationChain(parent_chain=self.chain, prompt=self.prompt, validation_llm=self.validation_llm, retry_llm=self.retry_llm)
            self.chain = self.hallunication_chain()

        return self.chain