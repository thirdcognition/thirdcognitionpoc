import random
import textwrap
from typing import Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, ChatMessage
from langchain_core.prompts.chat import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
# from fewshot_data import FewShotItem, example_tasks

pre_think_instruct = """Before starting plan how to proceed step by step and place your thinking between
[thinking_start] and [thinking_end]-tags. Then follow your plan and return only the expected output."""
    # For example:

    # Example 1:

    # [thinking_start] I have a long article about a recent scientific discovery. I will generate a summary
    # that highlights the main findings, the method used, and the implications of the discovery. [thinking_end]

    # A recent study published in the Journal of Neuroscience revealed that prolonged exposure to blue light
    # before sleep can disrupt circadian rhythms and lead to sleep disorders. The research, conducted on mice,
    # found that blue light exposure suppressed the production of melatonin, a hormone that regulates sleep-wake cycles.
    # These findings suggest that limiting blue light exposure, particularly in the evening, may help improve sleep
    # quality and reduce the risk of sleep disorders.

    # Example 2:

    # [thinking_start] I have a lengthy book review. I will create a summary that captures the main points of the book,
    # the author's style, and the reviewer's overall opinion. [thinking_end]

    # In 'The Catcher in the Rye', J.D. Salinger explores the disillusionment and alienation of a teenager named Holden Caulfield.
    # Through Holden's first-person narrative, the novel delves into themes of adolescence, identity, and the struggle for authenticity.
    # The book's raw and unfiltered language captures the angst and confusion of youth, making it a classic of modern literature.

    # Example 3:

    # [thinking_start] I have several research papers on a specific topic. I will generate a comprehensive report that synthesizes
    # the findings from each paper, identifies common themes, and discusses any discrepancies. [thinking_end]

    # A comprehensive analysis of the effects of climate change on global food security was conducted by synthesizing data from
    # multiple research papers. The synthesis revealed that climate change is likely to have a significant negative impact on food
    # security, particularly in low-income countries. The findings indicated that increased temperatures, droughts, and sea-level
    # rise will lead to reduced crop yields, increased food prices, and food insecurity. However, the report also highlighted potential
    # adaptation strategies and the importance of international cooperation to mitigate the risks associated with climate change.
    # """
keep_pre_think_together = """While following your plan don't explain what you are doing."""

class PromptFormatter(BaseModel):
    system: str = Field(description="The system message template")
    user: str = Field(description="The user message template")
    parser: Optional[BaseOutputParser] = Field(description="The parser for response", default=None)

    def format(
        self,
        system_format: Optional[Tuple[str, str]] = None,
        user_format: Optional[Tuple[str, str]] = None,
        system: Optional[str] = None,
        user: Optional[str] = None,
        use_format: bool = False
    ) -> "PromptFormatter":
        """
        Format the system and user messages with optional prefix and suffix.

        Args:
            system_format (Optional[Tuple[str, str]]): The prefix and suffix for the system message.
            user_format (Optional[Tuple[str, str]]): The prefix and suffix for the user message.
            system (Optional[str]): The system message. If not provided, the default system message is used.
            user (Optional[str]): The user message. If not provided, the default user message is used.
            use_format (bool): Whether to use the provided prefix and suffix.

        Returns:
            PromptFormatter: A new PromptFormatter instance with the formatted messages.
        """
        if system is None:
            system = self.system
        if user is None:
            user = self.user

        if (not use_format) or (system_format is None and user_format is None):
            return PromptFormatter(system=system, user=user)
        else:
            system_prefix, system_suffix = system_format if system_format else ("", "")
            user_prefix, user_suffix = user_format if user_format else ("", "")
            formatted_system = system_prefix + system + system_suffix
            formatted_user = user_prefix + user + user_suffix
            return PromptFormatter(
                system=formatted_system, user=formatted_user
            )

    def get_agent_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", self.user),
                # MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ]
        )
        if self.parser is not None:
            prompt.partial_variables["format_instructions"] = self.parser.get_format_instructions()

        return prompt

    def get_chat_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessagePromptTemplate.from_template(self.user)
            ]
        )
        if self.parser is not None:
            prompt.partial_variables["format_instructions"] = self.parser.get_format_instructions()

        return prompt

    def get_prompt_format(self) -> str:
        return self.parser.get_format_instructions()



text_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text specified by the user between the context start and context end in full detail using natural language.
        Don't use html tags or markdown. Remove all mentions of confidentiality. Use only information from the available in the text."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)

text_formatter_compress = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarise, compress and reduce the text specified by the user between the context start and context end in retaining details using natural language.
        Don't return the context text as it is, process it to shorter form.
        The context is a part of a longer document. Don't use html tags or markdown. Remove all mentions of confidentiality.
        Use only information from the available in the text."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context."""
    ),
)

text_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using only information and follow the instructions exactly.
        Don't use html tags or markdown."""
    ),
    user=textwrap.dedent(
        """
        Instructions: {question}

        Context start
        {context}
        Context end

        Format the text in the context."""
    ),
)

md_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using markdown syntax. Use only information from the context.
        Remove all mentions of confidentiality."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context."""
    ),
)

md_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using markdown syntax. Use only information from the context
        and follow the instructions exactly. Remove all mentions of confidentiality. Follow the instructions exactly."""
    ),
    user=textwrap.dedent(
        """
        Instructions: {question}

        Context start
        {context}
        Context end

        Format the text in the context."""
    ),
)

action = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a task completing machine.
        {pre_think_instruct}
        {keep_pre_think_together}
        Use the following pieces of information between the context start and context end to complete the task.
        If you don't know the how, just say that you don't know how."""
    ),
    user=textwrap.dedent(
        """
        Task: {action}

        Context start
        {context}
        Context end

        Complete the task with the details from context."""
    ),
)

grader = PromptFormatter(
    system=textwrap.dedent(
        f"""You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""
    ),
    user=textwrap.dedent(
        """
        Here is the retrieved document:
        {document}

        Here is the user question:
        {question}

        Grade the document based on the question."""
    ),
)
grader.parser = JsonOutputParser()

check = PromptFormatter(
    system=textwrap.dedent(
        f"""Act as a verification machine. Check that the items matches the format exactly and answer with one of the appropriate option only. Do not explain your answer or add anything else after or before the answer."""
    ),
    user=textwrap.dedent(
        """
        {format}

        {expected_count}

        Amount of items: {count}
        {context}

        Respond options: {options}

        Do the items match the format?
        Respond with one of the options only.
        Do not explain your answer or add anything else after or before the answer."""
    ),
)

helper = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a startup coach and answer questions thoroughly and exactly.
        {pre_think_instruct}
        Use the info between the context start and context end and previous discussion to answer the question.
        If you don't know the answer, just say that you don't know. Keep the answer concise.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Question: {question}"""
    ),
)

chat = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an helpful assistant answering in a lighthearted and funny way, but don't over do it.
        {pre_think_instruct}
        Use max three sentences maximum and keep the answer concise.
        Don't explain your responses, apologize or mention that you are an assistant."""
    ),
    user=textwrap.dedent(
        """
        {question}"""
    ),
)

question = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer
        the question. If you don't know the answer, say that you
        don't know. Limit your response to three sentences maximum
        and keep the answer concise."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Question: {question}"""
    ),
)

hyde = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Given a chat history and the latest user question
        which might reference the chat history,
        formulate a standalone answer which could be a result
        for a search engine query for the question.
        Use maximum of three sentences.
        """
    ),
    user=textwrap.dedent(
        """{question}"""
    ),
)

summary = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing texts.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarize the text between the context start and context end using natural language."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Summarize the text.
        """
    ),
)

summary_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing texts.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarize the text between the context start and context end using natural language
        and follow the instructions exactly."""
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Instructions: {question}
        Summarize the text.
        """
    ),
)

journey_steps = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a teacher who is planning a curriculum.
        Using the content between context start and end write a list
        with subjects limiting one to a single line.
        If instructions are provided follow them exactly.

        Example for 5 subjects:

        Title: Specific description
        Title: Specific description
        Title: Specific description
        Title: Specific description
        Title: Specific description
        """
    ),
    user=textwrap.dedent( # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        context start
        {context}
        context end

        Create a list of {amount} subjects with one item per line.
        Follow the format of the example exactly with title and description
        separated by : and keep each item within one line.
        Do not add "Here's the list" or any other text before or after the list.
        Only respond with the expected format.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic. Make sure the list has exactly {amount} items."""
    ),
)

journey_step_details = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are ThirdCognition Virtual Buddy.
        Act as a teacher who is planning the content for a class with a specific subject.
        Do not use code, or any markup, markdown or html. Just use natural spoken language divided
        into a clear structure.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring and you are making study materials for them.
        {pre_think_instruct}
        {keep_pre_think_together}
        Create the study material for the student with the following information between context start and end.
        If instructions are provided follow them exactly."""
    ),
    user=textwrap.dedent( # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        instructions end

        context start
        {context}
        context end

        Subject:
        {subject}


        Create study materials for the student defined by the subject. Don't include any other content outside of the subject.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic.
        The study materials should be exhaustive, detailed and generated from the context.
        If instructions are provided follow them exactly."""
    ),
)

journey_step_intro = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are ThirdCognition Virtual Buddy.
        Act as a teacher who is explaining the class with a specific subject for the student at the
        beginning of the class. Use an informal style and 3 sentences maximum.
        Do not use code, or any markup, markdown or html. Just use natural spoken language.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring so don't have to address more than one person.
        {pre_think_instruct}
        {keep_pre_think_together}
        Use the content for the class available between content start and end."""
    ),
    user=textwrap.dedent( # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        instructions end

        Subject:
        {subject}

        content start
        {context}
        content end

        Add an introduction to the class and explain the content of the class briefly.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic.
        Don't add anything new to the content, just explain it with 3 sentences maximum."""
    ),
)

journey_step_actions = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are ThirdCognition Virtual Buddy.
        Act as a teacher who planning 5 actions for teaching the student a specific subject and actions to verify that the student has learned the subject.
        You only have one student you're tutoring so don't have to address more than one person.
        {pre_think_instruct}
        {keep_pre_think_together}
        Use the content for the class available between content start and end."""
    ),
    user=textwrap.dedent( # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        instructions end

        Subject:
        {subject}

        content start
        {context}
        content end

        Write a list of actions to take to teach the subject to the student and how to verify that the student has learned the subject.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic."""
    ),
)

class ActionStructure(BaseModel):
    title: str = Field(description="Title for the step", title="Title")
    description: str = Field(description="Description for the teacher for what to do", title="Description")
    resources: List[str] = Field(description="List of content pieces to support the teacher in explaining and describing the step.", title="Resources")
    test: str = Field(description="Description on how to do a test to verify that the student has succeeded in learning the contents for the step.", title="Test")

class JourneyStructure(BaseModel):
    title: str = Field(description="Title of the class", title="Title")
    intro: str = Field(description="Introduction to the class", title="Intro")
    content: str = Field(description="Detailed content of the class", title="Content")
    actions: List[ActionStructure] = Field(description="List steps for the teacher to take within the class to teach the subject.", title="Actions")

journey_structured = PromptFormatter(
    system="""
    Act as a structured data formatter and use specified format instructions exactly
    to format the context data. Return only the JSON object with the formatted data.
    """,
    user="""
    context start
    {context}
    context end
    ----------------
    format instructions start
    {format_instructions}
    format instructions end
    ----------------
    Format the context data using the format instructions.
    Return only the properly formatted JSON object with the formatted data.
    """
)
journey_structured.parser=PydanticOutputParser(pydantic_object=JourneyStructure)

DEFAULT_PROMPT_FORMATTER = chat

# def generate_result_sample(tasks: List[FewShotItem], amount = 3) -> dict:
#     sample = random.sample(tasks, amount)
#     documents = [Document(page_content=task.document) for task in sample]
#     result = "\n".join([f"Task {i+1}:\n{task.result}" for i, task in enumerate(sample)])
#     return {"documents": documents, "result": result}

# def get_journey_format_example(amount=4) -> FewShotPromptTemplate:
#     instruction = textwrap.dedent(
#         """
#         Find tasks from the following documents and create a task list based on it.

#         Documents:
#         {documents}

#         Task:
#         {tasks}
#     """
#     )

#     task_lists = [lambda: generate_result_sample(example_tasks, amount) for _ in range(3)]

#     return FewShotPromptTemplate(
#         examples=task_lists,
#         example_prompt=PromptTemplate(
#             template=instruction,
#             input_variables=["documents", "tasks"],
#         ),
#         suffix=textwrap.dedent(
#             """
#             Documents:
#             {context}
#             """),
#         input_variables=["context"],
#     )
