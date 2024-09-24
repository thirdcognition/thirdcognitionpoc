import textwrap
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.source import ParsedTopic, ParsedTopicStructureList
from lib.prompts.base import (
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)

PAGE_INSTRUCT = """
        Use <thinking>-tag to consider the contents of the page and how to rewrite them. Explain your reasoning using
        <reflect>-tag and make sure to cover all of the content.

        Rewrite the page in a way that contains all the information about the topic that is available within the context.

        Use following format per topic:
        <output>
        <id>
        Human readable topic ID with letters, numbers and _-characters
        </id>
        <topic>
        Topic that covers the contents
        </topic>
        <instruct>
        Instructions on how to interpret the content
        </instruct>
        <summary>
        Summary of the content
        </summary>
        Formatted content in full detail.
        </output>

        All tags: <output>, <id>, <topic>, <instruct> and <summary> are required.

        Always use <[tag]> and </[tag]>-tags, e.g. <output> and </output>, or <topic> and </topic> when tags are specified.
        There should only be one of <output> structure.
        If the page contains multiple topics, combine them into one topic.
        If there's Previous or Next page defined, do not consider this to cover all content but just
        one slice of it and use the previous and next page to help formatting and writing of this page.
        Be verbose and include as much detail as possible.
        """
PAGE_INSTRUCT_TAGS = ["topic", "id", "instruct", "summary"]
TOPIC_INSTRUCT = """
        Use <thinking>-tag to identify different topics that are contained within the page. Explain your reasoning using
        <reflect>-tag and make sure to cover all topics separately.

        For each topic write content that covers all the information about the topic that is available within the context.

        Use following format per topic:
        <output>
        <id>
        Human readable topic ID with letters, numbers and _-characters
        </id>
        <topic>
        Topic that covers the content
        </topic>
        <instruct>
        Instructions on how to interpret the content
        </instruct>
        <summary>
        Summary of the content
        </summary>
        Formatted content in full detail.
        </output>

        All tags: <output>, <id>, <topic>, <instruct> and <summary> are required.
        If the content specifies multiple topics be sure to add an <output>-structure for each topic.
        If there's Previous or Next page defined, do not consider this to cover all content but just
        one slice of it and use the previous and next page to help identify the topics from current context.
        Always use <[tag]> and </[tag]>-tags, e.g. <output> and </output>, or <topic> and </topic> when tags are specified.
        Be verbose and include as much detail as possible.
        """
TOPIC_INSTRUCT_TAGS = ["id", "topic", "instruct", "summary"]

# Finally after writing <output>-tag per topic, write a final <output>-tag with specified format
# that covers the whole content within the context in full detail.

# TOPIC_COMBINE_INSTRUCT = """
#         Use <thinking>-tag to identify topics that cover the exact same subject contained within the content
#         to combine them. Also cover all topics which are not overlapping.
#         Explain your reasoning using <reflect>-tag and make sure to cover all topics.

#         For each topic write an item which contains the list of combined topics only.
#         Use following format per topic:
#         <item>
#         <id>
#         Human readable topic ID with letters, numbers and _-characters
#         </id>
#         <child_topic>
#         Topic id 1
#         </child_topic>
#         <child_topic>
#         Topic id 2
#         </child_topic>
#         ...
#         <child_topic>
#         Topic id N
#         </child_topic>
#         </item>

#         The item can contain 1 or more <child_topic>-tags. Only combine topics which are cover the exact same subject.
#         If there's no topics that cover the exact same subject, write an item per topic with one <child_topic>-tag.
#         Always use <[tag]> and </[tag]>-tags, e.g. <output> and </output>, or <topic> and </topic> when tags are specified.
#         """
# TOPIC_COMBINE_INSTRUCT_TAGS = ["item", "id", "child_topic"]


topic_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a topic extractor for the defined context.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {TOPIC_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        """
    ),
    user=textwrap.dedent(
        """
        Previous page ending start
        {prev_page}
        Previous page ending end
        Context start
        {context}
        Context end
        Next page beginning start
        {next_page}
        Next page beginning end

        Find the topics from the context and format them with
        <output>, <id>, <topic>, <instruct> and <summary>-tags per topic.
        """
    ),
)
topic_formatter.parser = TagsParser(
    min_len=100,
    tags=TOPIC_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

topic_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a topic extractor for the defined context.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {TOPIC_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        If there is instructions provided follow them exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}
        Previous page ending start
        {prev_page}
        Previous page ending end
        Context start
        {context}
        Context end
        Next page beginning start
        {next_page}
        Next page beginning end

        Find the topics from the context and format them with
        <output>, <id>, <topic>, <instruct> and <summary>-tags per topic.
        """
    ),
)
topic_formatter_guided.parser = TagsParser(
    min_len=100,
    tags=TOPIC_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

topic_hierarchy = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a topic hierachy compiler who is finding the hierachy of topics from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of topics extracted from a document.
        Find all connected topics and define a hierarchy for the topics.
        If there are topics which are relatively similar, combine them using joined list.
        """
    ),
    user=textwrap.dedent(
        """
        existing topics start
        {hierarchy_items}
        existing topics end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing topics find the hierachy of topics and extract it using the specified format.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
topic_hierarchy.parser = PydanticOutputParser(pydantic_object=ParsedTopicStructureList)

topic_combiner = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a topic combiner which is combining the topic list into a single topic.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of topics extracted from a document.
        Combine all topics into one topic using the defined format.
        """
    ),
    user=textwrap.dedent(
        """
        topics start
        {joined_items}
        topics end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the topics create a new topic which covers all the details from the topics.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
topic_combiner.parser = PydanticOutputParser(pydantic_object=ParsedTopic)


page_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {PAGE_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text specified by the user between the context start and context end in full detail using natural language.
        Remove all mentions of confidentiality. Use only information from the available in the text.
        """
    ),
    user=textwrap.dedent(
        """
        Previous page ending start
        {prev_page}
        Previous page ending end
        Context start
        {context}
        Context end
        Next page beginning start
        {next_page}
        Next page beginning end

        Format the text in the context using <output>, <id>, <topic>, <instruct> and <summary>-tags.
        """
    ),
)
page_formatter.parser = TagsParser(
    min_len=100,
    tags=PAGE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

page_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {PAGE_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text between the context start and context end using only information and follow the instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}
        Previous page ending start
        {prev_page}
        Previous page ending end
        Context start
        {context}
        Context end
        Next page beginning start
        {next_page}
        Next page beginning end

        Format the text in the context using <output>, <id>, <topic>, <instruct> and <summary>-tags.
        """
    ),
)
page_formatter_guided.parser = TagsParser(
    min_len=100,
    tags=PAGE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)
