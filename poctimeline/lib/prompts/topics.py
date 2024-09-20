import textwrap
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
        Use the <output>-tag to wrap the content, add <topic>-tag to specify the topic and <summary>-tag to specify the summary of the content.
        Also add an <id>-tag to identify the content and an <instruct>-tag to specify instructions and guidance on how to intrepret the content.

        Use following format for each topic:
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
        Formatted content in full detail.
        <summary>
        Summary of the content
        </summary>
        </output>

        Always use <[tag]> and </[tag]>-tags, e.g. <topic> and </topic> when tags are specified.
        There should only be one of <ouptut>, <topic> and <summary>-tags. If the page
        contains multiple topics, combine them into one topic.
        Be verbose and include as much detail as possible.
        """
PAGE_INSTRUCT_TAGS = ["topic", "id", "instruct", "summary"]
TOPIC_INSTRUCT = """
        Use <thinking>-tag to identify different topics that are contained within the page. Explain your reasoning using
        <reflect>-tag and make sure to cover all topics separately.

        For each topic write content that contains all the information about the topic that is available within the context.
        Use the <output>-tag to wrap the content, add <topic>-tag to specify the topic and <summary>-tag to specify the summary for the content.
        Use following format for each topic:
        <output>
        <id>
        Human readable topic ID with letters, numbers and _-characters
        </id>
        <topic>
        Topic that covers the content
        </topic>
        Formatted content in full detail.
        <instruct>
        Instructions on how to interpret the content
        </instruct>
        <summary>
        Summary of the content
        </summary>
        </output>

        If the content specifies multiple topics be sure to add a <topic>, <output> and <summary> for each topic.
        If there's Previous or Next page defined, do not consider this to cover all content but just one slice of
        it and use the previous and next page to help identify the topics from current context.
        Always use <[tag]> and </[tag]>-tags, e.g. <topic> and </topic> when tags are specified.
        """
TOPIC_INSTRUCT_TAGS = ["id", "topic", "instruct", "summary"]

        # Finally after writing <output>-tag for each topic, write a final <output>-tag with specified format
        # that covers the whole content within the context in full detail.

TOPIC_COMBINE_INSTRUCT = """
        Use <thinking>-tag to identify topics that cover the exact same subject contained within the content
        to combine them. Also cover all topics which are not overlapping.
        Explain your reasoning using <reflect>-tag and make sure to cover all topics.

        For each topic write an item which contains the list of combined topics only.
        Use following format for each topic:
        <item>
        <id>
        Human readable topic ID with letters, numbers and _-characters
        </id>
        <child_topic>
        Topic id 1
        </child_topic>
        <child_topic>
        Topic id 2
        </child_topic>
        ...
        <child_topic>
        Topic id N
        </child_topic>
        </item>

        The item can contain 1 or more <child_topic>-tags. Only combine topics which are cover the exact same subject.
        If there's no topics that cover the exact same subject, write an item for each topic with one <child_topic>-tag.
        Always use <[tag]> and </[tag]>-tags, e.g. <topic> and </topic> when tags are specified.
        """
TOPIC_COMBINE_INSTRUCT_TAGS = ["item", "id", "child_topic"]


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
        Next page beginning start
        {next_page}
        Next page beginning end
        Context start
        {context}
        Context end

        Find the topics from the context.
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
        Next page beginning start
        {next_page}
        Next page beginning end
        Context start
        {context}
        Context end

        Find the topics from the context.
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

topic_combiner = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a topic combiner for the list of topics provided by the user.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {TOPIC_COMBINE_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        """
    ),
    user=textwrap.dedent(
        """
        {context}
        """
    ),
)
topic_combiner.parser = TagsParser(
    min_len=100,
    tags=TOPIC_COMBINE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)


page_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {PAGE_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text specified by the user between the context start and context end in full detail using natural language.
        Don't use html tags or markdown. Remove all mentions of confidentiality. Use only information from the available in the text.
        """
    ),
    user=textwrap.dedent(
        """
        Previous page ending start
        {prev_page}
        Previous page ending end
        Next page beginning start
        {next_page}
        Next page beginning end
        Context start
        {context}
        Context end

        Format the text in the context.
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
        Don't use html tags or markdown.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}
        Previous page ending start
        {prev_page}
        Previous page ending end
        Next page beginning start
        {next_page}
        Next page beginning end
        Context start
        {context}
        Context end

        Format the text in the context.
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
