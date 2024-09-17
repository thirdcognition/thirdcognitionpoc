import asyncio
from collections import defaultdict
import os
import sys
from typing import Dict, List
import streamlit as st

from langchain_core.messages import BaseMessage


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from lib.helpers import pretty_print
from lib.models.sqlite_tables import ConceptData, ConceptDataTable, ConceptTaxonomy, SourceContents
from lib.chains.init import get_chain
from lib.document_tools import create_document_lists, split_text
from lib.load_env import SETTINGS
from lib.db_tools import (
    SourceDataTable,
    db_commit,
    delete_db_source,
    get_chroma_collections,
    get_concept_by_id,
    get_concept_category_tag_by_id,
    get_db_sources,
    init_db,
)
from lib.document_tools import markdown_to_text
from lib.streamlit_tools import check_auth, get_all_categories, llm_edit

st.set_page_config(
    page_title="TC POC: Manage Uploads",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


@st.fragment
async def manage_file(filename):
    database_session = init_db()
    file_categories = get_all_categories()
    file_entry = get_db_sources(source=filename)[filename]
    # Add Streamlit editors to edit 'disabled' and 'category_tags' fields

    if "rewrite_text" not in st.session_state:
        st.session_state.rewrite_text = {}

    rewrite_text = st.session_state.rewrite_text

    # for i in range(len(query)):
    col1, col2 = st.columns([1, 10], vertical_alignment="center")
    col2.subheader(filename, divider=True)
    with col1.popover(":x:"):
        if st.button(
            f"Are you sure you want to remove {filename}?",
            key=f"delete_button_{filename}",
            use_container_width=True,
        ):
            delete_db_source(filename)
            st.rerun(scope="fragment")
    header_col1, header_col2 = st.columns([1, 4], vertical_alignment="bottom")
    with header_col1:
        show_details = st.toggle("Extend", key=f"show_{filename}")
        disable_checkbox = st.toggle(
            f"Disable", value=file_entry.disabled, key=f"disable_{filename}"
        )
        st.download_button("Download", file_entry.file_data, file_name=filename)

    with header_col2:
        with st.form(f"edit_fields_{filename}", border=False):
            col1, col4 = st.columns([4, 1], vertical_alignment="bottom")

            with col1:
                category_tags = st.multiselect(
                    "Category",
                    file_categories,
                    default=file_entry.category_tags,
                    key=f"categories_{filename}",
                )

                chroma_collections = st.text_input(
                    f"Chroma collection",
                    value=", ".join(file_entry.chroma_collections),
                    key=f"chroma_{filename}",
                )
            with col4:
                submitted = st.form_submit_button("Save", use_container_width=True)

        if submitted:
            updates = {
                "disabled": disable_checkbox,
                "category_tags": category_tags,
                "chroma_collections": [
                    col.strip() for col in chroma_collections.split(",")
                ],
            }

            changes = False

            for key in updates.keys():
                changes = changes or updates[key] != file_entry.__getattribute__(key)

            if changes:
                instance = (
                    database_session.query(SourceDataTable)
                    .where(SourceDataTable.source == filename)
                    .first()
                )
                instance.disabled = updates["disabled"]
                instance.category_tags = updates["category_tags"]
                instance.chroma_collections = [
                    col.strip() for col in updates["chroma_collections"].split(",")
                ]

            if changes:
                database_session.commit()
                get_db_sources(reset=True)
                st.rerun(scope="fragment")

    if show_details:
        with st.container(border=True):
            filename = file_entry.source
            filetype = filename.split(".")[-1]
            summary = file_entry.source_contents.summary
            text = file_entry.source_contents.formatted_content
            combined_topics = file_entry.source_contents.topics
            topics = file_entry.source_contents.formatted_topics
            raw = file_entry.texts

            if filename in rewrite_text:
                text = rewrite_text[filename]

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Summary", "Formatted", "Unformatted", "Concepts", "RAG"]
            )

            with tab1:
                if st.button(
                    "Rewrite",
                    key=f"llm_summary_rewrite_{filename}",
                    use_container_width=True,
                ):
                    instance = (
                        database_session.query(SourceDataTable)
                        .where(SourceDataTable.source == filename)
                        .first()
                    )
                    contents = SourceContents(**instance.source_contents.__dict__)

                    with st.spinner("Rewriting"):
                        if instance.texts is not None:
                            text = await llm_edit(
                                (
                                    [contents.formatted_content]
                                    if len(contents.formatted_content) > 1000
                                    else instance.texts
                                ),
                                summarize=True,
                            )

                    st.success("Summary rewrite complete")

                    if text is not None and text != "" and text != contents.summary:
                        contents.summary = text
                        instance.source_contents = contents
                        database_session.commit()
                        get_db_sources(reset=True)
                    st.rerun()
                else:
                    with st.container():
                        st.write(summary, unsafe_allow_html=True)

            with tab2:
                update = False
                col1, col2, col3, col4 = st.columns(
                    [3, 1, 1, 1], vertical_alignment="bottom"
                )
                with col1:
                    guidance = st.text_input(
                        "Guidance for the rewriter:",
                        key=f"llm_md_guidance_{filename}",
                        value="",
                    )

                with col2:
                    rewrite = st.button(
                        "Rewrite",
                        key=f"llm_md_rewrite_{filename}",
                        use_container_width=True,
                    )

                with col3:
                    if st.button(
                        "Save", key=f"llm_md_save_{filename}", use_container_width=True
                    ):
                        update = True

                with col4:

                    if st.button(
                        "Clear",
                        key=f"llm_md_clear_{filename}",
                        use_container_width=True,
                    ):
                        text = ""
                        rewrite_text[filename] = ""

                if rewrite:
                    text = None
                    with st.spinner("Rewriting"):
                        if raw is not None and filetype != "md":
                            text = await llm_edit(
                                raw, guidance if 0 < len(guidance) else None
                            )
                        elif filetype == "md":
                            if len(raw) > 1 or len(raw[0]) > 1000:
                                text = await llm_edit(
                                    [markdown_to_text("\n".join(raw))],
                                    guidance if 0 < len(guidance) else None,
                                )
                            else:
                                text = markdown_to_text(raw[0])

                    st.success("Markdown rewrite complete")
                    rewrite_text[filename] = text

                text_tab1, text_tab2 = st.tabs(["Combined", "Pages"])
                with text_tab1:
                    if text != None and len(text) > 0:

                        st.write("##### Topics:\n - " + "\n - ".join(combined_topics))
                        st.write("##### Content:\n\n" + text, unsafe_allow_html=True)
                with text_tab2:
                    for topic in topics:
                        st.write("#### Page: " + str(topic.page_number) + " - " + str(topic.topic_index))
                        st.write("##### " + topic.topic)
                        st.write(topic.page_content, unsafe_allow_html=True)

                if update:
                    instance = (
                        database_session.query(SourceDataTable)
                        .where(SourceDataTable.source == filename)
                        .first()
                    )
                    contents = SourceContents(**instance.source_contents.__dict__)
                    contents.formatted_content = text
                    instance.source_contents = contents
                    database_session.commit()
                    get_db_sources(reset=True)

            with tab3:
                for i, text in enumerate(raw):
                    st.write(f"Page {i+1}/{len(raw)}")
                    st.write(text, unsafe_allow_html=True)

            with tab4:
                tagged_concepts: Dict[str, List[ConceptData]] = defaultdict(list)
                tags:Dict[str, ConceptTaxonomy] = {}
                for concept_id in file_entry.source_concepts:
                    db_concept = get_concept_by_id(concept_id)
                    concept_inst:ConceptData = db_concept.concept_contents
                    for tag in concept_inst.taxonomy:
                        tagged_concepts[tag].append(concept_inst)
                        if tag not in tags:
                            tag_inst = get_concept_category_tag_by_id(tag)
                            if tag_inst:
                                tags[tag] = tag_inst.concept_category_tag

                for concept_tag, concepts in tagged_concepts.items():
                    if concept_tag in tags:
                        st.write(f"### {tags[concept_tag].title}:")
                        st.write(tags[concept_tag].description)
                        # with st.expander("Concept instances"):
                        for concept_inst in concepts:
                            with st.expander(f"Concept: {concept_inst.title}"):
                                st.write(f"#### {concept_inst.id}")
                                st.code(concept_inst.id)
                                sub_col1, sub_col2  = st.columns([1,2])
                                sub_col1.write("References:")
                                sub_col1.write(ref for ref in concept_inst.references)
                                sub_col1.write("Taxonomy:")
                                sub_col1.write(concept_inst.taxonomy)
                                sub_col2.write(f"##### Summary:\n{concept_inst.summary}")
                                sub_col2.write(f"##### Content:\n{'\n'.join(concept_inst.contents)}")
                    else:
                        st.write(f"### Issue with: {concept_tag}:")
                        st.write(tags.keys())

            with tab5:
                # print(f"{file_entry.chroma_collections=}")
                rag_id = st.selectbox(
                    "RAG DB",
                    file_entry.chroma_collections,
                    placeholder="Choose one",
                    index=None,
                )
                # print("RAG ID:", rag_id)

                if rag_id:
                    chroma_collections = get_chroma_collections(rag_id)
                    st.write("##### RAG Items:")

                    rag_items = chroma_collections.get(
                        file_entry.chroma_ids,
                        include=["embeddings", "documents", "metadatas"],
                    )

                    for i, id in enumerate(file_entry.chroma_ids):
                        sub_col1, sub_col2 = st.columns([1, 3])
                        try:
                            index = rag_items["ids"].index(id)
                            with sub_col1:
                                st.write(id)
                                st.write(rag_items["metadatas"][index])
                            with sub_col2:
                                st.write("*Text:*")
                                st.write(rag_items["documents"][index])
                                st.write("_Embeddings:_")
                                st.write(rag_items["embeddings"][index])
                        except Exception as e:
                            st.write(f"Error: {e}")

                    for concept_id in file_entry.source_concepts:
                        concept:ConceptDataTable = get_concept_by_id(concept_id)
                        if any(collection in file_entry.chroma_collections for collection in concept.chroma_collections):
                            st.write(concept.id)
                            sub_col1, sub_col2 = st.columns([1, 3])
                            rag_items = chroma_collections.get(
                                concept.chroma_ids,
                                include=["embeddings", "documents", "metadatas"],
                            )
                            for i, id in enumerate(concept.chroma_ids):
                                sub_col1, sub_col2 = st.columns([1, 3])
                                try:
                                    index = rag_items["ids"].index(id)
                                    with sub_col1:
                                        st.write(id)
                                        st.write(rag_items["metadatas"][index])
                                    with sub_col2:
                                        st.write("*Text:*")
                                        st.write(rag_items["documents"][index])
                                        st.write("_Embeddings:_")
                                        st.write(rag_items["embeddings"][index])
                                except Exception as e:
                                    st.write(f"Error: {e}")
                else:
                    st.write("Select RAG DB first.")


async def main():
    init_db()
    st.title("Manage Uploads")

    if not check_auth():
        return

    file_categories = get_all_categories()
    del_col, col = st.columns([1, 8], vertical_alignment="bottom")
    categories = col.multiselect("Categories", file_categories)

    with del_col.popover(":x:", disabled=not categories, use_container_width=True):
        if st.button(
            f"Are you sure you want to remove {', '.join(categories)}?",
            key=f"delete_button_categories",
            use_container_width=True,
        ):
            files = get_db_sources(categories=categories)
            for file in files.keys():
                delete_db_source(file, commit=False)
            db_commit()
            st.rerun()

    if categories:
        files = get_db_sources(categories=categories)

        if files is not None and len(files) > 0:
            st.header("File database")
        else:
            st.header("No files uploaded yet.")

        for file in files.keys():
            await manage_file(file)
    else:
        st.header("First, choose a categories.")


if __name__ == "__main__":
    asyncio.run(main())
