import os
import sys
import streamlit as st

from langchain_core.messages import BaseMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from chains.init import get_chain
from lib.document_tools import create_document_lists, split_text
from lib.load_env import SETTINGS
from lib.db_tools import (
    FileDataTable,
    delete_db_file,
    get_chroma_collection,
    get_db_files,
    init_db,
)
from lib.document_parse import markdown_to_text
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
def manage_file(filename):
    database_session = init_db()
    file_categories = get_all_categories()
    file_entry = get_db_files()[filename]
    # Add Streamlit editors to edit 'disabled' and 'category_tag' fields

    if "rewrite_text" not in st.session_state:
        st.session_state.rewrite_text = {}
        st.session_state.rewrite_thoughts = {}

    rewrite_text = st.session_state.rewrite_text
    rewrite_thoughts = st.session_state.rewrite_thoughts

    # for i in range(len(query)):
    col1, col2 = st.columns([1, 10], vertical_alignment="center")
    col2.subheader(filename, divider=True)
    with col1.popover(":x:"):
        if st.button(
            f"Are you sure you want to remove {filename}?",
            key=f"delete_button_{filename}",
            use_container_width=True,
        ):
            delete_db_file(filename)
            st.rerun(scope="fragment")
    header_col1, header_col2 = st.columns([1, 4], vertical_alignment="bottom")
    with header_col1:
        show_details = st.toggle("Extend", key=f"show_{filename}")
        disable_checkbox = st.toggle(
            f"Disable", value=file_entry["disabled"], key=f"disable_{filename}"
        )
        st.download_button("Download", file_entry["file_data"], file_name=filename)

    with header_col2:
        with st.form(f"edit_fields_{filename}", border=False):
            col1, col4 = st.columns([4, 1], vertical_alignment="bottom")

            with col1:
                category_tags = st.multiselect(
                    "Category",
                    file_categories,
                    default=file_entry["category_tag"],
                    key=f"category_{filename}",
                )

                chroma_collection = st.text_input(
                    f"Chroma collection",
                    value=", ".join(file_entry["chroma_collection"]),
                    key=f"chroma_{filename}",
                )
            with col4:
                submitted = st.form_submit_button("Save", use_container_width=True)

        if submitted:
            updates = {
                "disabled": disable_checkbox,
                "category_tag": category_tags,
                "chroma_collection": [
                    col.strip() for col in chroma_collection.split(",")
                ],
            }

            changes = False

            for key in updates.keys():
                changes = changes or updates[key] != file_entry[key]

            if changes:
                instance = (
                    database_session.query(FileDataTable)
                    .where(FileDataTable.filename == filename)
                    .first()
                )

            if changes:
                instance = (
                    database_session.query(FileDataTable)
                    .where(FileDataTable.filename == filename)
                    .first()
                )
                instance.disabled = updates["disabled"]
                instance.category_tag = updates["category_tag"]
                instance.chroma_collection = [
                    col.strip() for col in updates["chroma_collection"].split(",")
                ]

            if changes:
                database_session.commit()
                get_db_files(reset=True)
                st.rerun(scope="fragment")

    if show_details:
        with st.container(border=True):
            filename = file_entry["filename"]
            filetype = filename.split(".")[-1]
            summary = file_entry["summary"]
            text = file_entry["formatted_text"]
            thoughts = None  # file_entry["format_thoughts"]
            raw = file_entry["texts"]

            if filename in rewrite_text:
                text = rewrite_text[filename]
                thoughts = rewrite_thoughts[filename]

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Summary", "Formatted", "Unformatted", "RAG"]
            )

            with tab1:
                if st.button(
                    "Rewrite",
                    key=f"llm_summary_rewrite_{filename}",
                    use_container_width=True,
                ):
                    instance = (
                        database_session.query(FileDataTable)
                        .where(FileDataTable.filename == filename)
                        .first()
                    )
                    text = instance.summary
                    summary_texts = instance.texts
                    if (
                        instance.formatted_text is not None
                        and instance.formatted_text != ""
                    ):
                        summary_texts = split_text(
                            instance.formatted_text,
                            SETTINGS.default_llms.instruct.char_limit,
                            128,
                        )
                    with st.spinner("Rewriting"):
                        if summary_texts is not None:
                            # split_texts = split_text("\n".join(texts), CHAR_LIMIT)
                            if len(summary_texts) == 1:
                                shorter_text, shorter_thoughts = llm_edit(
                                    "summary", summary_texts
                                )
                                if shorter_text is not None or shorter_text == "":
                                    text = shorter_text
                                else:
                                    text = summary_texts[0]
                            else:
                                list_of_docs = create_document_lists(
                                    summary_texts, source=filename
                                )
                                # print(f"{ list_of_docs = }")

                                results = get_chain("summary_documents").invoke(
                                    {"context": list_of_docs}
                                )

                                if isinstance(results, tuple) and len(results) == 2:
                                    shorter_text, shorter_thoughts = results
                                else:
                                    shorter_text = results
                                    shorter_thoughts = ""

                                shorter_text = (
                                    shorter_text.content
                                    if isinstance(shorter_text, BaseMessage)
                                    else shorter_text
                                )

                                # shorter_text, shorter_thoughts = llm_edit("summary", [summary_text])

                                if shorter_text is not None:
                                    text = shorter_text

                    st.success("Markdown rewrite complete")

                    if text is not None and text != "" and text != instance.summary:
                        instance.summary = text
                        database_session.commit()
                        get_db_files(reset=True)
                    st.rerun(scope="fragment")
                else:
                    with st.container():
                        st.write(summary, unsafe_allow_html=True)

            with tab2:
                update = False
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    guidance = st.text_input(
                        "Guidance for the rewriter:", key=f"llm_md_guidance_{filename}"
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
                        rewrite_thoughts[filename] = ""

                if rewrite:
                    text = None
                    with st.spinner("Rewriting"):
                        if raw is not None and filetype != "md":
                            text, thoughts = llm_edit("text_formatter", raw, guidance)
                        elif filetype == "md":
                            if len(raw) > 1 or len(raw[0]) > 1000:
                                text, thoughts = llm_edit(
                                    "text_formatter",
                                    [markdown_to_text("\n".join(raw))],
                                    guidance,
                                )
                            else:
                                text = markdown_to_text(raw[0])

                    st.success("Markdown rewrite complete")
                    rewrite_text[filename] = text
                    rewrite_thoughts[filename] = thoughts

                if thoughts != None and len(thoughts) > 0:
                    st.caption(thoughts, unsafe_allow_html=True)

                if text != None and len(text) > 0:
                    st.write(text, unsafe_allow_html=True)

                if update:
                    instance = (
                        database_session.query(FileDataTable)
                        .where(FileDataTable.filename == filename)
                        .first()
                    )
                    instance.formatted_text = text
                    database_session.commit()
                    get_db_files(reset=True)

            with tab3:
                for i, text in enumerate(raw):
                    st.write(f"Page {i+1}/{len(raw)}")
                    st.write(text, unsafe_allow_html=True)

            with tab4:
                # print(f"{file_entry["chroma_collection"]=}")
                rag_id = st.selectbox(
                    "RAG DB",
                    file_entry["chroma_collection"],
                    placeholder="Choose one",
                    index=None,
                )
                # print("RAG ID:", rag_id)

                if rag_id:
                    chroma_collection = get_chroma_collection(rag_id)
                    st.write("##### RAG Items:")

                    rag_items = chroma_collection.get(
                        file_entry["chroma_ids"],
                        include=["embeddings", "documents", "metadatas"],
                    )

                    for i, id in enumerate(file_entry["chroma_ids"]):
                        [sub_col1, sub_col2] = st.columns([1, 3])
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


def main():
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
            files = get_db_files(categories=categories)
            for file in files.keys():
                delete_db_file(file)
            st.rerun()

    if categories:
        files = get_db_files(categories=categories)

        if files is not None and len(files) > 0:
            st.header("File database")
        else:
            st.header("No files uploaded yet.")

        for file in files.keys():
            manage_file(file)
    else:
        st.header("First, choose a category.")


if __name__ == "__main__":
    main()
