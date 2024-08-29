from datetime import datetime
import os
import sys

import gc
from io import StringIO
import sqlalchemy as sqla
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_core.messages import BaseMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from chains.init import get_chain
from lib.db_tools import (
    FileDataTable,
    # JourneyDataTable,
    get_chroma_collection,
    get_db_files,
    init_db,
)

from lib.document_parse import load_pymupdf, markdown_to_text
from lib.document_tools import create_document_lists, split_markdown, split_text
from lib.load_env import SETTINGS
from lib.streamlit_tools import check_auth, get_all_categories, llm_edit

st.set_page_config(
    page_title="TC POC: Upload files",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        'About': """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    }
)

database_session = init_db()

def write_categories(add_new=True):
    st.subheader("File categories")

    file_categories = get_all_categories()

    st.write(f"Current categories: {', '.join(file_categories)}")

    cat_col1, cat_col2 = st.columns([3, 1], vertical_alignment="bottom")
    with cat_col1:
        new_categories_input = st.text_input(
            "Add" if add_new else "Show" + " more categories: _(split with ```,```)_",
            key="file_categories_" + "new" if add_new else "show",
        )

    with cat_col2:
        add_categories = st.button(
            "Add", key="add_file_categories_" + "new" if add_new else "show"
        )

    if add_categories:
        new_categories = new_categories_input.split(",")
        change = False
        for new_category in new_categories:
            if new_category not in file_categories:
                file_categories.append(new_category)
                change = True

        st.session_state.file_categories = file_categories

        if change:
            print(f"New categories {file_categories}")
            st.rerun()


def process_file_contents(
    uploaded_file: UploadedFile, filename, category, overwrite=False
):
    filetype = os.path.basename(uploaded_file.name).split(".")[-1]

    file_exists = database_session.query(
        sqla.exists().where(FileDataTable.filename == filename)
    ).scalar()

    if file_exists and overwrite is not True:
        return

    texts = None
    with st.status(f"Save to DB: {filename}"):

        if (
            filetype == "pdf"
            or filetype == "epub"
            or filetype == "xps"
            or filetype == "mobi"
            or filetype == "fb2"
            or filetype == "cbz"
            or filetype == "svg"
        ):
            parse_bar = st.progress(0, text="Parsing")
            texts = load_pymupdf(uploaded_file, filetype=filetype, progress_cb = lambda x, y: parse_bar.progress(x, text=y))
            if parse_bar:
                parse_bar.progress(1, text="Done.")
                parse_bar.empty()

        elif filetype == "md" or filetype == "txt":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            texts = split_markdown(text)
            # text = uploaded_file.read()
        else:
            st.warning("Unsupported file type")
            return  # Skip unsupported files

        split_texts = []
        for text in texts:
            if len(text) > SETTINGS.default_llms.instruct.char_limit:
                st.write("Splitting to fit context...")
                split_texts = split_texts + split_text(text)
            else:
                split_texts.append(text)

        texts = split_texts
        collections = []
        for cat in category:
            collections.append("rag_" + cat)
        with st.spinner("Saving to database..."):
            if file_exists:
                # If the file exists, get the row and update its text field
                existing_file = (
                    database_session.query(FileDataTable)
                    .filter(FileDataTable.filename == filename)
                    .first()
                )

                existing_file.texts = (
                    texts  # Update the text field with the new content
                )
                existing_file.category_tag = category
                existing_file.last_updated = datetime.now()
                existing_file.file_data = uploaded_file.getvalue()
                existing_file.chroma_collection = collections

                st.success(f"{filename} updated within database successfully.")
            else:


                # If the file does not exist, create a new row
                file = FileDataTable(
                    filename=filename,
                    texts=texts,
                    category_tag=category,
                    chroma_collection=collections,
                    last_updated=datetime.now(),
                    file_data=uploaded_file.getvalue(),
                )
                database_session.add(file)

                st.success(f"{filename} saved to database successfully.")

            database_session.commit()

def process_file_data(filename, category):
    with st.status(f"Document generation: {filename}"):
        file_entry = get_db_files()[filename]
        texts = file_entry["texts"]
        filetype = os.path.basename(filename).split(".")[-1]

        with st.spinner("Rewriting"):
            formatted_text = None
            format_thoughts = None
            if texts is not None and filetype != "md":
                if len(texts) > 1:
                    formatted_text, format_thoughts = llm_edit("text_formatter_compress", texts)
                else:
                    formatted_text, format_thoughts = llm_edit("text_formatter", texts)
            elif filetype == "md":
                if len(texts) > 1 or len(texts[0]) > 1000:
                    formatted_text, format_thoughts = llm_edit(
                        "text_formatter", [markdown_to_text("\n".join(texts))]
                    )
                else:
                    formatted_text = None
            st.write("### Formatted text:")
            if format_thoughts is not None and format_thoughts != "":
                st.write("#### Thoughts")
                st.caption(format_thoughts)
            st.write(formatted_text)

        st.success("Rewrite complete")

        with st.spinner("Summarizing text"):
            summary_text = None
            shorter_thoughts = None
            summary_texts = texts
            if formatted_text is not None and formatted_text != "":
                summary_texts = split_text(formatted_text, SETTINGS.default_llms.instruct.char_limit, 128)

            if summary_texts is not None:
                # split_texts = split_text("\n".join(texts), CHAR_LIMIT)
                if len(summary_texts) == 1:
                    shorter_text, shorter_thoughts = llm_edit("summary", summary_texts)
                    if shorter_text is not None or shorter_text == "":
                        summary_text = shorter_text
                    else:
                        summary_text = summary_texts[0]
                else:
                    list_of_docs = create_document_lists(summary_texts, source=filename)
                    # print(f"{ list_of_docs = }")

                    results = get_chain("summary_documents").invoke(
                        {"context": list_of_docs}
                    )

                    if isinstance(results, tuple) and len(results) == 2:
                        shorter_text, shorter_thoughts = results
                    else:
                        shorter_text = results
                        shorter_thoughts = ''

                    shorter_text = shorter_text.content if isinstance(shorter_text, BaseMessage) else shorter_text

                    # shorter_text, shorter_thoughts = llm_edit("summary", [summary_text])

                    if shorter_text is not None:
                        summary_text = shorter_text

        st.success("Summary complete")

        st.write(f"### Summary")
        if shorter_thoughts is not None and shorter_thoughts != "":
            st.write("#### Thoughts")
            st.caption(shorter_thoughts)
        st.write(summary_text)

        rag_split = []
        rag_ids = []
        rag_metadatas = []
        rag_documents = []
        with st.spinner("Creating RAG"):
            if texts is not None and filetype != "md":
                rag_split = split_text("\n".join(texts), SETTINGS.default_embeddings.default.char_limit, SETTINGS.default_embeddings.default.overlap)
            elif filetype == "md":
                rag_split = split_text(markdown_to_text("\n".join(texts)), SETTINGS.default_embeddings.default.char_limit, SETTINGS.default_embeddings.default.overlap)

            rag_ids = [filename + "_" + str(i) for i in range(len(rag_split))]
            rag_metadatas = [
                {
                    "file": filename,
                    "category": ", ".join(category),
                    "filetype": filetype,
                    "split": i,
                }
                for i in range(len(rag_split))
            ]

            if formatted_text is not None and formatted_text:
                if len(formatted_text) > SETTINGS.default_embeddings.default.char_limit:
                    formatted_split = split_text(formatted_text, SETTINGS.default_embeddings.default.char_limit, SETTINGS.default_embeddings.default.overlap)
                else:
                    formatted_split = [formatted_text]

                rag_split = rag_split + formatted_split
                rag_ids = rag_ids + [
                    filename + "_formatted_" + str(i)
                    for i in range(len(formatted_split))
                ]
                rag_metadatas = rag_metadatas + [
                    {
                        "file": "formatted_" + filename,
                        "thoughts": format_thoughts,
                        "category": ", ".join(category),
                        "filetype": filetype,
                        "split": i,
                    }
                    for i in range(len(formatted_split))
                ]
                # rag_documents = create_document_lists(rag_split, list_of_metadata=rag_metadatas, source=filename)
        st.success("RAG generation complete")
        # Save the filename and text into the database

        # file_exists = database_session.query(
        #     sqla.exists().where(FileDataTable.filename == filename)
        # ).scalar()

        with st.spinner("Saving to database..."):
            # if file_exists:
            # If the file exists, get the row and update its text field
            existing_file = (
                database_session.query(FileDataTable)
                .filter(FileDataTable.filename == filename)
                .first()
            )

            collections = existing_file.chroma_collection
            if len(existing_file.chroma_ids) > 0:
                for collection in collections:
                    vectorstore = get_chroma_collection(collection)
                    vectorstore.delete(existing_file.chroma_ids)

            collections = []

            for cat in category:
                collections.append("rag_" + cat)
                vectorstore = get_chroma_collection("rag_" + cat) #get_vectorstore("rag_" + cat)
                store_complete = False
                retries = 0
                while not store_complete and retries < 3:
                    if (retries > 0):
                        vectorstore.delete(rag_ids)
                    retries += 1
                    vectorstore.add(
                        ids=rag_ids, documents=rag_split, metadatas=rag_metadatas
                    )
                    rag_items = vectorstore.get(
                            rag_ids,
                            include=["embeddings", "documents", "metadatas"],
                        )
                    store_complete = True

                    for rag_id in rag_ids:
                        if rag_id not in rag_items["ids"]:
                            store_complete = False
                            print(f"{rag_id} not in {rag_items["ids"]} - retrying...")
                            break


            existing_file.texts = texts  # Update the text field with the new content
            existing_file.formatted_text = formatted_text
            existing_file.summary = summary_text
            existing_file.category_tag = category
            existing_file.chroma_ids = rag_ids
            existing_file.chroma_collection = collections
            existing_file.last_updated = datetime.now()

            st.success(f"{filename} updated within database successfully.")

            database_session.commit()


def main():
    init_db()
    st.title("Upload Files")

    if not check_auth():
        return

    write_categories()

    file_categories = get_all_categories()

    st.subheader("File uploader")

    default_category = st.multiselect("Default category", file_categories)

    # submitted = None
    uploaded_files = None

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "xps", "epub", "mobi", "fb2", "cbz", "svg", "txt", "md"],
        accept_multiple_files=True,
        disabled=default_category is None or len(default_category) == 0,
    )

    if "existing_files" not in st.session_state:
        st.session_state.existing_files = []
        st.session_state.existing_files_names = []
        st.session_state.existing_files_details = {}
        st.session_state.new_files = []
        st.session_state.new_files_names = []
        st.session_state.new_files_details = {}

    existing_files = st.session_state.existing_files
    existing_files_details = st.session_state.existing_files_details
    existing_files_names = st.session_state.existing_files_names
    new_files = st.session_state.new_files
    new_files_details = st.session_state.new_files_details
    new_files_names = st.session_state.new_files_names

    db_files = get_db_files()

    new_file_count = 0
    unique_files = len(existing_files_names) + len(new_files_names)

    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            # Convert the file to text based on its extension
            filename = os.path.basename(uploaded_file.name)
            filetype = filename.split(".")[-1]

            # file_exists = database_session.query(
            #     sqla.exists().where(FileDataTable.filename == filename)
            # ).scalar()

            file_exists = filename in db_files

            if file_exists and uploaded_file.name not in existing_files_names:
                existing_files.append(uploaded_file)
                existing_files_names.append(uploaded_file.name)
                new_file_count += 1
            elif not file_exists and uploaded_file.name not in new_files_names:
                new_files.append(uploaded_file)
                new_files_names.append(uploaded_file.name)
                new_file_count += 1

    if new_file_count > 0:
        st.write(f"Added {new_file_count} of {unique_files} files to list")

    if len(existing_files) > 0:
        st.subheader("Existing files:")

    for file in existing_files:
        filename = os.path.basename(file.name)
        if filename not in existing_files_details:
            existing_files_details[filename] = {
                "file": file,
                "name": filename,
                "category": db_files[filename]["category_tag"],
            }
        file_data = existing_files_details[filename]

        col1, col2 = st.columns([3, 1])

        with col1:
            file_data["name"] = st.text_input(
                label="Change name or leave it to replace file",
                value=file_data["name"] or filename,
                key=f"existing_filename_{filename}",
            )

        with col2:
            file_data["category"] = st.multiselect(
                "Category",
                file_categories,
                default=file_data["category"],
                key=f"existing_category_{filename}",
            )

        existing_files_details[filename] = file_data

    if len(new_files) > 0:
        st.subheader("New files")

    for file in new_files:
        filename = os.path.basename(file.name)

        col1, col2 = st.columns([3, 1])
        if filename in new_files_details:
            file_data = new_files_details[filename]
        else:
            file_data = {"file": file, "name": filename, "category": default_category}

        with col1:
            file_data["name"] = st.text_input(
                label="Filename",
                value=file_data["name"],
                key=f"filename_new_{filename}",
            )

        with col2:
            file_data["category"] = st.multiselect(
                "Category",
                file_categories,
                default=file_data["category"] or default_category,
                key=f"category_new_{filename}",
            )

        new_files_details[filename] = file_data

    if st.button("Start processing"):
        keys = list(existing_files_details.keys()) + list(new_files_details.keys())
        for filename in keys:
            det = None
            if filename in existing_files_details:
                det = existing_files_details[filename]
            elif filename in new_files_details:
                det = new_files_details[filename]

            process_file_contents(det["file"], det["name"], det["category"])
            # get_db_files(reset=True)
            gc.collect()
        get_db_files(reset=True)
        for filename in keys:
            det = None
            if filename in existing_files_details:
                det = existing_files_details[filename]
            elif filename in new_files_details:
                det = new_files_details[filename]

            process_file_data(det["name"], det["category"])
            gc.collect()

        get_db_files(reset=True)


if __name__ == "__main__":
    main()
