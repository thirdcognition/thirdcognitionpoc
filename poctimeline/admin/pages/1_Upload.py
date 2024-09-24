import asyncio
from datetime import datetime
import io
import os
import re
import sys

import gc
from io import StringIO
from typing import List, Union
import sqlalchemy as sqla
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.elements.lib.mutable_status_container import StatusContainer
from langchain_core.messages import BaseMessage


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from lib.graphs.handle_source import handle_source
from lib.helpers import pretty_print
from lib.db.source import get_db_sources
from lib.db.sqlite import init_db
from lib.models.source import SourceContents, SourceDataTable

from lib.document_parse import load_pymupdf
from lib.document_tools import split_markdown, split_text
from lib.load_env import SETTINGS
from lib.streamlit_tools import check_auth, get_all_categories, graph_call

st.set_page_config(
    page_title="TC POC: Upload files",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)

database_session = init_db()


def validate_category(category: str) -> bool:
    # Check length
    if not 3 <= len(category) <= 63:
        return False
    # Check start and end with alphanumeric character
    if not category[0].isalnum() or not category[-1].isalnum():
        return False
    # Check for valid characters
    if not re.match(r"^[A-Za-z0-9_\-]*$", category) or " " in category:
        return False
    # Check for consecutive periods
    if ".." in category:
        return False
    # Check for IPv4 address
    if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", category):
        return False
    return True


def write_categories(add_new=True) -> Union[List, None]:
    st.subheader("File categories")

    file_categories = get_all_categories()

    st.write(f"Current categories: {', '.join(file_categories)}")

    cat_col1, cat_col2 = st.columns([5, 1], vertical_alignment="bottom")
    new_categories: List[str] = []
    valid = True
    with cat_col1:
        new_categories_input = st.text_input(
            "Add" if add_new else "Show" + " more categories: _(split with ```,```)_",
            key="file_categories_" + "new" if add_new else "show",
        )

        if new_categories_input:
            _new_categories = new_categories_input.split(",")
            for new_category in _new_categories:
                new_category = new_category.strip().replace(" ", "_")

                if not validate_category(new_category):
                    valid = False
                    st.error(
                        f"Invalid category: {new_category}. Please ensure it meets the following requirements: "
                        "1. Contains 3-63 characters, "
                        "2. Starts and ends with an alphanumeric character, "
                        "3. Contains only alphanumeric characters, underscores or hyphens (-), "
                        "4. Does not contain two consecutive periods (..), "
                        "5. Is not a valid IPv4 address."
                    )
                    break
                if new_category not in new_categories:
                    new_categories.append(new_category)

    with cat_col2:
        add_categories = st.button(
            "Add",
            key="add_source_categories_" + "new" if add_new else "show",
            disabled=not valid,
            use_container_width=True,
        )

    if add_categories and valid:
        # new_categories = new_categories_input.replace(' ', '_').split(",")
        change = False
        for new_category in new_categories:
            cat = new_category.strip()
            if cat not in file_categories:
                file_categories.append(cat)
                change = True

        st.session_state.file_categories = file_categories

        if change:
            return new_categories
            # print(f"New categories {file_categories}")
            # st.rerun()


def process_source_contents(
    uploaded_file: UploadedFile, filename, categories, overwrite=False
):
    filetype = os.path.basename(uploaded_file.name).split(".")[-1]

    file_exists = database_session.query(
        sqla.exists().where(SourceDataTable.source == filename)
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
            texts = load_pymupdf(
                uploaded_file,
                filetype=filetype,
                progress_cb=lambda x, y: parse_bar.progress(x, text=y),
            )
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
        for cat in categories:
            collections.append("rag_" + cat)
        with st.spinner("Saving to database..."):
            if file_exists:
                # If the file exists, get the row and update its text field
                existing_file = (
                    database_session.query(SourceDataTable)
                    .filter(SourceDataTable.source == filename)
                    .first()
                )

                existing_file.texts = (
                    texts  # Update the text field with the new content
                )
                existing_file.category_tags = categories
                existing_file.last_updated = datetime.now()
                existing_file.file_data = uploaded_file.getvalue()
                existing_file.chroma_collections = collections

                st.success(f"{filename} updated within database successfully.")
            else:

                # If the file does not exist, create a new row
                file = SourceDataTable(
                    source=filename,
                    texts=texts,
                    category_tags=categories,
                    chroma_collections=collections,
                    last_updated=datetime.now(),
                    file_data=uploaded_file.getvalue(),
                )
                database_session.add(file)

                st.success(f"{filename} saved to database successfully.")

            database_session.commit()


async def process_source(
    categories, filename=None, url=None, file: io.BytesIO = None, overwrite=False
):
    with st.status(f"Document generation: {filename}"):
        # file_entry = get_db_sources()[filename]
        contents: str = None
        # texts = file_entry.texts
        # filetype = os.path.basename(filename).split(".")[-1]

        result = await graph_call(
            categories=categories,
            filename=filename,
            url=url,
            file=file,
            overwrite=overwrite,
            graph=handle_source,
            # collect_concepts=True,
        )

        pretty_print(result)

        # taxonomy = await graph_call(
        #     categories=categories,
        #     filename=filename,
        #     url=url,
        #     file=file,
        #     overwrite=overwrite,
        #     graph=find_concepts
        #     # collect_concepts=True,
        # )
        # with st.spinner("Processing"):

        if "process_text_result" in result:
            contents = result["process_text_result"]["summary"]
        else:
            contents = result["content_summaries"]

        # if "collected_concepts" in taxonomy:
        #     concepts = taxonomy["collected_concepts"]
        # else:
        #     concepts = taxonomy["concepts"]

        with st.container(height=400, border=False):
            st.write(f"### Summary")
            st.write(contents)
            # st.write(f"### Found concepts")
            # for concept in concepts:
            #     st.write(concept)


async def main():
    init_db()
    st.title("Upload Files")

    if not check_auth():
        return

    new_categories = write_categories()

    file_categories = get_all_categories()
    if new_categories is not None and len(new_categories) > 0:
        st.session_state.selected_new_categories = new_categories
        print("rerun")
        st.rerun()

    st.subheader("File uploader")

    default_categories = st.multiselect(
        "Default categories",
        file_categories,
        default=(
            st.session_state.selected_new_categories
            if "selected_new_categories" in st.session_state
            else None
        ),
    )

    # submitted = None
    uploaded_files = None

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "xps", "epub", "mobi", "fb2", "cbz", "svg", "txt", "md"],
        accept_multiple_files=True,
        disabled=default_categories is None or len(default_categories) == 0,
    )

    if "existing_files" not in st.session_state:
        st.session_state.existing_files = []
        st.session_state.existing_files_names = []
        st.session_state.existing_files_details = {}
        st.session_state.new_files = []
        st.session_state.new_files_names = []
        st.session_state.new_files_details = {}

    existing_files: List[UploadedFile] = st.session_state.existing_files
    existing_files_details = st.session_state.existing_files_details
    existing_files_names = st.session_state.existing_files_names
    new_files: List[UploadedFile] = st.session_state.new_files
    new_files_details = st.session_state.new_files_details
    new_files_names = st.session_state.new_files_names

    db_sources = get_db_sources()

    new_source_count = 0
    unique_files = len(existing_files_names) + len(new_files_names)

    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            # Convert the file to text based on its extension
            filename = os.path.basename(uploaded_file.name)
            # filetype = filename.split(".")[-1]

            # file_exists = database_session.query(
            #     sqla.exists().where(SourceDataTable.source == filename)
            # ).scalar()

            file_exists = filename in db_sources

            if file_exists and uploaded_file.name not in existing_files_names:
                existing_files.append(uploaded_file)
                existing_files_names.append(uploaded_file.name)
                new_source_count += 1
            elif not file_exists and uploaded_file.name not in new_files_names:
                new_files.append(uploaded_file)
                new_files_names.append(uploaded_file.name)
                new_source_count += 1

    if new_source_count > 0:
        st.write(f"Added {new_source_count} of {unique_files} files to list")

    if len(existing_files) > 0:
        st.subheader("Existing files:")

    for file in existing_files:
        filename = os.path.basename(file.name)
        if filename not in existing_files_details:
            existing_files_details[filename] = {
                "file": file,
                "name": filename,
                "categories": db_sources[filename].category_tags,
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
            file_data["categories"] = st.multiselect(
                "Category",
                file_categories,
                default=file_data["categories"],
                key=f"existing_categories_{filename}",
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
            file_data = {
                "file": file,
                "name": filename,
                "categories": default_categories,
            }

        with col1:
            file_data["name"] = st.text_input(
                label="Filename",
                value=file_data["name"],
                key=f"filename_new_{filename}",
            )

        with col2:
            file_data["categories"] = st.multiselect(
                "Category",
                file_categories,
                default=file_data["categories"] or default_categories,
                key=f"categories_new_{filename}",
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

            await process_source(
                det["categories"], filename=filename, file=det["file"], overwrite=True
            )

        get_db_sources(reset=True)


if __name__ == "__main__":
    asyncio.run(main())
