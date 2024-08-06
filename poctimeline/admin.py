from datetime import datetime
import gc
from io import StringIO
import numbers
import os
import re
import textwrap
from typing import Any, Dict, Iterable, List, Sequence, Union
from bs4 import BeautifulSoup
import fitz
from markdown import markdown
import numpy as np
import sqlalchemy as sqla

# import Column, Integer, String, Text, DateTime, Boolean, create_engine, exists
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from db_tables import (
    Base,
    FileDataTable,
    JourneyDataTable,
    get_db_files,
    get_db_journey,
    init_db,
)

from rapidocr_onnxruntime import RapidOCR

# from rapidocr_paddle import RapidOCR  # type: ignore
from chain import (
    INSTRUCT_CHAR_LIMIT,
    SQLITE_DB,
    create_document_lists,
    get_chain,
    get_chroma_collection,
    get_journey_format_example,
    get_vectorstore,
    handle_thinking,
    semantic_splitter,
    split_markdown,
    split_text,
    validate_json_format,
)

# Define the database and table
# SQLITE_DB = "../databases/files.db"
# SQLITE_DB = "d:/tmp/antler_db/files.db"

database_session = init_db()

# Create the database and table if they don't exist yet

ocr = None


def extract_from_images_with_rapidocr(
    images: Sequence[Union[Iterable[np.ndarray], bytes]],
    parse_bar: st.progress,
    start,
    step,
    ocrs,
) -> str:

    global ocr

    text = ""
    total = len(images)
    i = 0

    ocr = ocr or RapidOCR(
        use_gpu=True, det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True
    )

    for img in images:
        parse_bar.progress(start + step * i / total, "Analysing image")

        i = i + 1
        result = None

        if img["xref"] != None and img["xref"] in ocrs:
            result = ocrs[img["xref"]]
        else:
            result, _ = ocr(img["img"], use_det=True, use_cls=True, use_rec=True)

            if img["xref"] != None:
                ocrs[img["xref"]] = result

        if result and len(result) > 3:
            result = [text[1] for text in result]
            text += f"\nImage {i}:\n\n"
            text += " \n".join(result)
            text += "\n\n"
    return text


def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    parse_bar: st.progress,
    start,
    step,
    xrefs,
    ocrs,
) -> str:
    """Extract images from page and get the text with RapidOCR."""

    img_list = page.get_images()
    imgs = []
    total = len(img_list)
    if total < 1:
        return ""

    substep_total = step / 2
    substep = substep_total / total
    i = 0

    for img in img_list:
        parse_bar.progress(start + substep_total * i / total, "Loading image")
        i += 1

        xref = img[0]

        buf = None

        if xref == None or xref not in xrefs:
            pix = fitz.Pixmap(doc, xref)
            buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, -1
            )
            if xref != None:
                xrefs[xref] = buf
        else:
            buf = xrefs[xref]

        imgs.append({"xref": xref, "img": buf})
    return extract_from_images_with_rapidocr(
        imgs, parse_bar, start + substep_total, step - substep_total, ocrs
    )


def recursive_find(obj, key, _ret=[]):
    if key in obj:
        # print("found text: " + obj[key])
        return [str(obj[key])]
    resp = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) or isinstance(v, list):
                resp += recursive_find(v, key, _ret)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) or isinstance(v, list):
                resp += recursive_find(v, key, _ret)  # added return statement

    return _ret + resp


# Function to convert PDF to text using PyMuPDFLoader
def load_pymupdf(file: UploadedFile, filetype):
    st.write("Processing file... ")
    parse_bar = st.progress(0, text="Parsing")
    doc = fitz.open(stream=file.read(), filetype=filetype)
    text = ""

    page_percentage_total = 0.1
    total_left = 1 - page_percentage_total

    i = 0
    total = len(doc)
    step = total_left / total

    xrefs = {}
    ocrs = {}
    chunks = []

    text = ""

    for page in doc:
        parse_bar.progress(
            page_percentage_total / total * i + i * step, f"Parsing page {page.number}"
        )
        page_string = re.sub(" {2,}", " ", page.get_text())
        page_string += extract_images_from_page(
            doc,
            page,
            parse_bar,
            page_percentage_total / total * i + i * step,
            step,
            xrefs,
            ocrs,
        )

        text += page_string

        i += 1

    chunks = semantic_splitter(text)

    parse_bar.progress(1, text="Done.")
    parse_bar.empty()

    return chunks


def markdown_to_text(markdown_string):
    """Converts a markdown string to plaintext"""

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r"<pre>(.*?)</pre>", "\1", html)
    html = re.sub(r"<code>(.*?)</code>", "\1", html)
    html = html.replace("```markdown", "")
    html = html.replace("```", "")

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = "".join(soup.findAll(string=True))

    return text


def llm_edit(chain, texts, guidance=None, min_len=100, force=False) -> tuple[str, str]:
    text = ""
    thoughts = ""

    i = 0
    total = len(texts)

    if total > 1 and chain == "summary":
        total += 1

    if not force and (texts == None or texts[0] == None or total == 1 and len(texts[0]) < 1000):
        return None, None

    bar = st.progress(0, text="Processing...")

    if total > 1:
        inputs = []
        for sub_text in texts:
            bar.progress(i / total, "Processing...")
            i += 1
            _text = re.sub(r"[pP]age [0-9]+:", "", sub_text)
            _text = re.sub(r"[iI]mage [0-9]+:", "", _text)
            input = {"context": _text}

            guided_llm = ""
            if guidance is not None and guidance != "":
                input["question"] = guidance
                guided_llm = "_guided"

            inputs.append(input)

            mid_results, mid_thoughts = handle_thinking((lambda: get_chain(chain + guided_llm).invoke(input)["text"]), min_len=min_len)

            text += mid_results + "\n\n"
            thoughts += mid_thoughts + "\n\n"

    else:
        _text = re.sub(r"[pP]age [0-9]+", "", texts[0])
        _text = re.sub(r"[iI]mage [0-9]+", "", _text)
        text = _text

    bar.progress((total - 1) / total, text="Processing...")

    if chain == "summary":
        text = markdown_to_text(text)

        input = {"context": text}

        guided_llm = ""

        if guidance is not None and guidance != "":
            guided_llm = "_guided"
            input["question"] = guidance

        text, thoughts = handle_thinking((lambda: get_chain(chain + guided_llm).invoke(input)["text"]), min_len=min_len)

    bar.empty()

    return text, thoughts


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
            texts = load_pymupdf(uploaded_file, filetype=filetype)
        elif filetype == "md" or filetype == "txt":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            texts = split_markdown(text)
            # text = uploaded_file.read()
        else:
            st.warning("Unsupported file type")
            return  # Skip unsupported files

        split_texts = []
        for text in texts:
            if len(text) > INSTRUCT_CHAR_LIMIT:
                st.write("Splitting to fit context...")
                split_texts = split_texts + split_text(text)
            else:
                split_texts.append(text)

        texts = split_texts

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

                st.success(f"{filename} updated within database successfully.")
            else:
                # If the file does not exist, create a new row
                file = FileDataTable(
                    filename=filename,
                    texts=texts,
                    category_tag=category,
                    last_updated=datetime.now(),
                    file_data=uploaded_file.getvalue(),
                )
                database_session.add(file)

                st.success(f"{filename} saved to database successfully.")

            database_session.commit()


def process_file_data(filename, category):
    with st.status(f"RAG generation: {filename}"):
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
            if texts is not None:
                # split_texts = split_text("\n".join(texts), CHAR_LIMIT)
                if len(texts) == 1:
                    shorter_text = ''
                    while shorter_text == '' or len(shorter_text) < 10:
                        shorter_text, shorter_thoughts = llm_edit("summary", texts)
                    if shorter_text is not None:
                        summary_text = shorter_text
                    else:
                        summary_text = texts[0]
                else:
                    list_of_docs = create_document_lists(texts, source=filename)
                    # print(f"{ list_of_docs = }")

                    summary_text, shorter_thoughts = handle_thinking((lambda: get_chain("summary_documents").invoke(
                        {"input_documents": list_of_docs}
                    )["output_text"]))

                    # print(f"{ summary_text = } \n\n { shorter_thoughts = }")

                    shorter_text = ''
                    while shorter_text == '' or len(shorter_text) < 10:
                        shorter_text, shorter_thoughts = llm_edit("summary", [summary_text])

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
        with st.spinner("Creating RAG"):
            if texts is not None and filetype != "md":
                rag_split = split_text("\n".join(texts), 128 * 6, 128)
            elif filetype == "md":
                rag_split = split_text(markdown_to_text("\n".join(texts)), 128 * 6, 128)

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
                if len(formatted_text) > (128 * 6):
                    formatted_split = split_text(formatted_text, 128 * 6, 128)
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

            collections.append("rag_all")
            vectorstore = get_vectorstore("rag_all")
            vectorstore.add_texts(ids=rag_ids, texts=rag_split, metadatas=rag_metadatas)

            # print(f"existing_ids = {existing_file.chroma_ids}")
            # print(f"{rag_split =}")

            for cat in category:
                collections.append("rag_" + cat)
                vectorstore = get_vectorstore("rag_" + cat)
                vectorstore.add_texts(
                    ids=rag_ids, texts=rag_split, metadatas=rag_metadatas
                )

            existing_file.texts = texts  # Update the text field with the new content
            existing_file.formatted_text = formatted_text
            existing_file.summary = summary_text
            existing_file.category_tag = category
            existing_file.chroma_ids = rag_ids
            existing_file.last_updated = datetime.now()

            st.success(f"{filename} updated within database successfully.")

            database_session.commit()


def get_all_categories():
    if "file_categories" not in st.session_state:
        uniq_categories = []
        categories = database_session.query(FileDataTable.category_tag).distinct()
        for items in categories:
            for category_list in items:
                if len(category_list) > 0:
                    for category in category_list:
                        if category not in uniq_categories:
                            uniq_categories.append(category)

        st.session_state.file_categories = uniq_categories
    else:
        uniq_categories = st.session_state.file_categories

    return uniq_categories


def write_categories(add_new=True):
    st.subheader("File categories")

    file_categories = get_all_categories()

    st.write(f"Current categories: {', '.join(file_categories)}")

    cat_col1, cat_col2 = st.columns([3, 1])
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


def upload_files_ui():

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
                # database_session.query(FileDataTable)
                # .where(FileDataTable.filename == filename)
                # .first()
                # .category_tag,
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


def manage_file(filename):
    file_categories = get_all_categories()
    file_entry = get_db_files()[filename]
    # Add Streamlit editors to edit 'disabled' and 'category_tag' fields

    if "rewrite_text" not in st.session_state:
        st.session_state.rewrite_text = {}
        st.session_state.rewrite_thoughts = {}

    rewrite_text = st.session_state.rewrite_text
    rewrite_thoughts = st.session_state.rewrite_thoughts

    # for i in range(len(query)):

    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        show_details = st.toggle("Show file", key=f"show_{filename}")
        st.download_button("Download", file_entry["file_data"], file_name=filename)

    with header_col2:
        st.subheader(filename)

    with st.form(f"edit_fields_{filename}"):
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

        with col1:
            delete_checkbox = st.checkbox(
                f"Delete", value=False, key=f"delete_{filename}"
            )
            disable_checkbox = st.checkbox(
                f"Disabled", value=file_entry["disabled"], key=f"disable_{filename}"
            )

        with col2:
            category_tags = st.multiselect(
                "Category",
                file_categories,
                default=file_entry["category_tag"],
                key=f"category_{filename}",
            )

        with col3:
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
            "chroma_collection": [col.strip() for col in chroma_collection.split(",")],
        }

        changes = False

        for key in updates.keys():
            changes = changes or updates[key] != file_entry[key]

        if changes or delete_checkbox:
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

        if delete_checkbox:
            database_session.delete(instance)

        if delete_checkbox or changes:
            database_session.commit()
            get_db_files(reset=True)
            st.rerun()

    if show_details:
        with st.container(border=True):
            filename = file_entry["filename"]
            filetype = filename.split(".")[-1]
            summary = file_entry["summary"]
            text = file_entry["formatted_text"]
            raw = file_entry["texts"]

            rag_id = file_entry["chroma_collection"][0]

            chroma_collection = get_chroma_collection(rag_id)

            rag_items = chroma_collection.get(
                file_entry["chroma_ids"],
                include=["embeddings", "documents", "metadatas"],
            )

            if filename in rewrite_text:
                text = rewrite_text[filename]
                thoughts = rewrite_thoughts[filename]

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Summary", "Formatted", "Unformatted", "RAG"]
            )

            with tab1:
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
                    with st.spinner("Rewriting"):
                        text = None
                        thoughts = None
                        if raw is not None and filetype != "md":
                            text, thoughts = handle_thinking(llm_edit(
                                "text_formatter", raw, guidance
                            ))
                        elif filetype == "md":
                            if len(raw) > 1 or len(raw[0]) > 1000:
                                text, thoughts = llm_edit(
                                    "text_formatter",
                                    [markdown_to_text("\n".join(raw))],
                                    guidance
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
                # st.write(rag_items)
                for i, id in enumerate(rag_items["ids"]):
                    [sub_col1, sub_col2] = st.columns([1, 3])
                    with sub_col1:
                        st.write(id)
                        st.write(rag_items["metadatas"][i])
                    with sub_col2:
                        st.write("*Text:*")
                        st.write(rag_items["documents"][i])
                        st.write("_Embeddings:_")
                        st.write(rag_items["embeddings"][i])


def gen_journey_doc(list_of_strings = []) -> tuple[str, str]:
    text = "\n".join(list_of_strings)
    list_of_thoughts = []
    thoughts = ''

    bar = st.progress(0, text="Compressing journey document")

    reduce = False

    reduce = len(text) > INSTRUCT_CHAR_LIMIT

    # print(f"{reduce = } ({len(text)})")

    if reduce:
        list_of_docs = create_document_lists(list_of_strings)

        chain = get_chain("reduce_journey_documents")

        list_of_strings = []
        list_of_thoughts = []
        total = len(list_of_docs)
        for i, document in enumerate(list_of_docs):
            bar.progress(i / total, text=f"Compressing page {i+1}/{total}")
            result, thinking = handle_thinking((lambda: chain.invoke({"input_documents": [document]})["output_text"]))
            list_of_strings.append(result[0])
            list_of_thoughts.append(result[1])

        text = "\n".join(list_of_strings)
        thoughts = "\n".join(list_of_thoughts)

        reduce = len(text) > INSTRUCT_CHAR_LIMIT
        if reduce:
            bar.progress(1 - 1/total, text="Result too long, 2nd pass")
            list_of_docs = create_document_lists(list_of_strings, list_of_thoughts)
            text, thoughts = handle_thinking(
                (lambda: chain.invoke(
                    {
                        "input_documents": list_of_docs,
                        # "amount": amount,
                        # "format_example": get_journey_format_example(amount)
                    }
                )["output_text"])
            )
        bar.progress(1.0, text="Compression complete")

    bar.empty()

    return text, thoughts

    # return handle_thinking((lambda: get_chain("journey_text").invoke(
    #     {
    #         "context": text,
    #         "amount": amount,
    #         "format_example": get_journey_format_example(amount),
    #     }
    # )["text"]))

def gen_journey(content, amount=10) -> Dict:
    bar = st.progress(0, text="Generating curriculum")

    journey_steps_chain = get_chain("journey_steps")
    success = False
    steps: str = None
    retries = 0
    while not success and retries < 3:
        retries += 1
        bar.progress(0, text="Generate subjects for each day")
        steps = (journey_steps_chain.invoke(
            {
                "context": content,
                "amount": amount,
                # "format_example": get_journey_format_example(amount)
            }
        )["text"])
        steps = re.sub(r':\s*\n', ': ', steps)
        steps = re.sub(r'\n\s*:', ':', steps)
        steps = "\n".join([step.strip() for step in steps.split("\n") if step.strip()])
        correct_response = False
        resp_retr = 0
        while(not correct_response and resp_retr < 3):
            resp_retr += 1
            bar.progress(0.1, text="Verify the generated list of subjects")
            check_response = (get_chain("check").invoke({
                "context": steps,
                "options": "if matches the format respond: yes, if matches the format but not right amount of items respond: maybe, if does not match respond: no",
                # "expected_count": "",
                "expected_count": f"Expected approximately {amount} items.",
                "count": len(steps.split("\n")),
                "format":
"""
Format for 5 items:
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
"""
            })["text"])
            resp = check_response.lower().split('\n')[0].strip()
            print(f"{resp = }")
            correct_response = resp in ["yes", "y", "no", "n", "maybe", "m"]
            success = resp in ["yes", "y", "maybe", "m"]

    list_of_steps = [step.strip() for step in steps.split("\n") if step.strip()]

    days = []
    total_steps = len(list_of_steps)
    for i, subject in enumerate(list_of_steps):
        success = False
        retries = 0
        while not success and retries < 3:
            retries += 1
            bar.progress(0.1 + (0.25 * (i+1)/total_steps), text=f"Generating curriculum for day {i+1} of {total_steps}")
            sub_steps_response = (get_chain("journey_substeps").invoke(
                {
                    "context": content,
                    "subject": subject,
                }
            )["text"])
            sub_steps_response = re.sub(r':\s*\n', ': ', sub_steps_response)
            sub_steps_response = re.sub(r'\n\s*:', ':', sub_steps_response)
            sub_steps_response = "\n".join([step.strip() for step in sub_steps_response.split("\n") if step.strip()])
            correct_response = False
            success = True
            resp_retr = 0
            while(not correct_response and resp_retr < 3):
                resp_retr += 1
                bar.progress(0.1 + (0.25 * (i+1.5)/total_steps), text=f"Verify step for day {i+1} of {total_steps}")
                check_response: str = (get_chain("check").invoke({
                    "context": sub_steps_response,
                    "options": "if matches the format respond: yes, if matches the format but not right amount of items respond: maybe, if does not match respond: no",
                    "expected_count": f"Expected maximum of 5 items.",
                    "count": len(sub_steps_response.split("\n")),
                    "format":
"""
Format for 5 items:
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
Title: Description (optional)
"""
                })["text"])
                resp = check_response.lower().split('\n')[0].strip()
                correct_response = resp in ["yes", "y", "no", "n", "maybe", "m"]
                success = resp in ["yes", "y", "maybe", "m"]
        sub_steps = [step.strip() for step in sub_steps_response.split("\n") if step.strip()]
        if success or retries >= 3:
            days.append({"subject": subject, "steps": sub_steps})


    total_items = sum(len(item["steps"]) for item in days)
    cur_item = 0
    for i, day in enumerate(days):
        for j, item in enumerate(day["steps"]):
            bar.progress(0.35 + (0.65 * cur_item/total_items), text=f"Generating curriculum for day {i+1}: item {j+1} of {len(day["steps"])}")
            class_content, sub_steps_thoughts = handle_thinking(
                (lambda: get_chain("journey_step_details").invoke(
                    {
                        "context": content,
                        "subject": item,
                    }
                )["text"])
            )
            class_intro, sub_steps_thoughts = handle_thinking(
                (lambda: get_chain("journey_step_intro").invoke(
                    {
                        "context": class_content,
                        "subject": item,
                    }
                )["text"]), 50
            )
            days[i]["steps"][j] = {
                "subject": item,
                "content": class_content,
                "intro": class_intro
            }
            cur_item = cur_item + 1
        day_intro, sub_steps_thoughts = handle_thinking(
            (lambda: get_chain("journey_step_intro").invoke(
                {
                    "context": "\n".join([day["steps"][j]['intro'] for j in range(len(day["steps"]))]),
                    "subject": day["subject"],
                }
            )["text"]), 50
        )
        days[i] = {
            "title": day["subject"],
            "intro": day_intro,
            "steps": day["steps"]
        }

    title, title_thoughts = handle_thinking(
        (lambda: get_chain("action").invoke(
            {
                "context": "\n".join([f"Day {i+1}: {day['intro']}" for i, day in enumerate(days)]),
                "action": "Summarize context with 10 words or less to a title for the curriculum",
            }
        )["text"])
    )

    summary, summary_thoughts = llm_edit("summary", [f"Day {i+1}: {day['intro']}" for i, day in enumerate(days)], "Summarize the following list of daily intros into a description of the whole curriculum.", 10, force=True)

    bar.progress(1.0, text="Curriculum generation complete.")
    bar.empty()
    return {
        "title": title,
        "summary": summary,
        "days": days
    }

def gen_journey_json(source):
    result = get_chain("journey_json").invoke({"input": source})["text"]
    for i, item in enumerate(result):
        item["_source"] = source
        result[i] = validate_json_format(item)

    return result


def get_journey_gen(journey_name):
    # st.subheader("Journey generator")
    file_categories = get_all_categories()

    default_category = st.multiselect("Filter files by category", file_categories)

    db_files = get_db_files()
    shown_files = {}

    if default_category is None or len(default_category) < 1:
        st.write("Select category tag(s) first to see available files.")
    else:
        for filename in db_files.keys():
            category_tags = db_files[filename]["category_tag"]

            if len([i for i in category_tags if i in default_category]) > 0:
                shown_files[filename] = db_files[filename]

    if "gen_checklist_from" not in st.session_state:
        st.session_state.gen_checklist_from = {}

    gen_from = st.session_state.gen_checklist_from

    if len(shown_files) > 0:
        st.write("#### Available files")
        for filename in shown_files.keys():
            col1, col2 = st.columns([1, 5])
            file = shown_files[filename]

            with col1:
                select = st.checkbox(
                    "Use",
                    value=filename in gen_from.keys(),
                    key=f"select_for_journey_gen_{filename}",
                )

                if select:
                    gen_from[filename] = file

            with col2:
                st.write(filename)

    journey_details = {"journeyname": journey_name}
    if len(gen_from) > 0:
        journey_details["files"] = list(gen_from.keys())
        col1, col2 = st.columns([5, 1])
        amount = 0
        with col1:
            amount = st.number_input(
                "Number of subjects to generate (approximate)", min_value=1, max_value=20, value=5
            )
        if col2.button("Generate"):
            with st.status(f"Building journey"):
                list_of_strings = []
                for filename in gen_from.keys():
                    if (
                        gen_from[filename]["formatted_text"] is not None
                        and gen_from[filename]["formatted_text"] != ""
                    ):
                        list_of_strings.append(gen_from[filename]["formatted_text"])
                    else:
                        list_of_strings.append(
                            markdown_to_text("\n".join(gen_from[filename]["texts"]))
                        )

                compressed = ''

                with st.spinner("Generating journey document"):
                    compressed, compress_thoughts = gen_journey_doc(list_of_strings)

                st.success("Generating journey document done.")

                structured_journey = None
                with st.spinner("Generating curriculum"):
                    structured_journey = gen_journey(compressed, amount)

                journey_details["days"] = structured_journey["days"]
                journey_details["title"] = structured_journey["title"]
                journey_details["summary"] = structured_journey["summary"]

                save_journey(journey_name, journey_details)

                with st.spinner("Generating JSON for journey"):
                    bar = st.progress(0, "Generating JSON for journey")

                    journey_steps = []

                    total_items = sum(len(item["steps"]) for item in journey_details["days"])
                    cur_item = 0
                    for i, day in enumerate(journey_details["days"]):
                        for j, step in enumerate(day["steps"]):
                            journey_steps = journey_steps + gen_journey_json(
                                f"""
                                    Subject:
                                    {step["subject"]}
                                    Intro:
                                    {step["intro"]}
                                    Content:
                                    {step["content"]}
                                """
                            )
                            bar.progress(cur_item/total_items, f"Generating JSON for Day {i}: Class {j} of {len(day)}")
                        cur_item += 1

                    journey_details["days"][i]["json"] = journey_steps
                    bar.empty()
                st.success("Generating JSON for journey done.")

            journey_details["__complete"] = True

    return journey_details


def edit_journey_step(journey_name, journey, day_index, step_index):
    if journey["days"][day_index]["steps"][step_index].get("json", None):
        step = journey["days"][day_index]["steps"][step_index]["json"]
        col1, col2 = st.columns([1, 3])
        col1.write(f"##### Step {step_index+1}:")
        journey["days"][day_index]["steps"][step_index]["json"]["subject"] = col2.text_input(
            "Subject", value=step["subject"], key=f"journey_step_subject_{journey_name}_{step_index}"
        )

        journey["days"][day_index]["steps"][step_index]["json"]["intro"] = col2.text_area(
            "Intro",
            value=step["intro"],
            key=f"journey_step_intro_{journey_name}_{step_index}",
        )

        journey["days"][day_index]["steps"][step_index]["json"]["content"] = col2.text_area(
            "Content",
            value=step["content"],
            key=f"journey_step_content_{journey_name}_{step_index}",
        )

        if not isinstance(journey["days"][day_index]["steps"][step_index]["json"]["priority"], numbers.Number):
            journey["days"][day_index]["steps"][step_index]["json"]["priority"] = step["priority"] = 1

        journey["days"][day_index]["steps"][step_index]["json"]["priority"] = col2.select_slider(
            "Priority",
            options=list(range(1, 6)),
            value=max(1, min(5, step["priority"])),
            key=f"journey_step_priority_{journey_name}_{step_index}",
        )

        col1, col2 = st.columns([1, 3])
        col1.write(f"##### Actions:")
        for j, action in enumerate(journey["days"][day_index]["steps"][step_index]["json"]["actions"]):
            journey["days"][day_index]["steps"][step_index]["json"]["actions"][j] = col2.text_area(
                f"Action {j+1}",
                value=action,
                key=f"journey_step_actions_{journey_name}_{step_index}_{j}",
            )
    else:
        step = journey["days"][day_index]["steps"][step_index]
        col1, col2 = st.columns([1, 3])
        col1.write(f"##### Step {step_index+1}:")
        journey["days"][day_index]["steps"][step_index]["subject"] = col2.text_input(
            "Subject", value=step["subject"], key=f"journey_step_subject_{journey_name}_{step_index}"
        )

        journey["days"][day_index]["steps"][step_index]["intro"] = col2.text_area(
            "Intro",
            value=step["intro"],
            key=f"journey_step_intro_{journey_name}_{step_index}",
        )

    return journey


def edit_journey_details(journey_name, journey:Dict):
    # print(journey_name, journey)
    if "title" in journey.keys():
        col1, col2 = st.columns([1, 3])
        col1.write("Journey title:")
        journey["title"] = col2.text_input(
            f"Journey title", value=journey["title"], key=f"journey_title_{journey_name}"
        )
    if "summary" in journey.keys():
        col1, col2 = st.columns([1, 3])
        col1.write("Journey summary:")
        journey["summary"] = col2.text_input(
            f"Journey summary", value=journey["summary"], key=f"journey_summary_{journey_name}"
        )

    if "files" in journey.keys():
        col1, col2 = st.columns([1, 3])
        col1.write("Files used:")
        col2.write("* " + "\n* ".join(journey["files"]))

    return journey


def save_journey(journey_name, journey:Dict):
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        print("Modify journey")
        journey_db.files = journey.get("files", None)
        journey_db.days = journey.get("days", None)
        journey_db.title = journey.get("title", None)
        journey_db.summary = journey.get("summary", None)

        journey_db.last_updated = datetime.now()
        # st.success(f"{filename} updated within database successfully.")

    else:
        print("Create journey")
        journey_db = JourneyDataTable(
            journeyname=journey_name,
            files=journey.get("files", None),
            days=journey.get("days", None),
            title=journey.get("title", None),
            summary=journey.get("summary", None),
            last_updated=datetime.now(),
        )
        database_session.add(journey_db)
    database_session.commit()


def edit_journey(journey_name, journey:Dict):
    st.header(f"Edit journey: {journey_name}")

    journey = edit_journey_details(journey_name, journey)

    if "days" in journey.keys():
        st.subheader("Journey days")
        for i, day in enumerate(journey["days"]):
            st.write(f"### Journey day {i}: {day['title']}")
            for j, step in enumerate(day["steps"]):
                with st.expander(f"Day step {j}: {step['title']}"):
                    journey = edit_journey_step(journey_name, journey, i, j)

    if st.button("Save into database"):
        save_journey(journey_name, journey)


# Streamlit app
def main():
    init_db()
    st.title("Admin interface for TC POC")

    page_tab1, page_tab2, page_tab3 = st.tabs(
        ["Upload files", "Manage files", "Manage journeys"]
    )

    with page_tab1:
        upload_files_ui()

    with page_tab2:
        files = get_db_files()

        # st.header("Generate RAG")

        # file_categories = get_all_categories()

        # selected_categories = st.multiselect(
        #     "Select file categories to generate RAG from", file_categories
        # )

        st.header("File database")

        for file in files.keys():
            manage_file(file)
        # st.subheader("Chirp chirp...")

    with page_tab3:
        db_journey = get_db_journey()

        with st.container(border=True):
            if "journey_create_data" in st.session_state:
                journey_create = st.session_state.journey_create_data
            else:
                journey_create = None

            if "creating_journey" not in st.session_state:
                st.header("Create new journey")
                col1, col2 = st.columns([2, 1])

                journey_name = col1.text_input(
                    "Unique name for the journey", value="test"
                )
                if col2.button("Create", disabled=journey_name in db_journey.keys()):
                    st.session_state.creating_journey = journey_name
                    st.rerun()

            if "creating_journey" in st.session_state and (
                journey_create is None or "__complete" not in journey_create.keys()
            ):
                journey_name = st.session_state.creating_journey
                st.header(f"Create new journey: {journey_name}")
                journey_create = get_journey_gen(journey_name)
                st.session_state.journey_create_data = journey_create
                if "__complete" in journey_create.keys():
                    st.rerun()

            if journey_create != None and "__complete" in journey_create.keys():
                journey_name = st.session_state.creating_journey
                edit_journey(journey_name, journey_create)

        st.header("Journey database")

        for journey_name in db_journey.keys():
            journey = db_journey[journey_name]
            col1, col2 = st.columns([1, 3])
            col1.header(f"{journey_name}")
            if (
                col1.button("Edit details", key=f"edit_button_{journey_name}")
                or "editing_journey" in st.session_state
                and st.session_state.editing_journey == journey_name
                or "editing_journey_details" in st.session_state
                and st.session_state.editing_journey_details
            ):
                st.session_state.editing_journey = journey_name
                st.session_state.editing_journey_details = True
                journey_edit = edit_journey_details(journey_name, journey)
                if st.button("Save", key=f"save_button_{journey_name}"):
                    save_journey(journey_name, journey_edit)
                    st.session_state.editing_journey = None
                    st.session_state.editing_journey_details = None
                    st.rerun()
            else:
                col2.write(f'##### {journey["title"]}')
                col2.write(journey["summary"])

            i = 0
            if (col1.toggle("Show subjects", key=f"show_toggle_{journey_name}")):
                for i, day in enumerate(journey["days"]):
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        col1.write(f"#### Subject {i+1}:")
                        if (
                            col1.button(
                                "Edit subject", key=f"edit_button_{journey_name}_{i}"
                            )
                            or "editing_journey" in st.session_state
                            and st.session_state.editing_journey == journey_name
                            and "editing_journey_step" in st.session_state
                            and st.session_state.editing_journey_day == i
                            and st.session_state.editing_journey_step == None
                        ):
                            journey_edit = journey
                            st.session_state.editing_journey = journey_name
                            st.session_state.editing_journey_day = i
                            st.session_state.editing_journey_step = None
                            col1, col2 = st.columns([1, 3])
                            journey_edit["days"][i]["title"] = col2.text_input(
                                f"Day {i+1} Title", value=day["title"], key=f"day_title_{journey_name}_{i}"
                            )
                            journey_edit["days"][i]["intro"] = col2.text_area(
                                f"Day {i+1} intro", value=day["intro"], key=f"day_intro_{journey_name}_{i}"
                            )
                            if st.button("Save", key=f"save_button_{journey_name}_{i}"):
                                save_journey(journey_name, journey_edit)
                                st.session_state.editing_journey = None
                                st.session_state.editing_journey_details = None
                                st.session_state.editing_journey_day = None
                                st.session_state.editing_journey_step = None
                                st.rerun()
                        else:
                            col2.write(day["title"] + ":")
                            col2.write(day["intro"])

                        if (col1.toggle("Show steps", key=f"show_toggle_{journey_name}_{i}")):
                            for j, step in enumerate(day["steps"]):
                                with st.expander(f'##### Step {j+1}: {step["subject"]}'):
                                    if (
                                        st.button(
                                            "Edit step", key=f"edit_button_{journey_name}_{i}_{j}"
                                        )
                                        or "editing_journey" in st.session_state
                                        and st.session_state.editing_journey == journey_name
                                        and "editing_journey_step" in st.session_state
                                        and st.session_state.editing_journey_day == i
                                        and st.session_state.editing_journey_step == j
                                    ):
                                        st.session_state.editing_journey = journey_name
                                        st.session_state.editing_journey_day = i
                                        st.session_state.editing_journey_step = j
                                        journey_edit = edit_journey_step(journey_name, journey, i, j)
                                        if st.button("Save", key=f"save_button_{journey_name}_{i}_{j}"):
                                            save_journey(journey_name, journey_edit)
                                            st.session_state.editing_journey = None
                                            st.session_state.editing_journey_details = None
                                            st.session_state.editing_journey_day = None
                                            st.session_state.editing_journey_step = None
                                            st.rerun()
                                    else:
                                        if step.get("json", None):
                                            json = step["json"]
                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Intro:")
                                            col2.write(json["intro"])

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Content:")
                                            col2.write(json["content"])

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Actions:")
                                            col2.write("* " + "\n* ".join(json["actions"]))

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Priority:")
                                            col2.write(json["priority"])
                                        else:
                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Intro:")
                                            col2.write(step["intro"])

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Content:")
                                            col2.write(step["content"])


if __name__ == "__main__":
    main()
