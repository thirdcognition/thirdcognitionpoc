from datetime import datetime
import gc
from io import StringIO
import numbers
import os
import re
from typing import Iterable, Sequence, Union
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
    CHAR_LIMIT,
    SQLITE_DB,
    create_document_lists,
    get_chain,
    get_chroma_collection,
    get_journey_format_example,
    get_vectorstore,
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
def load_pymupdf(file: UploadedFile, filetype, context_size=CHAR_LIMIT):
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
        # page_string = f"Page {page.number}: \n\n"
        # blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        # lines = recursive_find(blocks, "text")

        # pprint.pprint(lines)

        # page_string: str = ""

        # for line in lines:
        #     if not line.isspace():
        #         page_string += str(line)
        #         if len(line) > 10:
        #             page_string += "\n"
        #         else:
        #             page_string += " "

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
        # if len(page_string.split()) < 200 and len(chunks) > 0 and len(chunks[-1]) + len(page_string) < context_size:
        #     chunks[-1] += "\n" + page_string
        # else:
        #     chunks.append(page_string)
        # page_string += "\n\n"
        # if len(text + page_string) < context_size:
        #     text += page_string
        # else:
        #     chunks.append(text)
        #     text = page_string
        i += 1

    # chunks.append(text)

    chunks = semantic_splitter(text)
    # chunks = split_text(text)

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


def llm_edit(chain, texts, guidance=None, strip_md=False):
    text = ""

    i = 0
    total = len(texts)

    if total > 1 and chain == "summary":
        total += 1

    if texts == None or texts[0] == None or total == 1 and len(texts[0]) < 1000:
        return None

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

            # st.write("{i}/{total} cycles required, this will take a while...")
            # if len(sub_text) > char_limit * 3:
            text += get_chain(chain + guided_llm).invoke(input)["text"] + "\n\n"
            # else:
            #     text += (
            #         get_chain(chain + guided_llm, "small").invoke(input)["text"]
            #         + "\n\n"
            #     )
        # texts = get_chain(chain + guided_llm).batch(inputs)
        # print(f'\n\n{ texts =}\n\n')
        # text = "\n\n".join([output["text"] for output in texts])

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

        # if len(text) > 1024 * 15.5 * 3:
        #     text = get_chain(chain + guided_llm, "large").invoke(input)["text"]
        # elif len(text) > 1024 * 7.5 * 3:
        #     text = get_chain(chain + guided_llm).invoke(input)["text"]
        # else:
        #     text = get_chain(chain + guided_llm, "small").invoke(input)["text"]
        text = get_chain(chain + guided_llm).invoke(input)["text"]

    bar.empty()

    return text


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
            if len(text) > CHAR_LIMIT:
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
            if texts is not None and filetype != "md":
                formatted_text = llm_edit("text_formatter", texts)
            elif filetype == "md":
                if len(texts) > 1 or len(texts[0]) > 1000:
                    formatted_text = llm_edit(
                        "text_formatter", [markdown_to_text("\n".join(texts))]
                    )
                else:
                    formatted_text = None
        st.success("Rewrite complete")

        with st.spinner("Summarizing text"):
            summary_text = None
            if texts is not None:
                split_texts = split_text("\n".join(texts), CHAR_LIMIT // 3 * 2)
                if len(split_texts) == 1:
                    shorter_text = llm_edit("summary", split_texts)
                    if shorter_text is not None:
                        summary_text = shorter_text
                    else:
                        summary_text = split_texts[0]
                else:
                    list_of_docs = create_document_lists(split_texts, source=filename)
                    summary_text = get_chain("summary_documents").invoke(
                        {"input_documents": list_of_docs}
                    )["output_text"]

                    shorter_text = llm_edit("summary", [summary_text])
                    if shorter_text is not None:
                        summary_text = shorter_text

        st.success("Summary complete")

        st.write(f"### Summary")
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
                formatted_split = split_text(formatted_text, 128 * 6, 128)

                rag_split = rag_split + formatted_split
                rag_ids = rag_ids + [
                    filename + "_formatted_" + str(i)
                    for i in range(len(formatted_split))
                ]
                rag_metadatas = rag_metadatas + [
                    {
                        "file": "formatted_" + filename,
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
            # else:
            #     # If the file does not exist, create a new row

            #     collections = []

            #     collections.append("rag_all")
            #     vectorstore = get_vectorstore("rag_all")
            #     vectorstore.add_texts(
            #         ids=rag_ids, texts=rag_split, metadatas=rag_metadatas
            #     )

            #     for cat in category:
            #         collections.append("rag_" + cat)
            #         vectorstore = get_vectorstore("rag_" + cat)
            #         vectorstore.add_texts(
            #             ids=rag_ids, texts=rag_split, metadatas=rag_metadatas
            #         )

            #     file = FileDataTable(
            #         filename=filename,
            #         texts=texts,
            #         formatted_text=formatted_text,
            #         summary=summary_text,
            #         category_tag=category,
            #         chroma_collection=collections,
            #         chroma_ids=rag_ids,
            #         last_updated=datetime.now(),
            #         file_data=uploaded_file.getvalue()
            #     )
            #     database_session.add(file)

            #     st.success(f"{filename} saved to database successfully.")

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

    # category = st.text_input(f"Category Tag", value="antler")

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

    # st.markdown(
    #     f"There's already a file with the name _{filename}_ in the database. \n\n Would you want to replace it?"
    # )
    # if st.button("Replace"):

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

    # query = database_session.query(FileDataTable).all()
    # df = pd.DataFrame(
    #     [(i.filename, i.text[:50]) for i in query], columns=["Filename", "sqla.Text"]
    # )
    # st.dataframe(df)

    file_entry = get_db_files()[filename]
    # Add Streamlit editors to edit 'disabled' and 'category_tag' fields

    if "rewrite_text" not in st.session_state:
        st.session_state.rewrite_text = {}

    rewrite_text = st.session_state.rewrite_text

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
                        rewrite_text[filename] = text

                if rewrite:
                    with st.spinner("Rewriting"):
                        text = None
                        if raw is not None and filetype != "md":
                            text = llm_edit(
                                "text_formatter", raw, guidance, strip_md=True
                            )
                        elif filetype == "md":
                            if len(raw) > 1 or len(raw[0]) > 1000:
                                text = llm_edit(
                                    "text_formatter",
                                    [markdown_to_text("\n".join(raw))],
                                    guidance,
                                    strip_md=True,
                                )
                            else:
                                text = markdown_to_text(raw[0])

                    st.success("Markdown rewrite complete")
                    rewrite_text[filename] = text

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


def gen_journey_doc(list_of_strings, amount=10):
    text = "\n".join(list_of_strings)

    reduce = False

    reduce = len(text) > 3 * 5 * 1024

    print(f"{reduce = } ({len(text)})")

    if reduce:
        list_of_docs = create_document_lists(list_of_strings)

        chain = get_chain("reduce_journey_documents")

        list_of_strings = [
            chain.invoke(
                {
                    "input_documents": [document],
                    "amount": amount,
                    "format_example": get_journey_format_example(amount),
                }
            )["output_text"]
            for document in list_of_docs
        ]

        text = "\n".join(list_of_strings)

        reduce = len(text) > 3 * 5 * 1024

        if reduce:
            list_of_docs = create_document_lists(list_of_strings)
            text = chain.invoke(
                {
                    "input_documents": list_of_docs,
                    "amount": amount,
                    "format_example": get_journey_format_example(amount),
                }
            )["output_text"]

    return get_chain("journey_text").invoke(
        {
            "context": text,
            "amount": amount,
            "format_example": get_journey_format_example(amount),
        }
    )["text"]


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
                "Number of journey to generate", min_value=1, max_value=20, value=10
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

                with st.spinner("Generating journey document"):
                    compressed = gen_journey_doc(list_of_strings, amount)
                    descriptions = get_chain("action").invoke(
                        {
                            "context": compressed,
                            "action": """Write me one descriptive title and one description for the whole context with following format.
Format:
Title: One sentence descriptive title for the context
Description: Up to 5 sentence description with relevant details from the context
""",
                        }
                    )["text"]

                st.success("Generating journey document done.")

                i = 0

                for line in descriptions.split("\n"):
                    text = line
                    if re.search("([:]?)", text, flags=re.IGNORECASE):
                        text = re.split("[:\\-]?", line, maxsplit=1)[1].strip()

                    if len(text) > 0:
                        key = "title" if i == 0 else "summary"
                        journey_details[key] = text
                    i += 1

                # print(f"{descriptions =}")

                st.write(
                    f'##### {journey_details["title"]} \n\n {journey_details["summary"]}'
                )

                with st.spinner("Generating JSON for journey"):
                    bar = st.progress(0)
                    journey = re.split(
                        r"task [0-9]+[_\-: ]?", compressed, flags=re.IGNORECASE
                    )
                    journey = list(
                        filter(
                            lambda item: len(str(item)) != 0
                            and not str(item).isspace(),
                            journey,
                        )
                    )
                    journey_steps = []
                    # print(f"{journey =}")
                    bar.progress(0.1)
                    total = len(journey)
                    i = 0
                    for step in journey:
                        if re.search(
                            "(description[_\\-: ]?)", step, flags=re.IGNORECASE
                        ):
                            journey_steps = journey_steps + gen_journey_json(step)
                            bar.progress(min(1, 0.1 + i / (total - 0.1)))
                        i += 1

                    bar.empty()
                st.success("Generating JSON for journey done.")

            journey_details["steps"] = journey_steps

            journey_details["__complete"] = True

    return journey_details


def edit_journey_step(journey_name, journey, index):
    step = journey["steps"][index]
    col1, col2 = st.columns([1, 3])
    col1.write(f"##### Step {index+1}:")
    journey["steps"][index]["name"] = col2.text_input(
        "Title", value=step["name"], key=f"journey_step_name_{journey_name}_{index}"
    )

    journey["steps"][index]["description"] = col2.text_area(
        "Description",
        value=step["description"],
        key=f"journey_step_description_{journey_name}_{index}",
    )

    if not isinstance(journey["steps"][index]["priority"], numbers.Number):
        journey["steps"][index]["priority"] = step["priority"] = 1

    journey["steps"][index]["priority"] = col2.select_slider(
        "Priority",
        options=list(range(1, 6)),
        value=max(1, min(5, step["priority"])),
        key=f"journey_step_priority_{journey_name}_{index}",
    )

    col1, col2 = st.columns([1, 3])
    col1.write(f"##### Actions:")
    for j, action in enumerate(journey["steps"][index]["actions"]):
        journey["steps"][index]["actions"][j] = col2.text_area(
            f"Action {j+1}",
            value=action,
            key=f"journey_step_actions_{journey_name}_{index}_{j}",
        )

    return journey


def edit_journey_details(journey_name, journey):
    col1, col2 = st.columns([1, 3])
    journey["title"] = col2.text_input(
        "Title", value=journey["title"], key=f"title_{journey_name}"
    )
    journey["summary"] = col2.text_area(
        "Summary", value=journey["summary"], key=f"summary_{journey_name}"
    )

    col1, col2 = st.columns([1, 3])
    col1.write("Files used:")
    col2.write("* " + "\n* ".join(journey["files"]))

    return journey


def save_journey(journey_name, journey):
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        print("Modify journey")
        journey_db.files = journey["files"]
        journey_db.steps = journey["steps"]
        journey_db.title = journey["title"]
        journey_db.summary = journey["summary"]

        journey_db.last_updated = datetime.now()
        # st.success(f"{filename} updated within database successfully.")

    else:
        print("Create journey")
        journey_db = JourneyDataTable(
            journeyname=journey_name,
            files=journey["files"],
            steps=journey["steps"],
            title=journey["title"],
            summary=journey["summary"],
            last_updated=datetime.now(),
        )
        database_session.add(journey_db)
    database_session.commit()


def edit_journey(journey_name, journey):
    st.header(f"Edit journey: {journey_name}")

    journey = edit_journey_details(journey_name, journey)

    st.subheader("Journey steps")

    for i, step in enumerate(journey["steps"]):
        with st.expander(f"Journey step {i}: {step['name']}"):
            journey = edit_journey_step(journey_name, journey, i)

    if st.button("Save into database"):
        save_journey(journey_name, journey)


# Streamlit app
def main():
    init_db()
    st.title("Admin interface for Antler buddy")

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

        # if "journey_edit_data" in st.session_state:
        #     journey_edit = st.session_state.journey_edit_data
        # else:
        #     journey_edit = None

        for journey_name in db_journey.keys():
            journey = db_journey[journey_name]
            col1, col2 = st.columns([2, 5])
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
            # st.write(journey)
            # st.write(f"Creating {journey_name}")

            # journey_steps = []
            # print(f"{journey = }")
            # for j in range(0, len(journey), 4):
            #     journey_steps.append({
            #         "task": journey[j].split(":", maxsplit=1)[1].strip(),
            #         "description": journey[j+1].split(":", maxsplit=1)[1].strip(),
            #         "actions": journey[j+2].split(":", maxsplit=1)[1].strip(),
            #         "priority": journey[j+3].split(":", maxsplit=1)[1].strip(),
            #     })

            # st.write("journey:")
            i = 0
            # print(f"{journey_steps =}")
            with st.container(border=True):
                for step in journey["steps"]:
                    with st.expander(f'##### Step {i+1}: {step["name"]}'):
                        if (
                            st.button(
                                "Edit step", key=f"edit_button_{journey_name}_{i}"
                            )
                            or "editing_journey" in st.session_state
                            and st.session_state.editing_journey == journey_name
                            and "editing_journey_step" in st.session_state
                            and st.session_state.editing_journey_step == i
                        ):
                            st.session_state.editing_journey = journey_name
                            st.session_state.editing_journey_step = i
                            journey_edit = edit_journey_step(journey_name, journey, i)
                            if st.button("Save", key=f"save_button_{journey_name}_{i}"):
                                save_journey(journey_name, journey_edit)
                                st.session_state.editing_journey = None
                                st.session_state.editing_journey_step = None
                                st.rerun()
                        else:
                            col1, col2 = st.columns([1, 5])
                            col1.write("##### Description:")
                            col2.write(step["description"])

                            col1, col2 = st.columns([1, 5])
                            col1.write("##### Actions:")
                            col2.write("* " + "\n* ".join(step["actions"]))

                            col1, col2 = st.columns([1, 5])
                            col1.write("##### Priority:")
                            col2.write(step["priority"])

                    i = i + 1

        # st.write("journey:")
        # st.write(compressed)
        # col1, col2 = st.columns([3, 1])
        # col1.write("Generating journey json")
        # resulting_json = get_chain("journey_json").invoke({"input": compressed})["text"]
        # col2.success("Complete")
        # print(f"{resulting_json =}")
        # foo = """
        # [{'name': 'Prepare Validation Story', 'description': 'Create an effective validation story that answers why you are solving this problem and how it solves the pain point. Provide evidence of traction to back up your claims.', 'actions': ['Gather data, user feedback, market research, financial projections, etc.'], 'priority': 1, 'length': 2}, {'name': 'Research Competitors & Market Size', 'description': 'Conduct thorough research on competitors and market size to understand the competitive landscape, potential moat against competition, and overall business viability.', 'actions': ['Identify key players in your industry', 'Analyze their strengths/weaknesses', "Assess target markets' sizes and growth rates"], 'priority': 2, 'length': 3}, {'name': 'Develop Team & Product Fit Narrative', 'description': "Create a compelling narrative that highlights your team's complementary skills and chemistry as well as the fit between you and this business idea.", 'actions': ["Identify each member's unique strengths & contributions to the project", 'Discuss how these abilities align with building, growing, or scaling the company effectively'], 'priority': 3, 'length': 3}, {'name': 'Create Data Management & Privacy Plan', 'description': 'Develop a comprehensive data management, privacy and security plan that addresses investor concerns regarding user consent collection methods, transparency about usage of collected information, adherence to international laws (GDPR/COPPA), encryption techniques for protecting sensitive info at rest or in transit.', 'actions': ['Research best practices & regulations'], 'priority': 4, 'length': 3}, {'name': 'Develop Data Security Measures & Compliance Plan', 'description': 'Create a data security measures and compliance plan that addresses investor questions regarding cybersecurity frameworks, international laws (ISO27001/NIST), incident response plans for breaches.', 'actions': ['Research best practices & regulations'], 'priority': 4, 'length': 3}, {'name': 'Identify & Assess Risks and Mitigants', 'description': 'Develop an understanding of potential risks to your data privacy, security practices as well as third-party vendors/partners. Create a plan for assessing these risks and mitigation strategies accordingly (risk assessment).', 'actions': ['Identify key areas where risk may arise'], 'priority': 5, 'length': 3}, {'name': 'Develop Workforce Awareness & Training Plan', 'description': 'Create a plan for ongoing training and awareness programs that ensure employees understand the importance of data protection, cybersecurity best practices.', 'actions': ['Research available resources'], 'priority': 6, 'length': 2}]}
        # """

        # if len(resulting_json) > 0:
        #     st.write("Data editor")
        #     st.data_editor(resulting_json)

    # # Display all files in the database
    # st.subheader("Saved Files")


if __name__ == "__main__":
    main()
