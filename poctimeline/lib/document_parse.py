import io
import re
from typing import Iterable, Sequence, Union
from bs4 import BeautifulSoup
import numpy as np
import fitz
from markdown import markdown
from rapidocr_onnxruntime import RapidOCR

from lib.document_tools import semantic_splitter

ocr = None

# def recursive_find(obj, key, _ret=[]):
#     if key in obj:
#         # print("found text: " + obj[key])
#         return [str(obj[key])]
#     resp = []
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if isinstance(v, dict) or isinstance(v, list):
#                 resp += recursive_find(v, key, _ret)
#     elif isinstance(obj, list):
#         for v in obj:
#             if isinstance(v, dict) or isinstance(v, list):
#                 resp += recursive_find(v, key, _ret)  # added return statement

#     return _ret + resp


def parse_images_to_text(
    images: Sequence[Union[Iterable[np.ndarray], bytes]],
    start,
    step,
    ocrs,
    progress_cb=None,
) -> str:

    global ocr

    text = ""
    total = len(images)
    i = 0

    ocr = ocr or RapidOCR(
        # use_gpu=True, det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True
    )

    for img in images:
        if progress_cb:
            progress_cb(start + step * i / total, "Analysing image")

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
    start,
    step,
    xrefs,
    ocrs,
    progress_cb=None,
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
        if progress_cb:
            progress_cb(start + substep_total * i / total, "Loading image")
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
    return parse_images_to_text(
        imgs, start + substep_total, step - substep_total, ocrs, progress_cb=progress_cb
    )


# Function to convert PDF to text using PyMuPDFLoader
def load_pymupdf(file: io.BytesIO, filetype, progress_cb=None):
    doc = fitz.open(stream=file.read(), filetype=filetype)
    text = ""

    page_percentage_total = 0.1
    total_left = 0.6 - page_percentage_total

    i = 0
    total = len(doc)
    step = total_left / total

    xrefs = {}
    ocrs = {}
    chunks = []

    text = ""

    for page in doc:
        if progress_cb:
            progress_cb(
                page_percentage_total / total * i + i * step,
                f"Parsing page {page.number}",
            )

        page_string = re.sub(" {2,}", " ", page.get_text())
        page_string += extract_images_from_page(
            doc,
            page,
            page_percentage_total / total * i + i * step,
            step,
            xrefs,
            ocrs,
            progress_cb=progress_cb,
        )

        text += page_string

        i += 1

    if progress_cb:
        progress_cb(0.6, "Splitting text...")

    chunks = semantic_splitter(
        text,
        progress_cb=lambda x, y: (
            progress_cb(min(1, 0.6 + 0.4 * y / x), f"Splitting text {y+1}/{x}")
            if progress_cb
            else None
        ),
    )

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
