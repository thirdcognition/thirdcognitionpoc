import asyncio
from collections import defaultdict
from optparse import Option
import os
import sys
import time
from typing import Dict, List, Union
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))
from admin.sidebar import init_sidebar
from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.user import check_auth
from lib.db.source import get_db_sources
from lib.db.taxonomy import get_db_taxonomy
from lib.db.concept import get_concept_by_id, get_db_concepts
from lib.helpers.shared import pretty_print
from lib.models.concepts import ConceptDataTable
from lib.models.taxonomy import TaxonomyDataTable


from lib.streamlit_tools import get_all_categories

st.set_page_config(
    page_title="TC POC: Browse found concepts",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


def display_concept(
    concept: ConceptDataTable = None,
    concepts: Dict[str, ConceptDataTable] = None,
    id: str = None,
    children: dict = None,
    header: str = None,
):
    if concept is None and concepts is not None and id is not None:
        concept = concepts[id]
    if concept is None:
        return

    with st.expander(header if header is not None else concept.title):
        st.code(concept.id)
        sub_col1, sub_col2 = st.columns([1, 2])
        sub_col1.write("References:")
        sub_col1.write(ref for ref in concept.references)
        sub_col1.write("Taxonomy:")
        sub_col1.write(concept.taxonomy)
        sub_col2.write(f"##### Summary:\n{concept.summary}")
        sub_col2.write(f"##### Content:\n{concept.content}")
    if children and concepts:
        for child in children.keys():
            child_item: ConceptDataTable = concepts[child]
            display_concept(
                concepts=concepts,
                header=f"{header} > {child_item.title}",
                id=child,
                children=(
                    children[child]["children"]
                    if "children" in children[child]
                    else None
                ),
            )


def build_hierarchy(
    items: Union[List[ConceptDataTable], List[TaxonomyDataTable]],
    parent_id=None,
):
    hierarchy = {}
    for item in items:
        children = None
        if item.parent_id == parent_id or item.parent_id == "" and parent_id is None:
            children = build_hierarchy(items, parent_id=item.id)
            if children:
                hierarchy[item.id] = {"children": children}
            else:
                hierarchy[item.id] = {}

    return hierarchy


def concept_hierarchy(categories: List[str]):
    for category in categories:
        st.header(category)
        concepts = get_db_concepts(categories=[category])
        concepts_by_id = {concept.id: concept for concept in concepts}
        hierarchy = build_hierarchy(concepts)

        for id in hierarchy.keys():
            children = hierarchy[id].get("children", [])
            if len(children) > 0:
                # st.write("---")
                st.subheader(f"{concepts_by_id[id].title}")

            display_concept(
                concepts=concepts_by_id,
                id=id,
                children=children,
                header=f"{concepts_by_id[id].title}",
            )

            if len(children) > 0:
                st.write("---")


def manage_taxonomy(
    taxonomy: TaxonomyDataTable = None,
    taxonomy_by_id: Dict[str, List[TaxonomyDataTable]] = None,
    id: str = None,
    children: dict = None,
    header: str = None,
):
    if taxonomy is None and taxonomy_by_id is not None:
        taxonomy = taxonomy_by_id[id]
    if taxonomy is None:
        return

    concepts = get_db_concepts(taxonomy=taxonomy)
    if len(concepts) > 0:
        st.subheader(header or taxonomy.title)
        st.write(taxonomy.description)
        for concept_data in concepts:
            display_concept(concept_data)

    if children and taxonomy_by_id:
        for child in children.keys():
            manage_taxonomy(
                taxonomy_by_id=taxonomy_by_id,
                header=f"{header} > {', '.join([taxonomy_item.title for taxonomy_item in taxonomy_by_id[str(child)]]) if isinstance(taxonomy_by_id[str(child)], list) else taxonomy_by_id[str(child)].title}",
                id=child,
                children=(
                    children[child]["children"]
                    if "children" in children[child]
                    else None
                ),
            )


def taxonomy_hierarchy(taxonomy: Dict[str, List[TaxonomyDataTable]]):
    for category in taxonomy.keys():
        taxonomy_by_id = {
            taxonomy_item.id: taxonomy_item for taxonomy_item in taxonomy[category]
        }

        st.header(category)
        hierarchy = build_hierarchy(taxonomy[category])
        # pretty_print({
        #     "hierarchy": hierarchy,
        #     "taxonomy": [taxonomy_item.model_dump_json(indent=2) for taxonomy_item in taxonomy[category]]
        # }, "TaxonomyDataTable Hierarchy", force=True)
        break_reset = True

        for id in hierarchy.keys():
            children = hierarchy[id].get("children", [])

            manage_taxonomy(
                taxonomy_by_id=taxonomy_by_id,
                id=id,
                children=children,
                header=f"{taxonomy_by_id[id].title}",
            )


def by_taxonomy_items(taxonomy: Dict[str, List[TaxonomyDataTable]]):
    sel_col1, sel_col2 = st.columns([1, 1])

    keys = ["parent_taxonomy", "taxonomy", "type", "tag"]
    desc = ["Parent taxonomy", "Taxonomy", "Type", "Tag"]
    structure_by = keys[desc.index(sel_col1.selectbox("Structure by", desc))]

    structured_taxonomy: Dict[str, List[TaxonomyDataTable]] = None
    for category in taxonomy.keys():
        structured_taxonomy = {}
        for item in taxonomy[category]:
            key = getattr(item, structure_by)
            if key not in structured_taxonomy:
                structured_taxonomy[key] = []
            structured_taxonomy[key].append(item)

    selected_item_type = None
    if structured_taxonomy is not None:
        selected_item_type = sel_col2.selectbox("", structured_taxonomy.keys())

    for category in taxonomy.keys():
        st.header(category)
        for item in structured_taxonomy[selected_item_type]:

            manage_taxonomy(item)


def by_source(source_name: str):
    file_entry = get_db_sources(source=source_name)[source_name]
    tagged_concepts: Dict[str, List[ConceptDataTable]] = defaultdict(list)
    tags: Dict[str, TaxonomyDataTable] = {}
    st.subheader(f"{source_name}")
    with st.container():
        for concept_id in file_entry.source_concepts:
            db_concept = get_concept_by_id(concept_id)
            # pretty_print(
            #     concept_inst.model_dump_json(indent=2), "Concept Data", force=True
            # )
            display_concept(db_concept)
        #     for tag in concept_inst.taxonomy:
        #         tagged_concepts[tag].append(concept_inst)
        #         if tag not in tags:
        #             tag_inst = get_taxonomy_by_id(tag)
        #             if tag_inst:
        #                 tags[tag] = tag_inst.concept_taxonomy

        # for concept_tag, concepts in tagged_concepts.items():
        #     if concept_tag in tags:
        #         st.write(f"### {tags[concept_tag].title}:")
        #         st.write(tags[concept_tag].description)
        #         # with st.expander("Concept instances"):
        #         for concept_inst in concepts:
        #             with st.expander(f"Concept: {concept_inst.title}"):
        #                 st.write(f"#### {concept_inst.id}")
        #                 st.code(concept_inst.id)
        #                 sub_col1, sub_col2  = st.columns([1,2])
        #                 sub_col1.write("References:")
        #                 sub_col1.write(ref for ref in concept_inst.references)
        #                 sub_col1.write("Taxonomy:")
        #                 sub_col1.write(concept_inst.taxonomy)
        #                 sub_col2.write(f"##### Summary:\n{concept_inst.summary}")
        #                 sub_col2.write(f"##### Content:\n{'\n'.join(concept_inst.contents)}")


async def main():

    st.title("Browse found concepts")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    file_categories = get_all_categories()
    categories = st.multiselect("Source categories", file_categories)

    # if not categories:
    #     st.header("First, choose a file category.")
    #     return

    if categories:
        taxonomy: Dict[str, List[TaxonomyDataTable]] = get_db_taxonomy(
            categories=categories
        )

        if taxonomy is not None and len(taxonomy) == 0:
            st.header("No concepts found.")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Concept hierarchy",
                "Taxonomy hierarchy",
                "By source",
                "By taxonomy selectors",
                "All",
            ]
        )

        with tab1:
            concept_hierarchy(categories)
        with tab2:
            taxonomy_hierarchy(taxonomy)
        with tab3:
            files = get_db_sources(categories=categories)
            source = st.selectbox("Available sources", files.keys(), index=None)
            # for file in files.keys():
            if source is not None:
                by_source(source)

        with tab4:
            by_taxonomy_items(taxonomy)
        with tab5:
            for category in categories:
                if len(taxonomy[category]) > 0:
                    st.header(category)
                    for item in taxonomy[category]:
                        manage_taxonomy(item)

    else:
        st.header("First, choose a categories.")


if __name__ == "__main__":
    asyncio.run(main())
