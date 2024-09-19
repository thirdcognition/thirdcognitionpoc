from datetime import datetime
from functools import cache
import io
import os
from typing import Dict, List, Union
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
from chromadb.config import Settings as ChromaSettings
import streamlit as st
from lib.chains.init import get_embeddings
from lib.document_tools import get_source_rag_chunks, get_concept_rag_chunks, get_topic_rag_chunks
from lib.load_env import SETTINGS
from lib.models.journey import JourneyModel
from lib.models.sqlite_tables import (
    Base,
    TaxonomyDataTable,
    Taxonomy,
    ConceptDataTable,
    ConceptData,
    SourceContentPage,
    SourceContents,
    SourceData,
    SourceDataTable,
    JourneyDataTable,
    SourceType,
)
from lib.chains.prompt_generator import CustomPrompt, CustomPromptContainer
from lib.helpers import pretty_print
