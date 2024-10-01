from typing import List, Optional

from lib.models.concepts import ConceptDataTable
from lib.models.reference import ReferenceType
from lib.models.source import SourceDataTable
from lib.models.taxonomy import TaxonomyDataTable
from lib.models.topics import TopicDataTable
from lib.models.user import user_db_get_session

def get_reference_item(id: str, type: ReferenceType) -> SourceDataTable | TopicDataTable | ConceptDataTable | TaxonomyDataTable | None:
    session = user_db_get_session()
    if type == ReferenceType.source:
        db_item = session.query(SourceDataTable).filter(SourceDataTable.source == id).first()
    elif type == ReferenceType.topic:
        db_item = session.query(TopicDataTable).filter(TopicDataTable.id == id).first()
    elif type == ReferenceType.concept:
        db_item = session.query(ConceptDataTable).filter(ConceptDataTable.id == id).first()
    elif type == ReferenceType.taxonomy:
        db_item = session.query(TaxonomyDataTable).filter(TaxonomyDataTable.id == id).first()
    else:
        raise ValueError(f"{type} is not a valid ReferenceType")

    if db_item:
        return db_item
    else:
        return None

