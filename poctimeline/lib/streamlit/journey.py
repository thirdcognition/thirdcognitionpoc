
from enum import Enum
import streamlit as st
from lib.models.journey import JourneyItem

class ChildPosition(Enum):
    FIRST = 1
    MIDDLE = 2
    LAST = 3

@st.cache_resource
def get_journey_item_cache() -> dict[str, JourneyItem]:
    return {}