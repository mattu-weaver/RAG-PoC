"""
Home page implementation
"""
import streamlit as st
from .base_page import StreamlitPage
from typing import Dict


class HomePage(StreamlitPage):
    """
    Home page for this streamlit application, derives from StreamlitPage.
    """

    @property
    def page_name(self) -> str:
        """
        Implementation of page_name from the page base class: StreamlitPage.
        Returns: The name of this page.
        """
        return "Home"

    def render_page(self, cfg_: Dict[str, any]) -> None:
        st.title("Document RAG System")
        st.write("Welcome to the RAG Document Processing System")
        st.markdown("""
        ### Features
        - Document processing and analysis
        - Question answering
        - Document search and retrieval
        """)
