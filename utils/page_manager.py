from typing import Dict, Type
from loguru import logger
import streamlit as st
from page_renderers import StreamlitPage


class PageManager:
    """
    Manages the available pages and their display.
    """

    def __init__(self, page_dict: Dict[str, Type[StreamlitPage]], cfg_: Dict[str, any]):
        self.pages = page_dict
        self.cfg = cfg_


    def set_global_sidebar_widgets(self) -> str:
        """Displays page navigation in the sidebar."""
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.selectbox("Select a page", list(self.pages.keys()))
        return selected_page

    def display_page(self, page_name: str, cfg_:Dict[str, any]) -> None:
        """Displays the selected page."""
        logger.info(f"Displaying page: {page_name}")
        page_instance = self.pages[page_name]()
        page_instance.render_page(cfg_)