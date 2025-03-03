"""
The application entry point with multi-page Streamlit interface.
"""

from typing import Dict, Type
import streamlit as st
from app_config import ConfigLoader, LogLoader
from page_renderers import StreamlitPage, HomePage, UploadPage, QueryPage
from utils.page_manager import PageManager


def get_pages() -> Dict[str, Type[StreamlitPage]]:
    """
    Get the mapping of page names to their class implementations.
    Returns a dictionary of the application's pages.
    """
    pages = [HomePage, UploadPage, QueryPage]
    return {page_class().page_name: page_class for page_class in pages}


def setup_streamlit_interface() -> None:
    """
    Sets up the basic Streamlit interface.
    """
    st.set_page_config(page_title="RAG Document System", page_icon="📚", layout="wide")


def main(cfg_: Dict[str, any], logger_configurator_: LogLoader, page_manager_: PageManager) -> None:
    """
    Main application entry point.
    param app_config: The application configuration object.
    param page_manager: The object to manage the pages.
    """
    logger =logger_configurator_.configure_logger(
        cfg_["logger"]["log_name"],
        cfg_["logger"]["format"],
        cfg_["logger"]["level"],
    )

    setup_streamlit_interface()
    # Sidebar navigation
    selected_page = page_manager_.set_global_sidebar_widgets()
    st.session_state.current_page = selected_page

    # Display selected page
    page_manager_.display_page(selected_page, cfg_)


if __name__ == "__main__":
    # Read app configuration and setup logger.
    config_loader = ConfigLoader("config/config.toml")
    cfg = config_loader.load_config()
    logger_configurator = LogLoader()

    # Get the app pages and display via the page manager.
    page_dictionary = get_pages()
    page_manager = PageManager(page_dictionary, cfg)
    main(cfg, logger_configurator, page_manager)
