"""
Query page implementation for searching through embedded documents
"""

from pathlib import Path
from typing import Dict
import streamlit as st
from loguru import logger
from sentence_transformers import SentenceTransformer
from utils.chunk_retrieval import ChunkRetrieval
from utils.response import LLMResponse
from .base_page import StreamlitPage


class QueryPage(StreamlitPage):
    @property
    def page_name(self) -> str:
        """Return the page name for navigation"""
        return "Query Documents"

    def render_page(self, cfg_: Dict[str, any]) -> None:
        """Render the query page with search functionality"""
        st.title(self.page_name)

        # Initialise components
        embedding_model = SentenceTransformer(cfg_["pages"]["embedding_model"])
        db_folder = Path(cfg_["databases"]["db_folder"])
        index_path = str(db_folder / cfg_["databases"]["faiss_db_index"])
        docs_path = str(db_folder / cfg_["databases"]["docs"])

        try:
            query_engine = ChunkRetrieval(index_path, docs_path)
            logger.info(
                f"Retrieving chunks from index: {index_path} and doc path: {docs_path}"
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"Error loading database: {str(e)}")
            logger.error(f"Error loading database: {str(e)}")
            return

        # Search interface
        query = st.text_input("Enter your query:")
        llm_model = st.sidebar.text_input("OpenAI Model", "gpt-3.5-turbo")
        st.sidebar.divider()
        k = st.sidebar.slider("Number of faiss results to retrieve", 
                               min_value=1, max_value=10, value=3)

        temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.4)
        max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 200)
        
        # Add system prompt editor in sidebar
        st.sidebar.divider()
        with st.sidebar.expander("System Prompt", expanded=False):
            default_system_prompt = cfg_["RAG"]["system_prompt"]
            system_prompt = st.text_area(
                "Edit the system prompt:",
                value=default_system_prompt,
                height=200,
                key="system_prompt_editor"
            )
            
            # Add a reset button to restore the default prompt
            if st.button("Reset to Default"):
                st.session_state.system_prompt_editor = default_system_prompt
                system_prompt = default_system_prompt

        if st.button("Search") and query:
            try:
                query_embedding = embedding_model.encode(query)
                results = query_engine.retrieve(query_embedding, k)

                # Display results
                st.subheader("Search Results")
                for i, doc in enumerate(results, 1):
                    with st.expander(f"Result {i}"):
                        st.markdown(doc.page_content)
                        st.markdown(
                            f"*Source: {doc.metadata.get('source', 'Unknown')}*"
                        )

                logger.info(
                    f"About to call generate_answer with model: {llm_model}, "
                    f"temp: {temperature}, and max_tokens: {max_tokens}"
                )

                # Use the edited system prompt
                custom_cfg = cfg_.copy()
                custom_cfg["RAG"]["system_prompt"] = system_prompt
                
                response_llm = LLMResponse(
                    query,
                    [doc.page_content for doc in results],
                    model_name=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cfg=custom_cfg,
                )

                final_answer = response_llm.generate_answer()

                st.subheader("OpenAI-based Answer")
                st.write(final_answer)

            except Exception as e:  # pylint: disable=broad-exception-caught
                st.error(f"Error while creating an LLM response: {str(e)}")
                logger.error(f"Error during search: {str(e)}")

