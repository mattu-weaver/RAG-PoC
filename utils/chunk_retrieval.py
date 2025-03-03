"""
Query engine for searching through FAISS database
"""
from typing import Dict, List
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
import numpy as np
import pickle
import logging
from typing import Optional
from copy import deepcopy
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkRetrieval:
    """
    Retrieves chunks of data from the faiss database that are related to the query.
    """
    def __init__(self, index_path, document_path):
        """
        Initialise the DocumentRetriever with the paths to the FAISS index and document chunks.
        param: index_path (str): Path to the FAISS index file (e.g., 'index.faiss').
        param: document_path (str): Path to the document chunks file (e.g., 'documents.npy').
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)

        # Load the document chunks
        self.documents = np.load(document_path, allow_pickle=True)

    def retrieve(self, query_embedding, k=5):
        """
        Retrieve the top-k document chunks that match the query embedding.
        param: query_embedding (np.ndarray): The embedding of the query.
        param: k (int): Number of top chunks to retrieve.
        Returns: list: List of the top-k document chunks.
        """
        # Ensure the query embedding is in the correct shape
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Search the FAISS index for the top-k nearest neighbors
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve the corresponding document chunks
        retrieved_chunks = [self.documents[i] for i in indices[0]]

        return retrieved_chunks
