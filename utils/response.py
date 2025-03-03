from typing import Dict, List
from loguru import logger
from openai import OpenAI # pylint: disable=import-error
import streamlit as st


class LLMResponse:
    def __init__(self,
    query: str,
    chunks: List[str],
    model_name: str,  # Updated default model
    temperature: float,
    max_tokens: int,
    cfg: Dict[str, any]
    ):
        self.query = query
        self.chunks = chunks
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cfg = cfg

    def generate_answer(self) -> str:
        client = OpenAI(api_key=st.secrets["openai_api_key"])
        combined_context = "\n".join(self.chunks)
        prompt_text = f"Context:\n{combined_context}\n\nQuery:\n{self.query}\n\nAnswer:"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
                ]
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"Error during search: {str(e)}")



def generate_answer(
    query: str,
    chunks: List[str],
    model_name: str = "gpt-3.5-turbo",  # Updated default model
    temperature: float = 0.7,
    max_tokens: int = 200
) -> str:
    """
    Generates a response using OpenAI with the provided query and chunks.
    param query: str User's query text.
    param chunks: List[str] List of relevant document chunks.
    param model_name: str Name of the model to use for generating the response.
    param temperature: float Sampling temperature to use for generating the response.
    param max_tokens: int Maximum number of tokens to generate in the response.
    Returns: Coherent answer text generated by OpenAI.
    """
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    combined_context = "\n".join(chunks)
    prompt_text = f"Context:\n{combined_context}\n\nQuery:\n{query}\n\nAnswer:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
            ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"Error during search: {str(e)}")