from typing import List
from sentence_transformers import SentenceTransformer
from contextchain.local_llm_client import OllamaClient
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """
    Split text into chunks based on token length for efficient LLM processing.

    Args:
        text (str): Input text to chunk.
        max_length (int): Maximum token length per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = text.split('. ')
        chunks, current_chunk = [], []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(model.tokenizer(sentence)["input_ids"])
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        logger.info(f"✓ Chunked text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"✗ Failed to chunk text: {e}")
        raise

def summarize_text(text: str, model: str = "mistral:7b") -> str:
    """
    Summarize text using an LLM to create a concise narrative for ChromaDB storage.

    Args:
        text (str): Input text to summarize.
        model (str): LLM model name (e.g., 'mistral:7b').

    Returns:
        str: Summarized text.
    """
    try:
        client = OllamaClient(model)
        prompt = f"Summarize the following text in 100 words or less:\n\n{text}"
        summary = client.generate(prompt, max_tokens=100)
        if summary:
            logger.info("✓ Text summarized successfully")
            return summary
        else:
            raise ValueError("Summarization failed")
    except Exception as e:
        logger.error(f"✗ Failed to summarize text: {e}")
        raise