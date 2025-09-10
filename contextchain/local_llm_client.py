import subprocess
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with a local Ollama LLM server."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            model (str): Name of the LLM model (e.g., 'mistral:7b').
            base_url (str): URL of the Ollama server.
        """
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized OllamaClient for model {model}")

    def setup(self) -> bool:
        """
        Pull and setup the specified model.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            subprocess.run(["ollama", "pull", self.model], check=True, capture_output=True)
            logger.info(f"✓ Pulled Ollama model: {self.model}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to pull model {self.model}: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """
        Generate text using the Ollama model.

        Args:
            prompt (str): Input prompt for generation.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            Optional[str]: Generated text or None if failed.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            result = response.json().get("response", "")
            logger.info(f"✓ Generated text for prompt: {prompt[:50]}...")
            return result
        except requests.RequestException as e:
            logger.error(f"✗ Failed to generate text: {e}")
            return None