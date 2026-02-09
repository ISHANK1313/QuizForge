import os
import yaml
import logging
from typing import Dict
from difflib import SequenceMatcher

def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {exc}")
        raise

def setup_logging(log_level: str = 'INFO'):
    """
    Configure logging with proper format.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_directories(base_path: str):
    """
    Create necessary directories if they don't exist.

    Args:
        base_path (str): Base path to create directories in.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        logging.info(f"Created directory: {base_path}")

def validate_pdf_file(file_path: str) -> bool:
    """
    Check if file exists and is a valid PDF.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return False

    if not file_path.lower().endswith('.pdf'):
        logging.error(f"File is not a PDF: {file_path}")
        return False

    return True

def calculate_text_statistics(text: str) -> Dict:
    """
    Calculate word count, sentence count, avg sentence length.

    Args:
        text (str): Input text.

    Returns:
        Dict: Dictionary containing statistics.
    """
    sentences = text.split('.') # Simple split for basic stats, more robust in preprocessor
    words = text.split()

    num_sentences = len([s for s in sentences if s.strip()])
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    return {
        'word_count': num_words,
        'sentence_count': num_sentences,
        'avg_sentence_length': round(avg_sentence_length, 2)
    }

def clean_special_characters(text: str) -> str:
    """
    Remove or replace special characters.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    # Basic cleaning, can be expanded based on specific needs
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text

def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate similarity between two strings using simple method.

    Args:
        text1 (str): First string.
        text2 (str): Second string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    return SequenceMatcher(None, text1, text2).ratio()
