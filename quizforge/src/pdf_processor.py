import PyPDF2
import logging
import re
import nltk
from typing import List

# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PDFExtractor:
    """
    Extract text from PDF files using PyPDF2.
    """
    def __init__(self, pdf_path: str):
        """
        Initialize PDFExtractor with PDF path.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.page_count = 0

    def extract_text(self) -> str:
        """
        Extract text from the PDF file.

        Returns:
            str: Extracted text content.
        """
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.page_count = len(reader.pages)
                logging.info(f"Processing PDF with {self.page_count} pages.")

                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logging.warning(f"No text found on page {i+1}")

        except FileNotFoundError:
            logging.error(f"PDF file not found: {self.pdf_path}")
            raise
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            raise

        return text

    def get_page_count(self) -> int:
        """
        Get the number of pages in the PDF.

        Returns:
            int: Number of pages.
        """
        return self.page_count


class TextPreprocessor:
    """
    Clean, normalize, and tokenize text.
    """
    def __init__(self, text: str):
        """
        Initialize TextPreprocessor with raw text.

        Args:
            text (str): Raw text content.
        """
        self.text = text
        self.cleaned_text = ""

    def clean_text(self) -> str:
        """
        Clean and normalize text.

        Returns:
            str: Cleaned text.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', self.text)

        # Remove special characters but keep punctuation needed for sentence splitting
        # Keeping basic punctuation and alphanumeric characters
        # text = re.sub(r'[^\w\s.,?!:;-]', '', text)

        # Strip leading/trailing whitespace
        self.cleaned_text = text.strip()

        return self.cleaned_text

    def tokenize_sentences(self) -> List[str]:
        """
        Split text into sentences.

        Returns:
            List[str]: List of sentences.
        """
        if not self.cleaned_text:
            self.clean_text()

        try:
            sentences = nltk.sent_tokenize(self.cleaned_text)
        except Exception as e:
            logging.warning(f"NLTK tokenization failed, falling back to regex: {e}")
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', self.cleaned_text)

        return sentences

    def remove_short_sentences(self, sentences: List[str], min_length: int = 10) -> List[str]:
        """
        Remove sentences shorter than min_length characters.

        Args:
            sentences (List[str]): List of sentences.
            min_length (int): Minimum character length.

        Returns:
            List[str]: Filtered list of sentences.
        """
        return [s for s in sentences if len(s.split()) >= min_length]
