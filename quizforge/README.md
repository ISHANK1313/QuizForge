# QuizForge ⚡

**AI-Powered Question Bank Generator for Lecture PDFs**

QuizForge is a production-ready machine learning system that automatically extracts text from lecture PDFs and generates high-quality quiz questions. It uses Natural Language Processing (NLP) with spaCy and TF-IDF to identify key concepts, entities, and definitions, and then constructs a variety of question types (Definition, Concept, Application, Fill-in-the-blank).

## 🚀 Features

*   **PDF Processing**: Robust text extraction and cleaning from PDF documents.
*   **ML-Powered Analysis**:
    *   **Named Entity Recognition (NER)** using `spaCy` to identify important people, organizations, and terms.
    *   **Keyword Extraction** using `TF-IDF` to find the most significant topics.
    *   **Difficulty Classification**: Heuristic-based classifier to categorize questions as Easy, Medium, or Hard.
*   **Intelligent Ranking**: Questions are ranked based on a composite score of TF-IDF importance, entity relevance, position, and diversity.
*   **Professional Output**: Generates a formatted PDF question bank with an answer key.
*   **Customizable**: Configurable via `config.yaml` to adjust difficulty distribution, question types, and more.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project root.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will also download necessary ML models (spaCy `en_core_web_sm`) and NLTK data automatically on first run.*

## 🏃 Quick Start

**Run the Demo:**
Generate a question bank from the included sample PDF:

```bash
python main.py --demo
```

**Run on your own PDF:**

```bash
python main.py --input path/to/lecture.pdf --output data/output/
```

## ⚙️ Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

**Options:**

*   `--input <path>`: Path to the input PDF file (Required, unless `--demo` is used).
*   `--output <path>`: Directory to save the generated question bank (Default: `data/output/`).
*   `--num-questions <int>`: Total number of questions to generate (Overrides config).
*   `--config <path>`: Path to a custom configuration file (Default: `config/config.yaml`).
*   `--demo`: Run in demo mode using a generated sample PDF.

### Configuration (`config/config.yaml`)

You can adjust the behavior of QuizForge by editing `config/config.yaml`:

```yaml
pdf_processing:
  max_pages: 50
  min_sentence_length: 10

ml_models:
  spacy_model: 'en_core_web_sm'
  tfidf_max_features: 50

question_generation:
  total_questions: 40
  difficulty_distribution:
    easy: 0.4
    medium: 0.4
    hard: 0.2
```

## 🧩 Architecture

1.  **Module 1: PDF Processing** (`src/pdf_processor.py`)
    *   Extracts raw text and cleans it.
    *   Tokenizes text into sentences.
2.  **Module 2: ML Features** (`src/ml_features.py`)
    *   Extracts entities (NER) and keywords (TF-IDF).
    *   Classifies sentence difficulty based on length and technical terms.
3.  **Module 3: Question Generation** (`src/question_generator.py`)
    *   Generates questions using templates.
    *   Ranks questions using a scoring algorithm.
    *   Filters duplicates and ensures difficulty distribution.
4.  **Module 4: Output Generation** (`src/output_handler.py`)
    *   Formats the selected questions into a professional PDF.
5.  **Module 5: Main Controller** (`main.py`)
    *   Orchestrates the entire pipeline.

## ⚠️ Troubleshooting

*   **Model Download Errors**: If you see errors related to `en_core_web_sm` or `nltk`, the script attempts to download them automatically. Ensure you have internet access.
*   **PDF Reading Errors**: Ensure the input PDF is not encrypted and contains selectable text (not just scanned images).
*   **Dependency Issues**: Python 3.8+ is required. If `numpy` installation fails, try upgrading `pip` and `setuptools`.

## 📄 License

MIT License. Built for the QuizForge Hackathon.
