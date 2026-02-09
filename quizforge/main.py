import os
import sys
import argparse
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import PDFExtractor, TextPreprocessor
from src.ml_features import NERExtractor, TFIDFAnalyzer, DifficultyClassifier
from src.question_generator import QuestionTemplates, MLQuestionRanker, QuestionValidator
from src.output_handler import PDFGenerator, Formatter
from src.utils import load_config, setup_logging, create_directories, validate_pdf_file

def generate_sample_pdf(output_dir: str) -> str:
    """Generate a sample PDF for demo purposes."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        create_directories(output_dir)
        path = os.path.join(output_dir, "sample_python_lecture.pdf")
        if os.path.exists(path):
            return path

        logging.info("Generating sample PDF...")
        c = canvas.Canvas(path, pagesize=letter)
        text = """
        Introduction to Python Programming

        Python is a high-level, interpreted programming language known for its readability and versatility.
        Guido van Rossum created Python, and it was first released in 1991.
        Python emphasizes code readability with its notable use of significant indentation.

        Key Concepts:

        1. Variables and Data Types:
        In Python, variables are created when you assign a value to them.
        Common data types include integers, floats, strings, and booleans.
        For example, x = 5 creates an integer variable.

        2. Control Structures:
        Python supports usual control flow statements like if, for, and while.
        The 'if' statement is used for conditional execution.
        The 'for' loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string).

        3. Functions:
        A function is a block of code which only runs when it is called.
        You can pass data, known as parameters, into a function.
        A function can return data as a result.
        In Python, a function is defined using the 'def' keyword.

        4. Object-Oriented Programming (OOP):
        Python is an object-oriented language.
        Almost everything in Python is an object, with its properties and methods.
        A Class is like an object constructor, or a "blueprint" for creating objects.

        5. Libraries:
        Python has a vast ecosystem of libraries.
        NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
        Pandas is a software library written for the Python programming language for data manipulation and analysis.

        Machine Learning:
        Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.
        """

        y = 750
        for line in text.split('\n'):
            if y < 50:
                c.showPage()
                y = 750
            c.drawString(50, y, line.strip())
            y -= 15

        c.save()
        return path
    except Exception as e:
        logging.error(f"Failed to generate sample PDF: {e}")
        return ""

def generate_question_bank(pdf_path: str, config: dict):
    """
    Main pipeline to generate question bank.
    """
    # STEP 1: PDF Extraction
    print("Step 1/5: Extracting text from PDF...")
    try:
        extractor = PDFExtractor(pdf_path)
        raw_text = extractor.extract_text()

        preprocessor = TextPreprocessor(raw_text)
        clean_text = preprocessor.clean_text()
        sentences = preprocessor.tokenize_sentences()

        # Filter short sentences
        sentences = preprocessor.remove_short_sentences(sentences, min_length=config['pdf_processing']['min_sentence_length'])

        logging.info(f"Extracted {len(sentences)} valid sentences.")
    except Exception as e:
        logging.error(f"Error in PDF processing: {e}")
        return None

    # STEP 2: ML Feature Extraction
    print("Step 2/5: Extracting ML features...")
    try:
        ner = NERExtractor(config['ml_models']['spacy_model'])
        all_entities = ner.extract_entities(clean_text)
        # Filter by type first
        entities = ner.filter_entities(all_entities, ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'LAW', 'NORP'])
        # Then filter for quality
        entities = ner.filter_quality_entities(entities)
        logging.info(f"Filtered to {len(entities)} quality entities.")

        tfidf = TFIDFAnalyzer(
            max_features=config['ml_models']['tfidf_max_features'],
            ngram_range=tuple(config['ml_models']['tfidf_ngram_range'])
        )
        tfidf.fit_transform(sentences)
        keywords = tfidf.get_top_keywords(n=20)

        # Calculate entity counts per sentence for difficulty classification
        sentence_entity_counts = {}
        for entity in entities:
            s = entity['sentence']
            sentence_entity_counts[s] = sentence_entity_counts.get(s, 0) + 1

        classifier = DifficultyClassifier()
        sentence_difficulties = {}
        for sent in sentences:
            ent_count = sentence_entity_counts.get(sent, 0)
            diff, conf = classifier.classify(sent, entity_count=ent_count)
            sentence_difficulties[sent] = diff

    except Exception as e:
        logging.error(f"Error in ML feature extraction: {e}")
        return None

    # STEP 3: Question Generation
    print("Step 3/5: Generating questions...")
    try:
        templates = QuestionTemplates()
        questions = []

        # Generate Definition Questions from Entities (with better answers)
        for entity in tqdm(entities, desc="Processing Entities"):
            q = templates.generate_definition_question(entity['text'], entity['sentence'])

            # Extract a better answer from context
            source_sent = entity['sentence']
            answer_text = f"{entity['text']}: {source_sent}"

            questions.append({
                'question': q,
                'type': 'definition',
                'difficulty': sentence_difficulties.get(entity['sentence'], 'medium'),
                'source': source_sent,
                'answer': answer_text
            })

        # Generate Concept Questions from Keywords
        keyword_map = tfidf.map_keywords_to_sentences([k for k, s in keywords], sentences)
        for keyword, score in tqdm(keywords[:15], desc="Processing Keywords"):
            # Only generate if we have good context
            source_sents = keyword_map.get(keyword, [])
            if not source_sents:
                continue

            # Pick the best source sentence (longest with keyword)
            source = max(source_sents, key=len) if source_sents else ""

            if len(source.split()) < 10:
                continue

            q = templates.generate_concept_question(keyword, source)
            answer_text = f"{keyword}: {source}"

            questions.append({
                'question': q,
                'type': 'concept',
                'difficulty': sentence_difficulties.get(source, 'medium'),
                'tfidf_score': score,
                'source': source,
                'answer': answer_text
            })

            # Generate Fill-in-Blank (only for good sentences)
            if len(source.split()) >= 10 and len(source.split()) <= 25:
                q_blank = templates.generate_fill_blank(source, keyword)
                if q_blank:
                    questions.append({
                        'question': q_blank,
                        'type': 'fill_blank',
                        'difficulty': sentence_difficulties.get(source, 'easy'),
                        'source': source,
                        'answer': f"Answer: {keyword}. Full context: {source}"
                    })

        # Generate Application Questions (only for top concepts)
        for keyword, score in keywords[:8]:
            source_sents = keyword_map.get(keyword, [])
            source = source_sents[0] if source_sents else "General application"

            q_app = templates.generate_application_question(keyword, source)
            answer_text = f"Application of {keyword}: This concept can be applied by understanding its context: {source}"

            questions.append({
                'question': q_app,
                'type': 'application',
                'difficulty': 'hard',
                'tfidf_score': score,
                'source': source,
                'answer': answer_text
            })

        # Add analytical "Why" questions for variety
        for entity in entities[:5]:
            q_why = templates.generate_why_question(entity['text'], entity['sentence'])
            questions.append({
                'question': q_why,
                'type': 'analytical',
                'difficulty': 'medium',
                'source': entity['sentence'],
                'answer': f"Analysis of {entity['text']}: {entity['sentence']}"
            })

    except Exception as e:
        logging.error(f"Error in question generation: {e}")
        return None

    # STEP 4: Ranking and Filtering
    print("Step 4/5: Ranking questions...")
    try:
        # Mock entity frequencies for ranking
        entity_freqs = {}
        for ent in entities:
            entity_freqs[ent['text']] = entity_freqs.get(ent['text'], 0) + 1

        ranker = MLQuestionRanker(tfidf_scores=dict(keywords), entity_frequencies=entity_freqs)
        ranked = ranker.rank_questions(questions)

        # Filter duplicates first?
        validator = QuestionValidator()
        unique_questions = validator.remove_duplicates(ranked)

        # Then ensure distribution
        filtered = ranker.ensure_distribution(unique_questions, config['question_generation']['difficulty_distribution'])

        # Finally slice
        final_questions = filtered[:config['question_generation']['total_questions']]

    except Exception as e:
        logging.error(f"Error in ranking: {e}")
        return None

    # STEP 5: Output Generation
    print("Step 5/5: Generating PDF...")
    try:
        formatter = Formatter()

        output_dir = config['output'].get('output_dir', 'data/output')
        create_directories(output_dir)
        output_path = os.path.join(output_dir, 'question_bank.pdf')

        generator = PDFGenerator(output_path)
        result_path = generator.generate(final_questions)

        # Summary
        stats = formatter.create_summary_stats(final_questions)
        print(f"\n✅ Success! Generated {len(final_questions)} questions")
        print(f"   Easy: {stats['easy']}, Medium: {stats['medium']}, Hard: {stats['hard']}")
        print(f"   Output: {result_path}")

        return result_path
    except Exception as e:
        logging.error(f"Error in output generation: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuizForge - AI Question Bank Generator")
    parser.add_argument('--input', help="Input PDF file path")
    parser.add_argument('--output', help="Output directory")
    parser.add_argument('--num-questions', type=int, help="Number of questions to generate")
    parser.add_argument('--config', default='config/config.yaml', help="Configuration file path")
    parser.add_argument('--demo', action='store_true', help="Run in demo mode with sample PDF")

    args = parser.parse_args()

    # Setup
    setup_logging()

    # Load Config
    try:
        config_path = os.path.join(os.path.dirname(__file__), args.config)
        # Adjust if running from root and config is passed relative
        if not os.path.exists(config_path):
             config_path = args.config

        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Override config with args
    if args.output:
        config['output']['output_dir'] = args.output
    if args.num_questions:
        config['question_generation']['total_questions'] = args.num_questions

    # Determine Input
    pdf_path = args.input
    if args.demo:
        print("🚀 Running in DEMO mode...")
        sample_dir = os.path.join(os.path.dirname(__file__), 'data/sample_pdfs')
        # Fix path if running from root
        if not os.path.exists(sample_dir):
            sample_dir = 'data/sample_pdfs'

        pdf_path = generate_sample_pdf(sample_dir)
        if not pdf_path:
            print("❌ Failed to generate sample PDF.")
            sys.exit(1)

    if not pdf_path:
        # Check if no input provided, show help
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"❌ Input file not found: {pdf_path}")
        sys.exit(1)

    # Run Pipeline
    result = generate_question_bank(pdf_path, config)

    if not result:
        sys.exit(1)
