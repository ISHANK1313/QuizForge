import spacy
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure spacy model is loaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class NERExtractor:
    """
    Use spaCy for Named Entity Recognition.
    """
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize NERExtractor.

        Args:
            model_name (str): Name of the spaCy model.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
             # Fallback if the global load didn't work for some reason or different model requested
            logging.info(f"Downloading model {model_name}")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Process text through spaCy and extract entities.

        Args:
            text (str): Input text.

        Returns:
            List[Dict]: List of entities with metadata.
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Basic context window (sentence)
            sentence = ent.sent.text.strip()
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'sentence': sentence,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })

        return entities

    def extract_entities_with_context(self, text: str, sentences: List[str]) -> List[Dict]:
        """
        Extract entities with extended paragraph context for better answers.

        Args:
            text (str): Full document text.
            sentences (List[str]): List of sentences.

        Returns:
            List[Dict]: Entities with paragraph context.
        """
        doc = self.nlp(text)
        entities = []

        # Build sentence to paragraph mapping
        paragraphs = text.split('\n\n')
        sentence_to_para = {}
        for para in paragraphs:
            para_sentences = [s.strip() for s in para.split('.') if s.strip()]
            for sent in para_sentences:
                sentence_to_para[sent] = para

        for ent in doc.ents:
            sentence = ent.sent.text.strip()

            # Find paragraph context (3-5 sentences around the entity)
            para_context = sentence_to_para.get(sentence, sentence)

            # If paragraph is too short, use surrounding sentences
            if len(para_context.split()) < 20:
                sent_idx = None
                for idx, s in enumerate(sentences):
                    if sentence in s:
                        sent_idx = idx
                        break

                if sent_idx is not None:
                    start = max(0, sent_idx - 2)
                    end = min(len(sentences), sent_idx + 3)
                    para_context = ' '.join(sentences[start:end])

            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'sentence': sentence,
                'paragraph_context': para_context,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })

        return entities

    def filter_entities(self, entities: List[Dict], entity_types: List[str]) -> List[Dict]:
        """
        Filter entities by type.

        Args:
            entities (List[Dict]): List of entities.
            entity_types (List[str]): List of allowed entity types (e.g., ['PERSON', 'ORG']).

        Returns:
            List[Dict]: Filtered list of entities.
        """
        return [ent for ent in entities if ent['label'] in entity_types]

    def filter_quality_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter entities for quality - remove short, common, or irrelevant ones.

        Args:
            entities (List[Dict]): List of entities.

        Returns:
            List[Dict]: High-quality entities only.
        """
        # Common words to exclude (too generic)
        exclude_terms = {
            'extraordinary', 'important', 'special', 'general', 'particular',
            'several', 'various', 'different', 'certain', 'other', 'such',
            'first', 'second', 'third', 'last', 'next', 'previous', 'new',
            'old', 'good', 'bad', 'great', 'small', 'large', 'many', 'few'
        }

        quality_entities = []
        for ent in entities:
            text = ent['text'].lower()

            # Skip if too short (< 3 characters)
            if len(text) < 3:
                continue

            # Skip common/generic terms
            if text in exclude_terms:
                continue

            # Skip if it's just a number or single letter
            if text.isdigit() or len(text) == 1:
                continue

            # Require meaningful context (sentence length > 20 chars)
            if len(ent['sentence']) < 20:
                continue

            # Keep only if entity appears in a substantial sentence
            quality_entities.append(ent)

        return quality_entities


class TFIDFAnalyzer:
    """
    Use sklearn TfidfVectorizer to extract keywords.
    """
    def __init__(self, max_features: int = 50, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TFIDFAnalyzer.

        Args:
            max_features (int): Maximum number of features (keywords) to extract.
            ngram_range (Tuple[int, int]): Range of n-grams.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.feature_names = []
        self.scores = {}

    def fit_transform(self, sentences: List[str]) -> Any:
        """
        Fit vectorizer and transform sentences.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            sparse matrix: TF-IDF matrix.
        """
        if not sentences:
            logging.warning("No sentences to analyze.")
            return None

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            self.feature_names = self.vectorizer.get_feature_names_out()

            # Calculate global importance of each feature (sum of scores across all docs)
            sums = tfidf_matrix.sum(axis=0)
            self.scores = {}
            for col, term in enumerate(self.feature_names):
                self.scores[term] = sums[0, col]

            return tfidf_matrix
        except ValueError as e:
            logging.error(f"Error in TF-IDF analysis: {e}")
            return None

    def get_top_keywords(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top n keywords by score.

        Args:
            n (int): Number of keywords to return.

        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples.
        """
        sorted_items = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:n]

    def map_keywords_to_sentences(self, keywords: List[str], sentences: List[str]) -> Dict[str, List[str]]:
        """
        Map keywords to sentences containing them.

        Args:
            keywords (List[str]): List of keywords.
            sentences (List[str]): List of sentences.

        Returns:
            Dict[str, List[str]]: Dictionary mapping keywords to list of source sentences.
        """
        mapping = {}
        for keyword in keywords:
            mapping[keyword] = []
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                     mapping[keyword].append(sentence)
        return mapping


class DifficultyClassifier:
    """
    Rule-based classifier for question difficulty.
    """
    def __init__(self):
        pass

    def extract_features(self, sentence: str, entity_count: int = 0) -> Dict[str, Any]:
        """
        Calculate features for difficulty classification.

        Args:
            sentence (str): Input sentence.
            entity_count (int): Number of named entities in the sentence.

        Returns:
            Dict: Feature dictionary.
        """
        words = sentence.split()
        sentence_length = len(words)
        avg_word_length = sum(len(word) for word in words) / sentence_length if sentence_length > 0 else 0

        # Count technical terms (simple heuristic: words > 8 characters)
        technical_terms = sum(1 for word in words if len(word) > 8)

        # Calculate entity density
        entity_density = entity_count / sentence_length if sentence_length > 0 else 0

        return {
            'sentence_length': sentence_length,
            'avg_word_length': avg_word_length,
            'technical_terms': technical_terms,
            'entity_density': entity_density
        }

    def classify(self, sentence: str, entity_count: int = 0) -> Tuple[str, float]:
        """
        Classify sentence difficulty.

        Args:
            sentence (str): Input sentence.
            entity_count (int): Number of named entities in the sentence.

        Returns:
            Tuple[str, float]: (difficulty_level, confidence_score)
        """
        features = self.extract_features(sentence, entity_count)
        length = features['sentence_length']
        tech_terms = features['technical_terms']
        entity_density = features['entity_density']

        # More lenient rules for better distribution
        if length < 15 and tech_terms <= 1:
            return 'easy', 0.85
        elif length < 25 and tech_terms <= 3:
            # High entity density makes it easier (factual recall)
            if entity_density > 0.2:
                return 'easy', 0.8
            return 'medium', 0.8
        else:
            # Very high technical term count makes it hard
            if tech_terms > 5 or length > 30:
                return 'hard', 0.9
            return 'medium', 0.75
