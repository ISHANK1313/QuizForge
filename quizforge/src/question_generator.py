import re
import random
from typing import List, Dict, Any, Tuple
from src.utils import similarity_score

class QuestionTemplates:
    """
    Template-based question creation.
    """
    def __init__(self):
        pass

    def generate_definition_question(self, entity: str) -> str:
        """
        Generate definition question: "What is {entity}?"
        """
        return f"What is {entity}?"

    def generate_concept_question(self, keyword: str) -> str:
        """
        Generate concept question: "Explain the concept of {keyword}."
        """
        return f"Explain the concept of {keyword}."

    def generate_application_question(self, concept: str) -> str:
        """
        Generate application question: "How would you apply {concept} in practice?"
        """
        return f"How would you apply {concept} in practice?"

    def generate_fill_blank(self, sentence: str, keyword: str) -> str:
        """
        Generate fill-in-blank question.
        Replace keyword with "_______".
        """
        # Case insensitive replacement, but keep original case in sentence except for the blank
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub("_______", sentence)


class MLQuestionRanker:
    """
    Score and rank questions.
    """
    def __init__(self, tfidf_scores: Dict[str, float] = None, entity_frequencies: Dict[str, int] = None):
        """
        Initialize ranker with scores.

        Args:
            tfidf_scores (Dict): Dictionary of TF-IDF scores for keywords.
            entity_frequencies (Dict): Dictionary of entity frequencies.
        """
        self.tfidf_scores = tfidf_scores or {}
        self.entity_frequencies = entity_frequencies or {}

    def calculate_score(self, question: Dict[str, Any]) -> float:
        """
        Calculate composite score for a question.

        Args:
            question (Dict): Question dictionary.

        Returns:
            float: Composite score.
        """
        # 1. TF-IDF Score (0.4)
        # Assuming the 'target' or 'answer' is the keyword/entity
        target = question.get('answer', '')
        tfidf_val = self.tfidf_scores.get(target, 0.5) # Default 0.5 if not found

        # Normalize tfidf (assuming max score is around 1.0, though it can be higher)
        # Using simple clipping for now
        tfidf_val = min(tfidf_val, 1.0)

        # 2. Entity Importance (0.3)
        # Using entity frequency if available
        entity_val = self.entity_frequencies.get(target, 1)
        entity_score = min(entity_val * 0.1, 1.0) # Simple normalization

        # 3. Sentence Position (0.2)
        # Prioritize questions from earlier in the document? Or later?
        # Let's assume 'source_index' is passed, lower index = higher importance (introduction)
        # Or just random for now if not provided
        position_score = 0.5

        # 4. Difficulty Diversity (0.1)
        # Bonus for harder questions? Or just standard weight.
        difficulty_map = {'easy': 0.5, 'medium': 0.8, 'hard': 1.0}
        diff_score = difficulty_map.get(question.get('difficulty', 'medium'), 0.5)

        composite_score = (
            tfidf_val * 0.4 +
            entity_score * 0.3 +
            position_score * 0.2 +
            diff_score * 0.1
        )
        return composite_score

    def rank_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank questions by score descending.

        Args:
            questions (List[Dict]): List of questions.

        Returns:
            List[Dict]: Ranked list.
        """
        for q in questions:
            q['score'] = self.calculate_score(q)

        return sorted(questions, key=lambda x: x['score'], reverse=True)

    def ensure_distribution(self, questions: List[Dict[str, Any]], target_dist: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Select questions to match distribution (e.g. 40% easy, 40% medium, 20% hard).
        This method assumes questions are already ranked.

        Args:
            questions (List[Dict]): Ranked list of questions.
            target_dist (Dict): Target distribution percentages.

        Returns:
            List[Dict]: List of questions interleaved to respect distribution.
        """
        if not questions:
            return []

        # Group by difficulty
        grouped = {'easy': [], 'medium': [], 'hard': []}
        for q in questions:
            diff = q.get('difficulty', 'medium')
            if diff in grouped:
                grouped[diff].append(q)
            else:
                grouped['medium'].append(q) # Fallback

        result = []

        # Create copies of lists to consume
        queues = {
            'easy': list(grouped['easy']),
            'medium': list(grouped['medium']),
            'hard': list(grouped['hard'])
        }

        # Interleave based on ratio: 4:4:2 -> 2:2:1
        # Simple cycle: Easy, Medium, Easy, Medium, Hard
        while any(queues.values()):
            added = False
            # Add Easy
            if queues['easy']:
                result.append(queues['easy'].pop(0))
                added = True
            # Add Medium
            if queues['medium']:
                result.append(queues['medium'].pop(0))
                added = True

            # Add Easy again (to approximate 40%)
            if queues['easy']:
                result.append(queues['easy'].pop(0))
                added = True
            # Add Medium again (to approximate 40%)
            if queues['medium']:
                result.append(queues['medium'].pop(0))
                added = True

            # Add Hard (20%)
            if queues['hard']:
                result.append(queues['hard'].pop(0))
                added = True

            if not added:
                break

        return result


class QuestionValidator:
    """
    Quality control for questions.
    """
    def __init__(self):
        pass

    def is_valid_question(self, question: str) -> bool:
        """
        Check if question is valid.

        Args:
            question (str): Question text.

        Returns:
            bool: True if valid.
        """
        if not question:
            return False
        if len(question) <= 10:
            return False
        if not question.strip().endswith('?'):
            # Fill-in-blank might not end with ?
            if "_______" not in question:
                 return False
        return True

    def remove_duplicates(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate questions using similarity checking.

        Args:
            questions (List[Dict]): List of questions.

        Returns:
            List[Dict]: Unique questions.
        """
        unique_questions = []

        for q in questions:
            is_duplicate = False
            for uq in unique_questions:
                # Check similarity of question text
                sim = similarity_score(q['question'], uq['question'])
                if sim > 0.85:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_questions.append(q)

        return unique_questions

    def ensure_diversity(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure variety in question types.

        Args:
            questions (List[Dict]): List of questions.

        Returns:
            List[Dict]: List of questions.
        """
        # This acts as a pass-through in this implementation,
        # as diversity is partly handled by the generation logic looping through different types.
        # But we could enforce max count per type here.
        return questions
