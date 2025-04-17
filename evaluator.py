from typing import List, Dict, Set
import statistics
import logging
from phoneme_control import RhymeController

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    def __init__(self, rhyme_controller: RhymeController, valid_pos_patterns: List[List[str]] = None):
        """Initialize evaluator with rhyme controller and optional POS patterns."""
        self.rhyme_controller = rhyme_controller
        # Define common Tamil POS patterns (simplified for demo)
        self.valid_pos_patterns = valid_pos_patterns or [
            ['NOUN', 'NOUN', 'VERB'],
            ['NOUN', 'ADJ', 'VERB'],
            ['PRON', 'VERB', 'ADP'],
            ['NOUN', 'NOUN', 'NOUN', 'VERB'],
            ['NOUN', 'NOUN', 'NOUN', 'NOUN', 'VERB'],
            ['PRON', 'NOUN', 'VERB'],
            ['ADJ', 'NOUN', 'VERB']
        ]
        logging.debug("Evaluator initialized")

    def check_pos_validity(self, pos_sequence: List[str]) -> bool:
        """Check if a POS sequence matches a valid Tamil pattern."""
        return pos_sequence in self.valid_pos_patterns

    def count_syllables(self, word: str) -> int:
        """Approximate syllable count based on Tamil vowels."""
        vowels = self.rhyme_controller.tamil_vowels.keys()
        syllable_count = sum(1 for char in word if char in vowels)
        return max(1, syllable_count)  # At least 1 syllable per word

    def evaluate_lyrics(self, lyrics: List[str], rhyme_scheme: str, keywords: List[str], 
                      genre: str, pos_sequences: List[List[str]], genre_words: Set[str]) -> Dict:
        """Evaluate lyrics across multiple metrics."""
        logging.debug(f"Evaluating lyrics: {lyrics}")
        metrics = {}

        # 1. POS Sequence Validity
        valid_lines = sum(1 for pos_seq in pos_sequences if self.check_pos_validity(pos_seq))
        metrics['pos_validity'] = (valid_lines / len(lyrics)) * 100 if lyrics else 0.0
        logging.debug(f"POS Validity: {metrics['pos_validity']}%")

        # 2. Rhyme Scheme Accuracy
        metrics['rhyme_accuracy'] = 100.0 if self.rhyme_controller.validate_rhyme_scheme(lyrics, rhyme_scheme) else 0.0
        logging.debug(f"Rhyme Accuracy: {metrics['rhyme_accuracy']}%")

        # 3. Rhyme Consistency
        rhyme_groups = {}
        for i, line in enumerate(lyrics):
            last_word = self.rhyme_controller._get_last_word(line)
            vowel_ending = self.rhyme_controller._get_vowel_ending(last_word)
            rhyme_letter = rhyme_scheme[i].upper() if i < len(rhyme_scheme) else 'X'
            if rhyme_letter not in rhyme_groups:
                rhyme_groups[rhyme_letter] = []
            rhyme_groups[rhyme_letter].append(vowel_ending)
        
        correct_rhymes = 0
        total_rhymed = 0
        for letter, endings in rhyme_groups.items():
            if letter == 'X' or not endings:
                continue
            first_ending = endings[0]
            if first_ending:  # Only count if there's a valid ending
                correct_rhymes += sum(1 for e in endings if e == first_ending)
                total_rhymed += len([e for e in endings if e])
        metrics['rhyme_consistency'] = (correct_rhymes / total_rhymed * 100) if total_rhymed > 0 else 100.0
        logging.debug(f"Rhyme Consistency: {metrics['rhyme_consistency']}%")

        # 4. Keyword Inclusion
        found_keywords = sum(1 for k in keywords if any(k in line for line in lyrics))
        metrics['keyword_inclusion'] = (found_keywords / len(keywords) * 100) if keywords else 100.0
        logging.debug(f"Keyword Inclusion: {metrics['keyword_inclusion']}%")

        # 5. Genre Consistency
        total_words = 0
        genre_specific_words = 0
        for line in lyrics:
            words = line.strip().split()
            total_words += len(words)
            genre_specific_words += sum(1 for w in words if w in genre_words)
        metrics['genre_consistency'] = (genre_specific_words / total_words * 100) if total_words > 0 else 0.0
        logging.debug(f"Genre Consistency: {metrics['genre_consistency']}%")

        # 6. Unique Word Ratio
        all_words = []
        for line in lyrics:
            all_words.extend(line.strip().split())
        unique_words = len(set(all_words))
        metrics['unique_word_ratio'] = (unique_words / len(all_words) * 100) if all_words else 0.0
        logging.debug(f"Unique Word Ratio: {metrics['unique_word_ratio']}%")

        # 7. Syllable Consistency
        syllable_counts = []
        for line in lyrics:
            words = line.strip().split()
            line_syllables = sum(self.count_syllables(word) for word in words)
            syllable_counts.append(line_syllables)
        mean_syllables = statistics.mean(syllable_counts) if syllable_counts else 1.0
        std_syllables = statistics.stdev(syllable_counts) if len(syllable_counts) > 1 else 0.0
        metrics['syllable_consistency'] = (1 - std_syllables / mean_syllables) * 100 if mean_syllables > 0 else 100.0
        logging.debug(f"Syllable Consistency: {metrics['syllable_consistency']}%")

        return metrics