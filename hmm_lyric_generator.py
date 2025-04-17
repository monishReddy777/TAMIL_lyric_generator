import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path
from phoneme_control import RhymeController
from evaluator import Evaluator
import logging
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data (run once)
nltk.download('punkt')

class HMMLyricGenerator:
    def __init__(self, processed_data_path: str = "processed_data"):
        """Initialize the HMM-based lyric generator."""
        logging.debug("Initializing HMMLyricGenerator")
        self.processed_data_path = Path(processed_data_path)
        try:
            self.lyrics_data = self._load_processed_data()
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")
        self.rhyme_controller = RhymeController()
        self.rhyme_controller.lyrics_data = self.lyrics_data
        self.evaluator = Evaluator(self.rhyme_controller)
        self.hmm = self._build_hmm()
        self.available_genres = self._get_available_genres()
        self.genre_words = self._build_genre_words()
        self.smooth = SmoothingFunction().method1  # For BLEU smoothing

    def _load_processed_data(self) -> List[Dict]:
        """Load processed lyrics data with POS tags from JSON file."""
        file_path = self.processed_data_path / 'processed_lyrics.json'
        logging.debug(f"Loading data from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)} at line {e.lineno}, column {e.colno}")

        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of entries")
        if not data:
            raise ValueError("No lyrics data found in the file")

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Entry {i} is not a dictionary")
            if 'genre' not in item:
                raise ValueError(f"Entry {i} missing 'genre' field")
            if 'text' not in item:
                raise ValueError(f"Entry {i} missing 'text' field")
            if 'pos_tags' not in item:
                raise ValueError(f"Entry {i} missing 'pos_tags' field")
            if not isinstance(item['pos_tags'], list):
                item['pos_tags'] = []
            for j, tag in enumerate(item['pos_tags']):
                if not isinstance(tag, list) or len(tag) != 2:
                    raise ValueError(f"Invalid pos_tag format at entry {i}, tag {j}")

        logging.debug(f"Loaded {len(data)} entries")
        return data

    def _get_available_genres(self) -> set:
        """Get all available genres from the lyrics data."""
        genres = set()
        for item in self.lyrics_data:
            genre = item['genre'].strip().lower()
            if genre:
                genres.add(genre)
        logging.debug(f"Available genres: {genres}")
        return genres

    def _build_genre_words(self) -> Dict[str, set]:
        """Build a set of genre-specific words."""
        genre_words = defaultdict(set)
        for item in self.lyrics_data:
            genre = item['genre'].lower()
            words = set(word for word, _ in item['pos_tags'])
            genre_words[genre].update(words)
        return genre_words

    def get_available_genres(self) -> List[str]:
        """Get sorted list of available genres."""
        return sorted(list(self.available_genres))

    def _build_hmm(self) -> Dict:
        """Build HMM for POS tag sequence modeling."""
        transitions = defaultdict(lambda: defaultdict(int))
        emissions = defaultdict(lambda: defaultdict(int))
        initial_probs = defaultdict(int)

        logging.debug("Building HMM...")
        for item in self.lyrics_data:
            pos_tags = item['pos_tags']
            if not pos_tags:
                continue

            prev_pos = None
            for word, pos in pos_tags:
                emissions[pos][word] += 1
                if prev_pos is None:
                    initial_probs[pos] += 1
                else:
                    transitions[prev_pos][pos] += 1
                prev_pos = pos

        hmm = {
            'transitions': {},
            'emissions': {},
            'initial_probs': {}
        }

        total_initial = sum(initial_probs.values()) or 1
        for pos in initial_probs:
            hmm['initial_probs'][pos] = initial_probs[pos] / total_initial

        for pos1 in transitions:
            total = sum(transitions[pos1].values()) or 1
            hmm['transitions'][pos1] = {pos2: count / total for pos2, count in transitions[pos1].items()}

        for pos in emissions:
            total = sum(emissions[pos].values()) or 1
            hmm['emissions'][pos] = {word: count / total for word, count in emissions[pos].items()}

        logging.debug("HMM built successfully")
        return hmm

    def _generate_pos_sequence(self, length: int) -> List[str]:
        """Generate a sequence of POS tags using the HMM."""
        pos_sequence = []
        pos_tags = list(self.hmm['initial_probs'].keys())
        if not pos_tags:
            logging.warning("No POS tags available, using fallback")
            return ['NOUN'] * length

        current_pos = random.choices(
            pos_tags,
            weights=[self.hmm['initial_probs'].get(pos, 0) for pos in pos_tags],
            k=1
        )[0]
        pos_sequence.append(current_pos)

        for _ in range(length - 1):
            if current_pos not in self.hmm['transitions']:
                current_pos = random.choice(pos_tags)
            else:
                next_pos = random.choices(
                    list(self.hmm['transitions'][current_pos].keys()),
                    weights=list(self.hmm['transitions'][current_pos].values()),
                    k=1
                )[0]
                pos_sequence.append(next_pos)
                current_pos = next_pos

        logging.debug(f"Generated POS sequence: {pos_sequence}")
        return pos_sequence

    def _get_word_for_pos(self, pos: str, genre: str = None, keywords: List[str] = None) -> str:
        """Get a word for a given POS tag."""
        if pos not in self.hmm['emissions']:
            logging.warning(f"No emissions for POS {pos}, using fallback")
            return "மழை"

        available_words = list(self.hmm['emissions'][pos].keys())

        if genre:
            genre = genre.lower()
            genre_words = []
            for item in self.lyrics_data:
                if item['genre'].lower() == genre:
                    for word, p in item['pos_tags']:
                        if p == pos:
                            genre_words.append(word)
            if genre_words:
                available_words = list(set(available_words).intersection(genre_words))

        if keywords:
            matching_keywords = [k for k in keywords if k in available_words]
            if matching_keywords:
                word = random.choice(matching_keywords)
                logging.debug(f"Selected keyword: {word}")
                return word

        if available_words:
            word = random.choices(
                available_words,
                weights=[self.hmm['emissions'][pos].get(word, 0) for word in available_words],
                k=1
            )[0]
            logging.debug(f"Selected word for POS {pos}: {word}")
            return word

        logging.warning("No available words, using fallback")
        return "மழை"

    def _generate_line(self, 
                      max_length: int = 10,
                      rhyme_pattern: str = None,
                      genre: str = None,
                      keywords: List[str] = None) -> Tuple[str, List[str]]:
        """Generate a single line of lyrics and return its POS sequence."""
        logging.debug(f"Generating line with max_length={max_length}, rhyme_pattern={rhyme_pattern}, genre={genre}, keywords={keywords}")
        pos_sequence = self._generate_pos_sequence(max_length)
        words = []

        keyword_used = False
        for pos in pos_sequence:
            if keywords and not keyword_used:
                word = self._get_word_for_pos(pos, genre, keywords)
                if word in keywords:
                    keyword_used = True
                words.append(word)
            else:
                words.append(self._get_word_for_pos(pos, genre))

        line = " ".join(words[:max_length])

        if rhyme_pattern:
            logging.debug(f"Applying rhyme pattern: {rhyme_pattern}")
            rhyming_words = self.rhyme_controller.get_rhyming_words(rhyme_pattern, genre)
            if rhyming_words:
                words[-1] = random.choice(rhyming_words)
                line = " ".join(words[:max_length])
                logging.debug(f"Rhymed line: {line}")
            else:
                logging.warning(f"No rhyming words found for pattern {rhyme_pattern}")

        return line, pos_sequence

    def _compute_perplexity(self, lyrics: List[str], pos_sequences: List[List[str]]) -> float:
        """Compute perplexity based on HMM probabilities."""
        logging.debug("Computing perplexity")
        total_log_prob = 0.0
        total_tokens = 0

        for line, pos_seq in zip(lyrics, pos_sequences):
            words = line.split()
            if len(words) != len(pos_seq) or len(words) < 2:
                continue

            for i in range(len(words) - 1):
                current_pos = pos_seq[i]
                next_pos = pos_seq[i + 1]
                current_word = words[i]
                next_word = words[i + 1]

                # Transition probability
                trans_prob = self.hmm['transitions'].get(current_pos, {}).get(next_pos, 1e-10)
                # Emission probability for next word given next POS
                emit_prob = self.hmm['emissions'].get(next_pos, {}).get(next_word, 1e-10)
                prob = trans_prob * emit_prob

                total_log_prob += math.log2(prob + 1e-10)  # Avoid log(0)
                total_tokens += 1

        if total_tokens == 0:
            logging.warning("No valid tokens for perplexity calculation")
            return float('inf')
        
        perplexity = 2 ** (-total_log_prob / total_tokens)
        logging.debug(f"Perplexity calculated: {perplexity}")
        return perplexity

    def generate_lyrics(self,
                       genre: str = None,
                       keywords: List[str] = None,
                       num_lines: int = 4,
                       rhyme_scheme: str = "AABB",
                       max_line_length: int = 10,
                       references: List[List[str]] = None) -> Tuple[List[str], Dict]:
        """Generate Tamil lyrics using HMM and evaluate them."""
        logging.debug(f"Generating lyrics: genre={genre}, keywords={keywords}, num_lines={num_lines}, rhyme_scheme={rhyme_scheme}, max_line_length={max_line_length}")

        if num_lines < 1:
            raise ValueError("Number of lines must be at least 1")
        if max_line_length < 1:
            raise ValueError("Maximum line length must be at least 1")
        if not isinstance(rhyme_scheme, str):
            raise ValueError(f"Rhyme scheme must be a string, got {type(rhyme_scheme)}")
        if not rhyme_scheme:
            raise ValueError("Rhyme scheme cannot be empty")
        if len(rhyme_scheme) < num_lines:
            raise ValueError(f"Rhyme scheme '{rhyme_scheme}' too short for {num_lines} lines")

        if genre:
            genre = genre.strip().lower()
            if genre not in self.available_genres:
                raise ValueError(f"Invalid genre: {genre}. Available genres: {', '.join(self.get_available_genres())}")

        available_lyrics = self.lyrics_data if not genre else [
            item for item in self.lyrics_data if item['genre'].lower() == genre
        ]
        if not available_lyrics:
            raise ValueError(f"No lyrics found for genre: {genre}")

        lines = []
        pos_sequences = []
        rhyme_patterns = {}

        for i in range(num_lines):
            try:
                rhyme_letter = rhyme_scheme[i].upper()
                logging.debug(f"Processing line {i+1}, rhyme_letter={rhyme_letter}")
            except IndexError:
                raise ValueError(f"Rhyme scheme '{rhyme_scheme}' too short for {num_lines} lines")
            if not rhyme_letter.isalpha():
                raise ValueError(f"Invalid rhyme scheme character at position {i}: {rhyme_letter}")

            if rhyme_letter not in rhyme_patterns:
                sample_text = random.choice(available_lyrics)['text']
                last_line = sample_text.split('\n')[-1] if '\n' in sample_text else sample_text
                rhyme_pattern = self.rhyme_controller._get_vowel_ending(
                    self.rhyme_controller._get_last_word(last_line)
                )
                rhyme_patterns[rhyme_letter] = rhyme_pattern if rhyme_pattern else 'a'
                logging.debug(f"Assigned rhyme pattern for {rhyme_letter}: {rhyme_patterns[rhyme_letter]}")

            line, pos_sequence = self._generate_line(
                max_length=max_line_length,
                rhyme_pattern=rhyme_patterns[rhyme_letter],
                genre=genre,
                keywords=keywords
            )
            lines.append(line)
            pos_sequences.append(pos_sequence)
            logging.debug(f"Generated line {i+1}: {line}")

        # Evaluate lyrics
        genre_word_set = self.genre_words.get(genre.lower(), set()) if genre else set()
        metrics = self.evaluator.evaluate_lyrics(
            lyrics=lines,
            rhyme_scheme=rhyme_scheme,
            keywords=keywords or [],
            genre=genre,
            pos_sequences=pos_sequences,
            genre_words=genre_word_set
        )

        # Compute BLEU if references are provided
        if references and references:
            bleu_scores = []
            for ref in references:
                if len(ref) > 0:  # Ensure reference is not empty
                    score = sentence_bleu([ref], lines[0].split(), smoothing_function=self.smooth)
                    bleu_scores.append(score)
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores) * 100 if bleu_scores else 0.0
            logging.debug(f"BLEU calculated: {metrics['bleu']}")

        # Compute perplexity
        metrics['perplexity'] = self._compute_perplexity(lines, pos_sequences)
        logging.debug(f"Perplexity added to metrics: {metrics['perplexity']}")

        logging.debug(f"Generated lyrics: {lines}, Metrics: {metrics}")
        return lines, metrics

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words (placeholder implementation)."""
        return [word.strip() for word in text.split() if word.strip()]


if __name__ == "__main__":
    # Example usage
    generator = HMMLyricGenerator()
    lyrics, metrics = generator.generate_lyrics(
        genre="love",
        keywords=["காதல்", "மழை", "நிலா"],
        num_lines=4,
        rhyme_scheme="AABB",
        max_line_length=10
    )
    for i, line in enumerate(lyrics, 1):
        print(f"{i}. {line}")
    print("Metrics:", metrics)