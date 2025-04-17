from typing import List, Dict, Optional
import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RhymeController:
    def __init__(self):
        self.tamil_vowels = {
            'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii', 'உ': 'u', 'ஊ': 'uu',
            'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
            'ா': 'aa', 'ி': 'i', 'ீ': 'ii', 'ு': 'u', 'ூ': 'uu', 'ெ': 'e',
            'ே': 'ee', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oo', 'ௌ': 'au'
        }
        self.rhyme_classes = {
            'a': ['அ', 'ா'], 'aa': ['ஆ', 'ா'], 'i': ['இ', 'ி'], 'ii': ['ஈ', 'ீ'],
            'u': ['உ', 'ு'], 'uu': ['ஊ', 'ூ'], 'e': ['எ', 'ெ'], 'ee': ['ஏ', 'ே'],
            'ai': ['ஐ', 'ை'], 'o': ['ஒ', 'ொ'], 'oo': ['ஓ', 'ோ'], 'au': ['ஔ', 'ௌ']
        }
        self.analyzer = self
        self.lyrics_data = None  # Will be set by HMMLyricGenerator
        logging.debug("Initialized RhymeController")

    def _get_last_word(self, line: str) -> str:
        if not line or not isinstance(line, str):
            logging.warning(f"Invalid line: {line}")
            return ""
        words = line.strip().split()
        return words[-1] if words else ""

    def _get_vowel_ending(self, word: str) -> Optional[str]:
        if not word or not isinstance(word, str):
            logging.warning(f"Invalid word: {word}")
            return None
        last_vowel = None
        for char in reversed(word):
            if char in self.tamil_vowels:
                last_vowel = self.tamil_vowels[char]
                break
        logging.debug(f"Vowel ending for '{word}': {last_vowel}")
        return last_vowel

    def get_rhyme_pattern(self, text: str) -> str:
        if not text or not isinstance(text, str):
            logging.warning(f"Invalid text for rhyme pattern: {text}")
            return ""
        
        lines = text.strip().split('\n')
        if not lines:
            logging.warning("No lines found for rhyme pattern")
            return ""
        
        rhyme_map = {}
        current_rhyme_id = 'A'
        
        for line in lines:
            last_word = self._get_last_word(line)
            vowel_ending = self._get_vowel_ending(last_word)
            
            if vowel_ending:
                found = False
                for rhyme_id, endings in rhyme_map.items():
                    if vowel_ending in endings:
                        found = True
                        break
                if not found:
                    rhyme_map[current_rhyme_id] = [vowel_ending]
                    current_rhyme_id = chr(ord(current_rhyme_id) + 1)
        
        pattern = []
        for line in lines:
            last_word = self._get_last_word(line)
            vowel_ending = self._get_vowel_ending(last_word)
            rhyme_id = None
            for r_id, endings in rhyme_map.items():
                if vowel_ending in endings:
                    rhyme_id = r_id
                    break
            pattern.append(rhyme_id or 'X')
        
        result = "".join(pattern)
        logging.debug(f"Rhyme pattern for text: {result}")
        return result

    def validate_rhyme_scheme(self, lyrics: List[str], rhyme_scheme: str) -> bool:
        if not lyrics or not rhyme_scheme or not isinstance(rhyme_scheme, str):
            logging.warning(f"Invalid inputs for validate_rhyme_scheme: lyrics={lyrics}, rhyme_scheme={rhyme_scheme}")
            return False
        
        generated_pattern = self.get_rhyme_pattern("\n".join(lyrics))
        
        if len(generated_pattern) != len(rhyme_scheme):
            logging.warning(f"Pattern length mismatch: generated={len(generated_pattern)}, expected={len(rhyme_scheme)}")
            return False
        
        rhyme_mapping = {}
        for gen_id, exp_id in zip(generated_pattern, rhyme_scheme.upper()):
            if gen_id == 'X':
                continue
            if gen_id not in rhyme_mapping:
                rhyme_mapping[gen_id] = exp_id
            elif rhyme_mapping[gen_id] != exp_id:
                logging.warning(f"Rhyme scheme mismatch: {gen_id} mapped to {rhyme_mapping[gen_id]}, expected {exp_id}")
                return False
        
        logging.debug(f"Rhyme scheme validated: {generated_pattern} matches {rhyme_scheme}")
        return True

    def get_rhyming_words(self, rhyme_pattern: str, genre: str = None) -> List[str]:
        if not rhyme_pattern or not isinstance(rhyme_pattern, str):
            logging.warning(f"Invalid rhyme_pattern: {rhyme_pattern}")
            return []
        
        logging.debug(f"Getting rhyming words for pattern={rhyme_pattern}, genre={genre}")
        rhyming_words = []
        
        if not self.lyrics_data:
            logging.warning("No lyrics data available")
            return []

        for item in self.lyrics_data:
            if genre and item['genre'].lower() != genre.lower():
                continue
            lines = item['text'].strip().split('\n')
            for line in lines:
                last_word = self._get_last_word(line)
                if last_word and self._get_vowel_ending(last_word) == rhyme_pattern:
                    rhyming_words.append(last_word)
        
        logging.debug(f"Found {len(rhyming_words)} rhyming words: {rhyming_words}")
        return rhyming_words