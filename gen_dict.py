#!/usr/bin/env python3
"""
Taiwanese Hokkien (Tâi-lô) Dictionary Generator

This script generates a Rime dictionary file (YAML format) from a Kautian ODS spreadsheet.
It extracts words and characters with their pronunciations, converts tone marks to
numerical tone annotations, and creates a sorted dictionary with proper weights.

Example output format:
TÂI-UÂN    TAI5 UAN5    1500
台灣        TAI5 UAN5    1500
"""

import pyexcel_ods3
import datetime
import re
import unicodedata
import sys
import argparse
import os
from typing import Dict, List, Tuple, Set, Optional, Any

class DictionaryConfig:
    """Configuration settings for the dictionary generation process."""
    
    def __init__(self, 
                 input_file: str = "kautian.ods", 
                 output_file: str = "hanlo.dict.yaml",
                 freq_file: str = "",
                 use_freq_weighting: bool = False,
                 base_weight: int = 0,
                 word_weight: int = 500,
                 char_weight: int = 0,
                 example_weight: int = 100,
                 alt_pronun_weight: int = 400,
                 colloq_pronun_weight: int = 450,
                 contract_pronun_weight: int = 470,
                 dialect_weight: int = 350,
                 vocab_comp_weight: int = 300,
                 min_occurrences: int = 5):
        # File paths
        self.input_file = input_file
        self.output_file = output_file
        self.freq_file = freq_file
        self.use_freq_weighting = use_freq_weighting
        self.base_weight = base_weight
        
        # Sheet names in the ODS file
        self.char_sheet = "漢字羅馬字對應"
        self.word_sheet = "詞目"
        self.example_sheet = "例句"
        self.alt_pronun_sheet = "又唸作"
        self.colloq_pronun_sheet = "俗唸作"
        self.contract_pronun_sheet = "合音唸作"
        self.dialect_sheet = "語音差異"
        self.vocab_comp_sheet = "詞彙比較"
        
        # Weights for different entry types
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.example_weight = example_weight
        self.alt_pronun_weight = alt_pronun_weight
        self.colloq_pronun_weight = colloq_pronun_weight
        self.contract_pronun_weight = contract_pronun_weight
        self.dialect_weight = dialect_weight
        self.vocab_comp_weight = vocab_comp_weight
        self.min_occurrences = min_occurrences
        
        # Define symbols that should not be in dictionary words
        self.invalid_symbols = set('，。？！；：「」『』、─,.?!;:()[]{}…"“‘’”\'')
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'DictionaryConfig':
        """Create a configuration instance from command-line arguments."""
        return cls(
            input_file=args.input_file,
            output_file=args.output_file,
            freq_file=args.freq_file,
            use_freq_weighting=args.use_freq_weighting,
            base_weight=args.base_weight,
            word_weight=args.word_weight,
            char_weight=args.char_weight,
            example_weight=args.example_weight,
            alt_pronun_weight=args.alt_pronun_weight,
            colloq_pronun_weight=args.colloq_pronun_weight,
            contract_pronun_weight=args.contract_pronun_weight,
            dialect_weight=args.dialect_weight,
            vocab_comp_weight=args.vocab_comp_weight,
            min_occurrences=args.min_occurrences
        )

class FrequencyLoader:
    """Handles loading and processing character frequency data."""
    
    def __init__(self, config: DictionaryConfig):
        self.config = config
        self.freq_map = {}
        self.pronun_freq_map = {}
        self.max_freq = 1
        self.min_weight = 50
        self.max_weight = 100000
        
    def load_frequency_data(self) -> Dict[str, int]:
        """Load character frequency data from file."""
        if not self.config.freq_file:
            print("No frequency file specified, skipping frequency loading")
            return {}
            
        try:
            print(f"Loading frequency data from {self.config.freq_file}...")
            processor = TextProcessor()
            
            with open(self.config.freq_file, "r", encoding="utf-8") as f:
                next(f)  # Skip header row
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        char, pronun, freq = parts[1], parts[2], int(parts[3])
                        self.freq_map[char] = freq
                        
                        # Store character-pronunciation pair
                        numeric_pronun, _ = processor.convert_tones(pronun)
                        pronun_key = f"{char}:{numeric_pronun.lower()}"
                        self.pronun_freq_map[pronun_key] = freq
                        
                        self.max_freq = max(self.max_freq, freq)
            
            print(f"Loaded frequency data for {len(self.freq_map)} characters")
            print(f"Loaded {len(self.pronun_freq_map)} character-pronunciation pairs")
            print(f"Maximum frequency: {self.max_freq}")
            return self.freq_map
            
        except IOError as e:
            print(f"Warning: Could not read frequency file {self.config.freq_file}. {str(e)}")
            return {}
    
    def get_char_weight(self, char: str, pronun: str = "", base_weight: int = 0) -> int:
        """Calculate weight for a character based on its frequency."""
        if not self.freq_map and self.config.freq_file:
            self.load_frequency_data()
            
        if not self.config.use_freq_weighting or not self.freq_map:
            return base_weight
        
        if pronun:
            pronun_key = f"{char}:{pronun.lower()}"
            if pronun_key in self.pronun_freq_map:
                freq = self.pronun_freq_map[pronun_key]
                relative_freq = freq / self.max_freq
                weight_range = self.max_weight - self.min_weight
                return self.min_weight + int(weight_range * relative_freq)
                
        return self.min_weight
    
    def get_word_weight(self, word: str, base_weight: int = 500) -> int:
        """Return the base weight for words."""
        return base_weight
            
    def adjust_weight(self, entry_text: str, pronun: str = "", base_weight: int = 0) -> int:
        """Adjust weight based on entry type and frequency."""
        return self.get_char_weight(entry_text, pronun, base_weight) if len(entry_text) == 1 else base_weight

class TextProcessor:
    """Utilities for processing and converting text between formats."""
    
    @staticmethod
    def split_romanizations(roman_str: str) -> List[str]:
        """Split romanization string by both '/' and ',' separators."""
        if not roman_str:
            return []
        
        # First split by '/' then by ',' for each part
        romanizations = []
        for part in roman_str.split('/'):
            for subpart in part.split(','):
                cleaned = subpart.strip()
                if cleaned:
                    romanizations.append(cleaned)
        return romanizations
    
    @staticmethod
    def convert_tones(text: str) -> Tuple[str, bool]:
        """Convert Tâi-lô romanization with tone marks to numerical tone representation."""
        # Note: This method processes a single romanization, not multiple alternatives
        # Use split_romanizations() first to get individual romanizations
        
        is_word = ' ' in text or '-' in text
        has_neutral_tone = '--' in text
        
        # Handle neutral tone markers
        text = text.replace(' -- ', ' --').replace('-- ', '--').replace('--', ' -- ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Process each syllable
        syllables = text.split()
        result_syllables = []
        
        for i, syllable in enumerate(syllables):
            if not syllable.strip() or syllable == '--':
                continue
                
            # Check for neutral tone
            is_neutral = i > 0 and syllables[i-1] == '--'
            tone = "4" if is_neutral else "1"  # Default tone
            
            # Handle special o͘ sequence
            syllable = syllable.replace('o͘', 'oo').replace('O͘', 'OO')
            
            # Handle hyphenated syllables
            if '-' in syllable and not syllable.startswith('-') and not syllable.endswith('-'):
                parts = syllable.split('-')
                hyphen_results = []
                
                for part in parts:
                    if not part:
                        continue
                    normalized = unicodedata.normalize('NFD', part)
                    base_part = ''.join([c for c in normalized if not unicodedata.combining(c)])
                    
                    # Determine tone for this part
                    part_tone = "1"  # Default tone
                    part_lower = part.lower()
                    
                    if any(c in part_lower for c in 'âêîôû') or b'\xcc\x82'.decode() in part_lower:
                        part_tone = "5"
                    elif any(c in part_lower for c in 'áéíóúḿń') or b'\xcc\x81'.decode() in part_lower:
                        part_tone = "2"
                    elif any(c in part_lower for c in 'àèìòùǹ') or b'\xcc\x80'.decode() in part_lower:
                        part_tone = "3"
                    elif any(c in part_lower for c in 'āēīōū') or b'\xcc\x84'.decode() in part_lower:
                        part_tone = "7"
                    elif any(c in part_lower for c in 'ăĕĭŏŭ') or b'\xcc\x86'.decode() in part_lower:
                        part_tone = "6"
                    elif any(c in part_lower for c in 'űő') or b'\xcc\x8b'.decode() in part_lower:
                        part_tone = "9"

                    # Handle stop consonants
                    if part_lower.endswith(('p', 't', 'k', 'h')):
                        part_tone = "8" if b'\xcc\x8d'.decode() in part_lower else "4"
                        
                    hyphen_results.append(base_part + part_tone)
                
                if hyphen_results:
                    result_syllables.extend(hyphen_results)
                continue
            
            # Process non-hyphenated syllables
            syllable_lower = syllable.lower()
            
            if not is_neutral:
                # Detect tone by examining the characters
                if syllable_lower.endswith(('p', 't', 'k', 'h')):
                    tone = "8" if b'\xcc\x8d'.decode() in syllable_lower else "4"
                elif any(c in syllable_lower for c in 'âêîôû') or b'\xcc\x82'.decode() in syllable_lower:
                    tone = "5"
                elif any(c in syllable_lower for c in 'áéíóúḿń') or b'\xcc\x81'.decode() in syllable_lower:
                    tone = "2"
                elif any(c in syllable_lower for c in 'àèìòùǹ') or b'\xcc\x80'.decode() in syllable_lower:
                    tone = "3"
                elif any(c in syllable_lower for c in 'āēīōū') or b'\xcc\x84'.decode() in syllable_lower:
                    tone = "7"
                elif any(c in syllable_lower for c in 'ăĕĭŏŭ') or b'\xcc\x86'.decode() in syllable_lower:
                    tone = "6"
                elif any(c in syllable_lower for c in 'űő') or b'\xcc\x8b'.decode() in syllable_lower:
                    tone = "9"
            
            # Get base form without diacritical marks
            normalized = unicodedata.normalize('NFD', syllable)
            base_syllable = ''.join([c for c in normalized if not unicodedata.combining(c)])
            
            if base_syllable.strip():
                result_syllables.append(base_syllable + tone)
        
        return " ".join(result_syllables).lower(), is_word

    @staticmethod
    def count_syllables(roman: str) -> int:
        """Count syllables in romanization text."""
        # Handle neutral tone markers
        counting_roman = roman
        if '--' in counting_roman:
            counting_roman = counting_roman.replace(' -- ', ' ').replace('--', ' ')
            counting_roman = re.sub(r'\s+', ' ', counting_roman).strip()
        
        # Count syllables based on format
        if ' ' not in counting_roman and '-' in counting_roman:
            return counting_roman.count('-') + 1
        elif ' ' in counting_roman and '-' not in counting_roman:
            return counting_roman.count(' ') + 1
        elif ' ' in counting_roman and '-' in counting_roman:
            syllable_count = 0
            for word in counting_roman.split():
                syllable_count += word.count('-') + 1 if '-' in word else 1
            return syllable_count
        else:
            return 1

    @staticmethod
    def is_valid_word_entry(hanzi: str, roman: str = "", invalid_symbols: Set[str] = None) -> bool:
        """Check if a word entry is valid based on various criteria."""
        if invalid_symbols is None:
            invalid_symbols = set('，。？！；：「」『』、─,.?!;:()[]{}…"“‘’”\'')

        if not hanzi or len(hanzi) == 1:
            return False

        if not roman:
            return True
        
        roman = roman.replace('"', '').replace('"', '').replace('"', '')
        
        # Special handling for entries with neutral tone markers
        if '--' in roman:
            normalized_roman = roman.replace(' -- ', ' --').replace('-- ', '--')
            clean_roman = normalized_roman.replace('--', ' ')
            clean_roman = re.sub(r'\s+', ' ', clean_roman).strip()
            
            syllable_count = TextProcessor.count_syllables(clean_roman)
            return abs(len(hanzi) - syllable_count) <= 1
            
        # Convert to tone numbers
        rime_roman, is_multi = TextProcessor.convert_tones(roman)
        
        if not rime_roman:
            return False
            
        # Check for single syllable without spaces
        if ' ' not in rime_roman and '-' not in roman and len(hanzi) > 1:
            return False
        
        syllable_count = TextProcessor.count_syllables(roman)
        
        # Allow small mismatch for hyphenated words
        if '-' in roman and abs(len(hanzi) - syllable_count) <= 1:
            return True
        
        return len(hanzi) == syllable_count

class DataLoader:
    """Handles loading data from ODS files and preprocessing."""
    
    def __init__(self, config: DictionaryConfig):
        self.config = config
        self.data = {}
        
    def load_data(self) -> Dict:
        """Load data from the ODS spreadsheet."""
        try:
            print(f"Loading data from {self.config.input_file}...")
            self.data = pyexcel_ods3.get_data(self.config.input_file)
            
            # Verify required sheets exist
            required_sheets = [self.config.char_sheet, self.config.word_sheet]
            missing_sheets = [sheet for sheet in required_sheets if sheet not in self.data]
            if missing_sheets:
                raise KeyError(f"Required sheet(s) not found: {', '.join(missing_sheets)}")
                
            return self.data
        except IOError as e:
            print(f"Error: Could not read file {self.config.input_file}. {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    def get_sheet(self, sheet_name: str, skip_header: bool = True) -> List[List[Any]]:
        """Get data from a specific sheet."""
        if not self.data:
            self.load_data()
            
        sheet_data = self.data.get(sheet_name, [])
        return sheet_data[1:] if skip_header and sheet_data else sheet_data
    
    def print_sheet_info(self) -> None:
        """Print information about all available sheets."""
        if not self.data:
            self.load_data()
            
        sheet_names = {
            "Character sheet": self.config.char_sheet,
            "Word sheet": self.config.word_sheet,
            "Example sheet": self.config.example_sheet,
            "Alternative pronunciations sheet": self.config.alt_pronun_sheet,
            "Colloquial pronunciations sheet": self.config.colloq_pronun_sheet,
            "Contracted pronunciations sheet": self.config.contract_pronun_sheet,
            "Dialect variations sheet": self.config.dialect_sheet,
            "Vocabulary comparisons sheet": self.config.vocab_comp_sheet
        }
        
        for desc, name in sheet_names.items():
            print(f"{desc} - Total entries: {len(self.data.get(name, []))}")

class DictionaryProcessor:
    """Processes various types of dictionary entries from different source sheets."""
    
    def __init__(self, config: DictionaryConfig, data_loader: DataLoader, freq_loader: Optional[FrequencyLoader] = None):
        self.config = config
        self.data_loader = data_loader
        self.text_processor = TextProcessor()
        self.freq_loader = freq_loader
        self.seen_entries = set()
        self.entry_id_map = {}  # Maps entry IDs to (hanzi, roman, weight) tuples
        # Add new attributes for word frequency tracking
        self.collected_words = {}  # Maps (hanzi, roman) to source info
        self.example_word_pairs = []  # All word pairs from examples for frequency counting

    def count_word_occurrences_in_examples(self) -> Dict[Tuple[str, str], int]:
        """Count how many times each word appears in example sentences."""
        if not self.example_word_pairs:
            self.example_word_pairs = self.extract_words_from_examples()
        
        word_counts = {}
        
        # Count occurrences of each word-pronunciation pair
        for hanzi, roman_str in self.example_word_pairs:
            if len(hanzi) < 2:  # Only count multi-character words
                continue
                
            # Handle multiple pronunciations
            romans = self.text_processor.split_romanizations(roman_str)
            for roman in romans:
                roman = roman.strip()
                if not roman:
                    continue
                    
                rime_roman, _ = self.text_processor.convert_tones(roman)
                word_key = (hanzi, rime_roman.lower())
                word_counts[word_key] = word_counts.get(word_key, 0) + 1
        
        print(f"Counted occurrences for {len(word_counts)} unique word-pronunciation pairs")
        return word_counts

    def process_collected_words_with_frequency(self, min_occurrences: int = 5) -> List[str]:
        """Process collected words, adjusting weights based on frequency but including all words."""
        if not self.collected_words:
            return []
            
        word_counts = self.count_word_occurrences_in_examples()
        word_dict_entries = []
        high_freq_count = 0
        low_freq_count = 0
        
        for (hanzi, rime_roman), (original_weight, entry_id) in self.collected_words.items():
            word_key = (hanzi, rime_roman.lower())
            occurrence_count = word_counts.get(word_key, 0)
            
            entry_key = f"{hanzi}:{rime_roman.lower()}"
            if entry_key in self.seen_entries:
                continue
                
            self.seen_entries.add(entry_key)
            
            # Calculate weight based on occurrence frequency
            if occurrence_count >= min_occurrences:
                # High frequency words: base weight + bonus based on frequency
                frequency_bonus = min(occurrence_count * 10, 500)  # Cap bonus at 500
                adjusted_weight = original_weight + frequency_bonus
                high_freq_count += 1
            else:
                # Low frequency words: weight = 1
                adjusted_weight = 1
                low_freq_count += 1
            
            # Store for alternative pronunciations if we have entry_id
            if entry_id:
                self.entry_id_map[entry_id] = (hanzi, rime_roman, adjusted_weight)
            
            word_dict_entries.append(f"{hanzi}\t{rime_roman}\t{adjusted_weight}")
        
        print(f"Added {high_freq_count} words with >= {min_occurrences} occurrences (high weight)")
        print(f"Added {low_freq_count} words with < {min_occurrences} occurrences (weight=1)")
        return word_dict_entries

    def get_weight(self, text: str, pronun: str = "", base_weight: int = 0, is_alt_pronun: bool = False, source_id: str = None) -> int:
        """Get weight for an entry, using frequency data if available."""
        # For alternative pronunciations, use half the original weight
        if is_alt_pronun and source_id and source_id in self.entry_id_map:
            _, _, original_weight = self.entry_id_map[source_id]
            min_weight = self.freq_loader.min_weight if self.freq_loader else 0
            return max(original_weight // 2, min_weight)
                
        # Calculate weight based on frequency
        if self.freq_loader and self.config.use_freq_weighting:
            return self.freq_loader.adjust_weight(text, pronun, base_weight)
        return base_weight
    
    def process_word_entries(self) -> List[str]:
        """Process word entries from the word sheet."""
        word_dict_entries = []
        word_entries = self.data_loader.get_sheet(self.config.word_sheet)
        
        for entry in word_entries:
            if len(entry) < 3:  # Need at least ID and hanzi fields
                continue
                
            # Extract entry data
            entry_id = str(entry[0]) if entry[0] else ""
            hanzi = str(entry[2])
            roman_str = str(entry[3]) if len(entry) > 3 else ""
            
            # Skip invalid entries
            if not hanzi or any(s in hanzi for s in self.config.invalid_symbols):
                continue
            
            # Handle entries with romanization
            if roman_str:
                # Split on both '/' and ',' separators to get all pronunciations
                romans = self.text_processor.split_romanizations(roman_str)
                
                for roman in romans:
                    roman = roman.strip()
                    if not roman:
                        continue
                        
                    # Convert to tone numbers
                    rime_roman, _ = self.text_processor.convert_tones(roman)
                    
                    # For multi-character words (>= 2), collect for frequency counting
                    if len(hanzi) >= 2:
                        # Store the original weight and entry_id for frequency counting
                        original_weight = self.get_weight(hanzi, rime_roman, self.config.word_weight)
                        self.collected_words[(hanzi, rime_roman.lower())] = (original_weight, entry_id)
                        continue
                    
                    # For single characters, add directly
                    entry_key = f"{hanzi}:{rime_roman.lower()}"
                    if entry_key in self.seen_entries:
                        continue
                        
                    self.seen_entries.add(entry_key)
                    weight = self.get_weight(hanzi, rime_roman, self.config.word_weight)
                    
                    # Store for alternative pronunciations
                    if entry_id:
                        self.entry_id_map[entry_id] = (hanzi, rime_roman, weight)
                    
                    word_dict_entries.append(f"{hanzi}\t{rime_roman}\t{weight}")
            else:
                # For entries without romanization
                entry_prefix = f"{hanzi}:"
                if not any(key.startswith(entry_prefix) for key in self.seen_entries):
                    self.seen_entries.add(entry_prefix)
                    word_dict_entries.append(f"{hanzi}\t")
        
        print(f"Added {len(word_dict_entries)} single character entries directly to dictionary")
        print(f"Collected {len(self.collected_words)} multi-character words for frequency analysis")
        return word_dict_entries

    def process_character_entries(self) -> List[str]:
        """Process character entries from the character sheet."""
        char_dict_entries = []
        char_entries = self.data_loader.get_sheet(self.config.char_sheet)
        
        for entry in char_entries:
            if len(entry) < 2:
                continue
                
            # Get data
            hanzi = str(entry[0]).lstrip('-')
            roman_str = str(entry[1]).lstrip('-')
            
            # Skip invalid entries
            if not hanzi or any(s in hanzi for s in self.config.invalid_symbols) or '--' in roman_str:
                continue
            
            # Handle entries without romanization
            if not roman_str:
                entry_prefix = f"{hanzi}:"
                if not any(key.startswith(entry_prefix) for key in self.seen_entries):
                    self.seen_entries.add(entry_prefix)
                    char_dict_entries.append(f"{hanzi}\t")
                continue
            
            # Process each romanization (handle both '/' and ',' separators)
            romans = self.text_processor.split_romanizations(roman_str)
            for roman in romans:
                roman = roman.strip()
                if not roman:
                    continue
                
                rime_roman, _ = self.text_processor.convert_tones(roman)
                entry_key = f"{hanzi}:{rime_roman.lower()}"
                
                if entry_key in self.seen_entries:
                    continue
                    
                self.seen_entries.add(entry_key)
                weight = self.get_weight(hanzi, roman, self.config.char_weight)
                char_dict_entries.append(f"{hanzi}\t{rime_roman}\t{weight}")
        
        print(f"Added {len(char_dict_entries)} character entries to dictionary")
        return char_dict_entries

    def process_pronunciation_sheet(self, sheet_name: str, weight: int, is_alt_pronun: bool = False) -> List[str]:
        """Process entries from pronunciation sheets."""
        pronun_dict_entries = []
        entries = self.data_loader.get_sheet(sheet_name)
        
        if not entries:
            return []
        
        for entry in entries:
            if len(entry) < 3:
                continue
            
            try:
                source_id = str(entry[0]) if entry[0] else ""
                hanzi = str(entry[1])
                roman_str = str(entry[2])
                
                if not hanzi or not roman_str or any(s in hanzi for s in self.config.invalid_symbols):
                    continue
                    
                # Handle both '/' and ',' separators
                romans = self.text_processor.split_romanizations(roman_str)
                for roman in romans:
                    roman = roman.strip()
                    if not roman:
                        continue
                    
                    rime_roman, _ = self.text_processor.convert_tones(roman)
                    
                    # For multi-character words (>= 2), collect for frequency counting
                    if len(hanzi) >= 2:
                        word_key = (hanzi, rime_roman.lower())
                        if word_key not in self.collected_words:
                            adjusted_weight = self.get_weight(
                                hanzi, rime_roman, weight, 
                                is_alt_pronun=is_alt_pronun, source_id=source_id
                            )
                            self.collected_words[word_key] = (adjusted_weight, source_id)
                        continue
                    
                    # For single characters, add directly
                    entry_key = f"{hanzi}:{rime_roman.lower()}"
                    if entry_key in self.seen_entries:
                        continue
                        
                    self.seen_entries.add(entry_key)
                    adjusted_weight = self.get_weight(
                        hanzi, rime_roman, weight, 
                        is_alt_pronun=is_alt_pronun, source_id=source_id
                    )
                    
                    pronun_dict_entries.append(f"{hanzi}\t{rime_roman}\t{adjusted_weight}")
            except (IndexError, ValueError, TypeError) as e:
                continue
        
        return pronun_dict_entries
    
    def process_alternative_pronunciations(self) -> List[str]:
        """Process alternative pronunciation entries."""
        entries = self.process_pronunciation_sheet(
            self.config.alt_pronun_sheet, 
            self.config.alt_pronun_weight,
            is_alt_pronun=True
        )
        print(f"Added {len(entries)} alternative pronunciation entries")
        return entries
    
    def process_colloquial_pronunciations(self) -> List[str]:
        """Process colloquial pronunciation entries."""
        entries = self.process_pronunciation_sheet(
            self.config.colloq_pronun_sheet, 
            self.config.colloq_pronun_weight,
            is_alt_pronun=True
        )
        print(f"Added {len(entries)} colloquial pronunciation entries")
        return entries
    
    def process_contracted_pronunciations(self) -> List[str]:
        """Process contracted pronunciation entries."""
        entries = self.process_pronunciation_sheet(
            self.config.contract_pronun_sheet, 
            self.config.contract_pronun_weight,
            is_alt_pronun=True
        )
        print(f"Added {len(entries)} contracted pronunciation entries")
        return entries
    
    def process_dialect_sheet(self) -> List[str]:
        """Process entries from the dialect sheet."""
        dialect_dict_entries = []
        entries = self.data_loader.get_sheet(self.config.dialect_sheet, skip_header=False)
        
        if not entries or len(entries) < 2:
            return []
            
        # Extract dialect names from header
        dialect_names = [str(name) for name in entries[0][2:]] if len(entries[0]) > 2 else []
        
        # Process entries (skip header)
        for i, entry in enumerate(entries):
            if i == 0 or len(entry) < 3:
                continue
                
            hanzi = str(entry[1])
            if any(s in hanzi for s in self.config.invalid_symbols):
                continue
                
            # Process each dialect pronunciation
            for j, dialect_roman in enumerate(entry[2:]):
                if not dialect_roman:
                    continue
                    
                roman_str = str(dialect_roman)
                if not roman_str:
                    continue
                
                dialect_name = dialect_names[j] if j < len(dialect_names) else f"方言{j+1}"
                
                # Handle both '/' and ',' separators
                romans = self.text_processor.split_romanizations(roman_str)
                for roman in romans:
                    roman = roman.strip()
                    if not roman:
                        continue
                    
                    rime_roman, _ = self.text_processor.convert_tones(roman)
                    
                    # For multi-character words (>= 2), collect for frequency counting
                    if len(hanzi) >= 2:
                        word_key = (hanzi, rime_roman.lower())
                        if word_key not in self.collected_words:
                            adjusted_weight = self.get_weight(hanzi, roman, self.config.dialect_weight)
                            self.collected_words[word_key] = (adjusted_weight, None)
                        continue
                    
                    # For single characters, add directly
                    entry_key = f"{hanzi}:{rime_roman.lower()}"
                    if entry_key in self.seen_entries:
                        continue
                
                    self.seen_entries.add(entry_key)
                    adjusted_weight = self.get_weight(hanzi, roman, self.config.dialect_weight)
                    dialect_dict_entries.append(f"{hanzi}\t{rime_roman}\t{adjusted_weight}")
        
        print(f"Added {len(dialect_dict_entries)} single character dialect entries")
        return dialect_dict_entries

    def process_vocab_comparison_sheet(self) -> List[str]:
        """Process entries from the vocabulary comparison sheet."""
        vocab_dict_entries = []
        entries = self.data_loader.get_sheet(self.config.vocab_comp_sheet)
        
        if not entries:
            print("No data in vocabulary comparison sheet")
            return []
        
        for entry in entries:
            if len(entry) < 5:
                continue
            
            try:
                hanzi = str(entry[3])
                roman_str = str(entry[4])
                
                if not hanzi or not roman_str or any(s in hanzi for s in self.config.invalid_symbols):
                    continue
                
                # Handle both '/' and ',' separators
                romans = self.text_processor.split_romanizations(roman_str)
                for roman in romans:
                    roman = roman.strip()
                    if not roman:
                        continue
                    
                    rime_roman, _ = self.text_processor.convert_tones(roman)
                    
                    # For multi-character words (>= 2), collect for frequency counting
                    if len(hanzi) >= 2:
                        word_key = (hanzi, rime_roman.lower())
                        if word_key not in self.collected_words:
                            adjusted_weight = self.get_weight(hanzi, roman, self.config.vocab_comp_weight)
                            self.collected_words[word_key] = (adjusted_weight, None)
                        continue
                    
                    # For single characters, add directly
                    entry_key = f"{hanzi}:{rime_roman.lower()}"
                    if entry_key in self.seen_entries:
                        continue
                        
                    self.seen_entries.add(entry_key)
                    adjusted_weight = self.get_weight(hanzi, roman, self.config.vocab_comp_weight)
                    vocab_dict_entries.append(f"{hanzi}\t{rime_roman}\t{adjusted_weight}")
            except (IndexError, ValueError, TypeError):
                continue
        
        print(f"Added {len(vocab_dict_entries)} single character vocabulary comparison entries")
        return vocab_dict_entries
    
    def extract_words_from_examples(self) -> List[Tuple[str, str]]:
        """Extract word-romanization pairs from example sentences."""
        word_pairs = []
        example_entries = self.data_loader.get_sheet(self.config.example_sheet, skip_header=False)
        
        if not example_entries or len(example_entries) < 2:
            print("No example entries found, skipping extraction from examples")
            return []
        
        try:
            example_count = 0
            print(f"Processing {len(example_entries)} example entries...")
            
            for i, entry in enumerate(example_entries):
                if i == 0 or len(entry) < 5:
                    continue
                
                # Extract example text
                hanzi = str(entry[3])
                roman_str = str(entry[4])
                
                if not hanzi or not roman_str:
                    continue
                    
                example_count += 1
                roman = roman_str.strip()
                
                # For clean entries with direct alignment
                if not any(s in hanzi for s in self.config.invalid_symbols):
                    if len(hanzi) == self.text_processor.count_syllables(roman):
                        word_pairs.append((hanzi, roman))
                    continue
                
                # For entries with symbols
                clean_hanzi = hanzi
                clean_roman = roman
                
                for sym in self.config.invalid_symbols:
                    clean_hanzi = clean_hanzi.replace(sym, '')
                    clean_roman = clean_roman.replace(sym, ' ')

                clean_roman = re.sub(r'\s+', ' ', clean_roman).strip()
                clean_hanzi = clean_hanzi.strip()
                
                # Pre-process to handle neutral tone markers
                normalized_romaji = clean_roman.replace('--', ' ')
                normalized_romaji = re.sub(r'\s+', ' ', normalized_romaji).strip()
                
                # Skip if lengths don't match
                total_syllables = self.text_processor.count_syllables(normalized_romaji)
                if len(clean_hanzi) != total_syllables:
                    print(f"Length mismatch: hanzi={len(clean_hanzi)}, syllables={total_syllables}")
                    print(f"hanzi: {clean_hanzi}")
                    print(f"roman: {normalized_romaji}")
                    continue
                    
                # Extract words
                pos = 0
                romaji_words = normalized_romaji.split()
                extracted_pairs = []
                prev_was_neutral_marker = False
                
                for word in romaji_words:
                    if word == '--':
                        prev_was_neutral_marker = True
                        continue
                        
                    # Clean the word
                    clean_word = word
                    for p in self.config.invalid_symbols:
                        clean_word = clean_word.replace(p, '')
                
                    # Skip invalid words
                    if any(p in clean_word for p in self.config.invalid_symbols):
                        prev_was_neutral_marker = False
                        if pos < len(clean_hanzi):
                            pos += 1
                        continue
                    
                    # Handle neutral tone words
                    if prev_was_neutral_marker:
                        prev_was_neutral_marker = False
                        if '-' in clean_word and pos > 0 and pos < len(clean_hanzi):
                            hanzi_segment = clean_hanzi[pos-1:pos]
                            if len(hanzi_segment) == 1:
                                extracted_pairs.append((hanzi_segment, f"{clean_word}--"))
                        continue
                    
                    # Handle hyphen-connected words
                    if '-' in clean_word and not clean_word.startswith('--'):
                        word_syllables = clean_word.count('-') + 1
                        
                        if 2 <= word_syllables <= 5 and pos + word_syllables <= len(clean_hanzi):
                            hanzi_segment = clean_hanzi[pos:pos + word_syllables]
                            
                            if len(hanzi_segment) == word_syllables:
                                extracted_pairs.append((hanzi_segment, clean_word))
                            
                            pos += word_syllables
                    elif clean_word.startswith('--'):
                        # Skip neutral tone words
                        continue
                    elif '-' not in clean_word and pos < len(clean_hanzi):
                        # Single syllable words
                        hanzi_segment = clean_hanzi[pos:pos+1]
                        extracted_pairs.append((hanzi_segment, clean_word))
                        pos += 1
                
                # Add extracted pairs
                word_pairs.extend(extracted_pairs)
            
            print(f"Processed {example_count} examples and found {len(word_pairs)} potential words")
            
        except Exception as e:
            print(f"Error processing examples: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return word_pairs
        
    def process_example_words(self) -> None:
        """Collect word pairs extracted from examples for frequency counting."""
        word_pairs = self.extract_words_from_examples()
        
        # Collect all word-pronunciation pairs from examples for frequency counting
        for hanzi, roman_str in word_pairs:
            if not roman_str or len(hanzi) < 2:  # Only collect multi-character words
                continue
                
            if hanzi and not self.text_processor.is_valid_word_entry(hanzi, roman_str, self.config.invalid_symbols):
                continue
                
            # Handle multiple pronunciations
            romans = self.text_processor.split_romanizations(roman_str)
            for roman in romans:
                roman = roman.strip()
                if not roman:
                    continue
                    
                rime_roman, _ = self.text_processor.convert_tones(roman)
            
                if not hanzi:
                    continue
                
                # Add to collected words if not already present (don't overwrite existing entries)
                word_key = (hanzi, rime_roman.lower())
                if word_key not in self.collected_words:
                    original_weight = self.config.example_weight
                    entry_id = None  # No direct entry ID for examples
                    self.collected_words[word_key] = (original_weight, entry_id)
        
        print(f"Collected {len(self.collected_words)} total words for frequency analysis")

class DictionaryWriter:
    """Handles writing the processed dictionary entries to a YAML file."""
    
    def __init__(self, config: DictionaryConfig):
        self.config = config
    
    def create_dictionary_header(self) -> str:
        """Create the YAML header for the Rime dictionary file."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        return f"""# Rime dictionary
# encoding: utf-8
#
# Taiwanese Hokkien (Tâi-lô) dictionary generated from {os.path.basename(self.config.input_file)}
# Generation date: {today}
#

---
name: hanlo
version: "1.0.0"
sort: by_weight
use_preset_vocabulary: false
import_tables:
  - lomaji_syllable
...

"""
    
    def write_dictionary(self, entries: List[str]) -> str:
        """Write dictionary entries to file."""
        dictionary_content = self.create_dictionary_header() + "\n".join(entries)
        
        try:
            with open(self.config.output_file, "w", encoding="utf-8") as f:
                f.write(dictionary_content)
            
            print(f"\nRime dictionary has been generated: {self.config.output_file}")
            return self.config.output_file
        except IOError as e:
            print(f"Error writing dictionary file: {str(e)}")
            sys.exit(1)
    
    def print_dictionary_sample(self, num_lines: int = 15) -> None:
        """Print a sample of entries from the generated dictionary."""
        print(f"\nSample from generated dictionary:")
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                # Skip header (approximately 15 lines)
                for _ in range(15):
                    next(f, None)
                # Print sample entries
                for _ in range(num_lines):
                    line = next(f, None)
                    if line:
                        print(line.strip())
        except IOError as e:
            print(f"Error reading dictionary file: {str(e)}")

class DictionaryGenerator:
    """Main class that orchestrates the dictionary generation process."""
    
    def __init__(self, config: DictionaryConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.freq_loader = FrequencyLoader(config) if config.freq_file else None
        self.processor = DictionaryProcessor(config, self.data_loader, self.freq_loader)
        self.writer = DictionaryWriter(config)
    
    def generate(self) -> str:
        """Generate the dictionary by processing all entry types."""
        try:
            # Load data
            self.data_loader.load_data()
            self.data_loader.print_sheet_info()
            
            # Load frequency data if specified
            if self.freq_loader and self.config.freq_file:
                self.freq_loader.load_frequency_data()
            
            # Process all entry types
            entries = []
            
            # Process single character and direct entries first
            entries.extend(self.processor.process_word_entries())
            entries.extend(self.processor.process_alternative_pronunciations())
            entries.extend(self.processor.process_colloquial_pronunciations())
            entries.extend(self.processor.process_contracted_pronunciations())
            entries.extend(self.processor.process_dialect_sheet())
            entries.extend(self.processor.process_vocab_comparison_sheet())
            
            # Collect words from examples for frequency counting
            self.processor.process_example_words()
            
            # Process collected multi-character words with frequency filtering
            frequency_filtered_entries = self.processor.process_collected_words_with_frequency(
                min_occurrences=self.config.min_occurrences
            )
            entries.extend(frequency_filtered_entries)
            
            # Add single character entries last
            entries.extend(self.processor.process_character_entries())
            
            print(f"Total dictionary entries: {len(entries)}")
            
            # Write to file
            output_file = self.writer.write_dictionary(entries)
            self.writer.print_dictionary_sample()
            
            return output_file
            
        except Exception as e:
            print(f"Error generating dictionary: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a Rime dictionary for Taiwanese Hokkien (Tâi-lô) from a Kautian ODS spreadsheet"
    )
    
    # File options
    parser.add_argument("-i", "--input-file", default="kautian.ods",
                        help="Path to the input ODS file (default: kautian.ods)")
    parser.add_argument("-o", "--output-file", default="hanlo.dict.yaml",
                        help="Path to the output YAML dictionary file (default: hanlo.dict.yaml)")
    parser.add_argument("-f", "--freq-file", default="char_freq_merged.txt",
                        help="Path to character frequency file (optional)")
    
    # Weight options
    parser.add_argument("--use-freq-weighting", action="store_true", default=True,
                        help="Use character frequency data for weighting entries")
    parser.add_argument("--base-weight", type=int, default=0,
                        help="Base weight for frequency calculations (default: 0)")
    parser.add_argument("--word-weight", type=int, default=75,
                        help="Weight for standard word entries (default: 500)")
    parser.add_argument("--char-weight", type=int, default=50,
                        help="Weight for character entries (default: 50)")
    parser.add_argument("--example-weight", type=int, default=75,
                        help="Weight for entries from examples (default: 100)")
    parser.add_argument("--alt-pronun-weight", type=int, default=75,
                        help="Weight for alternative pronunciations (default: 400)")
    parser.add_argument("--colloq-pronun-weight", type=int, default=75,
                        help="Weight for colloquial pronunciations (default: 100)")
    parser.add_argument("--contract-pronun-weight", type=int, default=75,
                        help="Weight for contracted pronunciations (default: 470)")
    parser.add_argument("--dialect-weight", type=int, default=75,
                        help="Weight for dialect variations (default: 350)")
    parser.add_argument("--vocab-comp-weight", type=int, default=75,
                        help="Weight for vocabulary comparisons (default: 300)")
    
    # Frequency filtering options
    parser.add_argument("--min-occurrences", type=int, default=5,
                        help="Minimum occurrences in examples for multi-character words to be included (default: 5)")
    
    return parser

def main() -> None:
    """Main function to handle command line arguments and run the generator."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    config = DictionaryConfig.from_args(args)
    generator = DictionaryGenerator(config)
    generator.generate()

if __name__ == "__main__":
    main()

