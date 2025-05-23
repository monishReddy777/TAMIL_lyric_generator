import tkinter as tk
from tkinter import ttk, messagebox
from hmm_lyric_generator import HMMLyricGenerator
import logging
import json
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class HMMLyricGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tamil HMM Lyric Generator")
        self.root.geometry("800x600")
        logging.debug("Initializing HMMLyricGeneratorGUI")

        try:
            self.generator = HMMLyricGenerator()
            self.available_genres = self.generator.get_available_genres()
            if not self.available_genres:
                raise ValueError("No genres found in the data")
            self.reference_lyrics = self._load_reference_lyrics()
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize generator: {str(e)}")
            self.root.destroy()
            return

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Genre:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.genre_var = tk.StringVar()
        self.genre_dropdown = ttk.Combobox(main_frame, textvariable=self.genre_var, width=37)
        self.genre_dropdown['values'] = self.available_genres
        self.genre_dropdown['state'] = 'readonly'
        if self.available_genres:
            self.genre_dropdown.set(self.available_genres[0])
        self.genre_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

        genre_count = len(self.available_genres)
        ttk.Label(main_frame, text=f"({genre_count} genres available)").grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)

        ttk.Label(main_frame, text="Keywords (comma-separated):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.keywords_var = tk.StringVar(value="காதல், மழை, நிலா")
        self.keywords_entry = ttk.Entry(main_frame, textvariable=self.keywords_var, width=40)
        self.keywords_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Number of Lines:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.num_lines_var = tk.StringVar(value="4")
        self.num_lines_entry = ttk.Entry(main_frame, textvariable=self.num_lines_var, width=10)
        self.num_lines_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Label(main_frame, text="Rhyme Scheme:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.rhyme_scheme_var = tk.StringVar(value="AABB")
        self.rhyme_scheme_entry = ttk.Entry(main_frame, textvariable=self.rhyme_scheme_var, width=10)
        self.rhyme_scheme_entry.grid(row=3, column=1, sticky=tk.W, pady=5)

        ttk.Label(main_frame, text="Max Line Length:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.max_length_var = tk.StringVar(value="10")
        self.max_length_entry = ttk.Entry(main_frame, textvariable=self.max_length_var, width=10)
        self.max_length_entry.grid(row=4, column=1, sticky=tk.W, pady=5)

        self.generate_button = ttk.Button(main_frame, text="Generate Lyrics", command=self.generate_lyrics)
        self.generate_button.grid(row=5, column=0, columnspan=3, pady=20)

        ttk.Label(main_frame, text="Generated Lyrics:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.output_text = tk.Text(main_frame, height=10, width=70)
        self.output_text.grid(row=7, column=0, columnspan=3, pady=5)

        ttk.Label(main_frame, text="Evaluation Metrics:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.metrics_text = tk.Text(main_frame, height=7, width=70)
        self.metrics_text.grid(row=9, column=0, columnspan=3, pady=5)

        self.copy_button = ttk.Button(main_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.grid(row=10, column=0, columnspan=3, pady=10)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.status_var.set(f"Ready - {genre_count} genres available")

    def _load_reference_lyrics(self) -> Dict[str, List[str]]:
        """Load reference lyrics from the specified JSON file."""
        file_path = Path(r"C:\Users\monis\OneDrive\Desktop\sem6\nlp\hu\processed_data\processed_lyrics.json")
        if not file_path.exists():
            logging.error(f"Reference file not found: {file_path}")
            messagebox.showerror("Error", f"Reference file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            reference_lyrics = {}
            for item in data:
                genre = item['metadata']['genre'].strip().lower()
                if genre not in reference_lyrics:
                    reference_lyrics[genre] = []
                tokenized_text = self.generator._tokenize(item['text'])  # Assuming _tokenize exists
                reference_lyrics[genre].append(tokenized_text)
            logging.debug(f"Loaded reference lyrics for genres: {list(reference_lyrics.keys())}")
            return reference_lyrics
        except Exception as e:
            logging.error(f"Failed to load reference lyrics: {str(e)}")
            messagebox.showerror("Error", f"Failed to load reference lyrics: {str(e)}")
            return {}

    def validate_inputs(self) -> bool:
        """Validate all input fields."""
        logging.debug("Validating inputs")
        try:
            genre = self.genre_var.get().strip()
            if not genre:
                messagebox.showerror("Error", "Please select a genre")
                logging.error("Validation failed: No genre selected")
                return False

            keywords = [k.strip() for k in self.keywords_var.get().split(",") if k.strip()]
            if not keywords:
                messagebox.showerror("Error", "Please enter at least one keyword")
                logging.error("Validation failed: No valid keywords")
                return False

            num_lines = int(self.num_lines_var.get())
            if not 1 <= num_lines <= 20:
                messagebox.showerror("Error", "Number of lines must be between 1 and 20")
                logging.error(f"Validation failed: Invalid num_lines {num_lines}")
                return False

            rhyme_scheme = self.rhyme_scheme_var.get().strip()
            if not rhyme_scheme:
                messagebox.showerror("Error", "Rhyme scheme cannot be empty")
                logging.error("Validation failed: Empty rhyme scheme")
                return False
            if len(rhyme_scheme) != num_lines:
                messagebox.showerror("Error", f"Rhyme scheme must be {num_lines} characters long")
                logging.error(f"Validation failed: Rhyme scheme length {len(rhyme_scheme)} != {num_lines}")
                return False
            if not all(c.isalpha() for c in rhyme_scheme):
                messagebox.showerror("Error", "Rhyme scheme must contain only letters")
                logging.error(f"Validation failed: Invalid rhyme scheme characters in {rhyme_scheme}")
                return False

            max_length = int(self.max_length_var.get())
            if not 3 <= max_length <= 15:
                messagebox.showerror("Error", "Maximum line length must be between 3 and 15")
                logging.error(f"Validation failed: Invalid max_length {max_length}")
                return False

            logging.debug(f"Inputs validated: genre={genre}, keywords={keywords}, num_lines={num_lines}, rhyme_scheme={rhyme_scheme}, max_length={max_length}")
            return True
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for numeric fields")
            logging.error(f"Validation failed: {str(e)}")
            return False

    def generate_lyrics(self):
        """Generate lyrics based on input parameters and display metrics."""
        if not self.validate_inputs():
            return

        try:
            genre = self.genre_var.get().strip()
            keywords = [k.strip() for k in self.keywords_var.get().split(",") if k.strip()]
            num_lines = int(self.num_lines_var.get())
            rhyme_scheme = self.rhyme_scheme_var.get().strip().upper()
            max_length = int(self.max_length_var.get())

            logging.debug(f"Calling generate_lyrics with genre={genre}, keywords={keywords}, num_lines={num_lines}, rhyme_scheme={rhyme_scheme}, max_length={max_length}")
            self.status_var.set("Generating lyrics with HMM...")
            self.root.update()

            # Use reference lyrics based on selected genre
            references = self.reference_lyrics.get(genre, [])
            if not references:
                logging.warning(f"No reference lyrics found for genre: {genre}")
                messagebox.showwarning("Warning", f"No reference lyrics found for genre: {genre}")

            lyrics, metrics = self.generator.generate_lyrics(
                genre=genre,
                keywords=keywords,
                num_lines=num_lines,
                rhyme_scheme=rhyme_scheme,
                max_line_length=max_length,
                references=references
            )

            self.output_text.delete(1.0, tk.END)
            for idx, line in enumerate(lyrics, 1):
                self.output_text.insert(tk.END, f"{idx}. {line}\n")

            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Evaluation Metrics:\n")
            self.metrics_text.insert(tk.END, f"POS Sequence Validity: {metrics['pos_validity']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Rhyme Scheme Accuracy: {metrics['rhyme_accuracy']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Rhyme Consistency: {metrics['rhyme_consistency']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Keyword Inclusion: {metrics['keyword_inclusion']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Genre Consistency: {metrics['genre_consistency']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Unique Word Ratio: {metrics['unique_word_ratio']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"Syllable Consistency: {metrics['syllable_consistency']:.2f}%\n")
            self.metrics_text.insert(tk.END, f"BLEU: {metrics.get('bleu', 0.0):.2f}\n")
            self.metrics_text.insert(tk.END, f"Perplexity: {metrics.get('perplexity', float('inf')):.2f}\n")

            self.status_var.set("Generation complete")
            logging.debug(f"Lyrics generated: {lyrics}, Metrics: {metrics}")
        except Exception as e:
            logging.error(f"Failed to generate lyrics: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate lyrics: {str(e)}")
            self.status_var.set("Generation failed")

    def copy_to_clipboard(self):
        """Copy generated lyrics to clipboard."""
        try:
            lyrics = self.output_text.get(1.0, tk.END).strip()
            if not lyrics:
                messagebox.showwarning("Warning", "No lyrics to copy")
                logging.warning("No lyrics to copy")
                return

            self.root.clipboard_clear()
            self.root.clipboard_append(lyrics)
            self.status_var.set("Lyrics copied to clipboard")
            logging.debug("Lyrics copied to clipboard")
        except Exception as e:
            logging.error(f"Failed to copy to clipboard: {str(e)}")
            messagebox.showerror("Error", f"Failed to copy to clipboard: {str(e)}")
            self.status_var.set("Copy failed")

def main():
    root = tk.Tk()
    app = HMMLyricGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()