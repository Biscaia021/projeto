import html
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextRegexPatterns:
    """Centralizes pre-compiled regex patterns for Portuguese text cleaning."""

    HTML_TAG = re.compile(r"<[^>]*>")
    METADATA = re.compile(r"\[.*?\]")

    TIMESTAMP_METADATA = re.compile(
        r"^(?:publicado\s+em:\s*)?\d{2}/\d{2}/\d{4}(?:\s*[-—–|]?\s*\d{2}h\d{2})?(?:\s*às\s*\d{2}h\d{2})?(?:\s*[-—–|]\s*[A-Za-zÀ-ÿ\s]+)?\s*",
        re.IGNORECASE,
    )

    FOOTER_TRASH = re.compile(
        r"(?:leia\s+o\s+comunicado\s+completo\s+em|acesse\s+a\s+pnad\s+contínua\s+completa|dados\s+completos\s+disponíveis\s+em|veja\s+a\s+cotação\s+completa\s+em|para\s+acessar\s+o\s+relatório\s+completo,\s+visite\s+o|fonte:\s*mdic\s*—)\s*",
        re.IGNORECASE,
    )

    # 100% Safe Regex patterns using Hexadecimal notation to prevent SyntaxErrors:
    # \x27 = Single Quote (') | \x22 = Double Quote (")
    REMOVE_SLASH_QUOTES = re.compile(r"\\+([\x27\x22])")
    CLEAN_ESCAPED_QUOTES = re.compile(r"\\\x22\s+\\\x22")
    CLEAN_EMPTY_QUOTES = re.compile(r"\x22\s+\x22")

    # Punctuation and spacing rules
    SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:?!])")
    SPACE_AFTER_PUNCTUATION = re.compile(
        r"([.,;:?!])(?=[^\s\d])(?!(?:gov|br|com|net|org|html|aspx))",
        re.IGNORECASE,
    )
    HYPHEN_CORRECTION = re.compile(r"\s+([—–-])\s+")
    DUPLICATED_SPACES = re.compile(r"\s+")

    # URL fixing rules
    URL_SPACE_CORRECTION = re.compile(
        r"(\b\w+)\s*\.\s*(bcb|gov|br|com|net|org|ibge|mdic)\b", re.IGNORECASE
    )
    URL_SLASH_CORRECTION = re.compile(r"(?<=[\w])\s*/\s*(?=[\w])")


class TextCleaner:
    """Handles text cleaning and normalization for Portuguese corpus."""

    @staticmethod
    def clean_portuguese_text(text: str) -> str:
        """Cleans input text using optimized pre-compiled regular expressions.

        Preserves original capitalization for acronym legibility and ensures URL
        and quote integrity without breaking python syntax evaluation.
        """
        if not isinstance(text, str):
            return ""

        # Initial decoding and stripping tags
        text = html.unescape(text)
        text = TextRegexPatterns.HTML_TAG.sub(" ", text)
        text = TextRegexPatterns.METADATA.sub(" ", text)
        text = TextRegexPatterns.REMOVE_SLASH_QUOTES.sub(r"\1", text)

        # Document structure cleaning
        text = TextRegexPatterns.TIMESTAMP_METADATA.sub("", text.strip())
        text = TextRegexPatterns.FOOTER_TRASH.sub("", text)

        # Cleaning specific empty/escaped quotes cases (e.g., \"   \" or "   ")
        text = TextRegexPatterns.CLEAN_ESCAPED_QUOTES.sub('""', text)
        text = TextRegexPatterns.CLEAN_EMPTY_QUOTES.sub('""', text)

        # Multi-pass loop to resolve nested broken URL domains
        for _ in range(3):
            text = TextRegexPatterns.URL_SPACE_CORRECTION.sub(r"\1.\2", text)
        text = TextRegexPatterns.URL_SLASH_CORRECTION.sub("/", text)

        # Typography and punctuation spacing normalization
        text = TextRegexPatterns.SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
        text = TextRegexPatterns.SPACE_AFTER_PUNCTUATION.sub(r"\1 ", text)
        text = TextRegexPatterns.HYPHEN_CORRECTION.sub(r" \1 ", text)

        return TextRegexPatterns.DUPLICATED_SPACES.sub(" ", text).strip()


class CorpusProcessor:
    """Handles data I/O operations and dataset validation."""

    def __init__(self, cleaner: TextCleaner):
        self.cleaner = cleaner

    def process(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        print("⏳ [Processing] Cleaning punctuation and structuring corpus...")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Missing input file: {input_file}")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            news_data = json.load(f)

        cleaned_data = []
        for item in news_data:
            cleaned_text = self.cleaner.clean_portuguese_text(
                item.get("texto", "")
            )
            cleaned_title = self.cleaner.clean_portuguese_text(
                item.get("titulo", "")
            )

            # Minimum length constraints validation
            if len(cleaned_text) > 10 and len(cleaned_title) > 3:
                cleaned_data.append(
                    {
                        "id": item.get("id"),
                        "titulo": cleaned_title,
                        "texto": cleaned_text,
                        "data": item.get("data"),
                        "fonte": item.get("fonte"),
                    }
                )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Corpus successfully processed and saved to: {output_file}\n")
        return cleaned_data


class OptimizedSemanticSearchEngine:
    """Vector search engine powered by SentenceTransformers and normalized embeddings."""

    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"⚙️ [Hardware] Initializing SentenceTransformer on device: {self.device.upper()}"
        )

        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus_data: List[Dict[str, Any]] = []
        self.normalized_embeddings: np.ndarray = np.empty((0, 0))

    def index(self, data: List[Dict[str, Any]]) -> None:
        """Generates semantic embeddings and normalizes them onto a hypersphere."""
        print(
            "⏳ [Spatial Mapping] Generating geometric embeddings on Hypersphere..."
        )
        self.corpus_data = data

        # CORREÇÃO: Removido o .lower() para preservar capitalização original (siglas, nomes próprios)
        texts_to_vectorize = [
            f"{d['titulo']} {d['texto']}" for d in data
        ]

        raw_embeddings = self.model.encode(
            texts_to_vectorize, show_progress_bar=True, convert_to_numpy=True
        )

        # L2 Normalization for Cosine Similarity via dot product
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = raw_embeddings / (norms + 1e-9)
        print("✅ Geometric indexing and normalization complete.\n")

    def search(self, query: str, top_k: int = 3) -> None:
        """Executes highly optimized vector search using argpartition."""
        # CORREÇÃO: Removido o .lower() para alinhar com o padrão do modelo e do corpus
        cleaned_query = TextCleaner.clean_portuguese_text(query)

        query_embedding = self.model.encode(
            [cleaned_query], convert_to_numpy=True
        )
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        # Vectorized dot product calculation
        similarities = np.dot(
            self.normalized_embeddings, query_norm.T
        ).flatten()

        # Selection optimization using Partitioning instead of full sort
        if len(similarities) > top_k:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[
                np.argsort(similarities[top_indices])[::-1]
            ]
        else:
            top_indices = np.argsort(similarities)[::-1]

        print(f"🔎 Optimized Results for: '{query}'")
        for idx in top_indices:
            item = self.corpus_data[idx]
            print(f"[{similarities[idx]:.4f}] {item['titulo']}")
            print(f"    Text: {item['texto'][:140]}...")
            print(f"    Source: {item['fonte']} | Date: {item['data']}")
        print("-" * 60)


if __name__ == "__main__":
    RAW_FILE_PATH = "dados/noticias_brutas.json"
    CLEANED_FILE_PATH = "dados/dados_limpos.json"

    try:
        if os.path.exists(RAW_FILE_PATH):
            text_cleaner = TextCleaner()
            processor = CorpusProcessor(cleaner=text_cleaner)

            corpus_data = processor.process(RAW_FILE_PATH, CLEANED_FILE_PATH)

            search_engine = OptimizedSemanticSearchEngine()
            search_engine.index(corpus_data)

            sample_queries = [
                "mudanças na taxa de juros",
                "mercado de trabalho e desemprego",
                "inflação e preços ao consumidor",
            ]

            print("🚀 Starting High-Performance Vector Search:")
            for target_query in sample_queries:
                search_engine.search(target_query, top_k=2)
        else:
            print(
                f"⚠️ Please place the '{RAW_FILE_PATH}' file in the directory to run the pipeline."
            )

    except Exception as error:
        print(f"❌ Error executing pipeline: {error}")