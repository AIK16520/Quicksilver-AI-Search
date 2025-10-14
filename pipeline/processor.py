# processor.py

import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

from openai import OpenAI
import nltk

from models import RawArticle, ProcessedArticle, Chunk
from config import OPENAI_API_KEY

# Download punkt model for sentence splitting, if not already available
nltk.download("punkt", quiet=True)
logger = logging.getLogger("processor")
logging.basicConfig(level=logging.INFO)

# ----- OpenAI Client -----
client = OpenAI(api_key=OPENAI_API_KEY)

# ----- GPT Extraction -----
def gpt_extract_summary_and_companies(article: RawArticle) -> Optional[dict]:
    """
    Calls OpenAI GPT to extract summary + company info (single call).
    Returns dictionary per mp.txt spec.
    """
    prompt = (
        f"Extract in a single JSON object:\n"
        f"1. 2-3 sentence summary\n"
        f"2. List of company names mentioned\n"
        f"3. For each company, what they build/who they sell to.\n\n"
        f"Title: {article.title}\n"
        f"Date: {article.published_date}\n"
        f"URL: {article.url}\n"
        f"Content: {article.content}\n\n"
        f"Return in this format:\n"
        f"{{\n"
        f'  "article_summary": "...",\n'
        f'  "company_names": ["...", "..."],\n'
        f'  "companies_context": {{"Company": "what they build, who they sell to"}}\n'
        f"}}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        import json, re
        content = response.choices[0].message.content
        content = re.sub(r"^```json|^```|```$", "", content.strip()).strip()
        data = json.loads(content)
        return data
    except Exception as e:
        logger.error(f"GPT extraction failed for '{article.title}': {e}")
        return None

# ----- Embedding Generation -----
def embed_text(text: str) -> Optional[List[float]]:
    """
    Generate OpenAI embeddings with 'text-embedding-3-small' (1536-dim).
    """
    try:
        resp = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None

# ----- Chunking -----
def chunk_article(content: str) -> Tuple[List[str], List[str]]:
    """
    Splits content into paragraphs and sentences.
    """
    # Paragraph splitting (by blank line or single \n between blocks)
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    # Sentence splitting (on all content, for completeness)
    sentences = []
    for para in paragraphs:
        sentences.extend(nltk.sent_tokenize(para))
    return paragraphs, sentences

def contextualize(article: RawArticle, summary: str, chunk_text: str) -> str:
    """
    Format chunk for contextualized embedding.
    """
    return (
        f"Title: {article.title}\n"
        f"Date: {article.published_date}\n"
        f"Article Summary: {summary}\n"
        f"Content: {chunk_text}"
    )

# ----- Processor Class -----
class Processor:
    """Main processor class for article processing"""

    def __init__(self):
        logger.info("Processor initialized with OpenAI API")

    def process_article(self, raw_dict: Dict[str, Any]) -> Tuple[Optional[ProcessedArticle], List[Chunk]]:
        """Process a single article - wrapper for process_article function"""
        return process_article(raw_dict)

    def create_chunks(self, processed: ProcessedArticle) -> List[Chunk]:
        """
        Create chunks from processed article.
        Note: chunks are already created in process_article, so this just extracts them.
        This method exists for API compatibility.
        """
        # Chunks are created within process_article, return empty list
        # (chunks are returned from process_article directly)
        return []

# ----- Main Processor Function -----
def process_article(raw_dict: Dict[str, Any]) -> Tuple[Optional[ProcessedArticle], List[Chunk]]:
    """
    Full pipeline: GPT, summary embedding, chunking, chunk embeddings.
    Returns (ProcessedArticle, [Chunk, ...]).
    """
    # Convert dict to RawArticle, handle missing keys for robustness
    article = RawArticle(
        title=raw_dict.get("title") or "",
        url=raw_dict.get("url") or "",
        content=raw_dict.get("content") or "",
        published_date=raw_dict.get("published_date"),
        author=None  # Not needed/available for now
    )

    logger.info(f"Processing article: {article.title} ({article.url})")

    # --- 1. GPT summary and companies ---
    gpt_data = gpt_extract_summary_and_companies(article)
    if not gpt_data or not gpt_data.get("article_summary"):
        logger.warning(f"Skipping article (GPT failure): {article.title}")
        return None, []

    summary = gpt_data["article_summary"]
    company_names = gpt_data.get("company_names") or []

    # --- 2. Article-level (parent) embedding ---
    summary_embedding = embed_text(summary)
    if not summary_embedding:
        logger.warning(f"Skipping article (embedding failure): {article.title}")
        return None, []

    processed = ProcessedArticle(
        raw_article=article,
        summary=summary,
        company_names=company_names,
        summary_embedding=summary_embedding,
    )

    # --- 3. Chunking (paragraphs & sentences, multi-granularity) ---
    paragraphs, sentences = chunk_article(article.content)
    chunks: List[Chunk] = []

    for level, blocks in [("paragraph", paragraphs), ("sentence", sentences)]:
        for idx, text in enumerate(blocks):
            ctxt = contextualize(article, summary, text)
            emb = embed_text(ctxt)
            if emb:  # Skip chunks that fail to embed
                chunks.append(Chunk(
                    chunk_text=text,
                    contextualized_text=ctxt,
                    chunk_level=level,
                    chunk_index=idx,
                    embedding=emb,
                ))

    logger.info(
        f" Processed '{article.title}' with {len(paragraphs)} para chunks, "
        f"{len(sentences)} sent. chunks, {len(chunks)} total."
    )
    return processed, chunks

