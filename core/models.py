from typing import List, Optional
from dataclasses import dataclass

@dataclass
class RawArticle:
    title: str
    url: str
    content: str
    published_date: Optional[str]  # ISO format recommended
    author: Optional[str] = None

@dataclass
class ProcessedArticle:
    raw_article: 'RawArticle'
    summary: str
    company_names: List[str]
    summary_embedding: List[float]

@dataclass
class Chunk:
    chunk_text: str
    contextualized_text: str
    chunk_level: str  # 'paragraph' or 'sentence'
    chunk_index: int
    embedding: List[float]