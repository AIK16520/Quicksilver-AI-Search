# storage.py

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import Client

from config import supabase_client
from models import ProcessedArticle, Chunk

logger = logging.getLogger("storage")
logging.basicConfig(level=logging.INFO)


class StorageManager:
    """
    Handles all Supabase database operations.
    Abstracts database logic from the rest of the application.
    """
    
    def __init__(self):
        """Initialize with Supabase client from config"""
        self.supabase: Client = supabase_client

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        limit: int = 20
    ) -> List[Dict]:
        """Search for similar chunks using embeddings"""
        try:
            response = self.supabase.rpc('search_chunks', {
                'query_embedding': query_embedding,
                'match_count': limit
            }).execute()

            results = []
            for row in response.data:
                if row['similarity'] >= threshold:
                    results.append({
                        'article_id': row['article_id'],
                        'similarity': row['similarity']
                    })

            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_newsletter(self, newsletter_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch newsletter metadata by ID.
        
        Args:
            newsletter_id: UUID of newsletter
        
        Returns:
            Dictionary with newsletter data or None if not found
        """
        try:
            response = self.supabase.table('newsletters').select('*').eq('id', newsletter_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            else:
                logger.warning(f"Newsletter not found: {newsletter_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get newsletter {newsletter_id}: {e}")
            return None
    
    def get_active_newsletters(self) -> List[Dict[str, Any]]:
        """
        Get all active newsletters.
        
        Returns:
            List of newsletter dictionaries
        """
        try:
            response = self.supabase.table('newsletters').select('*').execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to get newsletters: {e}")
            return []
    
    def update_newsletter_scraped(self, newsletter_id: str) -> bool:
        """
        Update newsletter's last_scraped timestamp to now.
        
        Args:
            newsletter_id: UUID of newsletter
        
        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now().isoformat()
            self.supabase.table('newsletters').update({
                'last_scraped': now
            }).eq('id', newsletter_id).execute()
            
            logger.info(f"Updated last_scraped for newsletter {newsletter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last_scraped for {newsletter_id}: {e}")
            return False
    def article_exists(self, url: str) -> bool:
        """
        Check if article with this URL already exists.
        Prevents duplicate processing.
        
        Args:
            url: Article URL
        
        Returns:
            True if exists, False otherwise
        """
        try:
            response = self.supabase.table('articles').select('id').eq('url', url).execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Failed to check article existence for {url}: {e}")
            # On error, assume it doesn't exist (safer to reprocess than skip)
            return False
    
    def store_article(
        self, 
        processed_article: Optional[ProcessedArticle],
        newsletter_id: str,
        raw_article_dict: Dict[str, Any]
    ) -> Optional[str]:
        """
        Store article to database.
        
        Args:
            processed_article: ProcessedArticle object (can be None if processor not run)
            newsletter_id: UUID of parent newsletter
            raw_article_dict: Dict with title, url, content, published_date
        
        Returns:
            Article UUID if successful, None otherwise
        """
        try:
            # Build article data
            article_data = {
                'newsletter_id': newsletter_id,
                'title': raw_article_dict.get('title', ''),
                'url': raw_article_dict.get('url', ''),
                'raw_content': raw_article_dict.get('content', ''),
                'published_date': raw_article_dict.get('published_date'),
            }
            
            # Add processed data if available
            if processed_article:
                article_data['summary'] = processed_article.summary
                article_data['company_names'] = processed_article.company_names
                article_data['summary_embedding'] = processed_article.summary_embedding
            
            # Insert article
            response = self.supabase.table('articles').insert(article_data).execute()
            
            if response.data and len(response.data) > 0:
                article_id = response.data[0]['id']
                logger.info(f"✓ Stored article: {article_data['title']} (ID: {article_id})")
                return article_id
            else:
                logger.error(f"Failed to store article: {article_data['title']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to store article {raw_article_dict.get('title')}: {e}")
            return None
    
    def store_chunks(self, chunks: List[Chunk], article_id: str) -> bool:
        """
        Store multiple chunks for an article.
        Uses batch insert for performance.
        
        Args:
            chunks: List of Chunk objects
            article_id: Parent article UUID
        
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.info("No chunks to store")
            return True
        
        try:
            # Convert chunks to dictionaries
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'article_id': article_id,
                    'chunk_text': chunk.chunk_text,
                    'contextualized_text': chunk.contextualized_text,
                    'chunk_level': chunk.chunk_level,
                    'chunk_index': chunk.chunk_index,
                    'embedding': chunk.embedding
                })
            
            # Batch insert
            response = self.supabase.table('article_chunks').insert(chunk_data).execute()
            
            logger.info(f" Stored {len(chunks)} chunks for article {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks for article {article_id}: {e}")
            return False
        