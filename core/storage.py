# storage.py

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import Client

from core.config import supabase_client
from core.models import ProcessedArticle, Chunk

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
        Get all newsletters (no is_active filter since column doesn't exist).
        
        Returns:
            List of newsletter dictionaries
        """
        try:
            response = self.supabase.table('newsletters').select('*').execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to get newsletters: {e}")
            return []
    
    def get_all_newsletters(self) -> List[Dict[str, Any]]:
        """Alias for get_active_newsletters (for backwards compatibility)"""
        return self.get_active_newsletters()
    
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

    def product_hunt_exists(self, producthunt_link: str) -> bool:
        """
        Check if product with this Product Hunt URL already exists.
        Prevents duplicate processing.

        Args:
            producthunt_link: Product Hunt URL

        Returns:
            True if exists, False otherwise
        """
        try:
            response = self.supabase.table('product_hunt_products').select('id').eq('producthunt_link', producthunt_link).execute()
            return len(response.data) > 0

        except Exception as e:
            logger.error(f"Failed to check product existence for {producthunt_link}: {e}")
            # On error, assume it doesn't exist (safer to reprocess than skip)
            return False

    def store_product_hunt(self, product_data: Dict[str, Any]) -> Optional[str]:
        """
        Store Product Hunt product to database with AI features.

        Args:
            product_data: Dict with product_name, producthunt_link, overview, description, 
                         product_link, ai_description, embedding, keywords, scraped_at

        Returns:
            Product UUID if successful, None otherwise
        """
        try:
            # Prepare the data for insertion
            insert_data = {
                'product_name': product_data.get('product_name', ''),
                'producthunt_link': product_data.get('producthunt_link'),
                'overview': product_data.get('overview'),
                'description': product_data.get('description'),
                'product_link': product_data.get('product_link'),
                'ai_description': product_data.get('ai_description'),
                'scraped_at': product_data.get('scraped_at', datetime.now().isoformat())
            }
            
            # Add embedding if available
            if 'embedding' in product_data and product_data['embedding']:
                insert_data['embedding'] = product_data['embedding']
            
            # Add keyword embedding if available
            if 'keyword_embedding' in product_data and product_data['keyword_embedding']:
                insert_data['keyword_embedding'] = product_data['keyword_embedding']
            
            # Add keywords if available
            if 'keywords' in product_data and product_data['keywords']:
                insert_data['keywords'] = product_data['keywords']

            # Add business model if available
            if 'Business' in product_data and product_data['Business']:
                insert_data['Business'] = product_data['Business']

            # Add technology if available
            if 'Tech' in product_data and product_data['Tech']:
                insert_data['Tech'] = product_data['Tech']

            # Add moat if available
            if 'Moat' in product_data and product_data['Moat']:
                insert_data['Moat'] = product_data['Moat']

            # Add moat embedding if available
            if 'moat_embedding' in product_data and product_data['moat_embedding']:
                insert_data['moat_embedding'] = product_data['moat_embedding']

            # Use upsert to handle duplicates (update if exists, insert if not)
            response = self.supabase.table('product_hunt_products').upsert(
                insert_data,
                on_conflict='producthunt_link'
            ).execute()

            if response.data and len(response.data) > 0:
                product_id = response.data[0]['id']
                logger.info(f"✓ Stored product: {product_data['product_name']} (ID: {product_id})")
                return product_id
            else:
                logger.error(f"Failed to store product: {product_data.get('product_name')}")
                return None

        except Exception as e:
            logger.error(f"Failed to store product {product_data.get('product_name')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_products_batch(self, products: List[Dict[str, Any]]) -> int:
        """
        Alias for store_product_hunt_batch for backwards compatibility.
        """
        return self.store_product_hunt_batch(products)

    def store_product_hunt_batch(self, products: List[Dict[str, Any]]) -> int:
        """
        Store multiple Product Hunt products in batch with AI features.
        Uses upsert to handle duplicates efficiently.

        Args:
            products: List of product dictionaries

        Returns:
            Number of products successfully stored
        """
        if not products:
            return 0

        try:
            # Prepare all products for batch insert
            batch_data = []
            seen_links = set()
            
            for product in products:
                # Remove metadata field if present (not stored in DB)
                product_clean = {k: v for k, v in product.items() if k != '_metadata'}
                
                # Skip duplicates within the batch
                link = product_clean.get('producthunt_link')
                if link in seen_links:
                    logger.warning(f"Skipping duplicate in batch: {link}")
                    continue
                seen_links.add(link)
                
                insert_data = {
                    'product_name': product_clean.get('product_name', ''),
                    'producthunt_link': link,
                    'overview': product_clean.get('overview'),
                    'description': product_clean.get('description'),
                    'product_link': product_clean.get('product_link'),
                    'ai_description': product_clean.get('ai_description'),
                    'scraped_at': product_clean.get('scraped_at', datetime.now().isoformat())
                }
                
                # Add embedding if available
                if 'embedding' in product_clean and product_clean['embedding']:
                    insert_data['embedding'] = product_clean['embedding']
                
                # Add keyword embedding if available
                if 'keyword_embedding' in product_clean and product_clean['keyword_embedding']:
                    insert_data['keyword_embedding'] = product_clean['keyword_embedding']
                
                # Add keywords if available
                if 'keywords' in product_clean and product_clean['keywords']:
                    insert_data['keywords'] = product_clean['keywords']
                
                batch_data.append(insert_data)

            # Batch upsert
            response = self.supabase.table('product_hunt_products').upsert(
                batch_data,
                on_conflict='producthunt_link'
            ).execute()

            if response.data:
                stored_count = len(response.data)
                logger.info(f"✓ Batch stored {stored_count} products")
                return stored_count
            else:
                logger.error("Failed to batch store products")
                return 0

        except Exception as e:
            logger.error(f"Failed to batch store products: {e}")
            import traceback
            traceback.print_exc()
            return 0

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
        