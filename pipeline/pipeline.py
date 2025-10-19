# pipeline.py
import sys
from pathlib import Path

# Add the parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.storage import StorageManager
from parsers.parserFactory import ParserFactory
from core.models import RawArticle

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)


class Pipeline:
    """
    Main orchestrator that coordinates parsing, processing, and storage.
    """
    
    def __init__(self):
        """Initialize pipeline with storage manager"""
        self.storage = StorageManager()
        # Initialize processor if OpenAI key is available
        try:
            from processor import Processor
            self.processor = Processor()
            logger.info("✓ Processor enabled (OpenAI API available)")
        except Exception as e:
            self.processor = None
            logger.warning(f"⚠ Processor disabled: {e}")
    
    def process_newsletter(self, newsletter_id: str, backfill: bool = False) -> Dict[str, Any]:
        """
        Process articles from a newsletter.
        
        Args:
            newsletter_id: UUID of newsletter to process
            backfill: If True, fetch ALL articles. If False, only fetch new ones.
        
        Returns:
            Dictionary with stats (articles_fetched, articles_stored, etc.)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting pipeline for newsletter: {newsletter_id}")
        logger.info(f"Mode: {'BACKFILL (all articles)' if backfill else 'INCREMENTAL (new only)'}")
        logger.info(f"{'='*60}\n")
        
        stats = {
            'newsletter_id': newsletter_id,
            'mode': 'backfill' if backfill else 'incremental',
            'articles_fetched': 0,
            'articles_stored': 0,
            'articles_skipped': 0,
            'errors': 0,
            'start_time': datetime.now(),
        }
        
        try:
            # Step 1: Get newsletter metadata
            newsletter = self.storage.get_newsletter(newsletter_id)
            if not newsletter:
                logger.error(f"Newsletter {newsletter_id} not found")
                return stats
            
            logger.info(f"Newsletter: {newsletter['name']}")
            logger.info(f"Source: {newsletter['url']}")
            logger.info(f"Type: {newsletter['source_type']}")
            
            # Step 2: Determine 'since' date for incremental mode
            since = None
            if not backfill and newsletter.get('last_scraped'):
                since = datetime.fromisoformat(newsletter['last_scraped'])
                logger.info(f"Fetching articles since: {since}")
            else:
                logger.info("Fetching ALL available articles")
            
            # Step 3: Create parser and fetch articles
            logger.info("\nFetching articles...")
            
            # Build config based on source type
            config = newsletter.get('_runtime_config', {})  # Get runtime config passed from run.py
            if newsletter['source_type'].lower() == 'producthunt' and not config:
                import os
                api_token = os.getenv('PRODUCTHUNT_API_TOKEN')
                if not api_token:
                    logger.error("PRODUCTHUNT_API_TOKEN environment variable not set")
                    return stats
                
                # Default config if not provided
                config = {
                    'limit': 25,
                    'days_back': 2,  # Default: last 2 days
                    'api_token': api_token
                }
            
            parser = ParserFactory.create(
                source_type=newsletter['source_type'],
                newsletter_id=newsletter_id,
                url=newsletter['url'],
                config=config
            )
            
            # Special handling for Product Hunt
            if newsletter['source_type'].lower() == 'producthunt':
                logger.info("Product Hunt source detected - using direct storage")
                # Product Hunt stores directly, doesn't return articles
                stored_count = parser.fetch_and_store(max_products=None)
                # Update newsletter timestamp
                self.storage.update_newsletter_scraped(newsletter_id)
                stats['articles_fetched'] = stored_count
                stats['articles_stored'] = stored_count
                stats['end_time'] = datetime.now()
                logger.info(f"✓ Product Hunt processing complete - {stored_count} products stored")
                return stats
            
            articles = parser.fetch_articles(since=since)
            stats['articles_fetched'] = len(articles)
            logger.info(f"Fetched {len(articles)} articles\n")
            
            if not articles:
                logger.info("No new articles found")
                self.storage.update_newsletter_scraped(newsletter_id)
                stats['end_time'] = datetime.now()
                return stats
            
            # Step 4: Process each article
            logger.info("Processing articles...\n")
            for idx, article in enumerate(articles, 1):
                logger.info(f"[{idx}/{len(articles)}] {article.title[:60]}...")
                
                # Check if article already exists
                if self.storage.article_exists(article.url):
                    logger.info(" Already exists, skipping")
                    stats['articles_skipped'] += 1
                    continue
                
                try:
                    # Convert RawArticle to dict for storage
                    article_dict = {
                        'title': article.title,
                        'url': article.url,
                        'content': article.content,
                        'published_date': article.published_date.isoformat() if article.published_date else None,
                        'author': article.author,
                    }

                    # Process article with OpenAI if processor is available
                    processed_article = None
                    chunks = []

                    if self.processor:
                        try:
                            processed_article, chunks = self.processor.process_article(article_dict)
                            if processed_article:
                                logger.info(f"     Generated summary & {len(chunks)} chunks")
                        except Exception as e:
                            logger.warning(f"     Processing failed: {e}")

                    # Store article (with or without processing)
                    article_id = self.storage.store_article(
                        processed_article=processed_article,
                        newsletter_id=newsletter_id,
                        raw_article_dict=article_dict
                    )

                    if article_id:
                        stats['articles_stored'] += 1
                        logger.info(f"  ✓ Stored (ID: {article_id[:8]}...)")

                        # Store chunks if available
                        if chunks:
                            self.storage.store_chunks(chunks, article_id)
                    else:
                        stats['errors'] += 1
                        logger.error(f"  ✗ Failed to store")
                
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"  ✗ Error processing article: {e}")
                    continue
            
            # Step 5: Update newsletter timestamp
            self.storage.update_newsletter_scraped(newsletter_id)
            
            # Calculate stats
            stats['end_time'] = datetime.now()
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            
            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info("Pipeline Complete")
            logger.info(f"{'='*60}")
            logger.info(f"Mode: {stats['mode']}")
            logger.info(f"Articles fetched: {stats['articles_fetched']}")
            logger.info(f"Articles stored: {stats['articles_stored']}")
            logger.info(f"Articles skipped: {stats['articles_skipped']}")
            logger.info(f"Errors: {stats['errors']}")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"{'='*60}\n")
            
            return stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            stats['end_time'] = datetime.now()
            stats['errors'] += 1
            return stats
    
    def process_all_active(self, backfill: bool = False, runtime_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Process all active newsletters.
        
        Args:
            backfill: If True, fetch all articles. If False, only new ones.
            runtime_config: Optional runtime configuration to pass to parsers
        
        Returns:
            List of stats dictionaries (one per newsletter)
        """
        logger.info("\n" + "="*60)
        logger.info("Processing ALL newsletters")
        logger.info("="*60 + "\n")
        
        newsletters = self.storage.get_active_newsletters()
        
        if not newsletters:
            logger.warning("No newsletters found")
            return []
        
        logger.info(f"Found {len(newsletters)} newsletter(s)\n")
        
        all_stats = []
        
        for idx, newsletter in enumerate(newsletters, 1):
            logger.info(f"\n{'*'*60}")
            logger.info(f"Newsletter {idx}/{len(newsletters)}: {newsletter['name']}")
            logger.info(f"{'*'*60}")
            
            # Inject runtime config if provided
            if runtime_config:
                newsletter['_runtime_config'] = runtime_config
            
            stats = self.process_newsletter(
                newsletter_id=newsletter['id'],
                backfill=backfill
            )
            all_stats.append(stats)
        
        # Print overall summary
        logger.info("\n" + "="*60)
        logger.info("ALL NEWSLETTERS COMPLETE")
        logger.info("="*60)
        
        total_fetched = sum(s['articles_fetched'] for s in all_stats)
        total_stored = sum(s['articles_stored'] for s in all_stats)
        total_skipped = sum(s['articles_skipped'] for s in all_stats)
        total_errors = sum(s['errors'] for s in all_stats)
        
        logger.info(f"Newsletters processed: {len(all_stats)}")
        logger.info(f"Total articles fetched: {total_fetched}")
        logger.info(f"Total articles stored: {total_stored}")
        logger.info(f"Total articles skipped: {total_skipped}")
        logger.info(f"Total errors: {total_errors}")
        logger.info("="*60 + "\n")
        
        return all_stats