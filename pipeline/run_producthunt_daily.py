#!/usr/bin/env python3
"""
Product Hunt Daily Top Products Pipeline

Fetches and stores today's top products from Product Hunt.
Perfect for daily cron jobs or scheduled tasks.

Usage:
    python pipeline/run_producthunt_daily.py
    python pipeline/run_producthunt_daily.py --limit 50
    python pipeline/run_producthunt_daily.py --test

Cron example (run daily at 9 AM):
    0 9 * * * cd /path/to/Quicksilver && python pipeline/run_producthunt_daily.py >> logs/producthunt.log 2>&1
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'pipeline' / 'parsers'))

from productHunt import ProductHuntScraper
from core.config import supabase_client
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"producthunt_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("producthunt_daily")


def generate_keyword_embedding(product_name, keywords, overview):
    """Generate embedding from product name + keywords + overview"""
    if not OPENAI_AVAILABLE or not openai_client:
        logger.error("OpenAI not available")
        return None
    
    # Combine for a focused search embedding
    if keywords:
        keywords_text = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    else:
        keywords_text = ""
    
    text_parts = [
        product_name,
        keywords_text,
        overview[:100] if overview else ""
    ]
    keyword_text = " | ".join([p for p in text_parts if p])
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=keyword_text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def backfill_keyword_embeddings():
    """Update existing products without keyword embeddings"""
    logger.info("\nFetching products from database...")
    
    response = supabase_client.table('product_hunt_products').select(
        'id, product_name, keywords, overview, keyword_embedding'
    ).execute()
    
    products = response.data
    logger.info(f"Found {len(products)} total products")
    
    # Filter products that need keyword embeddings
    products_to_update = [p for p in products if not p.get('keyword_embedding')]
    logger.info(f"{len(products_to_update)} products need keyword embeddings\n")
    
    if not products_to_update:
        logger.info("✓ All products already have keyword embeddings!")
        return 0
    
    # Process products
    updated_count = 0
    failed_count = 0
    
    for i, product in enumerate(products_to_update, 1):
        logger.info(f"[{i}/{len(products_to_update)}] {product['product_name'][:50]}...")
        
        # Generate keyword embedding
        keyword_embedding = generate_keyword_embedding(
            product['product_name'],
            product.get('keywords', []),
            product.get('overview', '')
        )
        
        if keyword_embedding:
            try:
                supabase_client.table('product_hunt_products').update({
                    'keyword_embedding': keyword_embedding
                }).eq('id', product['id']).execute()
                
                updated_count += 1
                logger.info(f"  ✓ Updated")
            except Exception as e:
                logger.error(f"  ✗ Failed to update: {e}")
                failed_count += 1
        else:
            logger.warning(f"  ✗ Failed to generate embedding")
            failed_count += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Updated: {updated_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("=" * 70)
    
    return 0 if failed_count == 0 else 1


def main():
    """Main execution function for daily Product Hunt scraping"""
    parser = argparse.ArgumentParser(
        description="Fetch today's top products from Product Hunt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Number of top products to fetch (default: 50)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Fetch products from last N days (default: 30 to get all recent top products)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode - fetch but do not store to database'
    )
    
    parser.add_argument(
        '--backfill-embeddings',
        action='store_true',
        help='Update existing products without keyword embeddings'
    )
    
    args = parser.parse_args()
    
    # Start
    logger.info("=" * 70)
    logger.info("Product Hunt Daily Top Products Pipeline")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Handle backfill mode
    if args.backfill_embeddings:
        logger.info("Mode: BACKFILL KEYWORD EMBEDDINGS")
        logger.info("=" * 70)
        return backfill_keyword_embeddings()
    
    logger.info(f"Fetching top {args.limit} products from last {args.days} day(s)")
    if args.test:
        logger.info("Mode: TEST (no database writes)")
    logger.info("=" * 70)
    
    try:
        # Create scraper configuration
        api_token = os.getenv('PRODUCTHUNT_API_TOKEN')
        if not api_token:
            logger.error("PRODUCTHUNT_API_TOKEN environment variable not set")
            return 1
        
        config = {
            'limit': args.limit,
            'days_back': args.days,
            'api_token': api_token
        }
        
        # Initialize scraper
        scraper = ProductHuntScraper(
            newsletter_id="producthunt-daily",
            url="https://www.producthunt.com",
            config=config
        )
        
        if args.test:
            # Test mode: fetch only
            logger.info("\nFetching products (test mode - not storing)...\n")
            products = scraper.fetch_products(max_products=args.limit)
            
            if products:
                logger.info(f"\n✓ Successfully fetched {len(products)} products")
                logger.info("\nTop 5 products:")
                for i, product in enumerate(products[:5], 1):
                    votes = product.get('_metadata', {}).get('votes', 0)
                    logger.info(f"  {i}. {product['product_name']} ({votes} votes)")
                
                logger.info("\n" + "=" * 70)
                logger.info("TEST SUCCESSFUL")
                logger.info("To store these products, run without --test flag")
                logger.info("=" * 70)
                return 0
            else:
                logger.warning("No products fetched")
                return 1
        else:
            # Production mode: fetch and store
            logger.info("\nFetching and storing products...\n")
            stored_count = scraper.fetch_and_store(max_products=args.limit)
            
            if stored_count > 0:
                logger.info("\n" + "=" * 70)
                logger.info("PIPELINE COMPLETE")
                logger.info("=" * 70)
                logger.info(f"Products stored: {stored_count}")
                logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
                logger.info("Status: SUCCESS")
                logger.info("=" * 70)
                return 0
            else:
                logger.warning("\n" + "=" * 70)
                logger.warning("PIPELINE FAILED")
                logger.warning("=" * 70)
                logger.warning("No products were stored")
                logger.warning("=" * 70)
                return 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        logger.info(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

