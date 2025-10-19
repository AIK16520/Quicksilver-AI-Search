#!/usr/bin/env python3
"""
Product Hunt Monthly Top Scraper

Fetches top products from Product Hunt using the official API
and stores them in Supabase.

Usage:
    python run_producthunt.py --limit 50 --days 30
    python run_producthunt.py --limit 100 --days 7
    python run_producthunt.py  # Uses defaults: 50 products, 30 days
"""

import sys
import os
from pathlib import Path
import argparse

# Add directories to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'pipeline' / 'parsers'))

from productHunt import ProductHuntScraper


def main():
    parser = argparse.ArgumentParser(
        description='Scrape top products from Product Hunt and store in Supabase'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=25,
        help='Number of products to fetch (default: 50)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Fetch products from last N days (default: 30)'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Product Hunt API token (or set in config)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: fetch but do not store to database'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Product Hunt Top Products Scraper")
    print("=" * 60)
    print(f"Fetching top {args.limit} products from last {args.days} days")
    if args.test:
        print("⚠️  TEST MODE: Will not store to database")
    print("=" * 60 + "\n")
    
    # Create scraper configuration
    config = {
        'limit': args.limit,
        'days_back': args.days
    }
    
    if args.token:
        config['api_token'] = args.token
    
    # Initialize scraper
    scraper = ProductHuntScraper(
        newsletter_id="producthunt-monthly",
        url="https://www.producthunt.com",
        config=config
    )
    
    if args.test:
        # Test mode: just fetch and display
        print("Fetching products (test mode)...\n")
        products = scraper.fetch_products()
        
        if products:
            print(f"\n✓ Fetched {len(products)} products\n")
            print("=" * 60)
            
            for i, product in enumerate(products[:10], 1):
                print(f"\n{i}. {product['product_name']}")
                print(f"   Votes: {product['_metadata']['votes']}")
                print(f"   Link: {product['producthunt_link']}")
            
            if len(products) > 10:
                print(f"\n... and {len(products) - 10} more products")
            
            print("\n" + "=" * 60)
            print(f"\n✓ Test successful! Found {len(products)} products")
            print("\nTo store to Supabase, run without --test flag")
        else:
            print("✗ No products fetched")
            return 1
    else:
        # Production mode: fetch and store
        stored_count = scraper.fetch_and_store()
        
        if stored_count > 0:
            print(f"\n✓ Success! Stored {stored_count} products to Supabase")
            return 0
        else:
            print("\n✗ Failed to store products")
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
