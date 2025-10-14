
"""
Cron example (run every hour):
    0 */ * * * cd /path/to/Quicksilver && python run.py >> logs/cron.log 2>&1
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from pipeline import Pipeline
from config import supabase_client

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("run")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Quicksilver Newsletter Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--backfill',
        action='store_true',
        help='Fetch ALL articles (instead of only new ones since last scrape)'
    )

    parser.add_argument(
        '--newsletter-id',
        type=str,
        help='Process specific newsletter by UUID (otherwise processes all active)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode with verbose logging'
    )

    args = parser.parse_args()

    # Set verbose logging for test mode
    if args.test:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Running in TEST mode")

    logger.info("="*70)
    logger.info("Quicksilver Newsletter Scraper Starting")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'BACKFILL' if args.backfill else 'INCREMENTAL'}")
    logger.info("="*70)

    try:
        # Test Supabase connection
        logger.info("Testing Supabase connection...")
        supabase_client.table('newsletters').select('id').limit(1).execute()
        logger.info(" Supabase connection successful\n")

        # Initialize pipeline
        pipeline = Pipeline()

        # Process newsletters
        if args.newsletter_id:
            # Process single newsletter
            logger.info(f"Processing single newsletter: {args.newsletter_id}\n")
            stats = pipeline.process_newsletter(
                newsletter_id=args.newsletter_id,
                backfill=args.backfill
            )
            results = [stats]
        else:
            # Process all active newsletters
            logger.info("Processing all active newsletters\n")
            results = pipeline.process_all_active(backfill=args.backfill)

        # Summary
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)

        if results:
            total_fetched = sum(s['articles_fetched'] for s in results)
            total_stored = sum(s['articles_stored'] for s in results)
            total_skipped = sum(s['articles_skipped'] for s in results)
            total_errors = sum(s['errors'] for s in results)

            logger.info(f"Newsletters processed: {len(results)}")
            logger.info(f"Articles fetched: {total_fetched}")
            logger.info(f"Articles stored: {total_stored}")
            logger.info(f"Articles skipped (duplicates): {total_skipped}")
            logger.info(f"Errors: {total_errors}")

            if total_errors > 0:
                logger.warning(f"� Completed with {total_errors} errors")
                sys.exit(1)
            else:
                logger.info(" Completed successfully")
                sys.exit(0)
        else:
            logger.warning("No newsletters were processed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n\n� Interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        logger.info("="*70)
        logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
