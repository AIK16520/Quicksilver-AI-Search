PRODUCT HUNT SCRAPER - AI ENABLED
==================================

SETUP:
1. Run schema_producthunt.sql in Supabase
2. Run setup_producthunt_newsletter.sql in Supabase

RUN:
- Pipeline: python pipeline/run.py
- Direct: python run_pipeline_producthunt.py

FEATURES:
✓ Fetches top products from Product Hunt API
✓ Generates embeddings (vector 1536)
✓ Generates keywords using GPT-4
✓ Stores in product_hunt_products table

SCHEMA:
- product_name, overview, description
- product_link, producthunt_link
- ai_description (combined text)
- embedding (VECTOR 1536)
- keywords (TEXT[])

REQUIREMENTS:
- OPENAI_API_KEY in .env
- SUPABASE_URL and SUPABASE_KEY in .env

