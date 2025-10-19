# Product Hunt API Scraper - Quick Start Guide

## Why Use the API Instead of Web Scraping?

✅ **Reliable**: No 403 errors or bot detection  
✅ **Official**: Supported by Product Hunt  
✅ **Fast**: Get data in seconds  
✅ **Complete**: Access to all fields including votes, makers, hunters  
❌ **Requires API Key**: Need to register (but it's free!)

## Quick Setup (5 minutes)

### Step 1: Get Your API Key

1. Go to https://www.producthunt.com/
2. Sign up or log in
3. Visit https://api.producthunt.com/v2/docs
4. Create an application to get your API key

### Step 2: Add API Key to Environment

Edit your `.env` file and add:

```env
PRODUCT_HUNT_API_KEY=your_api_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Step 3: Run the Scraper

```bash
# Test it first (fetches 5 products, doesn't store)
python -c "from pipeline.parsers.productHuntAPI import test_api_scraper; test_api_scraper()"

# Fetch and store 50 products
python -c "from pipeline.parsers.productHuntAPI import ProductHuntAPIScraper; s = ProductHuntAPIScraper('ph', '', {'limit': 50}); s.fetch_and_store()"
```

## Usage Examples

### Basic Usage (Python)

```python
from pipeline.parsers.productHuntAPI import ProductHuntAPIScraper

# Create scraper
scraper = ProductHuntAPIScraper(
    newsletter_id="producthunt",
    url="https://www.producthunt.com",  # Not used for API
    config={
        'limit': 100  # Number of products to fetch
    }
)

# Fetch and store to Supabase
stored_count = scraper.fetch_and_store()
print(f"Stored {stored_count} products")
```

### Using ParserFactory

```python
from parsers.parserFactory import ParserFactory

parser = ParserFactory.create(
    source_type='producthunt_api',
    newsletter_id='ph-api',
    url='https://www.producthunt.com',
    config={'limit': 100}
)

stored_count = parser.fetch_and_store()
```

### Fetch Specific Number of Products

```python
scraper = ProductHuntAPIScraper('ph', '', {'limit': 200})

# Fetch up to 50 products
products = scraper.fetch_products(max_products=50)

# Store to database
count = scraper.storage.store_products_batch(products)
```

## What Data Gets Extracted?

The API scraper extracts:

| Field | Description |
|-------|-------------|
| `name` | Product name |
| `link` | Product website URL |
| `producthunt_link` | Product Hunt page URL |
| `description` | Tagline + full description |
| `scraped_at` | Timestamp |

**Plus metadata** (stored in `_metadata` field):
- Vote count
- Created date
- Featured date
- Categories/topics
- Maker names
- Hunter names
- Thumbnail image URL

## Features

✅ **Automatic Pagination**: Fetches multiple pages if needed  
✅ **Duplicate Detection**: Skips already-stored products  
✅ **Batch Storage**: Efficient database inserts  
✅ **Error Handling**: Graceful error recovery  
✅ **Detailed Logging**: See what's happening  

## Troubleshooting

### Error: "API key required"

Make sure `PRODUCT_HUNT_API_KEY` is set in your `.env` file or passed in config:

```python
config = {
    'api_key': 'your_key_here',
    'limit': 50
}
```

### Error: "Unauthorized" or "Invalid API key"

1. Check your API key is correct
2. Make sure you're using the v2 API key (not v1)
3. Check if your API key has expired

### No products returned

The API returns newest products first. If you've already scraped them, they'll be skipped. Try:
- Clearing your database table
- Checking the Product Hunt website for new products

## API Limits

- **Rate Limits**: Product Hunt API has rate limits (usually 100-500 requests/hour)
- **Per Request**: Maximum ~50 products per API call
- **Total**: The scraper automatically handles pagination

## Extending the Schema

To store the additional metadata fields, you can extend your Supabase schema:

```sql
-- Add columns for additional Product Hunt data
ALTER TABLE product_hunt_products ADD COLUMN votes INTEGER;
ALTER TABLE product_hunt_products ADD COLUMN categories TEXT[];
ALTER TABLE product_hunt_products ADD COLUMN makers TEXT[];
ALTER TABLE product_hunt_products ADD COLUMN hunters TEXT[];
ALTER TABLE product_hunt_products ADD COLUMN thumbnail_url TEXT;
ALTER TABLE product_hunt_products ADD COLUMN featured_at TIMESTAMP;
```

Then modify the `store_product` method in `storage.py` to include these fields.

## Comparison: Web Scraping vs API

| Feature | Web Scraping | API |
|---------|-------------|-----|
| Reliability | ❌ Often blocked | ✅ Always works |
| Speed | 🐌 Slow (rate limiting) | ⚡ Fast |
| Setup | ✅ No API key needed | ❌ Requires API key |
| Data Quality | ⚠️ HTML parsing issues | ✅ Clean JSON |
| Maintenance | ❌ Breaks when HTML changes | ✅ Stable API |
| Legal | ⚠️ Gray area | ✅ Official |

## Recommended: Use the API!

For production use, **always prefer the API over web scraping**:
- More reliable
- Faster
- Better data quality
- Official and legal
- Won't break when Product Hunt updates their website

The only downside is needing an API key, but it's free and takes 5 minutes to set up!

## Next Steps

1. Get your API key from Product Hunt
2. Test with `test_api_scraper()`
3. Run a small batch (10-20 products)
4. Verify data in Supabase
5. Scale up to your desired volume

## Support

- Product Hunt API Docs: https://api.producthunt.com/v2/docs
- GraphQL Playground: https://api.producthunt.com/v2/api/graphql
- Product Hunt Support: hello@producthunt.com

