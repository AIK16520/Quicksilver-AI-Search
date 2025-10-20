"""
Product Hunt API Scraper
Fetches monthly top products from Product Hunt and stores in Supabase
"""

import sys
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import requests
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.storage import StorageManager

# OpenAI for embeddings and keyword generation
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    OPENAI_AVAILABLE = True
except Exception as e:
    OPENAI_AVAILABLE = False
    openai_client = None


class ProductHuntScraper:
    """
    Product Hunt scraper using the official GraphQL API.
    Fetches monthly top products and stores in Supabase.
    """
    
    def __init__(self, newsletter_id: str, url: str, config: dict):
        """
        Initialize Product Hunt API scraper.
        
        Args:
            newsletter_id: Identifier for this scraper instance
            url: Not used for API, kept for consistency
            config: Configuration dict with:
                - api_token: Product Hunt API token (required)
                - limit: Number of products to fetch (default: 25)
                - days_back: How many days back to fetch (default: 30)
        """
        self.newsletter_id = newsletter_id
        self.url = url
        self.config = config
        self.api_token = config.get('api_token')
        if not self.api_token:
            raise ValueError("PRODUCTHUNT_API_TOKEN is required but not provided")
        self.limit = config.get('limit', 25)
        self.days_back = config.get('days_back', 30)
        self.api_endpoint = "https://api.producthunt.com/v2/api/graphql"
        self.setup_logging()
        self.storage = StorageManager()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_posts(self, after_cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch posts using GraphQL API sorted by votes.
        
        Args:
            after_cursor: Pagination cursor for fetching next page
            
        Returns:
            API response with posts data
        """
        # GraphQL query to fetch top posts
        query = """
        query Posts($first: Int, $after: String, $order: PostsOrder) {
          posts(first: $first, after: $after, order: $order) {
            edges {
              cursor
              node {
                id
                name
                tagline
                description
                url
                votesCount
                commentsCount
                createdAt
                featuredAt
                website
                thumbnail {
                  url
                }
                media {
                  url
                  type
                }
                topics {
                  edges {
                    node {
                      name
                      id
                    }
                  }
                }
                makers {
                  id
                  name
                  username
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """
        
        variables = {
            "first": min(self.limit, 20),  # API typically limits to 20 per request
            "after": after_cursor,
            "order": "VOTES"  # Sort by votes to get top products
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Retry logic with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={"query": query, "variables": variables},
                    headers=headers,
                    timeout=30
                )

                # Log rate limit info from headers (available in all responses)
                rate_limit_limit = response.headers.get('X-Rate-Limit-Limit', 'unknown')
                rate_limit_remaining = response.headers.get('X-Rate-Limit-Remaining', 'unknown')
                rate_limit_reset = response.headers.get('X-Rate-Limit-Reset', 'unknown')

                self.logger.info(f"Rate limit status - Limit: {rate_limit_limit}, Remaining: {rate_limit_remaining}, Reset in: {rate_limit_reset}s")

                # Warn if running low on quota
                if rate_limit_remaining != 'unknown':
                    try:
                        remaining = int(rate_limit_remaining)
                        limit = int(rate_limit_limit) if rate_limit_limit != 'unknown' else 6250

                        # Warn if below 20% remaining
                        if remaining < (limit * 0.2):
                            self.logger.warning(f"⚠️ Low rate limit quota: {remaining}/{limit} remaining")

                        # If very low, add delay to avoid hitting limit
                        if remaining < 100:
                            self.logger.warning(f"Very low quota ({remaining}), adding 2s delay...")
                            time.sleep(2)
                    except ValueError:
                        pass

                # Handle rate limiting (429)
                if response.status_code == 429:
                    # Try to get the reset time from headers
                    reset_seconds = int(rate_limit_reset) if rate_limit_reset != 'unknown' else base_delay * (2 ** attempt)

                    if attempt < max_retries - 1:
                        # Use the reset time from API or exponential backoff
                        delay = min(reset_seconds, 60)  # Cap at 60 seconds
                        self.logger.warning(f"Rate limited (429). Waiting {delay} seconds until reset... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Rate limited - max retries exceeded. Reset in {reset_seconds}s")
                        return {}

                response.raise_for_status()
                data = response.json()

                # Check for GraphQL errors
                if 'errors' in data:
                    self.logger.error(f"GraphQL errors: {data['errors']}")
                    return {}

                return data

            except requests.exceptions.HTTPError as e:
                # Extract rate limit info even from error responses
                if hasattr(e, 'response') and e.response is not None:
                    rate_limit_reset = e.response.headers.get('X-Rate-Limit-Reset', 'unknown')

                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        reset_seconds = int(rate_limit_reset) if rate_limit_reset != 'unknown' else base_delay * (2 ** attempt)
                        delay = min(reset_seconds, 60)  # Cap at 60 seconds
                        self.logger.warning(f"Rate limited (HTTPError). Waiting {delay} seconds until reset...")
                        time.sleep(delay)
                        continue

                self.logger.error(f"API request failed: {e}")
                import traceback
                traceback.print_exc()
                return {}
            except Exception as e:
                self.logger.error(f"API request failed: {e}")
                import traceback
                traceback.print_exc()
                return {}

        return {}
    
    def parse_post(self, post_data: Dict, generate_embeddings: bool = True) -> Dict[str, Any]:
        """
        Parse a post from API response into our storage format.
        
        Args:
            post_data: Post node from GraphQL response
            generate_embeddings: Whether to generate AI embeddings (default True)
            
        Returns:
            Product dictionary ready for storage
        """
        # Extract topics/categories
        topics = post_data.get('topics', {}).get('edges', [])
        topic_names = [topic['node']['name'] for topic in topics]
        
        # Extract makers
        makers = post_data.get('makers', [])
        maker_names = [maker.get('name', '') for maker in makers if maker]
        
        # Build overview from tagline
        overview = post_data.get('tagline', '')
        
        # Build full description
        description = post_data.get('description', '')
        
        # Create a comprehensive description for AI processing
        ai_description_parts = [
            f"Product: {post_data.get('name', '')}",
            f"Overview: {overview}",
            f"Description: {description}",
            f"Categories: {', '.join(topic_names)}" if topic_names else "",
            f"Made by: {', '.join(maker_names)}" if maker_names else "",
            f"Website: {post_data.get('website', '')}"
        ]
        ai_description = " | ".join([part for part in ai_description_parts if part])
        
        # Extract business model and technology information
        business_model = self.extract_business_model({
            'product_name': post_data.get('name', 'Untitled'),
            'overview': overview,
            'description': description
        })

        technology = self.extract_technology({
            'product_name': post_data.get('name', 'Untitled'),
            'overview': overview,
            'description': description
        })

        # Generate competitive moat analysis
        moat = self.generate_moat({
            'product_name': post_data.get('name', 'Untitled'),
            'overview': overview,
            'description': description,
            'Business': business_model,
            'Tech': technology
        })

        # Generate moat embedding if moat is available
        moat_embedding = None
        if moat and OPENAI_AVAILABLE and openai_client:
            try:
                moat_embedding = self.generate_embedding(moat)
                self.logger.info(f"Generated moat embedding for {post_data.get('name', 'Untitled')}")
            except Exception as e:
                self.logger.warning(f"Failed to generate moat embedding: {e}")

        product = {
            'product_name': post_data.get('name', 'Untitled'),
            'producthunt_link': post_data.get('url', ''),
            'overview': overview,
            'description': description,
            'product_link': post_data.get('website', ''),
            'ai_description': ai_description,
            'scraped_at': datetime.now().isoformat(),
            # Business model and technology classification
            'Business': business_model,
            'Tech': technology,
            'Moat': moat,
            'moat_embedding': moat_embedding,
            # Metadata for reference
            '_metadata': {
                'votes': post_data.get('votesCount', 0),
                'comments': post_data.get('commentsCount', 0),
                'created_at': post_data.get('createdAt'),
                'featured_at': post_data.get('featuredAt'),
                'topics': topic_names,
                'makers': maker_names,
                'thumbnail': post_data.get('thumbnail', {}).get('url', ''),
                'media': [m.get('url', '') for m in post_data.get('media', [])]
            }
        }
        
        # Generate embeddings only if requested and OpenAI is available
        if generate_embeddings and OPENAI_AVAILABLE and openai_client:
            try:
                self.logger.info(f"Generating AI features for: {product['product_name'][:40]}...")
                
                # Generate full description embedding
                embedding = self.generate_embedding(ai_description)
                if embedding:
                    product['embedding'] = embedding
                
                # Generate keywords
                keywords = self.generate_keywords(product)
                if keywords:
                    product['keywords'] = keywords
                
                # Generate keyword embedding (name + keywords + overview for better search)
                keyword_embedding = self.generate_keyword_embedding(
                    product['product_name'],
                    keywords,
                    product['overview']
                )
                if keyword_embedding:
                    product['keyword_embedding'] = keyword_embedding
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate AI features: {e}")
        else:
            if not OPENAI_AVAILABLE:
                self.logger.debug("OpenAI not available - skipping embeddings")
            else:
                self.logger.debug(f"Product already has embeddings - skipping: {product['product_name'][:40]}")
        
        return product
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if not OPENAI_AVAILABLE or not openai_client:
            return None
        
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_keywords(self, product: Dict) -> Optional[List[str]]:
        """Generate search keywords using OpenAI"""
        if not OPENAI_AVAILABLE or not openai_client:
            return None

        try:
            prompt = f"""Extract 5-10 relevant search keywords for this product.
Return ONLY a comma-separated list of keywords, no explanations.

Product Name: {product['product_name']}
Overview: {product['overview']}
Description: {product['description'][:200]}

Keywords:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(',')]
            return keywords[:10]  # Limit to 10

        except Exception as e:
            self.logger.error(f"Failed to generate keywords: {e}")
            return None

    def extract_business_model(self, product: Dict) -> Optional[str]:
        """Extract business model from product description"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._extract_business_model_simple(product)

        try:
            prompt = f"""Analyze this product and determine its primary business model.
Return ONLY the business model type, no explanations.

Product Name: {product['product_name']}
Overview: {product['overview']}
Description: {product['description'][:300]}



Business Model:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            business_model = response.choices[0].message.content.strip()
            return business_model if business_model else None

        except Exception as e:
            self.logger.error(f"Failed to extract business model: {e}")
            return self._extract_business_model_simple(product)

    def extract_technology(self, product: Dict) -> Optional[str]:
        """Extract primary technology from product description"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._extract_technology_simple(product)

        try:
            prompt = f"""Analyze this product and determine its primary technology focus.
Return ONLY the technology type, no explanations.

Product Name: {product['product_name']}
Overview: {product['overview']}
Description: {product['description'][:300]}



Technology:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            technology = response.choices[0].message.content.strip()
            return technology if technology else None

        except Exception as e:
            self.logger.error(f"Failed to extract technology: {e}")
            return self._extract_technology_simple(product)

    def _extract_business_model_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based business model extraction"""
        text = f"{product['product_name']} {product['overview']} {product.get('description', '')}".lower()

        if any(word in text for word in ['saas', 'subscription', 'monthly', 'platform']):
            return 'SaaS Platform'
        elif any(word in text for word in ['enterprise', 'licensing', 'on-premise']):
            return 'Enterprise Software'
        elif any(word in text for word in ['api', 'data service', 'data provider']):
            return 'Data-as-a-Service'
        elif any(word in text for word in ['ai', 'machine learning', 'automation']):
            return 'AI-Powered Analytics'
        elif any(word in text for word in ['consulting', 'professional services']):
            return 'Consulting Services'
        elif any(word in text for word in ['open source', 'free']):
            return 'Open Source'
        else:
            return 'Other'

    def _extract_technology_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based technology extraction"""
        text = f"{product['product_name']} {product['overview']} {product.get('description', '')}".lower()

        if any(word in text for word in ['ai', 'artificial intelligence', 'machine learning', 'ml']):
            return 'AI/ML'
        elif any(word in text for word in ['data', 'analytics', 'dashboard', 'reporting']):
            return 'Data Analytics'
        elif any(word in text for word in ['automation', 'workflow', 'process']):
            return 'Automation'
        elif any(word in text for word in ['blockchain', 'crypto', 'web3']):
            return 'Blockchain'
        elif any(word in text for word in ['cloud', 'saas']):
            return 'Cloud/SaaS'
        elif any(word in text for word in ['mobile', 'ios', 'android']):
            return 'Mobile'
        else:
            return 'Other'

    def generate_moat(self, product: Dict) -> Optional[str]:
        """Generate competitive moat analysis for the product"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._generate_moat_simple(product)

        try:
            prompt = f"""Analyze this financial analytics product and identify its competitive moat/advantage.

Product Name: {product['product_name']}
Overview: {product['overview']}
Description: {product['description'][:300]}
Business Model: {product.get('Business', 'Unknown')}
Technology: {product.get('Tech', 'Unknown')}

Identify the key competitive advantages that make this product defensible:

1. Technology/IP advantages (unique algorithms, data sources, etc.)
2. Network effects (user base, data advantages, etc.)
3. Brand/reputation advantages
4. Cost advantages
5. Regulatory/compliance advantages
6. Switching costs for customers

Return a concise 2-3 sentence description of their competitive moat, focusing on how they make money and maintain their advantage.

Moat Description:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            moat = response.choices[0].message.content.strip()
            return moat if moat else None

        except Exception as e:
            self.logger.error(f"Failed to generate moat: {e}")
            return self._generate_moat_simple(product)

    def _generate_moat_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based moat generation"""
        text = f"{product['product_name']} {product.get('overview', '')} {product.get('description', '')}".lower()
        business_model = product.get('Business', '').lower()
        technology = product.get('Tech', '').lower()

        # Generate moat based on business model and technology
        if 'saas' in business_model and 'ai' in technology:
            return 'Proprietary AI algorithms and machine learning models provide predictive accuracy advantages, while subscription-based SaaS model creates recurring revenue and customer lock-in through data integration.'
        elif 'data' in business_model.lower():
            return 'Exclusive data partnerships and proprietary collection methods create data quality advantages, with API-based delivery enabling seamless integration and customer retention.'
        elif 'enterprise' in business_model.lower():
            return 'Deep regulatory compliance expertise and complex integration capabilities create high switching costs, while enterprise licensing provides stable, high-value revenue streams.'
        elif 'ai' in technology.lower():
            return 'Advanced machine learning models and continuous algorithm improvement provide accuracy advantages, with platform effects creating data network advantages over time.'
        else:
            return 'Innovative technology stack and market positioning provide differentiation, with scalable infrastructure enabling cost advantages at scale.'

    def generate_keyword_embedding(self, product_name: str, keywords: Optional[List[str]], overview: str) -> Optional[List[float]]:
        """Generate focused embedding from product name + keywords + overview for better search"""
        if not OPENAI_AVAILABLE or not openai_client:
            return None
        
        try:
            # Combine name, keywords, and overview for a focused search embedding
            keywords_text = ", ".join(keywords) if keywords else ""
            
            text_parts = [
                product_name,
                keywords_text,
                overview[:100] if overview else ""
            ]
            keyword_text = " | ".join([p for p in text_parts if p])
            
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=keyword_text
            )
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate keyword embedding: {e}")
            return None
    
    
    def is_within_date_range(self, post_data: Dict) -> bool:
        """
        Check if post is within the desired date range.
        
        Args:
            post_data: Post node from GraphQL response
            
        Returns:
            True if within range, False otherwise
        """
        created_at = post_data.get('createdAt')
        if not created_at:
            return True  # Include if no date available
        
        try:
            post_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            cutoff_date = datetime.now().astimezone() - timedelta(days=self.days_back)
            return post_date >= cutoff_date
        except Exception as e:
            self.logger.warning(f"Failed to parse date {created_at}: {e}")
            return True  # Include on error
    
    def fetch_products_raw(self, max_products: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch raw product data from Product Hunt API (no parsing, no embeddings).
        
        Args:
            max_products: Maximum number of products to fetch
            
        Returns:
            List of raw post_data dictionaries from API
        """
        products = []
        after_cursor = None
        total_to_fetch = max_products or self.limit
        
        self.logger.info(f"Fetching top {total_to_fetch} products from last {self.days_back} days...")
        
        while len(products) < total_to_fetch:
            self.logger.info(f"Progress: {len(products)}/{total_to_fetch} products fetched")
            
            # Fetch a page
            response = self.fetch_posts(after_cursor)
            
            if not response or 'data' not in response:
                self.logger.error("Failed to fetch data from API")
                break
            
            # Get posts
            edges = response.get('data', {}).get('posts', {}).get('edges', [])
            if not edges:
                self.logger.info("No more products available")
                break
            
            for edge in edges:
                post_data = edge.get('node', {})
                
                # Check if within date range
                if not self.is_within_date_range(post_data):
                    created_at = post_data.get('createdAt', 'unknown')
                    self.logger.debug(f"⊘ {post_data.get('name', 'Unknown')} (created {created_at}, outside {self.days_back} day range)")
                    continue
                
                # Just collect raw data, don't parse yet
                products.append(post_data)
                self.logger.info(f"✓ {post_data.get('name', 'Unknown')} ({post_data.get('votesCount', 0)} votes)")
                
                if len(products) >= total_to_fetch:
                    break
            
            # Check pagination
            page_info = response.get('data', {}).get('posts', {}).get('pageInfo', {})
            if not page_info.get('hasNextPage'):
                self.logger.info("Reached end of available products")
                break
            
            after_cursor = page_info.get('endCursor')
            
            # Rate limiting
            time.sleep(0.5)
        
        return products
    
    def fetch_articles(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch products and return them as article-like objects.
        This method is called by the pipeline to maintain compatibility.
        
        Args:
            since: Not used for Product Hunt (we fetch top products regardless)
            
        Returns:
            List of product dictionaries (empty list, products are stored directly)
        """
        # For Product Hunt, we fetch and store directly since products != articles
        # Return empty list so pipeline doesn't try to process as articles
        self.fetch_and_store()
        return []
    
    def fetch_and_store(self, max_products: Optional[int] = None) -> int:
        """
        Fetch products and store to Supabase.
        This is the main method called by the pipeline.
        
        Args:
            max_products: Maximum number of products to fetch
            
        Returns:
            Number of products stored
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Product Hunt API Scraper")
            self.logger.info("=" * 60)
            
            # Step 1: Fetch raw products from Product Hunt API
            raw_products = self.fetch_products_raw(max_products)
            
            if not raw_products:
                self.logger.warning("No products fetched from Product Hunt")
                return 0
            
            self.logger.info(f"Fetched {len(raw_products)} products from Product Hunt")
            
            # Step 2: Check which products already exist in DB (ONE batch query)
            self.logger.info("Checking which products are new...")
            product_links = [p.get('url', '') for p in raw_products]
            existing_links = self.get_existing_product_links(product_links)
            
            self.logger.info(f"  {len(existing_links)} already exist, {len(raw_products) - len(existing_links)} are new")
            
            # Step 3: Only parse and generate embeddings for NEW products
            new_products = []
            for post_data in raw_products:
                product_link = post_data.get('url', '')
                if product_link in existing_links:
                    self.logger.debug(f" {post_data.get('name', 'Unknown')} (already exists)")
                    continue
                
                # Parse and generate embeddings for new product
                product = self.parse_post(post_data, generate_embeddings=True)
                new_products.append(product)
            
            if not new_products:
                self.logger.info("No new products to store")
                return 0
            
            self.logger.info(f"Parsed {len(new_products)} new products with embeddings")
            
            # Step 4: Store only new products to database
            stored_count = self.storage.store_product_hunt_batch(new_products)
            
            self.logger.info("=" * 60)
            self.logger.info(f"✓ Stored {stored_count} new products")
            self.logger.info("=" * 60)
            return stored_count
            
        except Exception as e:
            self.logger.error(f"Error in fetch_and_store: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_existing_product_links(self, product_links: List[str]) -> set:
        """
        Check which products already exist in database (ONE batch query).
        
        Args:
            product_links: List of Product Hunt URLs
            
        Returns:
            Set of product links that already exist in DB
        """
        try:
            from core.config import supabase_client
            response = supabase_client.table('product_hunt_products').select(
                'producthunt_link'
            ).in_('producthunt_link', product_links).execute()
            
            # Return set of links that exist
            existing_links = set()
            for product in (response.data or []):
                existing_links.add(product['producthunt_link'])
            
            return existing_links
        except Exception as e:
            self.logger.warning(f"Failed to check existing products: {e}")
            return set()  # If check fails, assume all are new


def test_scraper():
    """Test the Product Hunt API scraper"""
    print("\n" + "=" * 60)
    print("Testing Product Hunt API Scraper")
    print("=" * 60 + "\n")
    
    config = {
        'limit': 10,  # Fetch 10 products for testing
        'days_back': 30  # Last 30 days
    }
    
    scraper = ProductHuntScraper(
        newsletter_id="producthunt-test",
        url="https://www.producthunt.com",
        config=config
    )
    
    print("Fetching top 10 products from last 30 days...\n")
    products = scraper.fetch_products(max_products=10)
    
    if products:
        print(f"\n✓ Fetched {len(products)} products\n")
        print("=" * 60)
        
        for i, product in enumerate(products, 1):
            print(f"\n{i}. {product['product_name']}")
            print(f"   Votes: {product['_metadata']['votes']}")
            print(f"   Overview: {product['overview'][:80]}...")
            print(f"   Link: {product['producthunt_link']}")
            print(f"   Topics: {', '.join(product['_metadata']['topics'][:3])}")
        
        print("\n" + "=" * 60)
        print("\nTo store these to Supabase, run:")
        print("  python run_producthunt.py --limit 50")
    else:
        print("✗ No products fetched. Check your API token.")


if __name__ == "__main__":
    test_scraper()
