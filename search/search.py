# search.py

import sys
import re
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from difflib import get_close_matches

from openai import OpenAI
from core.config import supabase_client, OPENAI_API_KEY, BRAVE_API_KEY, EMBEDDING_MODEL, GPT_MODEL
from .competitor_discovery import CompetitorDiscoveryService, DiscoveredCompetitor
from .competitor_config import CompetitorDiscoveryConfig, MODERATE_DISCOVERY
from .market_intelligence import MarketIntelligenceService, MarketIntelligenceReport
from .report_formatter import ReportFormatter
from .deep_dive import DeepDiveService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search")


class SearchService:
    """
    Hybrid search service for VCs: Curated database + Live web search + AI-powered insights

    Use cases:
    - Market intelligence: "What's happening in AI infrastructure?"
    - Competition tracking: "Recent developments in vertical SaaS"
    - Portfolio monitoring: "News about fintech companies"
    """

    def __init__(self):
        self.supabase = supabase_client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.brave_key = BRAVE_API_KEY

        # Initialize competitor discovery service with default configuration
        self.competitor_discovery = CompetitorDiscoveryService(MODERATE_DISCOVERY)
        
        # Initialize market intelligence service
        self.market_intelligence = MarketIntelligenceService()
        
        # Initialize deep dive service
        self.deep_dive_service = DeepDiveService()
        
        # Initialize report formatter
        self.report_formatter = ReportFormatter()

        # Cache portfolio companies for faster lookups
        self._portfolio_cache = None

        # Load portfolio mapping files for intelligent query detection
        self._load_portfolio_maps()

    def _load_portfolio_maps(self):
        """Load pre-built portfolio mapping files for intelligent query detection."""
        import json
        import os

        try:
            # Load keyword to companies mapping
            if os.path.exists('keyword_to_companies.json'):
                with open('keyword_to_companies.json', 'r', encoding='utf-8') as f:
                    self.keyword_to_companies = json.load(f)
            else:
                self.keyword_to_companies = {}

            # Load industry to companies mapping
            if os.path.exists('industry_to_companies.json'):
                with open('industry_to_companies.json', 'r', encoding='utf-8') as f:
                    self.industry_to_companies = json.load(f)
            else:
                self.industry_to_companies = {}

            # Load all keywords
            if os.path.exists('all_keywords.json'):
                with open('all_keywords.json', 'r', encoding='utf-8') as f:
                    self.all_keywords = json.load(f)
            else:
                self.all_keywords = []

            # Load all industries
            if os.path.exists('all_industries.json'):
                with open('all_industries.json', 'r', encoding='utf-8') as f:
                    self.all_industries = json.load(f)
            else:
                self.all_industries = []

            logger.info(f"Loaded portfolio maps: {len(self.all_keywords)} keywords, {len(self.all_industries)} industries")

        except Exception as e:
            logger.warning(f"Failed to load portfolio maps: {e}. Run build_portfolio_maps.py first.")
            self.keyword_to_companies = {}
            self.industry_to_companies = {}
            self.all_keywords = []
            self.all_industries = []

    def _get_portfolio_companies(self) -> List[Dict]:
        """Fetch all portfolio companies (with caching)"""
        if self._portfolio_cache is None:
            try:
                response = self.supabase.table('portfolio_companies').select(
                    'id, company_name, competitors, industry, keywords, what_they_build'
                ).execute()
                self._portfolio_cache = response.data or []
                logger.info(f"Loaded {len(self._portfolio_cache)} portfolio companies")
            except Exception as e:
                logger.error(f"Failed to fetch portfolio companies: {e}")
                self._portfolio_cache = []
        return self._portfolio_cache

    def _detect_query_type(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Intelligent query detection with priority:
        1. Check if asking about specific portfolio company
        2. Check if asking about industry/technology/trends
        3. Default to general portfolio monitoring

        Returns:
            (mode, matched_entity) where mode is one of:
            - 'company_mode': Query about a specific portfolio company
            - 'industry_mode': Query about industry/tech/business trends
            - 'portfolio_mode': General portfolio monitoring (all companies)

            matched_entity is:
            - Company name for company_mode
            - Industry/tech term for industry_mode
            - None for portfolio_mode
        """
        q = query.lower()
        companies = self._get_portfolio_companies()
        company_names = [c['company_name'] for c in companies]

        # PRIORITY 1: Check for specific portfolio company mention
        # This is the highest priority - if user mentions a specific company, focus on that

        # Check exact word match first (most reliable)
        query_words = set(q.split())
        for name in company_names:
            name_lower = name.lower()
            # Single word company name (e.g., "Stripe")
            if name_lower in query_words:
                logger.info(f"Detected: company_mode for '{name}' (exact word match)")
                return ("company_mode", name)

            # Multi-word company names (e.g., "Y Combinator")
            name_words = set(name_lower.split())
            if len(name_words) > 1 and name_words.issubset(query_words):
                logger.info(f"Detected: company_mode for '{name}' (multi-word match)")
                return ("company_mode", name)

        # Check word-bounded substring match (e.g., "news about OpenAI")
        for name in company_names:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            if re.search(pattern, q):
                logger.info(f"Detected: company_mode for '{name}' (substring match)")
                return ("company_mode", name)

        # PRIORITY 2: Check for industry/technology/business trend queries
        # These are queries about market trends, not specific companies

        # Explicit industry/trend keywords
        trend_keywords = [
            "industry", "sector", "market", "space", "landscape", "ecosystem",
            "trends", "trend", "developments", "happening in", "new in",
            "technology", "tech stack", "innovations", "emerging",
            "business model", "funding rounds", "m&a", "acquisitions",
            "ipo", "valuations", "startup", "startups"
        ]

        industries = list(set([c.get('industry', '') for c in companies if c.get('industry')]))

        # Check for explicit trend keywords first
        if any(keyword in q for keyword in trend_keywords):
            # Try to extract the specific topic
            # Check if query mentions a portfolio industry
            for industry in industries:
                if industry.lower() in q:
                    logger.info(f"Detected: industry_mode for '{industry}'")
                    return ("industry_mode", industry)

            # Extract technology/industry term from query
            # Remove common words to find the subject
            stopwords = {'the', 'in', 'about', 'for', 'on', 'what', 'is', 'are', 'whats',
                        'latest', 'recent', 'new', 'any', 'news', 'update', 'updates'}
            meaningful_words = [w for w in q.split() if w not in stopwords and len(w) > 2]

            if meaningful_words:
                # Take 2-3 words as the topic
                topic = " ".join(meaningful_words[:3])
                logger.info(f"Detected: industry_mode (extracted topic: '{topic}')")
                return ("industry_mode", topic)

        # NEW: Check for industry terms even without explicit trend keywords
        # Look for patterns like "in [industry]" or industry terms in query
        for industry in industries:
            # Check if industry name appears in query (case insensitive)
            if industry.lower() in q:
                logger.info(f"Detected: industry_mode for '{industry}' (industry term found)")
                return ("industry_mode", industry)

        # Check for pattern "in [industry]" or "[industry] updates/news/trends"
        query_words = q.split()
        for i, word in enumerate(query_words):
            if word.lower() == 'in' and i + 1 < len(query_words):
                # Look for next word that might be an industry
                next_word = query_words[i + 1].lower()
                for industry in industries:
                    if next_word in industry.lower() or industry.lower() in next_word:
                        logger.info(f"Detected: industry_mode for '{industry}' (pattern: in [industry])")
                        return ("industry_mode", industry)

        # PRIORITY 3: Explicit portfolio monitoring requests
        portfolio_keywords = [
            "portfolio", "our investments", "our companies", "all companies",
            "portfolio companies", "companies we invested", "our portcos"
        ]
        if any(keyword in q for keyword in portfolio_keywords):
            logger.info("Detected: portfolio_mode (explicit request)")
            return ("portfolio_mode", None)

        # DEFAULT: General portfolio monitoring
        # If query is vague (e.g., "latest news", "any updates"), monitor all portfolio
        logger.info("Detected: portfolio_mode (default - monitoring all companies)")
        return ("portfolio_mode", None)

    def _build_search_context(self, mode: str, entity: Optional[str]) -> Dict:
        """
        Build search context with relevant entities and keywords.

        Returns:
            {
                'entities': List of company names to search for,
                'keywords': List of technology/industry keywords,
                'description': Human-readable description,
                'mode': The query mode
            }
        """
        companies = self._get_portfolio_companies()

        if mode == "company_mode":
            # Single company + its competitors
            company = next((c for c in companies if c['company_name'] == entity), None)
            if company:
                entities = [company['company_name']]
                if company.get('competitors'):
                    entities.extend(company['competitors'])

                keywords = company.get('keywords', [])

                return {
                    'entities': list(set(entities)),
                    'keywords': keywords,
                    'description': f"Company: {entity} + {len(company.get('competitors', []))} competitors",
                    'mode': mode,
                    'company': entity,
                    'focus_company': company
                }

            return {'entities': [], 'keywords': [], 'description': 'Unknown company', 'mode': mode}

        elif mode == "industry_mode":
            # Find all companies related to this industry/technology/trend
            # Match by industry field OR keywords OR what_they_build
            relevant_companies = []
            entity_lower = (entity or '').lower()

            for c in companies:
                # Check industry field
                if c.get('industry', '').lower() == entity_lower:
                    relevant_companies.append(c)
                    continue

                # Check if entity keyword appears in industry
                if entity_lower in c.get('industry', '').lower():
                    relevant_companies.append(c)
                    continue

                # Check keywords
                if c.get('keywords'):
                    if any(entity_lower in kw.lower() for kw in c['keywords']):
                        relevant_companies.append(c)
                        continue

                # Check what they build
                if entity_lower in c.get('what_they_build', '').lower():
                    relevant_companies.append(c)
                    continue

            # Collect all entities and keywords
            company_names = [c['company_name'] for c in relevant_companies]
            competitors = []
            all_keywords = set()

            for c in relevant_companies:
                if c.get('competitors'):
                    competitors.extend(c['competitors'])
                if c.get('keywords'):
                    all_keywords.update(c['keywords'])

            # Add the industry/tech term itself as a keyword
            all_keywords.add(entity or '')

            all_entities = list(set(company_names + competitors))
            all_keywords = list(all_keywords)

            return {
                'entities': all_entities,
                'keywords': all_keywords,
                'description': f"Topic: {entity} ({len(company_names)} portfolio companies, {len(competitors)} competitors)",
                'mode': mode,
                'industry': entity,
                'relevant_companies': relevant_companies
            }

        elif mode == "portfolio_mode":
            # All portfolio companies - monitor everything
            company_names = [c['company_name'] for c in companies]

            # Collect all unique keywords from portfolio
            all_keywords = set()
            for c in companies:
                if c.get('keywords'):
                    all_keywords.update(c['keywords'])
                # Add industry as keyword
                if c.get('industry'):
                    all_keywords.add(c['industry'])

            return {
                'entities': company_names,
                'keywords': list(all_keywords),
                'description': f"All {len(company_names)} portfolio companies",
                'mode': mode,
                'all_companies': companies
            }

        return {'entities': [], 'keywords': [], 'description': 'Unknown', 'mode': mode}

    def search(
        self,
        query: str,
        include_web: bool = True,
        limit: int = 10,
        generate_insights: bool = True
    ) -> Dict:
        """
        Search database and web, then generate AI-powered insights.

        Args:
            query: Search query (e.g., "vertical SaaS trends", "fintech exits")
            include_web: Whether to include live web results
            limit: Number of results per source
            generate_insights: Whether to generate AI summary and insights

        Returns:
            Dictionary with 'database', 'web', and 'insights' results
        """
        logger.info(f"Searching for: '{query}'")

        # Detect query intent
        mode, entity = self._detect_query_type(query)
        context = self._build_search_context(mode, entity)

        logger.info(f"Search context: {context['description']}")

        results = {
            'query': query,
            'mode': mode,
            'context': context,
            'database': [],
            'web': [],
            'insights': None,
            'timestamp': datetime.now().isoformat()
        }

        # 1. Search curated database
        logger.info("Searching curated database...")
        try:
            db_results = self._search_database(query, limit)
            results['database'] = db_results
            logger.info(f"Found {len(db_results)} results from database")
        except Exception as e:
            logger.error(f"Database search failed: {e}")

        # 2. Search live web (with context-aware query expansion)
        if include_web and self.brave_key:
            logger.info("Searching live web...")
            try:
                web_results = self._search_brave_contextual(query, context, limit)
                results['web'] = web_results
                logger.info(f"Found {len(web_results)} results from web")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        elif include_web and not self.brave_key:
            logger.warning("Brave API key not set. Skipping web search.")

        # 3. Generate AI insights (context-aware)
        if generate_insights and self.openai_client:
            logger.info("Generating AI insights...")
            try:
                insights = self._generate_insights(query, results['database'], results['web'], mode, context)
                results['insights'] = insights
                logger.info("AI insights generated")
            except Exception as e:
                logger.error(f"Insights generation failed: {e}")

        return results

    def _search_database(self, query: str, limit: int) -> List[Dict]:
        """
        Search curated database using semantic search with vector embeddings.
        """
        if not self.openai_client:
            logger.warning("OpenAI key not set, using keyword search")
            return self._keyword_search_database(query, limit)

        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return self._keyword_search_database(query, limit)

        try:
            # Fetch chunks with embeddings from database
            response = self.supabase.table('article_chunks').select(
                '''
                id,
                chunk_text,
                embedding,
                article_id,
                articles!inner(
                    id,
                    title,
                    url,
                    company_names,
                    published_date
                )
                '''
            ).not_.is_('embedding', 'null').execute()

            if not response.data:
                logger.warning("No chunks with embeddings found")
                return self._keyword_search_database(query, limit)

            # Calculate cosine similarity in Python
            query_vec = np.array(query_embedding, dtype=np.float32)
            chunks_with_similarity = []

            for row in response.data:
                chunk_embedding = row.get('embedding')
                if not chunk_embedding:
                    continue

                try:
                    # FIX: Parse embedding (comes as string or list from Supabase)
                    if isinstance(chunk_embedding, str):
                        # Remove brackets and parse
                        import json
                        chunk_embedding = json.loads(chunk_embedding)

                    # Convert to numpy array
                    chunk_vec = np.array(chunk_embedding, dtype=np.float32)

                    # Validate dimensions match
                    if len(chunk_vec) != len(query_vec):
                        logger.warning(f"Dimension mismatch: query={len(query_vec)}, chunk={len(chunk_vec)}")
                        continue

                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, chunk_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                    )

                    chunks_with_similarity.append({
                        'chunk_text': row.get('chunk_text', ''),
                        'similarity': float(similarity),
                        'article_data': row.get('articles', {})
                    })

                except Exception as e:
                    logger.debug(f"Skipping chunk due to error: {e}")
                    continue

            if not chunks_with_similarity:
                logger.warning("No valid embeddings found, falling back to keyword search")
                return self._keyword_search_database(query, limit)

            # Sort by similarity
            chunks_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

            # Group by article and take top matches
            articles = {}
            for chunk in chunks_with_similarity[:limit * 3]:
                article_data = chunk['article_data']
                article_id = article_data.get('id')

                if article_id not in articles:
                    articles[article_id] = {
                        'title': article_data.get('title', 'Untitled'),
                        'url': article_data.get('url', ''),
                        'company_names': article_data.get('company_names', []),
                        'published_date': article_data.get('published_date'),
                        'similarity': chunk['similarity'],
                        'matched_chunks': [],
                        'source': 'database'
                    }

                # Update similarity to best match
                if chunk['similarity'] > articles[article_id]['similarity']:
                    articles[article_id]['similarity'] = chunk['similarity']

                # Add chunk preview
                chunk_text = chunk['chunk_text']
                articles[article_id]['matched_chunks'].append({
                    'text': chunk_text[:300] + ('...' if len(chunk_text) > 300 else ''),
                    'similarity': chunk['similarity']
                })

            # Convert to list and sort by best similarity
            results = list(articles.values())
            results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Falling back to keyword search.")
            import traceback
            traceback.print_exc()
            return self._keyword_search_database(query, limit)
        
    def _keyword_search_database(self, query: str, limit: int) -> List[Dict]:
        """
        Fallback: Simple keyword search in article content and titles.
        """
        # Search in both title and content
        response = self.supabase.table('articles').select(
            'id, title, url, company_names, raw_content, published_date'
        ).or_(
            f'title.ilike.%{query}%,raw_content.ilike.%{query}%'
        ).limit(limit).execute()

        results = []
        for article in response.data:
            # Extract snippet around the matched query
            content = article.get('raw_content', '')
            query_lower = query.lower()
            if query_lower in content.lower():
                idx = content.lower().find(query_lower)
                start = max(0, idx - 100)
                end = min(len(content), idx + 200)
                snippet = content[start:end]
            else:
                snippet = content[:300] if content else ''

            results.append({
                'title': article['title'],
                'url': article['url'],
                'company_names': article.get('company_names', []),
                'published_date': article.get('published_date'),
                'source': 'database',
                'match_type': 'keyword',
                'snippet': snippet + '...'
            })

        return results

    def _search_brave_contextual(self, query: str, context: Dict, limit: int) -> List[Dict]:
        """
        Context-aware web search with dynamic query expansion and intelligent filtering.

        For portfolio/company mode: Search for specific companies
        For industry mode: Search for topic + related companies + keywords
        """
        entities = context.get('entities', [])
        keywords = context.get('keywords', [])
        mode = context.get('mode')

        # Build expanded query based on mode
        if mode == "portfolio_mode":
            # Portfolio monitoring: Batch search companies to avoid API limits
            # Search 10 companies at a time, combine results
            all_results = []
            batch_size = 10

            logger.info(f"Portfolio mode: Searching {len(entities)} companies in batches of {batch_size}")

            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                terms_query = " OR ".join([f'"{term}"' for term in batch])
                expanded_query = f"{query} ({terms_query})"

                logger.info(f"Batch {i//batch_size + 1}/{(len(entities) + batch_size - 1)//batch_size}: Searching {len(batch)} companies")

                try:
                    batch_results = self._search_brave(expanded_query, limit)
                    all_results.extend(batch_results)
                    logger.info(f"  Found {len(batch_results)} results in this batch")
                except Exception as e:
                    logger.error(f"  Batch failed: {e}")
                    continue

            raw_results = all_results

        elif mode == "industry_mode":
            # Industry/tech trends: focus on keywords, mention companies
            if keywords:
                keyword_query = " OR ".join([f'"{kw}"' for kw in keywords[:10]])
                expanded_query = f"{query} ({keyword_query})"
                logger.info(f"Industry mode: Expanded with {len(keywords[:10])} keywords")
            else:
                expanded_query = query

            raw_results = self._search_brave(expanded_query, limit * 2)

        elif mode == "company_mode":
            # Specific company: search company + competitors
            if entities:
                entity_query = " OR ".join([f'"{e}"' for e in entities[:10]])
                expanded_query = f"{query} ({entity_query})"
                logger.info(f"Company mode: Expanded with {len(entities[:10])} companies")
            else:
                expanded_query = query

            raw_results = self._search_brave(expanded_query, limit * 2)

        else:
            expanded_query = query
            raw_results = self._search_brave(expanded_query, limit * 2)

        # Filter and score results based on entities AND keywords
        if entities or keywords:
            filtered_results = self._filter_and_score_results(
                raw_results,
                entities,
                keywords,
                mode
            )
            logger.info(f"Filtered {len(raw_results)} → {len(filtered_results)} relevant results")
            return filtered_results[:limit]
        else:
            return raw_results[:limit]

    def _search_brave(self, query: str, limit: int) -> List[Dict]:
        """
        Search live web using Brave Search API for real-time market intelligence.
        """
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_key
            },
            params={
                "q": query,
                "count": limit,
                "freshness": "pw"  # Past week for recent news
            }
        )

        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get('web', {}).get('results', []):
            results.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'description': item.get('description', ''),
                'age': item.get('age', ''),
                'source': 'web'
            })

        return results

    def _filter_and_score_results(
        self,
        results: List[Dict],
        entities: List[str],
        keywords: List[str] = None,
        mode: str = None
    ) -> List[Dict]:
        """
        Intelligent filtering based on entities and keywords.

        For company_mode: Require entity mention
        For industry_mode: Entity OR keyword match (more lenient)
        For portfolio_mode: Entity OR multiple keyword matches
        """
        if keywords is None:
            keywords = []

        filtered = []

        for result in results:
            text = (result.get('title', '') + ' ' + result.get('description', '')).lower()
            title_lower = result.get('title', '').lower()

            # Check entity matches
            matched_entities = []
            for entity in entities:
                if entity.lower() in text:
                    matched_entities.append(entity)

            # Check keyword matches
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text:
                    matched_keywords.append(keyword)

            # Apply mode-specific filtering
            should_include = False
            score = 0.0

            if mode == "company_mode":
                # Company mode: MUST mention the company or competitor
                if matched_entities:
                    should_include = True
                    score += min(len(matched_entities) * 0.3, 0.7)
                    if any(e.lower() in title_lower for e in matched_entities):
                        score += 0.3

            elif mode == "industry_mode":
                # Industry mode: Entity OR keyword (more lenient for trend discovery)
                if matched_entities or matched_keywords:
                    should_include = True
                    # Score based on both entities and keywords
                    score += min(len(matched_entities) * 0.2, 0.5)
                    score += min(len(matched_keywords) * 0.15, 0.4)
                    # Bonus for title mentions
                    if any(e.lower() in title_lower for e in matched_entities):
                        score += 0.2
                    if any(kw.lower() in title_lower for kw in matched_keywords):
                        score += 0.1

            elif mode == "portfolio_mode":
                # Portfolio mode: Entity OR multiple keywords (filter noise)
                if matched_entities:
                    should_include = True
                    score += min(len(matched_entities) * 0.3, 0.7)
                elif len(matched_keywords) >= 2:  # Require 2+ keywords to avoid noise
                    should_include = True
                    score += min(len(matched_keywords) * 0.2, 0.6)

                # Bonus for title mentions
                if any(e.lower() in title_lower for e in matched_entities):
                    score += 0.2

            else:
                # Default: require at least one match
                if matched_entities or matched_keywords:
                    should_include = True
                    score += min((len(matched_entities) + len(matched_keywords)) * 0.2, 0.8)

            if should_include:
                result['relevance_score'] = min(score, 1.0)
                result['matched_entities'] = matched_entities
                result['matched_keywords'] = matched_keywords
                filtered.append(result)

        # Sort by relevance score
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return filtered

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for search query."""
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def _generate_insights(
        self,
        query: str,
        db_results: List[Dict],
        web_results: List[Dict],
        mode: str,
        context: Dict
    ) -> Dict:
        """
        Generate AI-powered insights from search results.
        Tailored for VC use cases: market trends, competition, portfolio intelligence.
        Mode-aware prompt engineering for better results.
        """
        # Prepare context from results
        context_parts = []

        # Add database results
        if db_results:
            context_parts.append(" Curated Database Results:")
            for idx, item in enumerate(db_results[:5], 1):
                context_parts.append(f"\n{idx}. {item['title']}")
                if item.get('company_names'):
                    context_parts.append(f"   Companies: {', '.join(item['company_names'])}")
                if item.get('matched_chunks'):
                    context_parts.append(f"   Context: {item['matched_chunks'][0]['text']}")

        # Add web results
        if web_results:
            context_parts.append("\n\n Live Web Results:")
            for idx, item in enumerate(web_results[:5], 1):
                context_parts.append(f"\n{idx}. {item['title']}")
                context_parts.append(f"   {item['description']}")
                if item.get('age'):
                    context_parts.append(f"   Published: {item['age']}")

        results_context = "\n".join(context_parts)

        # Build mode-specific prompt
        base_instruction = "You are a senior VC analyst providing actionable intelligence. Focus on specific developments that could help or hurt portfolio companies. Be concrete, not generic. If no specific developments found, acknowledge but still provide strategic market overview."

        # Get portfolio companies for context
        portfolio_companies = self._get_portfolio_companies()
        portfolio_names = [c['company_name'] for c in portfolio_companies]

        # Common instruction for all modes
        mode_instruction = """
PRIORITY: Focus on developments that could impact our portfolio companies.
If no portfolio-specific updates found, clearly state this but provide relevant market context.
Be specific about which portfolio companies might find this interesting.
"""

        # Standard structure for all modes - using bold formatting instead of hashtags
        sections = """
**1. PORTFOLIO UPDATES**
   - Which of our portfolio companies have relevant news? Be specific.
   - Any funding rounds, product launches, or strategic moves?

**2. COMPETITIVE UPDATES**
   - What's happening with competitors that could affect our portfolio?
   - Any new entrants, market shifts, or competitive moves?

**3. TECH DEVELOPMENTS IN RELATED INDUSTRIES**
   - What new technologies or innovations are emerging that our portfolio companies should know about?
   - Any tech trends that could create opportunities or threats?

**4. BUSINESS DEVELOPMENTS IN RELATED INDUSTRIES**
   - What business model changes, partnerships, or market expansions are happening?
   - Any regulatory changes, M&A activity, or funding trends that matter?

**5. PORTFOLIO COMPANY IMPLICATIONS**
   - Which specific portfolio companies should pay attention to these developments?
   - What actions should we recommend to our portfolio companies?
   - Any follow-up actions for the VC team?

IMPORTANT: When mentioning company names in your response, format them in bold like **CompanyName**."""

        prompt = f"""{base_instruction}

{mode_instruction}

PORTFOLIO CONTEXT: Our portfolio companies include: {', '.join(portfolio_names[:10])}
{' and others...' if len(portfolio_names) > 10 else ''}

Query: "{query}"

CRITICAL: Focus on SPECIFIC, RECENT developments that could impact our portfolio companies.
Look for concrete examples like:
- "DoorDash launched new AI algorithm that reduced delivery times by 15%"
- "Uber Eats raised $200M for market expansion in Asia"
- "New regulation requiring all delivery apps to show carbon footprint"
- "Stripe launched embedded payments for food delivery platforms"

If search results are generic trends, acknowledge this but extract any specific company moves or product launches that could affect our portfolio.

PRIORITY: Always mention which of our portfolio companies (if any) might find this development particularly interesting or concerning.

Based on these search results, provide analysis:

{results_context}

Structure your response exactly as requested:

{sections}

Be concrete and actionable for VC portfolio management. Avoid generic statements. If no portfolio-specific updates found, clearly state this but still provide relevant market intelligence that our portfolio companies should know about."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior investment analyst helping VCs with market research and competitive intelligence. Provide clear, actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )

            insights_text = response.choices[0].message.content

            return {
                'summary': insights_text,
                'generated_at': datetime.now().isoformat(),
                'sources_analyzed': {
                    'database': len(db_results),
                    'web': len(web_results)
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    def analyze_competitors_enhanced(
        self,
        query: str,
        config: CompetitorDiscoveryConfig = None,
        include_discovery: bool = True
    ) -> Dict:
        """
        Enhanced competitor analysis that includes both mapped competitors and newly discovered ones

        Args:
            query: The search query or company name to analyze competitors for
            config: Configuration for competitor discovery behavior (uses default if None)
            include_discovery: Whether to include competitor discovery (beyond mapped competitors)

        Returns:
            Dictionary with:
            - mapped_competitors: Competitors already in portfolio map
            - discovered_competitors: Newly discovered competitors
            - all_competitors: Combined list for convenience
        """
        logger.info(f"Starting enhanced competitor analysis for: {query}")

        # Use provided config or default
        discovery_config = config or self.competitor_discovery.config

        # First, get the query type and matched entity
        mode, matched_entity = self._detect_query_type(query)

        mapped_competitors = []
        discovered_competitors = []

        if mode == "company_mode" and matched_entity:
            # Get competitors for a specific portfolio company
            company_data = self._find_company_by_name(matched_entity)
            if company_data and company_data.get('competitors'):
                mapped_competitors = company_data['competitors'].split(', ')

            # Extract keywords and industries from the company for discovery
            if company_data:
                keywords = company_data.get('keywords', '').split(', ') if company_data.get('keywords') else []
                industries = [company_data.get('industry', '')] if company_data.get('industry') else []

                if include_discovery and (keywords or industries):
                    # Discover new competitors using the company's keywords and industry
                    existing_competitor_names = set(mapped_competitors)
                    discovered_competitors = self.competitor_discovery.discover_competitors(
                        keywords=keywords,
                        industries=industries,
                        existing_competitors=existing_competitor_names,
                        config=discovery_config
                    )

        elif mode == "industry_mode" and matched_entity:
            # Get companies in the specified industry
            industry_companies = self._find_companies_by_industry(matched_entity)
            company_names = [c['company_name'] for c in industry_companies]

            # For industry queries, we'll use the industry as the search signal for discovery
            if include_discovery:
                existing_competitor_names = set(company_names)
                discovered_competitors = self.competitor_discovery.discover_competitors(
                    keywords=[matched_entity],  # Use industry as keyword
                    industries=[matched_entity],
                    existing_competitors=existing_competitor_names,
                    config=discovery_config
                )

            # Return industry companies as mapped competitors
            mapped_competitors = company_names

        # Convert discovered competitors to simple dict format for API compatibility
        discovered_competitors_simple = [
            {
                'name': comp.name,
                'industry': comp.industry,
                'keywords': comp.keywords,
                'description': comp.description,
                'confidence_score': comp.confidence_score,
                'source_url': comp.source_url,
                'search_signal': comp.search_signal,
                'is_discovered': True  # Flag to distinguish from mapped competitors
            }
            for comp in discovered_competitors
        ]

        # Combine all competitors
        all_competitors = mapped_competitors + [comp['name'] for comp in discovered_competitors_simple]

        result = {
            'query': query,
            'mode': mode,
            'matched_entity': matched_entity,
            'mapped_competitors': mapped_competitors,
            'discovered_competitors': discovered_competitors_simple,
            'all_competitors': all_competitors,
            'total_mapped': len(mapped_competitors),
            'total_discovered': len(discovered_competitors_simple),
            'total_competitors': len(all_competitors)
        }

        logger.info(f"Enhanced competitor analysis complete: {result['total_mapped']} mapped, {result['total_discovered']} discovered")
        return result

    def _find_company_by_name(self, company_name: str) -> Optional[Dict]:
        """Helper method to find a company by name"""
        companies = self._get_portfolio_companies()
        for company in companies:
            if company['company_name'].lower() == company_name.lower():
                return company
        return None

    def _find_companies_by_industry(self, industry: str) -> List[Dict]:
        """Helper method to find companies by industry"""
        companies = self._get_portfolio_companies()
        return [c for c in companies if c.get('industry', '').lower() == industry.lower()]

    def analyze_market_intelligence(
        self,
        query: str,
        max_results_per_dimension: int = 8,
        include_ai_insights: bool = True,
        format_type: str = "api"
    ) -> Dict:
        """
        Perform comprehensive multi-dimensional market intelligence analysis
        
        This is the new enhanced search that breaks down complex queries and
        analyzes the space across multiple dimensions: companies, technology,
        business models, innovations, and market trends.
        
        Args:
            query: Natural language description of company/space to analyze
            max_results_per_dimension: Max articles to collect per dimension
            include_ai_insights: Whether to generate AI-powered insights
            format_type: Output format ("api", "display", or "summary")
            
        Returns:
            Formatted market intelligence report
            
        Example:
            query = "I have a company that does hedge fund and trade analysis using AI. 
                     It has agents that scrape market data, pricing data, and financial news 
                     and gives funds automated updates if anything changes. Give me updates in the space."
        """
        logger.info(f"Starting market intelligence analysis: {query[:100]}...")
        
        # Perform the analysis
        report = self.market_intelligence.analyze_market(
            query=query,
            max_results_per_dimension=max_results_per_dimension,
            include_ai_insights=include_ai_insights
        )
        
        # Format based on requested type
        if format_type == "display":
            return {
                'formatted_output': self.report_formatter.format_for_display(report),
                'report': report
            }
        elif format_type == "summary":
            return self.report_formatter.format_summary(report)
        else:  # api (default)
            return self.report_formatter.format_for_api(report, extract_entities=True)
    
    def deep_dive_entity(
        self,
        entity: str,
        entity_type: str = "company",
        context: str = "",
        max_articles: int = 15
    ) -> Dict:
        """
        Perform deep dive on a specific entity (company, partnership, technology)
        
        This is triggered when user clicks "Deep Dive" button on an entity
        
        Args:
            entity: The entity to research (e.g., "Refinitiv", "Goldman Sachs AI partnerships")
            entity_type: Type ("company", "partnership", "technology", "vague_mention")
            context: Original search context for relevance
            max_articles: Maximum articles to analyze
            
        Returns:
            Formatted deep dive report
            
        Example:
            # User saw "Refinitiv" in market intelligence report
            service.deep_dive_entity("Refinitiv", "company", "hedge fund AI")
            
            # User saw "Goldman Sachs partnered with AI startups"
            service.deep_dive_entity(
                "Goldman Sachs AI startup partnerships",
                "partnership",
                "hedge fund AI"
            )
        """
        logger.info(f"Deep dive on entity: {entity}")
        
        report = self.deep_dive_service.deep_dive(
            entity=entity,
            entity_type=entity_type,
            context=context,
            max_articles=max_articles
        )
        
        # Format for API
        return {
            'entity': report.entity,
            'entity_type': report.entity_type,
            'context': report.context,
            'generated_at': report.generated_at,
            'overview': report.overview,
            'key_facts': report.key_facts,
            'recent_developments': report.recent_developments,
            'competitors': report.competitors,
            'technologies_used': report.technologies_used,
            'business_model': report.business_model,
            'partners': report.partners,
            'companies_using': report.companies_using,
            'related_entities': report.related_entities,
            'suggested_next_searches': report.suggested_next_searches
        }

    def display_results(self, results: Dict, show_full_insights: bool = True):
        """
        Pretty print search results with VC-focused formatting.
        """
        print("\n" + "="*80)
        print(f"SEARCH RESULTS: '{results['query']}'")
        print("="*80)

        # Log mode for debugging (don't show to user)
        if results.get('mode'):
            logger.info(f"Search mode: {results['mode']}")
            if results.get('context', {}).get('description'):
                logger.info(f"Search context: {results['context']['description']}")

        # Display AI insights first (most important for VCs)
        if results.get('insights') and show_full_insights:
            print("\n" + "="*80)
            print("PORTFOLIO-FOCUSED MARKET INTELLIGENCE")
            print("="*80)
            if results['insights'].get('summary'):
                print(results['insights']['summary'])
            elif results['insights'].get('error'):
                print(f"Error generating insights: {results['insights']['error']}")
            print()

        # Display database results
        if results['database']:
            print("-"*80)
            print(f"FROM YOUR CURATED DATABASE ({len(results['database'])} results)")
            print("-"*80)
            for idx, item in enumerate(results['database'], 1):
                print(f"\n{idx}. {item['title']}")
                print(f"   URL: {item['url']}")
                if item.get('company_names'):
                    print(f"   Companies: {', '.join(item['company_names'][:5])}")
                if item.get('published_date'):
                    print(f"   Date: {item['published_date']}")
                if item.get('similarity'):
                    print(f"   Relevance: {item['similarity']:.1%}")
                if item.get('matched_chunks'):
                    print(f"   Preview: {item['matched_chunks'][0]['text'][:200]}...")
                elif item.get('snippet'):
                    print(f"   Preview: {item['snippet'][:200]}...")
        else:
            print("\nFROM YOUR CURATED DATABASE: No results found")

        # Display web results
        if results['web']:
            print(f"\n{'-'*80}")
            print(f"FROM LIVE WEB ({len(results['web'])} results)")
            print("-"*80)
            for idx, item in enumerate(results['web'], 1):
                print(f"\n{idx}. {item['title']}")
                print(f"   URL: {item['url']}")
                print(f"   {item['description']}")
                if item.get('age'):
                    print(f"   Published: {item['age']}")

                # Show what matched
                matches = []
                if item.get('matched_entities'):
                    matches.append(f"Companies: {', '.join(item['matched_entities'][:3])}")
                if item.get('matched_keywords'):
                    matches.append(f"Keywords: {', '.join(item['matched_keywords'][:3])}")
                if matches:
                    print(f"   Matched: {' | '.join(matches)}")

                if item.get('relevance_score'):
                    print(f"   Relevance: {item['relevance_score']:.1%}")
        else:
            print("\nFROM LIVE WEB: No results found")

        print("\n" + "="*80)
        print(f"Search completed at {results['timestamp']}")
        print("="*80 + "\n")


# CLI Interface
def main():
    """
    Command-line interface for search.

    Examples:
        python search.py "AI infrastructure startups"
        python search.py "vertical SaaS trends" --no-web
        python search.py "fintech exits 2024" --no-insights
        python search.py "marketplace platforms" --limit 15
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='VC-focused search: Curated database + Live web + AI insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search.py "AI infrastructure trends"
  python search.py "vertical SaaS competition" --no-web
  python search.py "fintech startups" --limit 15
        """
    )

    parser.add_argument('query', help='Search query')
    parser.add_argument('--no-web', action='store_true', help='Skip live web search')
    parser.add_argument('--no-insights', action='store_true', help='Skip AI insights generation')
    parser.add_argument('--limit', type=int, default=10, help='Number of results per source (default: 10)')

    args = parser.parse_args()

    # Perform search
    searcher = SearchService()
    results = searcher.search(
        query=args.query,
        include_web=not args.no_web,
        limit=args.limit,
        generate_insights=not args.no_insights
    )

    # Display results
    searcher.display_results(results)


if __name__ == "__main__":
    main()
