# market_intelligence.py

"""
Multi-dimensional market intelligence service.
Performs comprehensive analysis across multiple dimensions: companies, tech, business models, trends.
"""

import logging
import requests
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from openai import OpenAI
from core.config import supabase_client, OPENAI_API_KEY, BRAVE_API_KEY, GPT_MODEL
from .query_decomposition import QueryComponents, QueryDecompositionService
from .competitor_discovery import CompetitorDiscoveryService
from .entity_extraction import EntityExtractionService

logger = logging.getLogger("market_intelligence")

@dataclass
class DimensionResults:
    """Results for a single search dimension"""
    dimension: str
    description: str
    queries_performed: List[str]
    articles: List[Dict[str, Any]]
    key_findings: List[str]
    companies_mentioned: List[str]
    technologies_mentioned: List[str]
    insights: Optional[str] = None

    # Enhanced fields for detailed information
    technology_usage_details: Dict[str, List[str]] = field(default_factory=dict)  # tech -> [specific usage examples]
    business_model_details: List[str] = field(default_factory=list)  # specific business model descriptions
    market_insights: List[str] = field(default_factory=list)  # specific market trend insights
    company_business_models: Dict[str, List[str]] = field(default_factory=dict)  # company -> [business models used]

@dataclass
class MarketIntelligenceReport:
    """Comprehensive market intelligence report"""
    query: str
    generated_at: str
    
    # Query analysis
    components: QueryComponents
    search_plan: Dict
    
    # Dimensional results
    companies_landscape: DimensionResults
    technology_landscape: DimensionResults
    business_models: DimensionResults
    innovations: DimensionResults
    market_trends: DimensionResults
    competitive_analysis: DimensionResults
    
    # Aggregated insights
    executive_summary: str
    key_players: List[Dict[str, Any]]
    emerging_trends: List[str]
    recommended_next_steps: List[str]


class MarketIntelligenceService:
    """
    Performs multi-dimensional market intelligence analysis.
    Goes beyond simple search to provide comprehensive market understanding.
    """
    
    def __init__(self):
        self.supabase = supabase_client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.brave_key = BRAVE_API_KEY
        
        # Initialize sub-services
        self.decomposition_service = QueryDecompositionService()
        self.competitor_service = CompetitorDiscoveryService()
        self.entity_extraction_service = EntityExtractionService()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds between requests
    
    def analyze_market(
        self,
        query: str,
        max_results_per_dimension: int = 8,
        include_ai_insights: bool = True
    ) -> MarketIntelligenceReport:
        """
        Perform comprehensive market intelligence analysis
        
        Args:
            query: Natural language description of the company/space
            max_results_per_dimension: Max articles per dimension
            include_ai_insights: Whether to generate AI-powered insights
            
        Returns:
            MarketIntelligenceReport with full analysis
        """
        logger.info(f"Starting market intelligence analysis for: {query[:100]}...")
        
        # Step 1: Decompose query
        components = self.decomposition_service.decompose_query(query)
        search_plan = self.decomposition_service.generate_search_plan(components)
        
        logger.info(f"Query decomposed into {len(components.search_dimensions)} dimensions")
        
        # Step 2: Execute multi-dimensional search
        dimension_results = self._execute_dimensional_search(
            components,
            search_plan,
            max_results_per_dimension
        )
        
        # Step 3: Discover competitors
        competitive_results = self._perform_competitive_analysis(
            components,
            max_results_per_dimension
        )
        
        # Step 4: Generate aggregated insights
        executive_summary = ""
        key_players = []
        emerging_trends = []
        recommended_steps = []
        
        if include_ai_insights and self.openai_client:
            insights = self._generate_aggregated_insights(
                query,
                components,
                dimension_results,
                competitive_results
            )
            executive_summary = insights.get('executive_summary', '')
            key_players = insights.get('key_players', [])
            emerging_trends = insights.get('emerging_trends', [])
            recommended_steps = insights.get('recommended_next_steps', [])
        
        # Step 5: Compile report
        report = MarketIntelligenceReport(
            query=query,
            generated_at=datetime.now().isoformat(),
            components=components,
            search_plan=search_plan,
            companies_landscape=dimension_results.get('companies', self._empty_dimension('companies')),
            technology_landscape=dimension_results.get('technology', self._empty_dimension('technology')),
            business_models=dimension_results.get('business_models', self._empty_dimension('business_models')),
            innovations=dimension_results.get('innovations', self._empty_dimension('innovations')),
            market_trends=dimension_results.get('market_trends', self._empty_dimension('market_trends')),
            competitive_analysis=competitive_results,
            executive_summary=executive_summary,
            key_players=key_players,
            emerging_trends=emerging_trends,
            recommended_next_steps=recommended_steps
        )
        
        logger.info("Market intelligence analysis complete")
        return report
    
    def _execute_dimensional_search(
        self,
        components: QueryComponents,
        search_plan: Dict,
        max_results: int
    ) -> Dict[str, DimensionResults]:
        """Execute searches for all dimensions in parallel"""
        
        results = {}
        dimensions = components.search_dimensions
        
        for dimension, queries in dimensions.items():
            if dimension == 'competitive':
                # Skip competitive here, handled separately
                continue
                
            logger.info(f"Searching dimension: {dimension}")
            
            articles = []
            companies_mentioned = set()
            technologies_mentioned = set()
            
            # Execute queries for this dimension
            for query in queries[:3]:  # Limit queries per dimension
                # Search web results
                search_results = self._brave_search(query, num_results=5)
                
                for result in search_results:
                    title = result.get('title', '')
                    description = result.get('description', '')
                    text = f"{title} {description}".lower()

                    articles.append({
                        'title': title,
                        'url': result.get('url', ''),
                        'description': description,
                        'query': query,
                        'source': 'web'
                    })

                    # Extract mentions (simple approach)
                    # Extract company names (basic pattern matching)
                    for word in text.split():
                        if word and word[0].isupper() and len(word) > 3:
                            companies_mentioned.add(word)

                    # Extract tech mentions
                    for tech in components.technologies:
                        if tech.lower() in text:
                            technologies_mentioned.add(tech)
                
                # Search Product Hunt products for this query
                ph_products = self._search_product_hunt_for_dimension(query, max_results=3)
                for product in ph_products:
                    title = product.get('product_name', '')
                    description = product.get('overview', '')
                    text = f"{title} {description}".lower()

                    articles.append({
                        'title': title,
                        'url': product.get('product_link', ''),
                        'description': description,
                        'query': query,
                        'source': 'product_hunt',
                        'producthunt_link': product.get('producthunt_link', ''),
                        'weight_boost': product.get('weight_boost', 1.5)
                    })

                    # Extract mentions from Product Hunt products
                    # Extract company names
                    for word in text.split():
                        if word and word[0].isupper() and len(word) > 3:
                            companies_mentioned.add(word)

                    # Extract tech mentions
                    for tech in components.technologies:
                        if tech.lower() in text:
                            technologies_mentioned.add(tech)
                
                time.sleep(self.min_request_interval)  # Rate limiting
            
            # Extract technology usage details with query context
            technology_usage = self._extract_technology_usage_details(articles, components)

            # Extract business model details with query context
            business_models, company_business_models = self._extract_business_model_details(articles, components)

            # Extract market insights with query context
            market_insights = self._extract_market_insights(articles, components)

            # Generate key findings for this dimension
            key_findings = self._extract_key_findings(articles, dimension)

            results[dimension] = DimensionResults(
                dimension=dimension,
                description=search_plan['dimensions'][dimension]['description'],
                queries_performed=queries[:3],
                articles=articles[:max_results],
                key_findings=key_findings,
                companies_mentioned=list(companies_mentioned)[:10],
                technologies_mentioned=list(technologies_mentioned),
                technology_usage_details=technology_usage,
                business_model_details=business_models[:5],  # Limit to top 5
                market_insights=market_insights[:3],  # Limit to top 3
                company_business_models=company_business_models
            )
        
        return results
    
    def _search_product_hunt_for_dimension(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Product Hunt products for a specific query/dimension"""
        if not self.openai_client:
            logger.warning("OpenAI client not available, skipping Product Hunt search")
            return []

        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []

            # Search both embedding types
            desc_results = self._search_product_embeddings(query_embedding, 'embedding', max_results)
            keyword_results = self._search_product_embeddings(query_embedding, 'keyword_embedding', max_results)

            # Merge and deduplicate results
            combined = self._merge_product_results(desc_results, keyword_results)

            # Add source type and weight boost
            for result in combined:
                result['source_type'] = 'product_hunt'
                result['weight_boost'] = 1.5

            logger.info(f"Found {len(combined)} Product Hunt products for query: '{query}'")
            return combined
            
        except Exception as e:
            logger.error(f"Product Hunt search failed for dimension: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for search query"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _search_product_embeddings(self, query_embedding: List[float],
                                 embedding_column: str, limit: int) -> List[Dict]:
        """Search specific embedding column in Product Hunt table"""
        try:
            import numpy as np
            
            # Fetch products with embeddings
            query = self.supabase.table('product_hunt_products').select(
                f'id, product_name, overview, description, product_link, producthunt_link, Business, Tech, {embedding_column}'
            ).not_.is_(embedding_column, 'null')

            response = query.execute()
            
            if not response.data:
                return []
            
            # Calculate cosine similarity in Python
            query_vec = np.array(query_embedding, dtype=np.float32)
            products_with_similarity = []
            
            for row in response.data:
                product_embedding = row.get(embedding_column)
                if not product_embedding:
                    continue
                
                try:
                    # Parse embedding
                    if isinstance(product_embedding, str):
                        import json
                        product_embedding = json.loads(product_embedding)
                    
                    product_vec = np.array(product_embedding, dtype=np.float32)
                    
                    if len(product_vec) != len(query_vec):
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, product_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(product_vec)
                    )
                    
                    products_with_similarity.append({
                        'id': row.get('id'),
                        'product_name': row.get('product_name', ''),
                        'overview': row.get('overview', ''),
                        'description': row.get('description', ''),
                        'product_link': row.get('product_link', ''),
                        'producthunt_link': row.get('producthunt_link', ''),
                        'similarity': float(similarity)
                    })
                    
                except Exception as e:
                    logger.debug(f"Skipping product due to error: {e}")
                    continue
            
            # Sort by similarity and return top results
            products_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)
            return products_with_similarity[:limit]
            
        except Exception as e:
            logger.error(f"Product Hunt embedding search failed: {e}")
            return []
    
    def _merge_product_results(self, desc_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Merge description and keyword results, removing duplicates"""
        seen_products = set()
        merged = []
        
        # Add description results first
        for result in desc_results:
            product_id = result.get('id')
            if product_id and product_id not in seen_products:
                result['search_type'] = 'description'
                merged.append(result)
                seen_products.add(product_id)
        
        # Add keyword results
        for result in keyword_results:
            product_id = result.get('id')
            if product_id and product_id not in seen_products:
                result['search_type'] = 'keyword'
                merged.append(result)
                seen_products.add(product_id)
        
        return merged
    
    def _perform_competitive_analysis(
        self,
        components: QueryComponents,
        max_results: int
    ) -> DimensionResults:
        """Perform competitive analysis using competitor discovery"""
        
        logger.info("Performing competitive analysis")
        
        # Use competitor discovery service
        discovered_competitors = self.competitor_service.discover_competitors(
            keywords=components.keywords,
            industries=components.industries,
            existing_competitors=set(),
            config=None
        )
        
        # Also do web searches for competitive intelligence
        articles = []
        competitive_queries = components.search_dimensions.get('competitive', [])
        
        for query in competitive_queries[:2]:
            search_results = self._brave_search(query, num_results=5)
            for result in search_results:
                articles.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'description': result.get('description', ''),
                    'query': query
                })
            time.sleep(self.min_request_interval)
        
        # Compile competitive findings
        key_findings = [
            f"Discovered {len(discovered_competitors)} potential competitors",
            f"Analyzed {len(articles)} competitive articles"
        ]
        
        companies_mentioned = [comp.name for comp in discovered_competitors[:15]]
        
        return DimensionResults(
            dimension='competitive',
            description='Competitive landscape and competitor identification',
            queries_performed=competitive_queries[:2],
            articles=articles[:max_results],
            key_findings=key_findings,
            companies_mentioned=companies_mentioned,
            technologies_mentioned=components.technologies,
            technology_usage_details={},
            business_model_details=[],
            market_insights=[],
            company_business_models={}
        )
    
    def _brave_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform Brave search (with rate limiting)"""
        if not self.brave_key:
            return []
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_key
        }
        params = {
            "q": query,
            "count": min(num_results, 10),
            "result_filter": "web"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            self.last_request_time = time.time()
            return data.get("web", {}).get("results", [])
            
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            self.last_request_time = time.time()
            return []
    
    def _extract_key_findings(self, articles: List[Dict], dimension: str) -> List[str]:
        """Extract key findings from articles for a dimension"""

        if not articles:
            return [f"No articles found for {dimension}"]

        findings = []

        # If no specific findings, fall back to basic analysis
        if not findings:
            # Count mentions in titles and descriptions
            content_words = {}
            for article in articles:
                content = f"{article.get('title', '')} {article.get('description', '')}".lower()
                words = [w for w in re.findall(r'\b\w{4,}\b', content) if len(w) > 3]
                for word in words:
                    content_words[word] = content_words.get(word, 0) + 1

            # Top mentioned terms
            if content_words:
                top_terms = sorted(content_words.items(), key=lambda x: x[1], reverse=True)[:3]
                findings.append(f"Key topics: {', '.join([t[0] for t in top_terms])}")

        # Always include article count
        findings.append(f"Found {len(articles)} relevant articles")

        return findings

    def _extract_technology_usage_details(self, articles: List[Dict], components: QueryComponents) -> Dict[str, List[str]]:
        """Extract detailed technology usage information from articles using AI"""
        if not self.openai_client or not articles:
            return {}

        # Compile article text for AI analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}"
            for article in articles[:10]  # Limit to top 10 articles
        ])

        # Build context filter from query components
        context_filter = []
        if components.domain:
            context_filter.extend(components.domain[:2])
        if components.keywords:
            context_filter.extend([kw for kw in components.keywords[:3] if len(kw) > 4])
        if components.technologies:
            context_filter.extend(components.technologies[:2])

        context_str = ', '.join(context_filter[:5])

        prompt = f"""Analyze these articles and extract SPECIFIC technology usage information RELEVANT to: {context_str}

Articles:
{articles_text}

CRITICAL FILTERING:
- ONLY extract technologies that are DIRECTLY RELATED to: {context_str}
- IGNORE generic AI/SaaS products unless they specifically serve the domain: {', '.join(components.domain) if components.domain else 'specified domain'}
- FOCUS on technologies that solve problems for: {', '.join(components.keywords[:3]) if components.keywords else 'the target use case'}

Extract and return ONLY valid JSON:
{{
  "technologies": [
    {{
      "name": "Technology Name (must be relevant to {context_str})",
      "usage_examples": ["Concrete usage example for {context_str}", "Another specific example"],
      "companies_using": ["Company 1", "Company 2"],
      "innovation_details": "Specific innovation details",
      "relevance_check": "Explain why this is relevant to {context_str}"
    }}
  ]
}}

STRICT REQUIREMENTS:
- Technologies MUST be related to the query context: {context_str}
- If a technology is generic (e.g., "AI invoicing", "SaaS starter kit"), SKIP it unless it specifically serves the target domain
- Return EMPTY list if no relevant technologies found"""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a technology analyst extracting usage patterns. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Convert to the expected format
            technology_usage = {}
            for tech in result.get('technologies', []):
                tech_name = tech.get('name', '')
                usage_examples = tech.get('usage_examples', [])
                if tech_name and usage_examples:
                    technology_usage[tech_name] = usage_examples
            
            return technology_usage
            
        except Exception as e:
            logger.error(f"AI technology extraction failed: {e}")
            return {}

    def _extract_business_model_details(self, articles: List[Dict], components: QueryComponents) -> tuple[List[str], Dict[str, List[str]]]:
        """Extract business model details using AI to discover NEW and INNOVATIVE models"""
        if not self.openai_client or not articles:
            return [], {}

        # Compile article text for AI analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}\nURL: {article.get('url', '')}"
            for article in articles[:10]  # Limit to top 10 articles
        ])

        # Build context filter
        context_filter = []
        if components.domain:
            context_filter.extend(components.domain[:2])
        if components.keywords:
            context_filter.extend([kw for kw in components.keywords[:3] if len(kw) > 4])

        context_str = ', '.join(context_filter[:5])

        prompt = f"""Analyze these articles and extract SPECIFIC business model information RELEVANT to: {context_str}

Articles:
{articles_text}

CRITICAL FILTERING:
- ONLY extract business models that are DIRECTLY RELATED to: {context_str}
- IGNORE generic subscription/SaaS models unless they specifically serve: {', '.join(components.domain) if components.domain else 'specified domain'}
- FOCUS on business models for companies serving: {', '.join(components.keywords[:3]) if components.keywords else 'the target market'}

Extract and return ONLY valid JSON:
{{
  "business_models": [
    {{
      "name": "Specific Business Model (must be relevant to {context_str})",
      "description": "Detailed business model explanation for {context_str}",
      "companies": ["Company 1", "Company 2"],
      "innovation_level": "new/emerging/innovative/traditional",
      "specific_approach": "Concrete innovation details",
      "relevance_check": "Explain why this is relevant to {context_str}"
    }}
  ]
}}

STRICT REQUIREMENTS:
- Business models MUST serve the target domain: {context_str}
- If a business model is generic (e.g., "subscription-based fintech"), SKIP it unless it specifically addresses the target use case
- Return EMPTY list if no relevant business models found"""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a business model analyst discovering innovative approaches. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Convert to the expected format
            business_models = []
            company_business_models = {}

            for model in result.get('business_models', []):
                model_name = model.get('name', '')
                model_description = model.get('description', '')
                companies = model.get('companies', [])

                if model_name and model_description:
                    # Create descriptive business model entry without innovation labels
                    full_description = f"{model_name}: {model_description}"

                    business_models.append(full_description)

                    # Associate companies with this business model
                    for company in companies:
                        if company not in company_business_models:
                            company_business_models[company] = []
                        if model_name not in company_business_models[company]:
                            company_business_models[company].append(model_name)
            
            return business_models, company_business_models
            
        except Exception as e:
            logger.error(f"AI business model extraction failed: {e}")
            return [], {}

    def _extract_market_insights(self, articles: List[Dict], components: QueryComponents) -> List[str]:
        """Extract market insights from articles using AI"""
        if not self.openai_client or not articles:
            return []

        # Compile article text for AI analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}"
            for article in articles[:10]  # Limit to top 10 articles
        ])

        # Build context filter
        context_filter = []
        if components.domain:
            context_filter.extend(components.domain[:2])
        if components.keywords:
            context_filter.extend([kw for kw in components.keywords[:3] if len(kw) > 4])

        context_str = ', '.join(context_filter[:5])

        prompt = f"""Analyze these articles and extract SPECIFIC market insights RELEVANT to: {context_str}

Articles:
{articles_text}

CRITICAL FILTERING:
- ONLY extract market insights that are DIRECTLY RELATED to: {context_str}
- IGNORE generic AI/technology trends unless they specifically impact: {', '.join(components.domain) if components.domain else 'specified domain'}
- FOCUS on insights for companies/trends in: {', '.join(components.keywords[:3]) if components.keywords else 'the target market'}

Extract and return ONLY valid JSON:
{{
  "market_insights": [
    "Specific market insight about {context_str} with concrete details",
    "Another trend affecting {context_str} with specific companies"
  ],
  "innovative_approaches": [
    "Specific innovative approach for {context_str}",
    "Another strategy specifically for {context_str}"
  ],
  "emerging_players": [
    "Specific emerging company in {context_str}",
    "Another new player serving {context_str}"
  ]
}}

STRICT REQUIREMENTS:
- Insights MUST be related to: {context_str}
- If an insight is generic (e.g., "AI funding doubled"), SKIP it unless it directly impacts the target domain
- Return EMPTY lists if no relevant insights found"""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a market analyst extracting insights. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Combine all insights into a single list
            all_insights = []
            all_insights.extend(result.get('market_insights', []))
            all_insights.extend(result.get('innovative_approaches', []))
            all_insights.extend(result.get('emerging_players', []))
            
            return all_insights
            
        except Exception as e:
            logger.error(f"AI market insights extraction failed: {e}")
            return []


    def _generate_aggregated_insights(
        self,
        query: str,
        components: QueryComponents,
        dimension_results: Dict[str, DimensionResults],
        competitive_results: DimensionResults
    ) -> Dict:
        """Generate AI-powered aggregated insights across all dimensions"""
        
        if not self.openai_client:
            return {}
        
        # Compile context from all dimensions
        context = f"""
Original Query: {query}

Domain: {', '.join(components.domain)}
Technologies: {', '.join(components.technologies)}
Problem Solved: {components.problem_solved}

Companies Found: {', '.join(competitive_results.companies_mentioned[:10])}

Key Findings by Dimension:
"""
        
        for dim_name, dim_result in dimension_results.items():
            context += f"\n{dim_name.upper()}:\n"
            context += f"- {'. '.join(dim_result.key_findings[:3])}\n"
        
        prompt = f"""{context}

Based on this market intelligence analysis, provide SPECIFIC insights about innovative approaches and emerging players:

1. EXECUTIVE SUMMARY (2-3 sentences): High-level overview highlighting SPECIFIC innovative approaches discovered
2. KEY PLAYERS (list 5-7 companies with SPECIFIC descriptions of their innovative approaches)
3. EMERGING TRENDS (list 3-5 SPECIFIC trends with concrete examples of innovative approaches)
4. RECOMMENDED NEXT STEPS (list 3-4 specific things to research deeper)

CRITICAL REQUIREMENTS:
- If you mention "innovative approaches" or "emerging players", you MUST provide SPECIFIC details about what these approaches are
- Extract concrete examples of HOW companies are innovating
- Provide specific company names and their unique strategies
- If you see vague terms, dig deeper to find the actual approaches
- Do NOT use generic phrases like "innovative approaches" without explaining what they are
- Be specific about technologies, business models, and strategies mentioned

Return ONLY valid JSON:
{{
  "executive_summary": "Market overview highlighting SPECIFIC innovative approaches and concrete examples",
  "key_players": [
    {{"name": "Specific Company Name", "description": "Specific innovative approach they use with concrete details"}},
    ...
  ],
  "emerging_trends": ["Specific innovative approach with concrete examples", "Another specific trend with details", ...],
  "recommended_next_steps": [
    "Deep dive on [specific innovative approach/company]",
    "Research [specific innovative strategy]",
    ...
  ]
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a market research analyst providing executive insights. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            import json
            insights = json.loads(response.choices[0].message.content.strip())
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {
                'executive_summary': 'Analysis complete. See detailed findings below.',
                'key_players': [],
                'emerging_trends': [],
                'recommended_next_steps': []
            }
    
    def _empty_dimension(self, dimension: str) -> DimensionResults:
        """Create an empty dimension result"""
        return DimensionResults(
            dimension=dimension,
            description='',
            queries_performed=[],
            articles=[],
            key_findings=[],
            companies_mentioned=[],
            technologies_mentioned=[],
            technology_usage_details={},
            business_model_details=[],
            market_insights=[],
            company_business_models={}
        )

