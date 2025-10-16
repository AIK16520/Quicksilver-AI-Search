# market_intelligence.py

"""
Multi-dimensional market intelligence service.
Performs comprehensive analysis across multiple dimensions: companies, tech, business models, trends.
"""

import logging
import requests
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
                search_results = self._brave_search(query, num_results=5)
                
                for result in search_results:
                    articles.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'description': result.get('description', ''),
                        'query': query
                    })
                    
                    # Extract mentions (simple approach)
                    text = f"{result.get('title', '')} {result.get('description', '')}".lower()
                    
                    # Extract company names (basic pattern matching)
                    for word in text.split():
                        if word and word[0].isupper() and len(word) > 3:
                            companies_mentioned.add(word)
                    
                    # Extract tech mentions
                    for tech in components.technologies:
                        if tech.lower() in text:
                            technologies_mentioned.add(tech)
                
                time.sleep(self.min_request_interval)  # Rate limiting
            
            # Generate key findings for this dimension
            key_findings = self._extract_key_findings(articles, dimension)
            
            results[dimension] = DimensionResults(
                dimension=dimension,
                description=search_plan['dimensions'][dimension]['description'],
                queries_performed=queries[:3],
                articles=articles[:max_results],
                key_findings=key_findings,
                companies_mentioned=list(companies_mentioned)[:10],
                technologies_mentioned=list(technologies_mentioned)
            )
        
        return results
    
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
            technologies_mentioned=components.technologies
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
        
        # Count mentions in titles
        title_words = {}
        for article in articles:
            title = article.get('title', '').lower()
            words = [w for w in title.split() if len(w) > 4]
            for word in words:
                title_words[word] = title_words.get(word, 0) + 1
        
        # Top mentioned terms
        if title_words:
            top_terms = sorted(title_words.items(), key=lambda x: x[1], reverse=True)[:3]
            findings.append(f"Top mentioned: {', '.join([t[0] for t in top_terms])}")
        
        findings.append(f"Found {len(articles)} relevant articles")
        
        return findings
    
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

Based on this market intelligence analysis, provide:

1. EXECUTIVE SUMMARY (2-3 sentences): High-level overview of the space and market dynamics
2. KEY PLAYERS (list 5-7 SOFTWARE/PLATFORM companies with brief descriptions of what they do)
3. EMERGING TRENDS (list 3-5 key trends or innovations in the market)
4. RECOMMENDED NEXT STEPS (list 3-4 specific things to research deeper)

CRITICAL RULES:
- Focus on PROVIDER companies (those that make/sell the product), NOT customer companies (those that use it)
- Examples:
  * Query about "hedge fund AI analysis" → List software companies like Kensho, AlphaSense (NOT hedge funds)
  * Query about "restaurant ordering systems" → List POS/ordering platforms (NOT restaurant chains)
  * Query about "healthcare diagnostics" → List medical tech companies (NOT hospitals)
- The KEY PLAYERS should be companies that BUILD/PROVIDE the solution
- Do NOT list the companies that are CUSTOMERS/USERS of the solution
- Do NOT mention "portfolio companies" or make investment recommendations
- Do NOT suggest "following up with companies" - this is pure market research
- Be specific about company names, technologies, and trends
- When mentioning partnerships, name the specific companies involved

Return ONLY valid JSON:
{{
  "executive_summary": "Market overview without portfolio references",
  "key_players": [
    {{"name": "Specific Company Name", "description": "What they actually do"}},
    ...
  ],
  "emerging_trends": ["Specific trend with examples", "trend 2", ...],
  "recommended_next_steps": [
    "Deep dive on [specific company/technology]",
    "Research [specific topic]",
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
            technologies_mentioned=[]
        )

