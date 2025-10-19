# query_decomposition.py

"""
Intelligent query decomposition for market intelligence searches.
Breaks down complex queries into searchable components and dimensions.
"""

import logging
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from openai import OpenAI
from core.config import OPENAI_API_KEY, GPT_MODEL

logger = logging.getLogger("query_decomposition")

@dataclass
class QueryComponents:
    """Structured representation of query components"""
    # Core business elements
    domain: List[str]  # Industry/sector (e.g., "hedge funds", "fintech")
    problem_solved: str  # What problem is being addressed
    value_proposition: str  # Key value delivered
    
    # Technology elements
    technologies: List[str]  # Tech stack mentioned (AI, agents, scraping)
    data_sources: List[str]  # Data inputs (market data, news, pricing)
    
    # Market intelligence dimensions
    search_dimensions: Dict[str, List[str]]  # Organized search queries
    
    # Keywords for discovery
    keywords: List[str]
    industries: List[str]


class QueryDecompositionService:
    """
    Decomposes complex market intelligence queries into structured components
    for comprehensive multi-dimensional search
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
    def decompose_query(self, query: str) -> QueryComponents:
        """
        Decompose a complex query into structured components
        
        Args:
            query: Natural language query describing a company/space
            
        Returns:
            QueryComponents with all extracted elements
        """
        logger.info(f"Decomposing query: {query[:100]}...")
        
        if self.openai_client:
            return self._ai_decomposition(query)
        else:
            return self._rule_based_decomposition(query)
    
    def _ai_decomposition(self, query: str) -> QueryComponents:
        """Use AI to intelligently decompose the query"""
        
        prompt = f"""Analyze this company/market description and extract structured information for market discovery:

Query: {query}

FOCUS ON MARKET DISCOVERY: Extract information to discover new and emerging companies, technologies, and business models in the market.

IMPORTANT: Distinguish between PROVIDER and CUSTOMER industries.
- If the query says "company that does X for Y" → Y is the customer, X is what the company does
- Domain should describe the PROVIDER company's industry, NOT the customer's industry
- Target customers should be WHO USES the product/service

CRITICAL: For FINANCIAL SERVICES queries, ALWAYS identify them correctly:

1. "hedge fund and trade analysis using AI agents"
   - Domain: "hedge fund technology", "financial analytics", "trading platforms", "fintech"
   - Target customers: "hedge funds", "trading firms", "investment managers"
   - Technologies: "AI agents", "data scraping", "market data", "financial news", "automation"
   - Keywords: "hedge fund analysis", "trade analysis", "market data scraping", "financial news monitoring", "automated updates"

2. "AI platform for investment management"
   - Domain: "financial technology", "investment management", "fintech"
   - Target customers: "investment firms", "asset managers", "financial advisors"
   - Technologies: "AI", "machine learning", "portfolio optimization"

3. "trading data analysis tools"
   - Domain: "financial data analytics", "trading technology"
   - Target customers: "trading desks", "investment banks", "hedge funds"
   - Technologies: "data analysis", "real-time processing", "financial APIs"

Extract and return ONLY valid JSON with this EXACT structure:
{{
  "domain": ["provider industry 1", "provider industry 2"],
  "technologies": ["tech1", "tech2", "tech3"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "target_customers": ["who uses this product/service"]
}}

Guidelines:
- Domain: The PROVIDER company's industry (what category of company they are)
- Technologies: Specific tech mentioned in the query
- Keywords: Features, capabilities, and market terms mentioned
- Target customers: Who BUYS/USES the product (these are NOT the companies we want to find)

Focus on discovering NEW companies, technologies, and business models in this space."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a market research analyst expert at extracting structured information from company descriptions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Generate search dimensions based on extracted components
            search_dimensions = self._generate_search_dimensions(result)
            
            return QueryComponents(
                domain=result.get('domain', []),
                problem_solved="",  # Not needed for market discovery
                value_proposition="",  # Not needed for market discovery
                technologies=result.get('technologies', []),
                data_sources=[],  # Not needed for market discovery
                search_dimensions=search_dimensions,
                keywords=result.get('keywords', []),
                industries=result.get('domain', [])  # Use domain as industries
            )
            
        except Exception as e:
            logger.error(f"AI decomposition failed: {e}")
            return self._rule_based_decomposition(query)
    
    def _rule_based_decomposition(self, query: str) -> QueryComponents:
        """Fallback rule-based decomposition if AI is unavailable - optimized for financial services"""

        # Financial services specific keywords
        tech_keywords = ['AI', 'ML', 'machine learning', 'agent', 'automation', 'scraping', 'NLP', 'neural network', 'data analysis', 'real-time processing']
        domain_keywords = ['hedge fund', 'trading', 'financial analytics', 'fintech', 'investment management', 'asset management', 'financial technology']
        data_keywords = ['market data', 'pricing', 'financial news', 'real-time data', 'trading data', 'financial data', 'portfolio data']
        customer_keywords = ['hedge funds', 'trading firms', 'investment managers', 'asset managers', 'financial institutions', 'investment banks']

        query_lower = query.lower()
        
        technologies = [kw for kw in tech_keywords if kw.lower() in query_lower]
        domain = [kw for kw in domain_keywords if kw.lower() in query_lower]
        data_sources = [kw for kw in data_keywords if kw.lower() in query_lower]
        target_customers = [kw for kw in customer_keywords if kw.lower() in query_lower]
        
        # Extract potential keywords (simple word extraction)
        words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        keywords = list(set(words))[:10]
        
        search_dimensions = self._generate_search_dimensions({
            'domain': domain,
            'technologies': technologies,
            'data_sources': data_sources,
            'keywords': keywords,
            'target_customers': target_customers
        })
        
        return QueryComponents(
            domain=domain,
            problem_solved="",  # Not needed for market discovery
            value_proposition="",  # Not needed for market discovery
            technologies=technologies,
            data_sources=[],  # Not needed for market discovery
            search_dimensions=search_dimensions,
            keywords=keywords,
            industries=domain
        )
    
    def _generate_search_dimensions(self, components: Dict) -> Dict[str, List[str]]:
        """
        Generate targeted search queries for market discovery across multiple dimensions
        Focus on discovering NEW companies, technologies, and business models
        """
        dimensions = {}
        
        domain = components.get('domain', [])
        technologies = components.get('technologies', [])
        keywords = components.get('keywords', [])
        target_customers = components.get('target_customers', [])
        
        # Dimension 1: Emerging Companies & Startups
        dimensions['companies'] = []
        
        if domain:
            for d in domain[:2]:
                dimensions['companies'].extend([
                    f"new startups in {d}",
                    f"emerging companies {d}",
                    f"recent {d} companies launched",
                    f"innovative {d} solutions"
                ])
        
        if target_customers:
            for customer in target_customers[:2]:
                dimensions['companies'].extend([
                    f"new companies serving {customer}",
                    f"startups for {customer}",
                    f"innovative solutions for {customer}"
                ])
        
        if technologies and domain:
            dimensions['companies'].extend([
                f"new {technologies[0]} companies in {domain[0]}",
                f"startups using {technologies[0]} for {domain[0]}"
            ])
        
        # Dimension 2: Technology Innovations
        dimensions['technology'] = []
        if technologies:
            for tech in technologies[:3]:
                dimensions['technology'].extend([
                    f"new {tech} innovations",
                    f"emerging {tech} technologies",
                    f"latest {tech} developments",
                    f"cutting-edge {tech} solutions"
                ])
        
        if domain:
            dimensions['technology'].extend([
                f"new technologies in {domain[0]}",
                f"innovative tech solutions for {domain[0]}",
                f"emerging tech trends in {domain[0]}"
            ])
        
        # Dimension 3: Business Model Innovation
        dimensions['business_models'] = []
        if domain:
            dimensions['business_models'].extend([
                f"new business models in {domain[0]}",
                f"innovative monetization {domain[0]}",
                f"emerging revenue models {domain[0]}",
                f"disruptive business models {domain[0]}"
            ])
        
        if keywords:
            dimensions['business_models'].extend([
                f"new business models for {' '.join(keywords[:2])}",
                f"innovative pricing {' '.join(keywords[:2])}"
            ])
        
        # Dimension 4: Market Innovations & Trends
        dimensions['innovations'] = []
        if domain:
            dimensions['innovations'].extend([
                f"latest innovations in {domain[0]}",
                f"new approaches to {domain[0]}",
                f"disruptive innovations {domain[0]}",
                f"breakthrough solutions {domain[0]}"
            ])
        
        if technologies:
            dimensions['innovations'].extend([
                f"innovative applications of {technologies[0]}",
                f"new use cases for {technologies[0]}"
            ])
        
        # Dimension 5: Market Trends & Opportunities
        dimensions['market_trends'] = []
        if domain:
            dimensions['market_trends'].extend([
                f"emerging trends in {domain[0]}",
                f"new opportunities in {domain[0]}",
                f"market gaps in {domain[0]}",
                f"future of {domain[0]}"
            ])
        
        if target_customers:
            dimensions['market_trends'].extend([
                f"new trends affecting {target_customers[0]}",
                f"emerging needs of {target_customers[0]}"
            ])
        
        # Dimension 6: Competitive Landscape Discovery
        dimensions['competitive'] = []
        if keywords:
            dimensions['competitive'].extend([
                f"new players in {' '.join(keywords[:2])}",
                f"emerging competitors {' '.join(keywords[:2])}",
                f"recent entrants {' '.join(keywords[:2])}"
            ])
        
        if domain:
            dimensions['competitive'].extend([
                f"new competitors in {domain[0]}",
                f"emerging players {domain[0]}"
            ])
        
        return dimensions
    
    def generate_search_plan(self, components: QueryComponents) -> Dict:
        """
        Generate a comprehensive search plan based on decomposed components
        
        Returns:
            Dictionary with search strategy and prioritized queries
        """
        plan = {
            'summary': {
                'domain': ', '.join(components.domain),
                'key_technologies': ', '.join(components.technologies[:5]),
                'search_scope': len(components.search_dimensions),
                'total_queries': sum(len(queries) for queries in components.search_dimensions.values())
            },
            'dimensions': {},
            'priority_order': [
                'companies',
                'competitive', 
                'technology',
                'innovations',
                'business_models',
                'market_trends'
            ]
        }
        
        # Organize search queries by dimension with descriptions
        dimension_descriptions = {
            'companies': 'Key players, established companies, and emerging startups in the space',
            'technology': 'Technology stack, tools, and platforms being used',
            'business_models': 'Pricing strategies, monetization approaches, and revenue models',
            'innovations': 'Recent innovations, new approaches, and differentiators',
            'market_trends': 'Market dynamics, growth trends, and future outlook',
            'competitive': 'Competitor identification and landscape mapping'
        }
        
        for dimension, queries in components.search_dimensions.items():
            plan['dimensions'][dimension] = {
                'description': dimension_descriptions.get(dimension, ''),
                'queries': queries,
                'priority': plan['priority_order'].index(dimension) if dimension in plan['priority_order'] else 99
            }
        
        return plan

